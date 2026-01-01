# -*- coding: utf-8 -*-
import json
import traceback
import numpy as np
import textwrap
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

from ahe_whisper.config import ExportConfig
import logging

# GUIワーカー(`pipeline_worker.py`)と同じロガーを使う
LOGGER = logging.getLogger("ahe_whisper_worker")

# Sudachi (日本語形態素解析) は存在すれば使う / なければ何もしない
try:
    from sudachipy import dictionary as sudachi_dictionary
    from sudachipy import tokenizer as sudachi_tokenizer
except Exception:
    sudachi_dictionary = None
    sudachi_tokenizer = None

_SUDACHI_TOKENIZER = None
_SUDACHI_MODE = None

_CJK_RE = re.compile(r'[\u3000-\u303F\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]')


def _is_hiragana(ch: str) -> bool:
    return len(ch) == 1 and "\u3040" <= ch <= "\u309F"


# 「文末っぽい」定型：ここで終わっている「。」は基本的に正しいとみなす
SENTENCE_FINAL_PATTERNS: List[str] = [
    "です",
    "ます",
    "でした",
    "ました",
    "だ",
    "だった",
    "である",
    "ません",
    "でしょう",
    "ですよね",
    "ですね",
    "ですよ",
    "だよ",
    "だね",
    "ですが",
    "だけど",
    "けれど",
    "かもしれません",
]

def _get_sudachi_tokenizer():
    """Sudachiトークナイザを遅延初期化する。失敗したら None を返す。"""
    global _SUDACHI_TOKENIZER, _SUDACHI_MODE
    if _SUDACHI_TOKENIZER is not None:
        return _SUDACHI_TOKENIZER
    if sudachi_dictionary is None or sudachi_tokenizer is None:
        LOGGER.warning("Sudachi is not available; cross-segment fix is disabled.")
        return None
    try:
        _SUDACHI_TOKENIZER = sudachi_dictionary.Dictionary().create()
        _SUDACHI_MODE = sudachi_tokenizer.Tokenizer.SplitMode.C
        return _SUDACHI_TOKENIZER
    except Exception as e:
        LOGGER.warning("Failed to initialise Sudachi tokenizer: %s", e)
        _SUDACHI_TOKENIZER = None
        return None

def _wrap_cjk(text: str, width: int) -> str:
    if width <= 0 or not text: return text
    lines, current_line, current_width = [], "", 0
    for char in text:
        char_width = 2 if _CJK_RE.match(char) else 1
        if current_width + char_width > width:
            lines.append(current_line)
            current_line, current_width = char, char_width
        else:
            current_line += char
            current_width += char_width
    if current_line:
        lines.append(current_line)
    return '\n'.join(lines)

def _normalize_segment_text(text: str) -> str:
    """
    - 日本語(CJK)同士の間にだけ入っているスペースを削除する
    - それ以外のスペース（英数字や記号の前後）はそのまま残す
    """
    if not text:
        return text

    chars = list(text)
    n = len(chars)
    out = []

    for i, ch in enumerate(chars):
        if ch == " ":
            prev = None
            nxt = None

            # 左側の直近の非スペース
            j = i - 1
            while j >= 0 and chars[j] == " ":
                j -= 1
            if j >= 0:
                prev = chars[j]

            # 右側の直近の非スペース
            k = i + 1
            while k < n and chars[k] == " ":
                k += 1
            if k < n:
                nxt = chars[k]

            # 直前・直後どちらも CJK なら、このスペースは捨てる
            if prev and nxt and _CJK_RE.match(prev) and _CJK_RE.match(nxt):
                continue

        out.append(ch)

    return "".join(out)

def _tag_words_with_speaker(words: List[Dict[str, Any]], speaker_segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not speaker_segs:
        for word in words:
            word['speaker'] = None
        return words
    
    segs = sorted(speaker_segs, key=lambda s: s['start'])
    seg_starts = [s['start'] for s in segs]
    
    EPS = 1e-6
    
    for word in words:
        start_raw = word.get('start')
        end_raw = word.get('end')
        
        if start_raw is None or end_raw is None:
            word['speaker'] = None
            continue
        
        try:
            start = float(start_raw)
            end = float(end_raw)
        except (TypeError, ValueError):
            word['speaker'] = None
            continue
        
        if end <= start:
            word['speaker'] = None
            continue
        
        word_mid = (start + end) / 2.0
        idx = int(np.searchsorted(seg_starts, word_mid, side='right')) - 1
        
        if 0 <= idx < len(segs):
            seg = segs[idx]
            if (seg['start'] - EPS) <= word_mid <= (seg['end'] + EPS):
                word['speaker'] = seg['speaker']
            else:
                word['speaker'] = None
        else:
            word['speaker'] = None
            
    return words

def _snap_segments_to_word_boundaries(
    segments: List[Dict[str, Any]],
    words: List[Dict[str, Any]],
    max_snap_sec: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    DP などで決まったセグメント境界を、近傍の単語終端 (word['end']) にスナップする。
    - segments: {"start", "end", "speaker", "text"} を含む dict のリスト
    - words   : {"start", "end", "text"} を含む dict のリスト
    """
    if not segments or not words:
        return segments

    # 念のため時間順ソート
    segs = sorted(segments, key=lambda s: float(s.get("start", 0.0)))
    # 単語終端だけを見る（Whisper の word-level timestamp を利用）
    word_ends: List[Tuple[float, bool]] = []
    boundary_marks = ("？", "?", "。", "！", "!")
    for w in words:
        end_raw = w.get("end")
        try:
            t_end = float(end_raw)
        except (TypeError, ValueError):
            continue
        wtxt = (w.get("text") or w.get("word") or "").strip()
        has_mark = wtxt.endswith(boundary_marks)
        word_ends.append((t_end, has_mark))

    if not word_ends:
        return segs

    max_snap = float(max_snap_sec)
    eps = 1e-3

    for i in range(len(segs) - 1):
        left = segs[i]
        right = segs[i + 1]

        try:
            t_boundary = float(left.get("end", 0.0))
            left_start = float(left.get("start", 0.0))
            right_end = float(right.get("end", left_start))
        except (TypeError, ValueError):
            continue

        best_t = None
        best_mark_t = None

        # 「最近傍」ではなく、「境界より前の最後の word.end」にスナップする
        for t_word, has_mark in word_ends:
            # 境界より後ろの単語終端は無視
            if t_word > t_boundary:
                continue
            dist = t_boundary - t_word
            if dist > max_snap:
                continue
            # t_word は t_boundary より前なので、「一番大きい t_word」を選ぶ
            if best_t is None or t_word > best_t:
                best_t = t_word
            if has_mark and (best_mark_t is None or t_word > best_mark_t):
                best_mark_t = t_word

        if best_t is None:
            continue

        new_t = float(best_mark_t if best_mark_t is not None else best_t)

        # 境界が逆転しないように安全チェック
        if new_t <= left_start + eps:
            continue
        if new_t >= right_end - eps:
            continue

        # 実際にスナップ
        left["end"] = new_t
        right["start"] = new_t

    return segs

def _assign_words_to_segments(
    words: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    """
    各単語をちょうど 1 つのセグメントに割り当てる。
      - 時間的に重なっているセグメントのみ候補
      - 重なり時間が最大のセグメントに所属させる
      - 割り当てたセグメントの speaker を word['speaker'] に付与
      - セグメントの text は割り当てられた単語のみから再構成
    """
    if not segments or not words:
        for w in words:
            w["speaker"] = None
        return words, []

    # start 昇順に並べ替えて扱う
    segs = sorted(segments, key=lambda s: float(s.get("start", 0.0)))
    seg_word_buckets: List[List[Dict[str, Any]]] = [[] for _ in segs]
    midpoint_override_ratio = 0.55

    for w in words:
        start_raw = w.get("start")
        end_raw = w.get("end")
        try:
            w_start = float(start_raw)
            w_end = float(end_raw)
        except (TypeError, ValueError):
            w["speaker"] = None
            continue

        if w_end <= w_start:
            w["speaker"] = None
            continue

        best_idx = -1
        best_overlap = 0.0
        candidate_idxs: List[int] = []

        for idx, seg in enumerate(segs):
            try:
                s_start = float(seg.get("start", 0.0))
                s_end = float(seg.get("end", s_start))
            except (TypeError, ValueError):
                continue

            if s_end <= s_start:
                continue

            # 時間的重なり量
            overlap = min(s_end, w_end) - max(s_start, w_start)
            if overlap > best_overlap and overlap > 0.0:
                best_overlap = overlap
                best_idx = idx
            if overlap > 0.0:
                candidate_idxs.append(idx)

        if best_idx >= 0:
            word_dur = w_end - w_start
            mid = (w_start + w_end) / 2.0
            mid_idx = -1
            for idx in candidate_idxs:
                seg = segs[idx]
                try:
                    s_start = float(seg.get("start", 0.0))
                    s_end = float(seg.get("end", s_start))
                except (TypeError, ValueError):
                    continue
                if s_start <= mid <= s_end:
                    mid_idx = idx
                    break
            if mid_idx >= 0 and word_dur > 0.0:
                if (best_overlap / word_dur) < midpoint_override_ratio:
                    best_idx = mid_idx
            seg_word_buckets[best_idx].append(w)
            w["speaker"] = segs[best_idx].get("speaker")
        else:
            w["speaker"] = None

    def _wtext_local(w: Dict[str, Any]) -> str:
        return (w.get("text") or w.get("word") or "").strip()

    new_segments: List[Dict[str, Any]] = []
    for seg, seg_words in zip(segs, seg_word_buckets):
        if not seg_words:
            continue

        text = " ".join(_wtext_local(w) for w in seg_words).strip()
        text = _normalize_segment_text(text)  # ← ここで CJK 間の余計なスペースを除去
        if not text:
            continue

        try:
            s_start = float(seg.get("start", 0.0))
            s_end = float(seg.get("end", s_start))
        except (TypeError, ValueError):
            continue

        speaker_label = seg.get("speaker") or "SPEAKER_00"
        new_segments.append(
            {
                "start": s_start,
                "end": s_end,
                "speaker": speaker_label,
                "text": text,
            }
        )

    return words, new_segments


_CROSS_SEGMENT_PATTERNS: List[Tuple[str, str]] = []


def _fix_cross_segment_fragments(
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Sudachiベースで、セグメント跨ぎの mid-word 分割を保守的に補正する。

    - 時間・speaker は一切変更せず、text のみを変更する。
    - 「...ひらがな。」＋「ひらがな...」の境界のみを基本候補とする。
    - かつ、文末っぽい定型（です・ます・でした 等）で終わる場合は触らない。
    - 上記に加えて、_CROSS_SEGMENT_PATTERNS にあるピンポイントなパターンも補正対象とする。
    """
    if not segments:
        return segments

    tok = _get_sudachi_tokenizer()
    if tok is None:
        # Sudachi が使えない場合は、何もせずそのまま返す
        return segments

    window = 8  # 境界前後で見る文字数

    for i in range(len(segments) - 1):
        cur = segments[i]
        nxt = segments[i + 1]

        # 話者が違う場合は触らない
        if cur.get("speaker") != nxt.get("speaker"):
            continue

        cur_text = (cur.get("text") or "").strip()
        nxt_text = (nxt.get("text") or "").strip()
        if not cur_text or not nxt_text:
            continue

        # 対象は「。で終わる」セグメントのみ
        if not cur_text.endswith("。"):
            continue

        # 末尾の「。」を一旦取った文字列
        base = cur_text[:-1].rstrip()
        if not base:
            continue

        # 文末っぽい定型（です・ました 等）で終わっている場合は正しい句点とみなしてスキップ
        if any(base.endswith(pat) for pat in SENTENCE_FINAL_PATTERNS):
            continue

        last_char = base[-1]
        first_char_next = nxt_text[0]

        # 「ひらがな。[。]ひらがな」以外は基本スキップ。
        # ただし、_CROSS_SEGMENT_PATTERNS に明示的に載っているものは例外として扱う。
        pattern_hit = any(
            cur_text.endswith(suffix) and nxt_text.startswith(prefix)
            for suffix, prefix in _CROSS_SEGMENT_PATTERNS
        )

        if not pattern_hit:
            if not (_is_hiragana(last_char) and _is_hiragana(first_char_next)):
                continue

        # --- Sudachi で「句点を残した場合」と「句点を消した場合」をスコアリング ---
        # 句点あり
        left_keep = cur_text[-window:] + nxt_text[:window]
        # 句点なし
        left_drop = base[-window:] + nxt_text[:window]

        tokens_keep = tok.tokenize(left_keep, _SUDACHI_MODE)
        tokens_drop = tok.tokenize(left_drop, _SUDACHI_MODE)

        def _score(tokens) -> Tuple[float, int]:
            """
            badness が小さい方を「より自然」とみなす。
            - 記号トークン（特に句読点）が多いほど badness を増やす
            - 1 文字ひらがなトークン（「い」「て」単体など）が多いほど badness を増やす
            - tie-breaker としてトークン数も見る
            """
            bad = 0.0
            for t in tokens:
                surf = t.surface()
                pos = t.part_of_speech()
                # 記号（特に句読点）が多いほど悪い
                if surf in ("。", "．", ".", "、", "，") or (pos and pos[0] == "記号"):
                    bad += 0.5
                # 単独ひらがなトークン（「い」「て」など）が連発していると怪しい
                if len(surf) == 1 and _is_hiragana(surf):
                    bad += 0.8
            return bad, len(tokens)

        score_keep = _score(tokens_keep)
        score_drop = _score(tokens_drop)

        # 「句点を消した方」が明らかにシンプルなら、末尾の「。」だけ落とす
        if score_drop < score_keep:
            cur["text"] = base

    return segments

class Exporter:
    def __init__(self, output_dir: str, config: ExportConfig) -> None:
        self.output_dir = Path(output_dir)
        self.config = config

    def _format_time(self, seconds_total: float) -> str:
        try:
            sec = max(0.0, float(seconds_total))
            td = timedelta(seconds=sec)
        except (TypeError, ValueError):
            return "00:00:00,000"
            
        mm, ss = divmod(td.seconds, 60)
        hh, mm = divmod(mm, 60)
        hh += td.days * 24
        ms = td.microseconds // 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
    
    def _format_time_no_ms(self, seconds_total: float) -> str:
        """
        TXT表示用: HH:MM:SS まで（ミリ秒なし）
        """
        try:
            sec = max(0.0, float(seconds_total))
            td = timedelta(seconds=sec)
        except (TypeError, ValueError):
            return "00:00:00"

        mm, ss = divmod(td.seconds, 60)
        hh, mm = divmod(mm, 60)
        hh += td.days * 24
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    
    def _default_serializer(self, o: Any) -> Any:
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.bool_): return bool(o)
        return str(o)
        
    def _wrap_text(self, text: str) -> str:
        width = self.config.srt_max_line_width
        if width <= 0:
            return text
        # 日本語など CJK を含む場合は、自動改行しない
        if _CJK_RE.search(text):
            return text
        # CJK を含まない（英語など）の場合だけ、指定幅で wrap する
        return "\n".join(
            textwrap.wrap(
                text,
                width=width,
                replace_whitespace=False,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )

    def save(self, result: Dict[str, Any], basename: str) -> Path:
        
        LOGGER.info(f"[DEBUG-EXPORT] save() called, keys={list(result.keys())}")
        LOGGER.info(f"[DEBUG-EXPORT] output_dir={self.output_dir}, exists={self.output_dir.exists()}")
        
        safe_basename = re.sub(r'[^\w.\-]+', '_', basename).strip('._') or "run"
            
        run_dir = None
        run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        for i in range(100):
            suffix = "" if i == 0 else f"-{i+1}"
            run_dir_name = f"{run_ts}_{safe_basename}{suffix}"
            candidate_dir = self.output_dir / run_dir_name
            try:
                candidate_dir.mkdir(parents=True, exist_ok=False)
                run_dir = candidate_dir
                break
            except FileExistsError:
                continue
        
        if run_dir is None:
            raise IOError("Could not create a unique output directory after 100 attempts.")
        
        error_log_path = run_dir / "export_error.log"
        words, speaker_segs_raw = result.get('words', []), result.get('speaker_segments', [])
        
        final_segs = []
        if speaker_segs_raw:
            for seg in speaker_segs_raw:
                if isinstance(seg, (list, tuple)) and len(seg) >= 3:
                    start, end, spk_idx = seg[:3]  # 最初の3つだけを取り出す
                elif isinstance(seg, dict):
                    start = seg.get("start", 0.0)
                    end = seg.get("end", start)
                    spk_idx = seg.get("speaker", 0)
                else:
                    continue
                                
                try:
                    start = float(start)
                    end = float(end)
                except (TypeError, ValueError):
                    continue  # skip invalid segment
                
                seg_words: List[Dict[str, Any]] = []
                for w in words:
                    w_start_raw = w.get('start')
                    w_end_raw = w.get('end')
                    
                    if w_start_raw is None or w_end_raw is None:
                        continue
                    
                    try:
                        w_start = float(w_start_raw)
                        w_end = float(w_end_raw)
                    except (TypeError, ValueError):
                        continue
                    
                    if w_end <= w_start:
                        continue
                    
                    # --- 型安全比較 ---
                    if max(0.0, min(end, w_end) - max(start, w_start)) > 0.0:
                        seg_words.append(w)
                
                def _wtext(w):
                    return (w.get('text') or w.get('word') or '').strip()
                
                if text := " ".join(_wtext(w) for w in seg_words).strip():
                    if isinstance(spk_idx, str) and spk_idx.startswith("SPEAKER_"):
                        speaker_label = spk_idx
                    else:
                        try:
                            speaker_label = f"SPEAKER_{int(spk_idx):02d}"
                        except Exception:
                            speaker_label = "SPEAKER_00"
                    final_segs.append({
                        "start": float(start),
                        "end": float(end),
                        "speaker": speaker_label,
                        "text": text
                    })
        
        if words and final_segs:
            valid_words = [w for w in words if w.get("start") is not None and w.get("end") is not None]
            if valid_words:
                w_min = float(min(w["start"] for w in valid_words))
                w_max = float(max(w["end"] for w in valid_words))
                clamped_segs = []
                for seg in final_segs:
                    s = max(float(seg["start"]), w_min)
                    e = min(float(seg["end"]), w_max)
                    if e > s:
                        seg["start"], seg["end"] = s, e
                        clamped_segs.append(seg)
                final_segs = clamped_segs

                # ★ ここで「境界スナップ」を実行：
                #    DP で決まった start/end を、近傍の単語終端に寄せる。
                #    評価用のフレームラベルは変えず、出力用のセグメントだけ整形する。
                if bool(getattr(self.config, "enable_boundary_snap", True)):
                    final_segs = _snap_segments_to_word_boundaries(
                        final_segs,
                        valid_words,
                        max_snap_sec=float(getattr(self.config, "boundary_snap_max_sec", 0.5))
                    )

            # 単語→セグメントを一意に割り当てて text を再構成
            words, final_segs = _assign_words_to_segments(words, final_segs)
        else:
            # セグメントが無い場合でも speaker フィールドだけは整理しておく
            words, final_segs = _assign_words_to_segments(words, final_segs)

        # セグメントまたぎの mid-word 分割を、ごく簡単なパターンだけ軽く補正する
        final_segs = _fix_cross_segment_fragments(final_segs)

        result["words"] = words
        result["speaker_segments"] = final_segs
        
        # --- optional punctuation / truecasing pass ---
        if getattr(self.config, "enable_punctuation", False) and final_segs:
            try:
                from ahe_whisper.punctuator_xlmroberta import get_punctuator
                project_root = getattr(self.config, "project_root", None)
                punctuator = get_punctuator(project_root)
                if punctuator is not None:
                    LOGGER.info("[PUNCT] applying punctuator to %d segments", len(final_segs))
                    final_segs = punctuator.punctuate_segments(final_segs)
                    final_segs = _fix_cross_segment_fragments(final_segs)
                    result["speaker_segments"] = final_segs
            except Exception as e:
                with open(error_log_path, "a", encoding="utf-8") as ef:
                    ef.write(f"Punctuation post-processing failed: {e}\n{traceback.format_exc()}\n")

        try:
            raw_path = run_dir / f"{safe_basename}_raw_output.json"
            with open(raw_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=self._default_serializer)
        except Exception as e:
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(f"Raw JSON export failed: {e}\n{traceback.format_exc()}\n")

        LOGGER.info(f"[DEBUG-EXPORT] final_segs count={len(final_segs)}")

        if not final_segs: return run_dir

        try:
            if "txt" in self.config.output_formats:
                txt_lines: List[str] = []
                for s in final_segs:
                    header = f"[{self._format_time_no_ms(s['start'])}] {s['speaker']}:"
                    body = self._wrap_text(s["text"])
                    txt_lines.append(header)
                    txt_lines.append(body)
                    txt_lines.append("")  # セグメント間を 1 行空ける
                (run_dir / f"{safe_basename}.txt").write_text(
                    "\n".join(txt_lines), "utf-8"
                )
        except Exception as e:
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(f"TXT export failed: {e}\n{traceback.format_exc()}\n")
            
        try:
            if "srt" in self.config.output_formats:
                srt_blocks: List[str] = []
                for i, s in enumerate(final_segs):
                    idx = i + 1
                    t_start = self._format_time(s["start"])  # ミリ秒あり
                    t_end = self._format_time(s["end"])
                    header = f"{idx}\n{t_start} --> {t_end}"
                    # 話者行と本文行を分ける
                    body = f"{s['speaker']}:\n{self._wrap_text(s['text'])}"
                    srt_blocks.append(header + "\n" + body)
                (run_dir / f"{safe_basename}.srt").write_text(
                    "\n\n".join(srt_blocks), "utf-8"
                )
        except Exception as e:
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(f"SRT export failed: {e}\n{traceback.format_exc()}\n")

        try:
            if "json" in self.config.output_formats:
                json_output = {
                    "segments": final_segs,
                    "words": words,
                    "metrics": result.get("metrics", {}),
                }
                with open(run_dir / f"{safe_basename}.json", 'w', encoding='utf-8') as f:
                    json.dump(
                        json_output,
                        f,
                        indent=2,
                        ensure_ascii=False,
                        default=self._default_serializer,
                    )
        except Exception as e:
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(f"Final JSON export failed: {e}\n{traceback.format_exc()}\n")
        
        return run_dir
