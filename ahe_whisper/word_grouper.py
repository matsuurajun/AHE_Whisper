# -*- coding: utf-8 -*-
import traceback
from typing import List, Dict, Any
from threading import Lock
import logging

LOGGER = logging.getLogger("ahe_whisper_worker")

_PUNCT_CHARS = set("。、？！?!,.，．・：；…")

try:
    from sudachipy import Dictionary, Tokenizer
except ImportError:
    Dictionary = None
    Tokenizer = None

_sudachi_tokenizer = None
_sudachi_lock = Lock()

def _get_text(w: Dict[str, Any]) -> str:
    text = w.get('text', w.get('word'))
    return "" if text is None else str(text)

def _is_cjk_char(ch: str) -> bool:
    return (
        "\u3040" <= ch <= "\u30FF"
        or "\u4E00" <= ch <= "\u9FFF"
        or "\u3400" <= ch <= "\u4DBF"
        or "\uF900" <= ch <= "\uFAFF"
        or "\u3000" <= ch <= "\u303F"
    )

def _is_punct_text(text: str) -> bool:
    return bool(text) and all(ch in _PUNCT_CHARS for ch in text)

def _is_cjk_text(text: str) -> bool:
    if not text:
        return False
    for ch in text:
        if ch.isspace() or ch in _PUNCT_CHARS:
            return False
        if not (_is_cjk_char(ch) or ch == "ー"):
            return False
    return True

def _get_tokenizer():
    global _sudachi_tokenizer
    if _sudachi_tokenizer is None:
        with _sudachi_lock:
            if _sudachi_tokenizer is None:
                if Dictionary is None:
                    raise ImportError("SudachiPy is not installed but is required for word grouping.")
                _sudachi_tokenizer = Dictionary().create()
    return _sudachi_tokenizer

def group_words_sudachi(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    修正版: 単語のグループ化を行うが、すべての単語を確実に保持する
    """
    # 正規化: text/wordと時間情報を持つ単語のみ
    norm_words = [w for w in words if _get_text(w) and w.get('start') is not None and w.get('end') is not None]
    
    if len(words) != len(norm_words):
        LOGGER.debug(f"Dropped {len(words) - len(norm_words)} word entries lacking text or timestamps.")
    
    # 単語が少ない場合は元のまま返す
    if len(norm_words) < 10:
        LOGGER.warning(f"Too few words ({len(norm_words)}) for grouping, returning original")
        return norm_words

    def _group_words_heuristic(words_in: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        max_gap_sec = 0.12
        max_group_chars = 8
        max_group_sec = 1.5
        max_token_chars = 2

        def _get_prob(w: Dict[str, Any]) -> float:
            for key in ("prob", "confidence", "avg_logprob"):
                val = w.get(key)
                if val is None:
                    continue
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
            return 0.8

        grouped: List[Dict[str, Any]] = []
        cur_words: List[Dict[str, Any]] = []
        cur_text = ""
        cur_start = None
        cur_end = None

        def _flush():
            if not cur_words:
                return
            probs = [_get_prob(w) for w in cur_words]
            grouped.append(
                {
                    "text": cur_text,
                    "start": cur_start,
                    "end": cur_end,
                    "prob": sum(probs) / float(len(probs)),
                    "words": list(cur_words),
                }
            )

        for w in words_in:
            text = _get_text(w).strip()
            if not text:
                continue
            try:
                w_start = float(w.get("start"))
                w_end = float(w.get("end"))
            except (TypeError, ValueError):
                _flush()
                grouped.append(w)
                cur_words = []
                cur_text = ""
                cur_start = None
                cur_end = None
                continue
            if w_end <= w_start:
                continue

            if not cur_words:
                cur_words = [w]
                cur_text = text
                cur_start = w_start
                cur_end = w_end
                continue

            prev = cur_words[-1]
            prev_text = _get_text(prev).strip()
            try:
                prev_end = float(prev.get("end"))
            except (TypeError, ValueError):
                prev_end = None

            gap = None if prev_end is None else (w_start - prev_end)
            if gap is None or gap < 0.0:
                gap = 0.0

            can_merge = (
                gap <= max_gap_sec
                and not _is_punct_text(prev_text)
                and not _is_punct_text(text)
                and _is_cjk_text(prev_text)
                and _is_cjk_text(text)
                and (len(prev_text) <= max_token_chars or len(text) <= max_token_chars)
                and (len(cur_text) + len(text)) <= max_group_chars
                and (w_end - float(cur_start)) <= max_group_sec
            )

            if can_merge:
                cur_words.append(w)
                cur_text += text
                cur_end = w_end
            else:
                _flush()
                cur_words = [w]
                cur_text = text
                cur_start = w_start
                cur_end = w_end

        _flush()
        return grouped if grouped else words_in

    if Tokenizer is None or not words:
        grouped = _group_words_heuristic(norm_words)
        if len(grouped) != len(norm_words):
            LOGGER.info("[WORD-GROUP] heuristic: %d -> %d", len(norm_words), len(grouped))
        return grouped

    # テキスト全体を構築
    full_text = "".join(_get_text(w) for w in norm_words).replace(" ", "")
    
    try:
        tokenizer = _get_tokenizer()
        mode = Tokenizer.SplitMode.C
        morphemes = list(tokenizer.tokenize(full_text, mode))
    except Exception as e:
        LOGGER.error(f"SudachiPy tokenization failed, using heuristic grouping. Error: {e}")
        grouped = _group_words_heuristic(norm_words)
        if len(grouped) != len(norm_words):
            LOGGER.info("[WORD-GROUP] heuristic: %d -> %d", len(norm_words), len(grouped))
        return grouped

    grouped_words = []
    word_idx = 0
    unmatched_start = 0  # マッチしなかった単語の開始位置
    
    for m_idx, m in enumerate(morphemes):
        if word_idx >= len(norm_words):
            break

        m_text = m.surface()
        current_word_group = []
        temp_text = ""
        
        # この形態素に対応する単語を収集
        while word_idx < len(norm_words):
            word_to_add = norm_words[word_idx]
            word_text_stripped = _get_text(word_to_add).strip()
            
            if not m_text.startswith(temp_text + word_text_stripped):
                break

            temp_text += word_text_stripped
            current_word_group.append(word_to_add)
            word_idx += 1
            
            if temp_text == m_text:
                break

        # グループが作成できた場合
        if current_word_group:
            # マッチしなかった単語があれば先に追加
            if unmatched_start < word_idx - len(current_word_group):
                for i in range(unmatched_start, word_idx - len(current_word_group)):
                    grouped_words.append(norm_words[i])
                LOGGER.debug(f"Added {word_idx - len(current_word_group) - unmatched_start} unmatched words")
            
            # グループ化された単語を追加
            start_time = current_word_group[0]['start']
            end_time = current_word_group[-1]['end']
            avg_prob = sum(w.get('prob', 0.8) for w in current_word_group) / len(current_word_group)
            
            grouped_words.append({
                "text": m_text,
                "start": start_time,
                "end": end_time,
                "prob": avg_prob,
                "words": current_word_group
            })
            
            unmatched_start = word_idx
    
    # 残りの単語をすべて追加
    if word_idx < len(norm_words):
        LOGGER.warning(f"Adding {len(norm_words) - word_idx} remaining words that couldn't be grouped")
        grouped_words.extend(norm_words[word_idx:])
    
    # グループ化で単語が大幅に減少した場合は警告して元の単語を返す
    if len(grouped_words) < len(norm_words) * 0.5:
        LOGGER.error(f"Grouping reduced words too much: {len(norm_words)} -> {len(grouped_words)}. Returning original.")
        return norm_words
    
    LOGGER.info(f"Word grouping: {len(norm_words)} -> {len(grouped_words)} words")
    return grouped_words if grouped_words else norm_words
