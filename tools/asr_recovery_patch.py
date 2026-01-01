#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# --- project root on sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# --- 1) Patch: disable alignment (prevents long-hang for JP long-form) ---
try:
    import mlx_whisper.transcribe as _mlx_transcribe
    import mlx_whisper.timing as _mlx_timing

    def _noop(*args, **kwargs):
        return None

    # 防御的に両方潰す（実装バージョン差異吸収）
    _mlx_transcribe.add_word_timestamps = _noop
    _mlx_timing.add_word_timestamps = _noop
except Exception as e:
    print(f"[WARN] Failed to patch word_timestamps alignment: {e}")

# --- 2) Patch: wrap transcribe() to enforce safer kwargs ---
try:
    import functools
    import mlx_whisper.transcribe as _mlx_transcribe

    _orig_transcribe = _mlx_transcribe  # これが関数本体

    @functools.wraps(_orig_transcribe)
    def _wrapped_transcribe(*args, **kwargs):
        # ここで Whisper 内部VAD / フィルタ設定を調整
        kwargs.setdefault("vad_filter", True)                   # ← 内部VADを保持
        kwargs.setdefault("logprob_threshold", -1.0)
        kwargs.setdefault("compression_ratio_threshold", 10.0)
        return _orig_transcribe(*args, **kwargs)

    # 関数本体を差し替え
    import sys
    sys.modules["mlx_whisper.transcribe"] = _wrapped_transcribe
    print("[INFO] Wrapped mlx_whisper.transcribe() successfully")
except Exception as e:
    print(f"[WARN] Failed to wrap transcribe(): {e}")

# --- 3) AHE pipeline imports ---
from ahe_whisper.config import AppConfig, TranscriptionConfig
from ahe_whisper.pipeline_worker import run_ai_pipeline

# --- 4) utilities ---
def _summarize(result: Dict[str, Any]) -> Dict[str, Any]:
    segs = result.get("segments") or []
    if not segs:
        return {"segments": 0, "start": 0.0, "end": 0.0, "duration": 0.0, "tokens": 0}
    start = segs[0].get("start", 0.0)
    end = segs[-1].get("end", start)
    dur = max(0.0, end - start)
    tokens = sum(len(s.get("text", "").split()) for s in segs)
    return {"segments": len(segs), "start": start, "end": end, "duration": dur, "tokens": tokens}

def _write_outputs(stem: str, result: Dict[str, Any], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    (out_dir / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # TXT（行ごと）
    lines = []
    for i, s in enumerate(result.get("segments") or []):
        lines.append(f"{i+1:04d} [{s.get('start',0):.2f} - {s.get('end',0):.2f}] {s.get('text','').strip()}")
    (out_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    # SRT（簡易）
    def _fmt(t):
        ms = int((t - int(t)) * 1000)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    srt = []
    for i, s in enumerate(result.get("segments") or []):
        srt.append(str(i+1))
        srt.append(f"{_fmt(s.get('start',0))} --> {_fmt(s.get('end',0))}")
        srt.append((s.get("text") or "").strip())
        srt.append("")
    (out_dir / f"{stem}.srt").write_text("\n".join(srt), encoding="utf-8")

# --- 5) recovery: direct whisper call if pipeline empty/too short ---
def _fallback_whisper(audio_path: str, language: str = "ja") -> Dict[str, Any]:
    """
    Whisper 単体直叩き。外部VAD・Aligner・Exporter を完全にバイパスして
    “素のASRセグメント”を回収する。
    """
    try:
        result = _mlx_transcribe(
            audio_path,
            language=language,
            # 上で wrap 済み：vad_filter=False, filters relaxed
            # その他は既定のままでOK（モデル/weightsは env/キャッシュ準拠）
        )
        # 返却形式はすでに segments を含む dict のはず
        return result or {}
    except Exception as e:
        print(f"[ERROR] fallback whisper failed: {e}")
        return {}

# --- 6) main run for multiple thresholds ---
def run_once(audio: str, th: float, out_dir: Path, min_ok_sec: float) -> Dict[str, Any]:
    print(f"\n=== RUN (no_speech_threshold={th}) ===")
    cfg = AppConfig(
        transcription=TranscriptionConfig(
            language="ja",
            no_speech_threshold=th
        )
    )
    t0 = time.time()
    result = {}
    err = None
    try:
        result = run_ai_pipeline(audio, cfg, PROJECT_ROOT)
    except Exception as e:
        err = str(e)
    elapsed = time.time() - t0

    summary = _summarize(result or {})
    print(f"[PIPELINE] elapsed={elapsed:.1f}s, segments={summary['segments']}, dur={summary['duration']:.2f}s, tokens={summary['tokens']}")
    if summary["segments"] == 0 or summary["duration"] < min_ok_sec:
        print(f"[WARN] pipeline result too short (<{min_ok_sec:.1f}s) or empty; try fallback whisper")
        fb = _fallback_whisper(audio, language="ja")
        fb_summary = _summarize(fb)
        print(f"[FALLBACK] segments={fb_summary['segments']}, dur={fb_summary['duration']:.2f}s, tokens={fb_summary['tokens']}")
        if fb_summary["segments"] > 0:
            stem = f"recovered_ns{str(th).replace('.','_')}"
            _write_outputs(stem, fb, out_dir)
            print(f"[OK] fallback outputs written: {out_dir}/{stem}.(json|txt|srt)")
            return {"mode": "fallback", "elapsed": elapsed, "summary": fb_summary}
        else:
            print("[ERR] fallback also empty. Keeping pipeline result (empty).")
            return {"mode": "pipeline_empty", "elapsed": elapsed, "summary": summary, "error": err}
    else:
        stem = f"pipeline_ns{str(th).replace('.','_')}"
        _write_outputs(stem, result, out_dir)
        print(f"[OK] pipeline outputs written: {out_dir}/{stem}.(json|txt|srt)")
        return {"mode": "pipeline", "elapsed": elapsed, "summary": summary}

def main():
    ap = argparse.ArgumentParser(description="AHE-Whisper recovery runner (safe monkey-patch)")
    ap.add_argument("--input", "-i", required=True, help="audio file path (e.g. wada_25m.mp3)")
    ap.add_argument("--thresholds", "-t", nargs="+", type=float, default=[0.4, 0.6, 0.7], help="no_speech_threshold list")
    ap.add_argument("--min-ok-sec", type=float, default=30.0, help="pipeline result shorter than this triggers fallback")
    ap.add_argument("--out", type=str, default="outputs_recovered", help="output dir")
    args = ap.parse_args()

    audio = str(Path(args.input).expanduser().resolve())
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== AHE-Whisper Recovery Patch Runner ===")
    print(f"audio={audio}")
    print(f"thresholds={args.thresholds}")
    print(f"min_ok_sec={args.min_ok_sec}s")
    print(f"out_dir={out_dir}")

    reports: List[Dict[str, Any]] = []
    for th in args.thresholds:
        reports.append({"th": th, **run_once(audio, th, out_dir, args.min_ok_sec)})

    # 簡易サマリを保存
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[SUMMARY] written -> {summary_path}")

if __name__ == "__main__":
    main()
