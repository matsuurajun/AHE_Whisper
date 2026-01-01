#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AHE-Whisper v90 用 埋め込みヒートマップ診断スクリプト
(Silero-VAD + ERes2NetV2 / ECAPA-TDNN / ResNet)

- 入力:  音声ファイル（.wav / .mp3 など）
- 出力:
    - diag_XXXX/segments.json           : 使用したセグメント一覧
    - diag_XXXX/embeddings.npy          : 各セグメントの埋め込みベクトル (N, D)
    - diag_XXXX/similarity_matrix.npy   : コサイン類似度マトリクス (N, N)
    - diag_XXXX/similarity_heatmap.png  : 類似度ヒートマップ

(実行例)
cd /Users/mrjn/Projects/transcription/whisper0814/AHE-Whisper-v90

# Titanet-L
uv run \
  --with sherpa-onnx --with numpy --with matplotlib --with librosa \
  python tools/diagnose_sherpa_speakers_v90.py \
    miyazaki_7m.mp3 \
    --model models/k2-fsa/sherpa-onnx/nemo_en_titanet_large.onnx \
    --out-dir diag_miyazaki_7m_titanet \
    --win-sec 3.0 \
    --hop-sec 1.5 \
    --title "Cosine similarity (Titanet-L, fixed segments)"

# CAM++_LM
uv run \
  --with sherpa-onnx --with numpy --with matplotlib --with librosa \
  python tools/diagnose_sherpa_speakers_v90.py \
    miyazaki_7m.mp3 \
    --model models/k2-fsa/sherpa-onnx/wespeaker_en_voxceleb_CAM++_LM.onnx \
    --out-dir diag_miyazaki_7m_campp_lm \
    --win-sec 3.0 \
    --hop-sec 1.5 \
    --title "Cosine similarity (CAM++_LM, fixed segments)"

# Speakernet
uv run \
  --with sherpa-onnx --with numpy --with matplotlib --with librosa \
  python tools/diagnose_sherpa_speakers_v90.py \
    miyazaki_7m.mp3 \
    --model models/k2-fsa/sherpa-onnx/nemo_en_speakerverification_speakernet.onnx \
    --out-dir diag_miyazaki_7m_speakernet \
    --win-sec 3.0 \
    --hop-sec 1.5 \
    --title "Cosine similarity (Speakernet, fixed segments)"

"""

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
except ImportError:
    sns = None

from ahe_whisper.vad import VAD
from ahe_whisper.embedding import (
    build_er2v2_session,
    er2v2_embed_batched,
    build_ecapa512_session,
    ecapa512_embed_batched,
)

LOGGER = logging.getLogger("diag_embeddings_v90")


# ==============================
# ロガー
# ==============================

def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


# ==============================
# Silero-VAD からセグメント生成
# ==============================

def segments_from_vad_probs(
    probs: np.ndarray,
    times: np.ndarray,
    threshold: float,
    min_segment_sec: float,
) -> List[Dict[str, float]]:
    segments: List[Dict[str, float]] = []
    if probs.size == 0 or times.size == 0:
        return segments

    assert probs.shape[0] == times.shape[0]

    in_speech = False
    seg_start_t = 0.0

    for i, p in enumerate(probs):
        t = float(times[i])
        if p >= threshold and not in_speech:
            in_speech = True
            seg_start_t = t
        elif p < threshold and in_speech:
            in_speech = False
            seg_end_t = t
            dur = seg_end_t - seg_start_t
            if dur >= min_segment_sec:
                segments.append({"start": seg_start_t, "end": seg_end_t})

    if in_speech:
        seg_end_t = float(times[-1])
        dur = seg_end_t - seg_start_t
        if dur >= min_segment_sec:
            segments.append({"start": seg_start_t, "end": seg_end_t})

    return segments


def collect_segments_with_vad(
    audio: np.ndarray,
    sr: int,
    vad_model_path: Path,
    window_size_samples: int,
    grid_hz: int,
    prob_threshold: float,
    min_segment_sec: float,
) -> Tuple[List[Dict[str, float]], List[np.ndarray]]:
    vad_config = SimpleNamespace(window_size_samples=int(window_size_samples))
    vad = VAD(vad_model_path, vad_config)

    probs, grid_times = vad.get_speech_probabilities(audio, sr, grid_hz)
    LOGGER.info(
        "VAD probs: shape=%s, grid_times.shape=%s",
        probs.shape,
        grid_times.shape,
    )

    segments = segments_from_vad_probs(
        probs=probs,
        times=grid_times,
        threshold=prob_threshold,
        min_segment_sec=min_segment_sec,
    )
    LOGGER.info("Segments from VAD: %d", len(segments))

    waveforms: List[np.ndarray] = []
    used_segments: List[Dict[str, float]] = []

    for seg in segments:
        s = int(seg["start"] * sr)
        e = int(seg["end"] * sr)
        if e <= s:
            continue
        w = audio[s:e].astype(np.float32)
        if len(w) < int(0.3 * sr):
            continue
        waveforms.append(w)
        used_segments.append(seg)

    LOGGER.info("Segments after length filter: %d", len(used_segments))
    return used_segments, waveforms


# ==============================
# 埋め込み抽出
# ==============================

def extract_embeddings(
    model_type: str,
    embedding_model_path: Path,
    waveforms: List[np.ndarray],
    sr: int,
    embedding_dim: int,
    prefer_coreml_ep: bool,
    min_frames: int,
    batch_cap: int,
    bucket_step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    指定された model_type に応じてセッション構築と推論を行う
    """
    emb_config = SimpleNamespace(
        embedding_dim=int(embedding_dim),
        intra_threads=None,
        inter_threads=None,
        prefer_coreml_ep=bool(prefer_coreml_ep),
        min_frames=int(min_frames),
        batch_cap=int(batch_cap),
        bucket_step=int(bucket_step),
        cmvn_policy=None,
    )

    if not waveforms:
        raise ValueError("waveforms が空です。")

    if model_type == "wespeaker":
        # ECAPA-TDNN / ResNet (WeSpeaker系)
        # input layout 自動判定機能付き
        session = build_ecapa512_session(embedding_model_path, emb_config)
        embeddings, valid_mask = ecapa512_embed_batched(
            session=session,
            model_path=embedding_model_path,
            chunks=waveforms,
            sr=sr,
            config=emb_config,
        )
    else:
        # 3D-Speaker / ERes2Net系 (Default)
        # (B, T, F) 固定レイアウト
        session = build_er2v2_session(embedding_model_path, emb_config)
        embeddings, valid_mask = er2v2_embed_batched(
            session=session,
            model_path=embedding_model_path,
            audio_chunks=waveforms,
            sr=sr,
            config=emb_config,
        )

    LOGGER.info(
        "Embeddings extracted: shape=%s, valid=%d/%d",
        embeddings.shape,
        int(valid_mask.sum()) if valid_mask is not None else len(waveforms),
        len(waveforms),
    )
    return embeddings, valid_mask


# ==============================
# 簡易 VAD (エネルギーベース) とフォールバック用
# ==============================

def naive_vad_segments(
    audio: np.ndarray,
    sr: int,
    window_sec: float = 0.8,
    hop_sec: float = 0.4,
    energy_thresh_db: float = -35.0,
    min_segment_sec: float = 1.0,
) -> List[Dict[str, float]]:
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    n_samples = len(audio)

    energies_db: List[float] = []
    frame_starts: List[int] = []

    i = 0
    while i + win <= n_samples:
        frame = audio[i: i + win]
        power = np.mean(frame ** 2) + 1e-10
        db = 10.0 * np.log10(power)
        energies_db.append(db)
        frame_starts.append(i)
        i += hop

    speech_flags = [e > energy_thresh_db for e in energies_db]

    segments: List[Dict[str, float]] = []
    in_speech = False
    seg_start_sample = 0

    for idx, is_speech in enumerate(speech_flags):
        if is_speech and not in_speech:
            in_speech = True
            seg_start_sample = frame_starts[idx]
        elif not is_speech and in_speech:
            in_speech = False
            seg_end_sample = frame_starts[idx] + win
            dur = (seg_end_sample - seg_start_sample) / sr
            if dur >= min_segment_sec:
                segments.append(
                    {
                        "start": seg_start_sample / sr,
                        "end": seg_end_sample / sr,
                    }
                )

    if in_speech:
        seg_end_sample = n_samples
        dur = (seg_end_sample - seg_start_sample) / sr
        if dur >= min_segment_sec:
            segments.append(
                {
                    "start": seg_start_sample / sr,
                    "end": seg_end_sample / sr,
                }
            )

    return segments


def collect_segments_with_naive_vad(
    audio: np.ndarray,
    sr: int,
    min_segment_sec: float,
) -> Tuple[List[Dict[str, float]], List[np.ndarray]]:
    segments = naive_vad_segments(
        audio=audio,
        sr=sr,
        window_sec=0.8,
        hop_sec=0.4,
        energy_thresh_db=-35.0,
        min_segment_sec=min_segment_sec,
    )
    LOGGER.info("Naive-VAD segments: %d", len(segments))

    waveforms: List[np.ndarray] = []
    used_segments: List[Dict[str, float]] = []

    for seg in segments:
        s = int(seg["start"] * sr)
        e = int(seg["end"] * sr)
        if e <= s:
            continue
        w = audio[s:e].astype(np.float32)
        if len(w) < int(0.3 * sr):
            continue
        waveforms.append(w)
        used_segments.append(seg)

    LOGGER.info("Naive-VAD segments after length filter: %d", len(used_segments))
    return used_segments, waveforms


# ==============================
# 類似度マトリクス & ヒートマップ
# ==============================

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    n, d = embeddings.shape
    LOGGER.info("Embeddings shape (for similarity): N=%d, D=%d", n, d)

    if n < 2:
        raise ValueError("セグメント数が 2 未満のため、類似度マトリクスを作れません。")

    sim = embeddings @ embeddings.T
    sim = np.clip(sim, -1.0, 1.0)
    return sim


def plot_similarity_heatmap(sim: np.ndarray, out_path: Path, title: str) -> None:
    n = sim.shape[0]
    plt.figure(figsize=(max(6, n * 0.25), max(5, n * 0.25)))

    if sns is not None:
        sns.heatmap(
            sim,
            vmin=-1.0,
            vmax=1.0,
            cmap="viridis",
            square=True,
        )
    else:
        im = plt.imshow(sim, vmin=-1.0, vmax=1.0, cmap="viridis")
        plt.colorbar(im)

    plt.title(title)
    plt.xlabel("Segment index")
    plt.ylabel("Segment index")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ==============================
# メイン処理
# ==============================

def run(
    audio_path: Path,
    out_dir: Path,
    vad_model_path: Path,
    embedding_model_path: Path,
    model_type: str,
    max_segments: int,
    target_sr: int,
    vad_window_samples: int,
    vad_grid_hz: int,
    vad_threshold: float,
    min_segment_sec: float,
    embedding_dim: int,
    prefer_coreml_ep: bool,
    min_frames: int,
    batch_cap: int,
    bucket_step: int,
) -> None:
    LOGGER.info("Audio: %s", audio_path)
    LOGGER.info("Out dir: %s", out_dir)
    LOGGER.info("VAD model: %s", vad_model_path)
    LOGGER.info("Embedding model: %s (Type=%s)", embedding_model_path, model_type)

    out_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
    audio = audio.astype(np.float32)
    duration = len(audio) / sr
    LOGGER.info("Audio loaded: sr=%d, duration=%.2fs", sr, duration)

    try:
        segments, waveforms = collect_segments_with_vad(
            audio=audio,
            sr=sr,
            vad_model_path=vad_model_path,
            window_size_samples=vad_window_samples,
            grid_hz=vad_grid_hz,
            prob_threshold=vad_threshold,
            min_segment_sec=min_segment_sec,
        )
    except Exception as e:
        LOGGER.error(
            "Silero-VAD でエラーが発生しました (%s)。診断用途のため、Naive VAD にフォールバックします。",
            e,
        )
        segments, waveforms = collect_segments_with_naive_vad(
            audio=audio,
            sr=sr,
            min_segment_sec=min_segment_sec,
        )

    if not waveforms:
        LOGGER.error("有効なセグメントが 1 つもありませんでした。")
        return

    if len(waveforms) > max_segments:
        seg_lens = [s["end"] - s["start"] for s in segments]
        order = np.argsort(seg_lens)[::-1][:max_segments]
        order = sorted(order, key=lambda i: segments[i]["start"])
        segments = [segments[i] for i in order]
        waveforms = [waveforms[i] for i in order]
        LOGGER.info("Segments truncated to max_segments=%d", max_segments)

    embeddings, valid_mask = extract_embeddings(
        model_type=model_type,
        embedding_model_path=embedding_model_path,
        waveforms=waveforms,
        sr=sr,
        embedding_dim=embedding_dim,
        prefer_coreml_ep=prefer_coreml_ep,
        min_frames=min_frames,
        batch_cap=batch_cap,
        bucket_step=bucket_step,
    )

    if valid_mask is not None and valid_mask.shape[0] == embeddings.shape[0]:
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 2:
            LOGGER.error("有効な埋め込みが 2 未満です。")
            return
        embeddings = embeddings[valid_indices]
        segments = [segments[i] for i in valid_indices]
    else:
        if embeddings.shape[0] < 2:
            LOGGER.error("埋め込み数が 2 未満です。")
            return

    sim = compute_similarity_matrix(embeddings)

    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "similarity_matrix.npy", sim)

    with (out_dir / "segments.json").open("w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    title = f"Cosine Similarity ({embedding_model_path.stem})"
    plot_similarity_heatmap(sim, out_dir / "similarity_heatmap.png", title)

    LOGGER.info("Done. Saved to: %s", out_dir)


def main() -> None:
    setup_logger()

    parser = argparse.ArgumentParser(
        description="AHE-Whisper v90 用 埋め込みヒートマップ診断ツール",
    )
    parser.add_argument("audio", type=str, help="入力音声ファイルパス")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--vad-model", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="Embeddingモデルパス")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["er2v2", "wespeaker"],
        default="er2v2",
        help="er2v2 (Cam++/ERes2Net) or wespeaker (ECAPA/ResNet)",
    )
    parser.add_argument("--max-segments", type=int, default=200)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--vad-window-samples", type=int, default=5120)
    parser.add_argument("--vad-grid-hz", type=int, default=50)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--min-segment-sec", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--no-coreml", action="store_true")
    parser.add_argument("--min-frames", type=int, default=40)
    parser.add_argument("--batch-cap", type=int, default=16)
    parser.add_argument("--bucket-step", type=int, default=40)

    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    vad_model_path = Path(args.vad_model).expanduser().resolve()
    embedding_model_path = Path(args.model).expanduser().resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not vad_model_path.exists():
        raise FileNotFoundError(f"VAD model not found: {vad_model_path}")
    if not embedding_model_path.exists():
        raise FileNotFoundError(f"Embedding model not found: {embedding_model_path}")

    if args.out_dir is not None:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = audio_path.parent / f"diag_{audio_path.stem}"

    run(
        audio_path=audio_path,
        out_dir=out_dir,
        vad_model_path=vad_model_path,
        embedding_model_path=embedding_model_path,
        model_type=args.model_type,
        max_segments=args.max_segments,
        target_sr=args.sr,
        vad_window_samples=args.vad_window_samples,
        vad_grid_hz=args.vad_grid_hz,
        vad_threshold=args.vad_threshold,
        min_segment_sec=args.min_segment_sec,
        embedding_dim=args.embedding_dim,
        prefer_coreml_ep=not args.no_coreml,
        min_frames=args.min_frames,
        batch_cap=args.batch_cap,
        bucket_step=args.bucket_step,
    )


if __name__ == "__main__":
    main()