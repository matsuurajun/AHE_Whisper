# tools/diagnose_sherpa_speakers_v90.py
# -*- coding: utf-8 -*-

"""
Sherpa-ONNX Speaker Embedding Diagnostics for AHE-Whisper v90

- 対象:
    - nemo_en_titanet_large.onnx          (192-d)
    - wespeaker_en_voxceleb_CAM++_LM.onnx (512-d)
    - nemo_en_speakerverification_speakernet.onnx (256-d)
  など、SpeakerEmbeddingExtractorConfig で扱える sherpa-onnx モデル。

- 入力:
    - 音声ファイル（.wav / .mp3 など）

- 出力 (out-dir 配下):
    - segments.json           : 各セグメントの start/end（秒・サンプル）
    - embeddings.npy          : (N, D) embedding 行列
    - similarity_matrix.npy   : (N, N) コサイン類似度マトリクス
    - similarity_heatmap.png  : 類似度ヒートマップ
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import sherpa_onnx


LOGGER = logging.getLogger("diag_sherpa_speakers")


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )


def load_audio(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != target_sr:
        LOGGER.info("Resampling audio from %d Hz to %d Hz", sr, target_sr)
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    LOGGER.info(
        "Audio loaded: %s (duration=%.2f sec, sr=%d)",
        path,
        len(y) / sr,
        sr,
    )
    return y.astype(np.float32), sr


def segment_audio(
    y: np.ndarray,
    sr: int,
    win_sec: float,
    hop_sec: float,
) -> List[Tuple[int, int]]:
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    segments: List[Tuple[int, int]] = []

    if len(y) < win:
        LOGGER.warning(
            "Audio shorter than one window: len=%.2fs, win=%.2fs; using whole audio.",
            len(y) / sr,
            win_sec,
        )
        segments.append((0, len(y)))
        return segments

    for start in range(0, len(y) - win + 1, hop):
        end = start + win
        segments.append((start, end))

    LOGGER.info(
        "Segmented audio into %d segments (win=%.2fs, hop=%.2fs)",
        len(segments),
        win_sec,
        hop_sec,
    )
    return segments


def build_extractor(
    model_path: Path,
    num_threads: int,
    provider: str,
    debug: bool,
) -> sherpa_onnx.SpeakerEmbeddingExtractor:
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_path),
        num_threads=num_threads,
        provider=provider,
        debug=1 if debug else 0,
    )
    if not cfg.validate():
        raise RuntimeError(f"Invalid SpeakerEmbeddingExtractorConfig for {model_path}")

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    LOGGER.info(
        "SpeakerEmbeddingExtractor created: model=%s, dim=%d",
        model_path,
        extractor.dim,
    )
    return extractor


def extract_embeddings(
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
    y: np.ndarray,
    sr: int,
    segments: List[Tuple[int, int]],
) -> np.ndarray:
    embs: List[np.ndarray] = []

    for idx, (start, end) in enumerate(segments):
        seg = y[start:end]
        stream = extractor.create_stream()
        stream.accept_waveform(sr, seg)
        stream.input_finished()
        emb_list = extractor.compute(stream)
        emb = np.asarray(emb_list, dtype=np.float32)
        embs.append(emb)
        LOGGER.debug(
            "Segment %d: len=%.2fs, emb_norm=%.4f",
            idx,
            len(seg) / sr,
            float(np.linalg.norm(emb)),
        )

    embeddings = np.stack(embs, axis=0)
    LOGGER.info(
        "Extracted embeddings: shape=%s (num_segments=%d, dim=%d)",
        embeddings.shape,
        embeddings.shape[0],
        embeddings.shape[1],
    )
    return embeddings


def compute_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms
    sim = np.matmul(normed, normed.T)
    return sim


def summarize_similarity(sim: np.ndarray) -> None:
    n = sim.shape[0]
    if n < 2:
        LOGGER.warning("Only one segment; skipping similarity summary.")
        return

    mask = ~np.eye(n, dtype=bool)
    flat = sim[mask]

    global_mean = float(np.mean(flat))
    global_std = float(np.std(flat))

    LOGGER.info(
        "[SIM] Global cosine: mean=%.4f, std=%.4f (over %d pairs)",
        global_mean,
        global_std,
        flat.size,
    )

    row_sims = np.where(mask, sim, -np.inf)
    top1 = row_sims.max(axis=1)
    top2 = np.partition(row_sims, -2, axis=1)[:, -2]
    gap = top1 - top2

    LOGGER.info(
        "[SIM] Row-wise max: mean=%.4f, median=%.4f",
        float(np.mean(top1)),
        float(np.median(top1)),
    )
    LOGGER.info(
        "[SIM] Row-wise (top1 - top2): mean=%.4f, median=%.4f",
        float(np.mean(gap)),
        float(np.median(gap)),
    )
    LOGGER.info(
        "[SIM] Rows with max < 0.4: %d / %d",
        int(np.sum(top1 < 0.4)),
        n,
    )


def save_heatmap(sim: np.ndarray, out_png: Path, title: str) -> None:
    if sim.size == 0:
        LOGGER.warning("Empty similarity matrix; skipping heatmap.")
        return

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(sim, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Segment index")
    ax.set_ylabel("Segment index")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    LOGGER.info("Saved similarity heatmap to %s", out_png)


def save_outputs(
    out_dir: Path,
    y: np.ndarray,
    sr: int,
    segments: List[Tuple[int, int]],
    embeddings: np.ndarray,
    sim: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_entries = []
    for idx, (start, end) in enumerate(segments):
        seg_entries.append(
            {
                "index": idx,
                "start_sec": float(start / sr),
                "end_sec": float(end / sr),
                "start_sample": int(start),
                "end_sample": int(end),
            }
        )

    with (out_dir / "segments.json").open("w", encoding="utf-8") as f:
        json.dump(seg_entries, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved segments.json")

    np.save(out_dir / "embeddings.npy", embeddings)
    LOGGER.info("Saved embeddings.npy")

    np.save(out_dir / "similarity_matrix.npy", sim)
    LOGGER.info("Saved similarity_matrix.npy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose sherpa-onnx speaker-embedding models "
            "(Titanet-L / CAM++_LM / Speakernet) using fixed windows."
        )
    )
    parser.add_argument("audio", type=str, help="Input audio file path")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to sherpa-onnx speaker embedding model (.onnx)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="diag_sherpa_speakers",
        help="Output directory for diagnostics",
    )
    parser.add_argument(
        "--win-sec",
        type=float,
        default=3.0,
        help="Segment window size in seconds",
    )
    parser.add_argument(
        "--hop-sec",
        type=float,
        default=1.5,
        help="Segment hop size in seconds",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for sherpa-onnx",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="ONNX runtime provider (cpu / coreml / cuda etc.)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable sherpa-onnx debug logs",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Target sampling rate (Hz)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for the similarity heatmap",
    )
    return parser.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()

    audio_path = Path(args.audio)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)

    LOGGER.info("Audio : %s", audio_path)
    LOGGER.info("Model : %s", model_path)
    LOGGER.info("Out   : %s", out_dir)

    y, sr = load_audio(audio_path, target_sr=args.sr)
    segments = segment_audio(
        y=y,
        sr=sr,
        win_sec=args.win_sec,
        hop_sec=args.hop_sec,
    )

    extractor = build_extractor(
        model_path=model_path,
        num_threads=args.num_threads,
        provider=args.provider,
        debug=args.debug,
    )

    embeddings = extract_embeddings(
        extractor=extractor,
        y=y,
        sr=sr,
        segments=segments,
    )

    norms = np.linalg.norm(embeddings, axis=1)
    LOGGER.info(
        "[EMB] L2 norm: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        float(np.mean(norms)),
        float(np.std(norms)),
        float(np.min(norms)),
        float(np.max(norms)),
    )

    sim = compute_cosine_similarity(embeddings)
    summarize_similarity(sim)

    save_outputs(out_dir, y, sr, segments, embeddings, sim)

    title = (
        args.title
        if args.title is not None
        else "Cosine similarity (sherpa-onnx, fixed segments)"
    )
    heatmap_path = out_dir / "similarity_heatmap.png"
    save_heatmap(sim, heatmap_path, title)


if __name__ == "__main__":
    main()
