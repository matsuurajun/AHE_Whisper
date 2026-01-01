# tools/diagnose_sherpa_from_segments_v90.py
# -*- coding: utf-8 -*-

"""
AHE-Whisper v90: sherpa-onnx speaker embedding diagnostics from precomputed segments.

想定ユースケース
----------------
- 既存の diagnose_embeddings_v90.py（Silero-VAD + ERes2NetV2 など）で
  すでに作成済みの `segments.json` を再利用して、
  Titanet-L / Speakernet などの sherpa-onnx モデルで
  同じセグメントに対する埋め込みと類似度ヒートマップを確認する。

入力
----
- 音声ファイル (.wav/.mp3 など)
- segments.json (start_sec / end_sec または start_sample / end_sample を含む)
- sherpa-onnx speaker embedding モデル (.onnx)

出力
----
- out_dir/segments.json           : 入力をそのままコピー
- out_dir/embeddings.npy          : (N, D) 埋め込み行列
- out_dir/similarity_matrix.npy   : (N, N) コサイン類似度マトリクス
- out_dir/sherpa_similarity_heatmap.png : 類似度ヒートマップ
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import sherpa_onnx


LOGGER = logging.getLogger("diag_sherpa_from_segments")


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


def load_segments_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"segments.json must be a list, got {type(data)}")
    if not data:
        raise ValueError("segments.json is empty")
    LOGGER.info("Loaded %d segments from %s", len(data), path)
    return data


def segments_to_sample_ranges(
    segments: List[Dict[str, Any]],
    sr: int,
) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for seg in segments:
        if "start_sec" in seg and "end_sec" in seg:
            # sherpa / WavLM 系の固定窓セグメントなど
            start = int(round(float(seg["start_sec"]) * sr))
            end = int(round(float(seg["end_sec"]) * sr))
        elif "start_sample" in seg and "end_sample" in seg:
            # サンプル番号で指定されている場合
            start = int(seg["start_sample"])
            end = int(seg["end_sample"])
        elif "start" in seg and "end" in seg:
            # diagnose_embeddings_v90.py（CampPlus 既存 diag）の形式
            start = int(round(float(seg["start"]) * sr))
            end = int(round(float(seg["end"]) * sr))
        else:
            raise ValueError(
                "Segment entry must contain one of:\n"
                "  - ('start_sec','end_sec')\n"
                "  - ('start_sample','end_sample')\n"
                "  - ('start','end')\n"
                f"got keys={list(seg.keys())}"
            )

        if end <= start:
            LOGGER.warning("Invalid segment with end <= start: %s", seg)
            continue

        ranges.append((start, end))

    LOGGER.info("Converted to %d sample ranges (sr=%d)", len(ranges), sr)
    return ranges


def build_sherpa_extractor(
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


def extract_embeddings_from_segments(
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
    y: np.ndarray,
    sr: int,
    ranges: List[Tuple[int, int]],
) -> np.ndarray:
    embs: List[np.ndarray] = []
    for idx, (start, end) in enumerate(ranges):
        start = max(0, min(start, len(y)))
        end = max(start + 1, min(end, len(y)))
        seg = y[start:end]
        stream = extractor.create_stream()
        stream.accept_waveform(sr, seg)
        stream.input_finished()
        emb_list = extractor.compute(stream)
        emb = np.asarray(emb_list, dtype=np.float32)
        embs.append(emb)
        LOGGER.debug(
            "Segment %d: start=%.3fs end=%.3fs len=%.3fs emb_norm=%.4f",
            idx,
            start / sr,
            end / sr,
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
    segments: List[Dict[str, Any]],
    embeddings: np.ndarray,
    sim: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "segments.json").open("w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved segments.json")
    np.save(out_dir / "embeddings.npy", embeddings)
    LOGGER.info("Saved embeddings.npy")
    np.save(out_dir / "similarity_matrix.npy", sim)
    LOGGER.info("Saved similarity_matrix.npy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose sherpa-onnx speaker embeddings (Titanet-L / Speakernet etc.) "
            "using pre-computed segments.json (e.g., from diagnose_embeddings_v90.py)."
        )
    )
    parser.add_argument("audio", type=str, help="Input audio file path")
    parser.add_argument(
        "--segments-json",
        type=str,
        required=True,
        help="Path to segments.json produced by another diagnostic (Silero-VAD, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to sherpa-onnx speaker embedding model (.onnx)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for diagnostics",
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
    seg_path = Path(args.segments_json)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)

    LOGGER.info("Audio    : %s", audio_path)
    LOGGER.info("Segments : %s", seg_path)
    LOGGER.info("Model    : %s", model_path)
    LOGGER.info("Out      : %s", out_dir)

    y, sr = load_audio(audio_path, target_sr=args.sr)
    segments = load_segments_json(seg_path)
    ranges = segments_to_sample_ranges(segments, sr=sr)

    extractor = build_sherpa_extractor(
        model_path=model_path,
        num_threads=args.num_threads,
        provider=args.provider,
        debug=args.debug,
    )

    embeddings = extract_embeddings_from_segments(
        extractor=extractor,
        y=y,
        sr=sr,
        ranges=ranges,
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

    save_outputs(out_dir, segments, embeddings, sim)

    title = (
        args.title
        if args.title is not None
        else "Cosine similarity (sherpa-onnx, VAD-based segments)"
    )
    save_heatmap(sim, out_dir / "sherpa_similarity_heatmap.png", title)


if __name__ == "__main__":
    main()
