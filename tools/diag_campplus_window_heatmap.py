# -*- coding: utf-8 -*-
"""
diag_campplus_window_heatmap.py

短い時間窓（例: 0〜30秒）について、
- CampPlus (3D-Speaker) 埋め込みに基づく話者確率ヒートマップ
- hard label (argmax) タイムライン
- Silero VAD の speech probability

を可視化するための超限定診断スクリプト。

前提:
- AHE-Whisper の ahe_whisper パッケージが import できる状態
- embedding.py / vad.py / config.py は v90 相当のもの
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from ahe_whisper.config import AppConfig
from ahe_whisper.vad import VAD
from ahe_whisper.embedding import (
    build_er2v2_session,
    warmup_er2v2,
    er2v2_embed_batched,
)

LOGGER = logging.getLogger("diag_campplus_window")


# -------------------------------------------------------------------------
# 共通ユーティリティ
# -------------------------------------------------------------------------

def setup_logger() -> None:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(fmt)
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(handler)


def build_embedding_segments(
    t0: float,
    t1: float,
    frame_sec: float,
    hop_sec: float,
) -> np.ndarray:
    """
    [t0, t1] の範囲を一様な長さ frame_sec, hop_sec で刻んだセグメント配列を返す。
    shape: (N, 2) [start_sec, end_sec]
    """
    segments: List[Tuple[float, float]] = []
    t = t0
    while t < t1:
        end = min(t + frame_sec, t1)
        if end - t >= 0.3 * frame_sec:
            segments.append((t, end))
        t += hop_sec

    if not segments:
        return np.zeros((0, 2), dtype=np.float32)

    return np.asarray(segments, dtype=np.float32)


def segments_from_vad_probs(
    probs: np.ndarray,
    times: np.ndarray,
    threshold: float,
    min_segment_sec: float,
) -> List[Dict[str, float]]:
    """
    VAD 確率列から単純なしきい値ベースで有声区間を切り出す。
    diagnose_embeddings_v90.py のロジックを簡略化したもの。
    （※現バージョンでは直接は使っていないが、今後の分析用に残している）
    """
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


def softmax(x: np.ndarray, axis: int = -1, tau: float = 1.0) -> np.ndarray:
    z = x * tau
    z = z - np.max(z, axis=axis, keepdims=True)
    exp = np.exp(z)
    return exp / (np.sum(exp, axis=axis, keepdims=True) + 1e-8)


def build_prob_heatmap(
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    probs: np.ndarray,
    t0: float,
    t1: float,
    grid_hz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    segment 中心に定義された probs を、一定ステップの時間グリッドに写像する。

    Returns:
        grid_times: (G,)
        heatmap:   (G, S)
        labels:    (G,)
    """
    assert probs.ndim == 2
    n_segments, n_speakers = probs.shape
    if n_segments == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, n_speakers), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    seg_centers = 0.5 * (seg_starts + seg_ends)
    seg_centers = np.clip(seg_centers, t0, t1)

    step = 1.0 / grid_hz
    grid_times = np.arange(t0, t1, step, dtype=np.float32)
    if grid_times.size == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, n_speakers), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    heatmap = np.zeros((grid_times.size, n_speakers), dtype=np.float32)
    labels = np.zeros((grid_times.size,), dtype=np.int64)

    for i, t in enumerate(grid_times):
        idx = int(np.argmin(np.abs(seg_centers - t)))
        heatmap[i] = probs[idx]
        labels[i] = int(np.argmax(probs[idx]))

    return grid_times, heatmap, labels


def plot_heatmap(grid_times: np.ndarray, heatmap: np.ndarray, out_png: Path) -> None:
    if grid_times.size == 0:
        LOGGER.warning("No data to plot heatmap; skipping")
        return

    n_speakers = heatmap.shape[1]
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        heatmap.T,
        origin="lower",
        aspect="auto",
        extent=[
            float(grid_times[0]),
            float(grid_times[-1]),
            -0.5,
            float(n_speakers) - 0.5,
        ],
    )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speaker id")
    ax.set_title("CampPlus speaker probabilities (window)")
    fig.colorbar(im, ax=ax, label="prob")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_labels(grid_times: np.ndarray, labels: np.ndarray, out_png: Path) -> None:
    if grid_times.size == 0:
        LOGGER.warning("No data to plot labels; skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 1.5))
    img = labels[np.newaxis, :]
    ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=[
            float(grid_times[0]),
            float(grid_times[-1]),
            -0.5,
            0.5,
        ],
    )
    ax.set_yticks([])
    ax.set_xlabel("time [s]")
    ax.set_title("hard speaker labels (argmax prob)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_vad(grid_times: np.ndarray, vad_probs: np.ndarray, out_png: Path) -> None:
    if grid_times.size == 0 or vad_probs.size == 0:
        LOGGER.warning("No data to plot VAD; skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 2.0))
    ax.plot(grid_times, vad_probs)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speech prob")
    ax.set_title("VAD speech probabilities (window)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------------
# メイン処理
# -------------------------------------------------------------------------

def main() -> None:
    setup_logger()

    parser = argparse.ArgumentParser(
        description=(
            "CampPlus 確率と話者ラベルの超限定可視化 "
            "(短い時間窓, pipeline VAD + pipeline embedding)"
        )
    )
    parser.add_argument("audio", type=Path, help="入力音声ファイル (wav/mp3 など)")
    parser.add_argument(
        "--vad-model",
        type=Path,
        required=True,
        help="Silero VAD ONNX パス (ahe_whisper と同じもの)",
    )
    parser.add_argument(
        "--spk-model",
        type=Path,
        required=True,
        help="CampPlus (3dspeaker) ONNX パス",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("diag_campplus_window"),
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="分析開始秒 (デフォルト: 0.0)",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=30.0,
        help="分析終了秒 (デフォルト: 30.0)",
    )
    parser.add_argument(
        "--n-speakers",
        type=int,
        default=2,
        help="クラスタ数 (=話者数想定, デフォルト: 2)",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=200,
        help="診断用の最大セグメント数 (デフォルト: 200)",
    )
    parser.add_argument(
        "--logit-scale",
        type=float,
        default=3.0,
        help="類似度 → 確率変換のスケール (デフォルト: 3.0)",
    )
    parser.add_argument(
        "--grid-hz",
        type=float,
        default=20.0,
        help="ヒートマップ・VAD の時間解像度 [Hz] (デフォルト: 20)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="※将来用: VAD から有声セグメントを切るしきい値 (デフォルト: 0.5)",
    )
    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=1.0,
        help="※将来用: 有声セグメント最小長 [sec] (デフォルト: 1.0)",
    )

    args = parser.parse_args()

    sr_target = 16000
    LOGGER.info("Loading audio: %s", args.audio)
    wav, _ = librosa.load(str(args.audio), sr=sr_target, mono=True)

    # 時間窓の決定
    t0 = max(float(args.start_sec), 0.0)
    t1 = min(float(args.end_sec), len(wav) / float(sr_target))
    if t1 <= t0:
        raise ValueError("Invalid time window: end_sec must be > start_sec")

    # 対象窓の波形（相対時間 0〜(t1-t0)）を作る
    s_idx = int(t0 * sr_target)
    e_idx = int(t1 * sr_target)
    wav_win = wav[s_idx:e_idx]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # AppConfig から embedding / VAD / aligner 設定を取得
    app_cfg = AppConfig()
    emb_cfg = app_cfg.embedding
    vad_cfg = app_cfg.vad
    align_cfg = app_cfg.aligner

    frame_sec = float(getattr(emb_cfg, "embedding_win_sec", 1.5))
    hop_sec = float(getattr(emb_cfg, "embedding_hop_sec", 0.75))
    min_chunk_speech_prob = float(getattr(emb_cfg, "min_chunk_speech_prob", 0.4))

    LOGGER.info(
        "Using embedding window=%.2fs, hop=%.2fs, min_chunk_speech_prob=%.2f",
        frame_sec,
        hop_sec,
        min_chunk_speech_prob,
    )
    LOGGER.info(
        "Aligner grid_hz (config)=%d, grid_hz (args)=%.1f",
        getattr(align_cfg, "grid_hz", 50),
        args.grid_hz,
    )

    # まず VAD をパイプラインと同じ実装で走らせる（対象窓だけ）
    grid_hz = int(args.grid_hz)
    LOGGER.info("Running pipeline VAD on window (grid_hz=%d)...", grid_hz)
    pipeline_vad = VAD(args.vad_model, vad_cfg)
    vad_probs_local, vad_times_local = pipeline_vad.get_speech_probabilities(
        waveform=wav_win,
        sr=sr_target,
        grid_hz=grid_hz,
    )
    # VAD の時間軸をグローバルな絶対時刻に合わせる
    vad_times_abs = vad_times_local + t0

    LOGGER.info(
        "VAD probs: shape=%s, times range=[%.2f, %.2f]",
        vad_probs_local.shape,
        float(vad_times_abs[0]) if vad_times_abs.size > 0 else -1.0,
        float(vad_times_abs[-1]) if vad_times_abs.size > 0 else -1.0,
    )

    # Embedding 用の一様セグメント（絶対時刻）を作る
    LOGGER.info(
        "Building embedding segments in window [%.2f, %.2f] with frame_sec=%.2f, hop_sec=%.2f",
        t0,
        t1,
        frame_sec,
        hop_sec,
    )
    segs_abs_all = build_embedding_segments(
        t0,
        t1,
        frame_sec=frame_sec,
        hop_sec=hop_sec,
    )
    if segs_abs_all.shape[0] == 0:
        LOGGER.warning("No segments built in window; aborting")
        return

    # VAD スコアの平均が低いセグメントは embedding 対象から外す
    seg_starts: List[float] = []
    seg_ends: List[float] = []
    seg_vad_means: List[float] = []
    audio_chunks: List[np.ndarray] = []

    for (seg_s_abs, seg_e_abs) in segs_abs_all:
        seg_s_local = seg_s_abs - t0
        seg_e_local = seg_e_abs - t0
        if vad_times_local.size == 0:
            mean_prob = 0.0
        else:
            mask = (vad_times_local >= seg_s_local) & (vad_times_local < seg_e_local)
            if not np.any(mask):
                # セグメント内にサンプルがない場合は中心に最も近い点を使う
                center = 0.5 * (seg_s_local + seg_e_local)
                idx = int(np.argmin(np.abs(vad_times_local - center)))
                mean_prob = float(vad_probs_local[idx])
            else:
                mean_prob = float(vad_probs_local[mask].mean())

        if mean_prob < min_chunk_speech_prob:
            continue

        s_global = int(seg_s_abs * sr_target)
        e_global = int(seg_e_abs * sr_target)
        if e_global <= s_global:
            continue
        chunk = wav[s_global:e_global].astype(np.float32)
        if len(chunk) < int(0.3 * sr_target):
            continue

        seg_starts.append(seg_s_abs)
        seg_ends.append(seg_e_abs)
        seg_vad_means.append(mean_prob)
        audio_chunks.append(chunk)

    if not audio_chunks:
        LOGGER.warning(
            "No segments passed VAD gating (min_chunk_speech_prob=%.2f); aborting",
            min_chunk_speech_prob,
        )
        return

    seg_starts_arr = np.asarray(seg_starts, dtype=np.float32)
    seg_ends_arr = np.asarray(seg_ends, dtype=np.float32)
    seg_vad_means_arr = np.asarray(seg_vad_means, dtype=np.float32)

    if seg_starts_arr.shape[0] > args.max_segments:
        idx = np.linspace(
            0, seg_starts_arr.shape[0] - 1, num=args.max_segments, dtype=np.int64
        )
        seg_starts_arr = seg_starts_arr[idx]
        seg_ends_arr = seg_ends_arr[idx]
        seg_vad_means_arr = seg_vad_means_arr[idx]
        audio_chunks = [audio_chunks[i] for i in idx]
        LOGGER.info(
            "Too many segments; downsampled to %d for diagnostics",
            args.max_segments,
        )

    LOGGER.info("Final segments for embedding: %d", seg_starts_arr.shape[0])

    # CampPlus 埋め込みセッションを構築
    LOGGER.info("Building CampPlus embedding session...")
    spk_sess = build_er2v2_session(args.spk_model, emb_cfg)
    warmup_er2v2(spk_sess)

    # 本番と同じロジックで embedding 抽出
    LOGGER.info("Extracting embeddings (er2v2_embed_batched)...")
    emb_all, valid_mask = er2v2_embed_batched(
        spk_sess,
        args.spk_model,
        audio_chunks,
        sr_target,
        emb_cfg,
    )

    valid_mask = np.asarray(valid_mask, dtype=bool)
    if emb_all.shape[0] == 0 or not np.any(valid_mask):
        LOGGER.warning("No valid embeddings extracted; aborting")
        return

    emb = emb_all[valid_mask]
    seg_starts_valid = seg_starts_arr[valid_mask]
    seg_ends_valid = seg_ends_arr[valid_mask]
    seg_vad_means_valid = seg_vad_means_arr[valid_mask]

    LOGGER.info(
        "Embeddings: shape=%s, segments(valid)=%d",
        emb.shape,
        seg_starts_valid.shape[0],
    )

    # KMeans (話者数 = n_speakers)
    LOGGER.info("Running KMeans clustering (n_speakers=%d)...", args.n_speakers)
    kmeans = KMeans(
        n_clusters=args.n_speakers,
        n_init=10,
        random_state=0,
    )
    kmeans.fit(emb)
    centroids = kmeans.cluster_centers_

    # コサイン類似度 → 確率
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    cen_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    sims = emb_norm @ cen_norm.T  # [N, S]
    probs = softmax(sims, axis=1, tau=args.logit_scale)

    # 時間グリッドに写像してヒートマップを作る
    grid_times, heatmap, labels = build_prob_heatmap(
        seg_starts_valid,
        seg_ends_valid,
        probs,
        t0,
        t1,
        grid_hz=float(grid_hz),
    )

    # npz でデバッグ情報を保存
    npz_path = args.out_dir / "campplus_window_debug.npz"
    LOGGER.info("Saving debug npz to: %s", npz_path)
    np.savez(
        npz_path,
        grid_times=grid_times,
        heatmap=heatmap,
        labels=labels,
        seg_starts=seg_starts_valid,
        seg_ends=seg_ends_valid,
        seg_vad_means=seg_vad_means_valid,
        probs=probs,
        sims=sims,
        vad_probs=vad_probs_local,
        vad_times=vad_times_abs,
    )

    # 図を保存
    LOGGER.info("Plotting heatmap, labels, and VAD...")
    plot_heatmap(
        grid_times,
        heatmap,
        args.out_dir / "campplus_probs_heatmap.png",
    )
    plot_labels(
        grid_times,
        labels,
        args.out_dir / "campplus_labels_timeline.png",
    )
    plot_vad(
        vad_times_abs,
        vad_probs_local,
        args.out_dir / "vad_probs_window.png",
    )

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
