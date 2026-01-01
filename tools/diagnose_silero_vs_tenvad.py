#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_silero_vs_tenvad.py

Silero VAD (ONNX) と TEN VAD (C binding: TenVad) を
同じ音声に対して比較するスクリプト。

- 16kHz mono に変換
- Silero / TEN それぞれでフレームごとの VAD 確率列を取得
- 閾値に基づくセグメント統計 (speech_ratio, num_segments, 平均/中央値長)
- 共通 time grid 上での IoU を計算
- JSON と PNG に結果保存
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

# リポジトリ直下 (ahe_whisper パッケージの親ディレクトリ) を import path に追加
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ten_vad import TenVad  # C バインディング版 TEN VAD
from ahe_whisper.config import VadConfig, AlignerConfig
from ahe_whisper.vad import VAD as AHEVAD

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("diagnose_silero_vs_tenvad")


class VadResult:
    """
    VAD 出力コンテナ。
    probs: shape=(T,) の [0,1] 確率列
    times: shape=(T,) の秒単位時刻
    name : "silero" / "ten" など
    """

    def __init__(self, probs: np.ndarray, times: np.ndarray, name: str) -> None:
        assert probs.shape == times.shape
        self.probs = probs.astype(np.float32)
        self.times = times.astype(np.float32)
        self.name = name

    @property
    def frame_dt(self) -> float:
        if self.times.size <= 1:
            return 0.0
        return float(self.times[1] - self.times[0])

    @property
    def duration(self) -> float:
        if self.times.size == 0:
            return 0.0
        return float(self.times[-1])


def load_audio_mono(file_path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    音声ファイルを target_sr Hz mono で読み込む。
    """
    LOGGER.info("Loading audio: %s", file_path)
    y, sr = librosa.load(str(file_path), sr=None, mono=True)
    if sr != target_sr:
        LOGGER.info("Resampling from %d Hz to %d Hz", sr, target_sr)
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    LOGGER.info("Audio loaded: duration=%.2f sec, sr=%d", len(y) / sr, sr)
    return y.astype(np.float32), sr


# =========================
# Silero VAD (ONNX)
# =========================

class SileroOnnxRunner:
    """
    Silero VAD の ONNXRuntime 実装（簡易版）。
    - audio: 16kHz mono float32 波形
    - win_samples / hop_samples: サンプル数
    """

    def __init__(self, model_path: Path, win_samples: int = 512, hop_samples: int = 256) -> None:
        self.model_path = model_path
        self.win_samples = win_samples
        self.hop_samples = hop_samples

        LOGGER.info("Loading Silero VAD ONNX model: %s", model_path)
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        self.inputs = self.session.get_inputs()
        self.input_names = [i.name for i in self.inputs]
        self.output_names = [o.name for o in self.session.get_outputs()]
        LOGGER.info("Silero IO: inputs=%s, outputs=%s", self.input_names, self.output_names)

        # state 入力を使うかどうか
        self.uses_state = any("state" in n.lower() for n in self.input_names)

        # state の shape をモデルから取得
        self.state_shape = None
        if self.uses_state:
            for inp in self.inputs:
                if "state" in inp.name.lower():
                    # inp.shape は [2,1,128] など。None が混じる場合は 1 に潰す
                    shape = []
                    for d in inp.shape:
                        if isinstance(d, int) and d > 0:
                            shape.append(d)
                        else:
                            shape.append(1)
                    self.state_shape = tuple(shape)
                    LOGGER.info("Silero state input: name=%s, shape=%s", inp.name, self.state_shape)
                    break

        self.state = None
        self._reset_state()

    def _reset_state(self) -> None:
        if self.uses_state and self.state_shape is not None:
            self.state = np.zeros(self.state_shape, dtype=np.float32)
        else:
            self.state = None

    def __call__(self, audio: np.ndarray, sr: int) -> VadResult:
        """
        音声全体を win/hop でスライドさせながら Silero VAD を実行。
        戻り値は frame ごとの平均確率と、その中心時刻。
        """
        assert sr == 16000, "Silero VAD は 16kHz 前提です"

        win = self.win_samples
        hop = self.hop_samples

        probs_list: List[float] = []
        times_list: List[float] = []

        self._reset_state()

        total_len = len(audio)
        pos = 0

        # input 名の推定
        audio_name = None
        for n in self.input_names:
            ln = n.lower()
            if "input" in ln or "audio" in ln or "samples" in ln:
                audio_name = n
                break
        if audio_name is None:
            audio_name = self.input_names[0]

        # state 入力名の推定
        state_name = None
        if self.uses_state:
            for n in self.input_names:
                if "state" in n.lower():
                    state_name = n
                    break

        # sr 入力名の推定
        sr_name = None
        for n in self.input_names:
            ln = n.lower()
            if ln in ("sr", "sample_rate", "sampling_rate"):
                sr_name = n
                break

        while pos < total_len:
            chunk = audio[pos:pos + win]
            if len(chunk) < win:
                pad = np.zeros((win - len(chunk),), dtype=np.float32)
                chunk = np.concatenate([chunk, pad])

            inputs: Dict[str, np.ndarray] = {}
            # メイン音声
            inputs[audio_name] = chunk.reshape(1, -1).astype(np.float32)

            # state
            if self.uses_state and self.state is not None:
                if state_name is None:
                    state_name = self.input_names[-1]
                inputs[state_name] = self.state

            # sr (int64)
            if sr_name is not None:
                inputs[sr_name] = np.array([sr], dtype=np.int64)

            # 推論
            outputs = self.session.run(None, inputs)

            prob_arr = None
            next_state = None
            for out, name in zip(outputs, self.output_names):
                if "state" in name.lower():
                    next_state = out
                else:
                    prob_arr = out

            if prob_arr is None:
                prob_arr = outputs[0]

            prob_arr = np.asarray(prob_arr, dtype=np.float32).reshape(-1)
            p_mean = float(prob_arr.mean())
            probs_list.append(p_mean)

            center_t = (pos + win / 2) / float(sr)
            times_list.append(center_t)

            if next_state is not None:
                self.state = next_state.astype(np.float32)

            pos += hop

        probs = np.array(probs_list, dtype=np.float32)
        times = np.array(times_list, dtype=np.float32)

        LOGGER.info(
            "Silero VAD: frames=%d, duration=%.2f sec",
            probs.shape[0],
            times[-1] if times.size else 0.0,
        )

        return VadResult(probs=probs, times=times, name="silero")

def run_silero_vad(
    audio: np.ndarray,
    sr: int,
    model_path: Path,
    win_samples: int = 512,
    hop_samples: int = 256,
) -> VadResult:
    """
    本番パイプラインの VAD と同じコードパス
    (ahe_whisper.vad.VAD.get_speech_probabilities) を使って
    Silero の確率列を取得する。
    """

    # まず 16kHz mono に揃える（TEN 側と同じ前処理）
    if sr != 16000:
        LOGGER.info("Silero VAD: resampling from %d Hz to 16kHz", sr)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # 本番と同じ VAD 設定
    # ※ VadConfig は単なる class なので、インスタンス生成後に属性を上書きする
    vad_cfg = VadConfig()
    vad_cfg.window_size_samples = win_samples

    align_cfg = AlignerConfig()
    grid_hz = align_cfg.grid_hz

    vad = AHEVAD(Path(model_path), vad_cfg)
    probs, grid_times = vad.get_speech_probabilities(audio, sr, grid_hz=grid_hz)

    LOGGER.info(
        "Silero VAD (AHE pipeline): frames=%d, duration=%.2f sec, grid_hz=%d",
        probs.shape[0],
        grid_times[-1] if grid_times.size > 0 else 0.0,
        grid_hz,
    )

    return VadResult(probs=probs, times=grid_times, name="silero")


# =========================
# TEN VAD (C binding)
# =========================

def calibrate_ten_probs(
    probs: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """
    TEN の生確率列 probs を
    p_calib = sigmoid(a * (p_raw - b))
    でキャリブレーションする。
    """
    z = (probs - b) * a
    z = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


# =========================
# TEN VAD (C binding)
# =========================

def calibrate_ten_probs(
    probs: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """
    TEN の生確率列 probs を
        p_calib = sigmoid(a * (p_raw - b))
    でキャリブレーションする。
    """
    z = (probs - b) * a
    z = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


def smooth_probs(
    probs: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """
    時間方向の移動平均で平滑化する。
    kernel_size はフレーム数（奇数・偶数どちらでも可）。
    """
    if kernel_size <= 1 or probs.size == 0:
        return probs

    k = int(kernel_size)
    if k <= 1:
        return probs

    # 端は edge で伸ばしてから 1D 畳み込み
    pad = k // 2
    padded = np.pad(probs, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / float(k)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def run_ten_vad(
    audio: np.ndarray,
    sr: int,
    hop_samples: int = 256,
    threshold: float = 0.5,
    calib_a: float = 5.0,
    calib_b: float = 0.5,
    smooth_kernel: int = 5,
) -> VadResult:
    """
    TEN VAD を C バインディング (TenVad) で実行。
    - audio: float32 [-1,1] 波形
    - sr   : 16kHz 前提（違う場合は内部で変換）
    """
    if sr != 16000:
        LOGGER.info("TEN VAD: resampling from %d Hz to 16kHz", sr)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    if audio.ndim == 2:
        audio = audio.mean(axis=0)

    y = np.clip(audio, -1.0, 1.0)
    y_int16 = (y * 32767.0).astype(np.int16)

    vad = TenVad(hop_samples, float(threshold))

    num_frames = y_int16.shape[0] // hop_samples
    probs_raw = np.zeros(num_frames, dtype=np.float32)
    flags = np.zeros(num_frames, dtype=np.int8)

    for i in range(num_frames):
        frame = y_int16[i * hop_samples:(i + 1) * hop_samples]
        p, f = vad.process(frame)
        probs_raw[i] = float(p)
        flags[i] = int(f)

    # 1) ロジスティックでキャリブレーション
    if calib_a != 1.0 or calib_b != 0.5:
        probs_calib = calibrate_ten_probs(probs_raw, calib_a, calib_b)
    else:
        probs_calib = probs_raw

    # 2) 時間方向の移動平均で平滑化
    probs = smooth_probs(probs_calib, smooth_kernel)

    frame_dt = hop_samples / float(sr)
    times = np.arange(num_frames, dtype=np.float32) * frame_dt

    LOGGER.info(
        "TEN VAD: frames=%d, duration=%.2f sec, speech_ratio=%.3f",
        num_frames,
        times[-1] if times.size else 0.0,
        float(flags.mean()) if num_frames > 0 else 0.0,
    )

    return VadResult(probs=probs, times=times, name="ten")


# =========================
# セグメント & 統計
# =========================

def create_segments(probs: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int]]:
    """
    VAD確率からセグメント (start_idx, end_idx) を生成。
    """
    segments: List[Tuple[int, int]] = []
    in_segment = False
    start_idx = 0

    for i, p in enumerate(probs):
        if p > threshold and not in_segment:
            start_idx = i
            in_segment = True
        elif p <= threshold and in_segment:
            segments.append((start_idx, i))
            in_segment = False

    if in_segment:
        segments.append((start_idx, len(probs)))

    return segments


def calculate_statistics(
    segments: List[Tuple[int, int]],
    total_frames: int,
    frame_dt: float,
) -> Dict[str, float]:
    """
    セグメント統計を計算。
    """
    if total_frames <= 0:
        return {
            "speech_ratio": 0.0,
            "num_segments": 0.0,
            "avg_segment_frames": 0.0,
            "median_segment_frames": 0.0,
            "avg_segment_sec": 0.0,
            "median_segment_sec": 0.0,
        }

    if not segments:
        return {
            "speech_ratio": 0.0,
            "num_segments": 0.0,
            "avg_segment_frames": 0.0,
            "median_segment_frames": 0.0,
            "avg_segment_sec": 0.0,
            "median_segment_sec": 0.0,
        }

    lengths = np.array([end - start for start, end in segments], dtype=np.float32)
    total_speech_frames = float(lengths.sum())
    speech_ratio = total_speech_frames / float(total_frames)

    avg_frames = float(lengths.mean())
    median_frames = float(np.median(lengths))

    avg_sec = avg_frames * frame_dt
    median_sec = median_frames * frame_dt

    return {
        "speech_ratio": speech_ratio,
        "num_segments": float(len(segments)),
        "avg_segment_frames": avg_frames,
        "median_segment_frames": median_frames,
        "avg_segment_sec": avg_sec,
        "median_segment_sec": median_sec,
    }

# =========================
# IoU (time grid 揃え)
# =========================

def compute_aligned_iou(
    vad1: VadResult,
    vad2: VadResult,
    threshold: float,
) -> float:
    """
    2つの VAD 出力を共通 time grid に射影して IoU を計算。
    - grid dt は両者の frame_dt のうち小さい方
    - 区間は [0, min(t1_end, t2_end)] を使用
    """
    if vad1.probs.size == 0 or vad2.probs.size == 0:
        return 0.0

    dt1 = max(vad1.frame_dt, 1e-6)
    dt2 = max(vad2.frame_dt, 1e-6)
    dt = min(dt1, dt2)

    t_end = min(vad1.duration, vad2.duration)
    if t_end <= 0.0:
        return 0.0

    t_grid = np.arange(0.0, t_end, dt, dtype=np.float32)

    p1 = np.interp(t_grid, vad1.times, vad1.probs, left=vad1.probs[0], right=vad1.probs[-1])
    p2 = np.interp(t_grid, vad2.times, vad2.probs, left=vad2.probs[0], right=vad2.probs[-1])

    m1 = p1 >= threshold
    m2 = p2 >= threshold

    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 0.0
    return float(inter / union)


# =========================
# 可視化
# =========================

def plot_vad_results(
    audio: np.ndarray,
    sr: int,
    silero: VadResult,
    ten: VadResult,
    output_file: Path,
) -> None:
    """
    波形 + Silero/TEN の VAD 確率をプロット。
    """
    duration = len(audio) / float(sr)
    t_audio = np.linspace(0.0, duration, num=len(audio), endpoint=False)

    plt.figure(figsize=(15, 10))

    # 元の音声
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t_audio, audio)
    ax1.set_title("Original Audio Waveform")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0.0, duration)

    # Silero
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(silero.times, silero.probs, label="Silero VAD")
    ax2.set_title("Silero VAD Output")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0.0, 1.0)
    ax2.legend()

    # TEN
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(ten.times, ten.probs, label="TEN VAD")
    ax3.set_title("TEN VAD Output")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Probability")
    ax3.set_ylim(0.0, 1.0)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    LOGGER.info("Saved plot to %s", output_file)


# =========================
# メイン
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Silero VAD (ONNX) and TEN VAD (C binding) on a single audio file"
    )
    parser.add_argument("input_file", help="Input audio file path")
    parser.add_argument(
        "--silero-model-path",
        type=str,
        required=True,
        help="Path to Silero VAD ONNX model (e.g. models/onnx-community/silero-vad/onnx/model.onnx)",
    )
    parser.add_argument(
        "--silero-win-samples",
        type=int,
        default=512,
        help="Silero VAD window size in samples at 16kHz (default: 512 ≈ 32ms)",
    )
    parser.add_argument(
        "--silero-hop-samples",
        type=int,
        default=256,
        help="Silero VAD hop size in samples at 16kHz (default: 256 ≈ 16ms)",
    )
    parser.add_argument(
        "--ten-hop-samples",
        type=int,
        default=256,
        help="TEN VAD hop size in samples at 16kHz (default: 256 ≈ 16ms)",
    )
    parser.add_argument(
        "--ten-threshold",
        type=float,
        default=0.5,
        help="TEN VAD internal decision threshold (0–1)",
    )
    parser.add_argument(
        "--ten-calib-a",
        type=float,
        default=5.0,
        help="TEN calibration scale a in p_calib = sigmoid(a * (p_raw - b)) (default: 5.0)",
    )
    parser.add_argument(
        "--ten-calib-b",
        type=float,
        default=0.5,
        help="TEN calibration bias b in p_calib = sigmoid(a * (p_raw - b)) (default: 0.5)",
    )
    parser.add_argument(
        "--ten-mask-threshold",
        type=float,
        default=0.4,
        help="Threshold for TEN mask after calibration (default: 0.4; Silero uses --threshold)",
    )
    parser.add_argument(
        "--ten-smooth-kernel",
        type=int,
        default=5,
        help="TEN 確率列の時間方向平滑化に使うカーネル幅 [フレーム数] (default: 5 ≒ 80ms)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for segment statistics / IoU (0–1)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="vad_results",
        help="Output directory path",
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    silero_model_path = Path(args.silero_model_path)
    if not silero_model_path.exists():
        raise FileNotFoundError(f"Silero model not found: {silero_model_path}")

    # 音声読み込み
    audio, sr = load_audio_mono(input_file, target_sr=16000)

    # VAD 実行
    silero_res = run_silero_vad(
        audio,
        sr,
        model_path=silero_model_path,
        win_samples=args.silero_win_samples,
        hop_samples=args.silero_hop_samples,
    )
    ten_res = run_ten_vad(
        audio,
        sr,
        hop_samples=args.ten_hop_samples,
        threshold=args.ten_threshold,
        calib_a=args.ten_calib_a,
        calib_b=args.ten_calib_b,
        smooth_kernel=args.ten_smooth_kernel,        
    )

    # セグメント & 統計
    silero_segments = create_segments(silero_res.probs, threshold=args.threshold)
    ten_thr = args.ten_mask_threshold if args.ten_mask_threshold is not None else args.threshold
    ten_segments = create_segments(ten_res.probs, threshold=ten_thr)

    silero_stats = calculate_statistics(
        silero_segments,
        total_frames=silero_res.probs.size,
        frame_dt=silero_res.frame_dt,
    )
    ten_stats = calculate_statistics(
        ten_segments,
        total_frames=ten_res.probs.size,
        frame_dt=ten_res.frame_dt,
    )

    # IoU
    iou_score = compute_aligned_iou(silero_res, ten_res, threshold=args.threshold)

    # JSON 保存（Silero / TEN の確率列も含める）
    results: Dict[str, Any] = {
        "input_file": str(input_file),
        "sample_rate": sr,
        "threshold": float(args.threshold),
        "silero": {
            "frame_dt": silero_res.frame_dt,
            "duration": silero_res.duration,
            "times": silero_res.times.tolist(),
            "probs": silero_res.probs.tolist(),
            "stats": silero_stats,
        },
        "ten": {
            "frame_dt": ten_res.frame_dt,
            "duration": ten_res.duration,
            "times": ten_res.times.tolist(),
            "probs": ten_res.probs.tolist(),
            "stats": ten_stats,
        },
        "iou_score": float(iou_score),
    }

    output_json = out_dir / "vad_comparison.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved JSON to %s", output_json)

    # 可視化
    output_plot = out_dir / "vad_comparison.png"
    plot_vad_results(audio, sr, silero_res, ten_res, output_plot)

    # コンソール表示
    print(f"VAD比較結果:")
    print(f"  入力ファイル: {input_file}")
    print(f"  閾値: {args.threshold:.2f}")
    print(f"  IoUスコア: {iou_score:.4f}")

    print("\nSilero VAD:")
    print(f"  スピーチ比率:        {silero_stats['speech_ratio']:.4f}")
    print(f"  セグメント数:        {int(silero_stats['num_segments'])}")
    print(f"  平均セグメント長(s): {silero_stats['avg_segment_sec']:.2f}")
    print(f"  中央セグメント長(s): {silero_stats['median_segment_sec']:.2f}")

    print("\nTEN VAD:")
    print(f"  スピーチ比率:        {ten_stats['speech_ratio']:.4f}")
    print(f"  セグメント数:        {int(ten_stats['num_segments'])}")
    print(f"  平均セグメント長(s): {ten_stats['avg_segment_sec']:.2f}")
    print(f"  中央セグメント長(s): {ten_stats['median_segment_sec']:.2f}")

    print(f"\n結果は {output_json} と {output_plot} に保存されました")


if __name__ == "__main__":
    main()
