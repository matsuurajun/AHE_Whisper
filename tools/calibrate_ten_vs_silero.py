#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Silero マスクを擬似ラベルにして TEN VAD の a,b,t をグリッドサーチで最適化するスクリプト"
    )
    p.add_argument(
        "json_path",
        type=Path,
        help="diagnose_silero_vs_tenvad.py が出力した vad_comparison.json",
    )
    p.add_argument(
        "--silero-threshold",
        type=float,
        default=0.5,
        help="Silero マスク生成に使うしきい値 (default: 0.5)",
    )
    p.add_argument(
        "--a-grid",
        type=str,
        default="0.5,1.0,2.0,3.0,4.0,5.0",
        help="TEN のスケーリング係数 a の候補 (カンマ区切り)",
    )
    p.add_argument(
        "--b-grid",
        type=str,
        default="0.2,0.3,0.4,0.5,0.6,0.7,0.8",
        help="TEN のバイアス b の候補 (カンマ区切り)",
    )
    p.add_argument(
        "--t-grid",
        type=str,
        default="0.4,0.5,0.6",
        help="キャリブレーション後のしきい値 t の候補 (カンマ区切り)",
    )
    return p.parse_args()


def parse_grid(s: str) -> np.ndarray:
    vals = [float(x) for x in s.split(",") if x.strip() != ""]
    return np.asarray(vals, dtype=np.float32)


def load_probs_from_json(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    sil = data["silero"]
    ten = data["ten"]

    sil_times = np.asarray(sil["times"], dtype=np.float32)
    sil_probs = np.asarray(sil["probs"], dtype=np.float32)

    ten_times = np.asarray(ten["times"], dtype=np.float32)
    ten_probs = np.asarray(ten["probs"], dtype=np.float32)

    return sil_times, sil_probs, ten_times, ten_probs


def align_ten_to_silero(
    sil_times: np.ndarray,
    ten_times: np.ndarray,
    ten_probs: np.ndarray,
) -> np.ndarray:
    # VadResult.times は単調増加のはずなので、そのまま np.interp に渡す
    ten_probs_interp = np.interp(sil_times, ten_times, ten_probs)
    return ten_probs_interp.astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray):
    assert y_true.shape == y_pred.shape
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    if tp == 0.0 and fp == 0.0 and fn == 0.0:
        return 1.0, 1.0, 1.0

    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def main() -> None:
    args = parse_args()

    a_grid = parse_grid(args.a_grid)
    b_grid = parse_grid(args.b_grid)
    t_grid = parse_grid(args.t_grid)

    sil_times, sil_probs, ten_times, ten_probs = load_probs_from_json(args.json_path)
    ten_probs_aligned = align_ten_to_silero(sil_times, ten_times, ten_probs)

    # Silero を擬似ラベルにした 0/1 マスク
    sil_mask = (sil_probs >= args.silero_threshold).astype(np.int8)

    best = {
        "a": None,
        "b": None,
        "t": None,
        "precision": -1.0,
        "recall": -1.0,
        "f1": -1.0,
    }

    for a in a_grid:
        for b in b_grid:
            z = (ten_probs_aligned - b) * a
            p_calib = sigmoid(z)

            for t in t_grid:
                y_pred = (p_calib >= t).astype(np.int8)
                precision, recall, f1 = compute_f1(sil_mask, y_pred)

                if f1 > best["f1"]:
                    best.update(
                        {
                            "a": float(a),
                            "b": float(b),
                            "t": float(t),
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                        }
                    )

    print("=== TEN calibration result (Silero pseudo-label) ===")
    print(f"  json_path        : {args.json_path}")
    print(f"  silero_threshold : {args.silero_threshold:.3f}")
    print(f"  search space     : |a|={len(a_grid)}, |b|={len(b_grid)}, |t|={len(t_grid)}")
    print()
    print(f"  best a           : {best['a']}")
    print(f"  best b           : {best['b']}")
    print(f"  best t           : {best['t']}")
    print(f"  precision        : {best['precision']:.4f}")
    print(f"  recall           : {best['recall']:.4f}")
    print(f"  F1               : {best['f1']:.4f}")


if __name__ == "__main__":
    main()
