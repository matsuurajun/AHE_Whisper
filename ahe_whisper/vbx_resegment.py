# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import numpy as np

_EPS = 1e-9
_NEG_INF = -1e30


def _count_switches(path: np.ndarray) -> int:
    if path.size <= 1:
        return 0
    return int(np.sum(path[1:] != path[:-1]))


def _runs(path: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    if path.size == 0:
        return runs
    start = 0
    cur = int(path[0])
    for t in range(1, path.size):
        val = int(path[t])
        if val != cur:
            runs.append((start, t, cur))
            start = t
            cur = val
    runs.append((start, path.size, cur))
    return runs


def _minrun_absorb(path: np.ndarray, log_em: np.ndarray, min_run_frames: int) -> np.ndarray:
    if min_run_frames <= 0 or path.size == 0:
        return path

    out = path.copy()
    changed = True
    while changed:
        changed = False
        runs = _runs(out)
        if len(runs) <= 1:
            break

        for idx, (start, end, _mid) in enumerate(runs):
            if (end - start) >= min_run_frames:
                continue

            left = runs[idx - 1][2] if idx - 1 >= 0 else None
            right = runs[idx + 1][2] if idx + 1 < len(runs) else None

            if left is None and right is None:
                continue

            score_left = float(np.mean(log_em[start:end, left])) if left is not None else None
            score_right = float(np.mean(log_em[start:end, right])) if right is not None else None

            if score_left is None:
                target = int(right)
            elif score_right is None:
                target = int(left)
            else:
                target = int(left) if score_left >= score_right else int(right)

            out[start:end] = target
            changed = True
            break
    return out


def vbx_resegment(
    spk_probs: np.ndarray,
    vad_probs: Optional[np.ndarray],
    grid_hz: int,
    p_stay_speech: float,
    p_stay_silence: float,
    speech_th: float,
    out_hard_mix: float,
    min_run_sec: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if spk_probs.ndim != 2:
        raise ValueError(f"spk_probs must be 2D, got {spk_probs.ndim}D")

    T, K = spk_probs.shape
    if T == 0 or K == 0:
        return spk_probs, {"ok": True, "T": int(T), "K": int(K)}

    probs = np.clip(spk_probs.astype(np.float32), _EPS, 1.0)
    probs = probs / np.clip(np.sum(probs, axis=1, keepdims=True), _EPS, None)
    log_em = np.log(probs)

    def _log_trans(p_stay: float) -> np.ndarray:
        p_stay = float(np.clip(p_stay, 0.0, 1.0))
        if K == 1:
            return np.zeros((1, 1), dtype=np.float32)
        p_switch = (1.0 - p_stay) / float(max(1, K - 1))
        lt = np.full((K, K), np.log(np.clip(p_switch, _EPS, 1.0)), dtype=np.float32)
        np.fill_diagonal(lt, np.log(np.clip(p_stay, _EPS, 1.0)))
        return lt

    lt_speech = _log_trans(p_stay_speech)
    lt_silence = _log_trans(p_stay_silence)

    log_pi = np.full(K, -np.log(float(K)), dtype=np.float32)
    dp = np.full((T, K), _NEG_INF, dtype=np.float32)
    bp = np.zeros((T, K), dtype=np.int32)
    dp[0] = log_pi + log_em[0]

    use_vad = vad_probs is not None and vad_probs.shape[0] == T
    for t in range(1, T):
        lt = lt_speech
        if use_vad and float(vad_probs[t]) < speech_th:
            lt = lt_silence
        scores = dp[t - 1][:, None] + lt
        bp[t] = np.argmax(scores, axis=0)
        dp[t] = scores[bp[t], np.arange(K)] + log_em[t]

    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = int(np.argmax(dp[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]

    switches_pre = _count_switches(np.argmax(probs, axis=1))
    switches_viterbi = _count_switches(path)

    min_run_frames = int(round(float(min_run_sec) * float(grid_hz)))
    if min_run_frames > 1:
        path = _minrun_absorb(path, log_em, min_run_frames)
    switches_post = _count_switches(path)

    out_hard_mix = float(np.clip(out_hard_mix, 0.0, 1.0))
    hard = np.zeros((T, K), dtype=np.float32)
    hard[np.arange(T), path] = 1.0
    out_probs = (1.0 - out_hard_mix) * probs + out_hard_mix * hard
    out_probs = out_probs / np.clip(np.sum(out_probs, axis=1, keepdims=True), _EPS, None)

    diag = {
        "ok": True,
        "T": int(T),
        "K": int(K),
        "switches_pre": int(switches_pre),
        "switches_viterbi": int(switches_viterbi),
        "switches_post": int(switches_post),
        "min_run_frames": int(min_run_frames),
    }
    return out_probs.astype(np.float32), diag
