#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


Role = int  # 0=GUEST, 1=HOST


@dataclass
class FrameGrid:
    times: np.ndarray  # shape (T,)
    roles_gold: np.ndarray  # shape (T,), values in {0,1} or -1 for "no label"


@dataclass
class UpperBoundResult:
    accuracy: float
    confusion: np.ndarray  # shape (2,2) gold x pred, in seconds
    per_role_acc: Dict[str, float]
    best_mapping: Dict[int, Role]


def parse_hms(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_gold_segments(path: Path) -> List[Dict]:
    segments: List[Dict] = []
    cur_speaker: str | None = None
    cur_start: float | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("[") and "]" in line:
                # e.g. "[00:00:07] SPEAKER_00:"
                ts = line[1:9]
                start = parse_hms(ts)
                parts = line.split("]")
                rest = parts[1].strip() if len(parts) > 1 else ""
                speaker = None
                if rest:
                    # "SPEAKER_00:" -> "SPEAKER_00"
                    speaker = rest.split()[0].rstrip(":")
                if speaker is None:
                    continue
                if cur_speaker is not None and cur_start is not None:
                    segments.append(
                        {
                            "start": cur_start,
                            "end": start,
                            "speaker": cur_speaker,
                        }
                    )
                cur_speaker = speaker
                cur_start = start
            else:
                # text line -> ignore for timing purposes
                continue

    if cur_speaker is not None and cur_start is not None:
        segments.append(
            {
                "start": cur_start,
                "end": None,  # to be filled later with duration
                "speaker": cur_speaker,
            }
        )
    return segments


def finalize_segment_ends(
    segments: List[Dict],
    audio_duration: float,
) -> List[Dict]:
    fixed: List[Dict] = []
    for i, seg in enumerate(segments):
        start = float(seg["start"])
        if i + 1 < len(segments):
            end = float(segments[i + 1]["start"])
        else:
            end = float(audio_duration)
        if end <= start:
            continue
        fixed.append({"start": start, "end": end, "speaker": seg["speaker"]})
    return fixed


def build_frame_grid(
    segments: List[Dict],
    host_speakers: List[str],
    duration_sec: float,
    frame_hz: float,
) -> FrameGrid:
    dt = 1.0 / frame_hz
    num_frames = int(np.round(duration_sec / dt))
    times = np.arange(num_frames, dtype=np.float32) * dt
    roles = np.full(num_frames, -1, dtype=np.int8)

    host_set = set(host_speakers)

    for seg in segments:
        s = float(seg["start"])
        e = float(seg["end"])
        spk = seg["speaker"]
        role: Role = 1 if spk in host_set else 0
        i0 = int(np.floor(s / dt))
        i1 = int(np.ceil(e / dt))
        i0 = max(i0, 0)
        i1 = min(i1, num_frames)
        if i1 <= i0:
            continue
        roles[i0:i1] = role

    return FrameGrid(times=times, roles_gold=roles)


def best_oracle_from_spk_probs(
    frame_times: np.ndarray,
    spk_probs: np.ndarray,
    grid: FrameGrid,
    frame_hz: float,
) -> UpperBoundResult:
    dt = 1.0 / frame_hz
    num_frames = grid.roles_gold.shape[0]

    if frame_times.shape[0] != num_frames:
        raise ValueError(
            f"frame_times length {frame_times.shape[0]} "
            f"!= expected {num_frames} (from duration & frame_hz)"
        )
    if spk_probs.shape[0] != num_frames:
        raise ValueError(
            f"spk_probs length {spk_probs.shape[0]} "
            f"!= expected {num_frames} (from duration & frame_hz)"
        )
    if spk_probs.shape[1] != 2:
        raise ValueError("upper bound script currently assumes 2 speakers (2 columns).")

    # raw cluster argmax per frame: 0 or 1
    cluster_idx = np.argmax(spk_probs, axis=1).astype(np.int8)

    # gold-labeled frames only
    mask = grid.roles_gold >= 0
    gold = grid.roles_gold[mask]
    cluster = cluster_idx[mask]

    # two possible mappings: cluster 0/1 -> role 0/1
    mappings: List[Dict[int, Role]] = [
        {0: 0, 1: 1},
        {0: 1, 1: 0},
    ]

    best_acc = -1.0
    best_mapping: Dict[int, Role] = {}
    best_pred_roles: np.ndarray | None = None

    for mapping in mappings:
        pred_roles = np.vectorize(lambda c: mapping[int(c)])(cluster)
        correct = (pred_roles == gold)
        acc = float(correct.mean())
        if acc > best_acc:
            best_acc = acc
            best_mapping = mapping
            best_pred_roles = pred_roles

    assert best_pred_roles is not None

    confusion = np.zeros((2, 2), dtype=np.float64)
    for g, p in zip(gold, best_pred_roles):
        confusion[g, p] += dt

    per_role_acc: Dict[str, float] = {}
    for role_val, name in [(0, "GUEST"), (1, "HOST")]:
        total = confusion[role_val, :].sum()
        correct = confusion[role_val, role_val]
        per_role_acc[name] = float(correct / total) if total > 0.0 else float("nan")

    return UpperBoundResult(
        accuracy=best_acc,
        confusion=confusion,
        per_role_acc=per_role_acc,
        best_mapping=best_mapping,
    )


def print_report(
    result: UpperBoundResult,
    frame_hz: float,
    duration_sec: float,
) -> None:
    dt = 1.0 / frame_hz
    total_frames = int(np.round(duration_sec / dt))
    total_sec = total_frames * dt

    print("=== CampPlus Upper-Bound Oracle (frame-wise, argmax(spk_probs)) ===")
    print(f"- Frame step        : {dt:.3f} sec ({frame_hz:.1f} Hz)")
    print(f"- Total frames      : {total_frames}  ({total_sec:.1f} sec)")
    print(
        "- Best cluster→role mapping (0=GUEST, 1=HOST): "
        + ", ".join(
            f"cluster_{k}→{v} ({'GUEST' if v == 0 else 'HOST'})"
            for k, v in sorted(result.best_mapping.items())
        )
    )
    print()
    print(f"Upper-bound accuracy: {result.accuracy:.4f}")
    print()
    print("Confusion matrix (durations in seconds)")
    print("  gold\\pred   GUEST       HOST")
    g_guest = result.confusion[0, 0]
    g_guest_host = result.confusion[0, 1]
    g_host_guest = result.confusion[1, 0]
    g_host = result.confusion[1, 1]
    print(f"  GUEST     {g_guest:8.1f}  {g_guest_host:8.1f}")
    print(f"  HOST      {g_host_guest:8.1f}  {g_host:8.1f}")
    print()
    print("Per-role accuracy (by gold role)")
    print(
        f"  GUEST: {result.per_role_acc['GUEST']:.4f}  "
        f"(correct {g_guest:.1f}s / total {(g_guest + g_guest_host):.1f}s)"
    )
    print(
        f"  HOST : {result.per_role_acc['HOST']:.4f}  "
        f"(correct {g_host:.1f}s / total {(g_host + g_host_guest):.1f}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_transcript", type=Path)
    parser.add_argument("spk_probs_npz", type=Path)
    parser.add_argument("--host-speakers", type=str, required=True)
    parser.add_argument("--frame-hz", type=float, default=10.0)
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Optional override. If omitted, taken from npz['duration_sec'] or "
        "last frame time.",
    )
    args = parser.parse_args()

    host_speakers = [s.strip() for s in args.host_speakers.split(",") if s.strip()]

    data = np.load(args.spk_probs_npz)
    frame_times = data["frame_times"].astype(np.float32)
    spk_probs = data["spk_probs"].astype(np.float32)
    if "duration_sec" in data:
        duration_sec = float(data["duration_sec"])
    else:
        duration_sec = float(frame_times[-1])

    if args.duration_sec is not None:
        duration_sec = float(args.duration_sec)

    segments = load_gold_segments(args.gold_transcript)
    segments = finalize_segment_ends(segments, duration_sec)

    grid = build_frame_grid(
        segments=segments,
        host_speakers=host_speakers,
        duration_sec=duration_sec,
        frame_hz=args.frame_hz,
    )

    result = best_oracle_from_spk_probs(
        frame_times=frame_times,
        spk_probs=spk_probs,
        grid=grid,
        frame_hz=args.frame_hz,
    )

    print_report(result, frame_hz=args.frame_hz, duration_sec=duration_sec)


if __name__ == "__main__":
    main()
