# -*- coding: utf-8 -*-
"""
eval_two_speaker_from_transcripts.py

人力で修正したタイムコード付きトランスクリプト（gold）と、
システム出力のタイムコード付きトランスクリプト（system）を比較して、

  - gold 側の複数話者（例: SPEAKER_00, SPEAKER_03）を「HOST」役
  - それ以外を「GUEST」役

という 2 クラスにまとめた上で、
0.1 秒グリッド（デフォルト）でのラベル一致率を評価する。

さらに、誤分類区間（GUEST→HOST, HOST→GUEST）を
連続区間として抽出して表示する。
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np


TS_RE = re.compile(
    r"^\[(\d+):(\d+):(\d+)(?:\.(\d+))?\]\s+([A-Z0-9_]+):"
)


@dataclass
class Segment:
    start: float
    speaker: str
    text: str


# ---------------------------------------------------------------------
# パーサ
# ---------------------------------------------------------------------


def parse_segments(path: Path) -> List[Segment]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    segments: List[Segment] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = TS_RE.match(line)
        if not m:
            i += 1
            continue

        hh = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3))
        ms = m.group(4)
        t = hh * 3600 + mm * 60 + ss + (int(ms) / 1000.0 if ms else 0.0)
        spk = m.group(5)

        j = i + 1
        texts: List[str] = []
        while j < len(lines):
            line_j = lines[j]
            if TS_RE.match(line_j.strip()):
                break
            if line_j.strip() == "":
                j += 1
                break
            texts.append(line_j.rstrip())
            j += 1

        seg_text = " ".join(texts).strip()
        segments.append(Segment(start=float(t), speaker=spk, text=seg_text))
        i = j

    segments.sort(key=lambda s: s.start)
    return segments


def estimate_tail_margin(starts: List[float]) -> float:
    if len(starts) < 2:
        return 1.0
    gaps = [b - a for a, b in zip(starts[:-1], starts[1:]) if b > a]
    if not gaps:
        return 1.0
    gaps.sort()
    mid = gaps[len(gaps) // 2]
    return float(max(1.0, min(15.0, mid)))


def build_time_grid(
    gold_segments: List[Segment],
    sys_segments: List[Segment],
    frame_hz: float,
) -> np.ndarray:
    assert gold_segments and sys_segments
    start = min(gold_segments[0].start, sys_segments[0].start)
    gold_tail = estimate_tail_margin([s.start for s in gold_segments])
    sys_tail = estimate_tail_margin([s.start for s in sys_segments])
    tail = max(gold_tail, sys_tail)
    end = max(gold_segments[-1].start, sys_segments[-1].start) + tail
    dt = 1.0 / frame_hz
    return np.arange(start, end, dt, dtype=np.float32)


def segments_to_labels(
    segments: List[Segment],
    times: np.ndarray,
    default_label: str = "UNKNOWN",
) -> List[str]:
    if not segments:
        return [default_label] * len(times)

    segs = sorted(segments, key=lambda s: s.start)
    labels: List[str] = []
    seg_idx = 0
    for t in times:
        while seg_idx + 1 < len(segs) and t >= segs[seg_idx + 1].start:
            seg_idx += 1
        if t >= segs[0].start:
            labels.append(segs[seg_idx].speaker)
        else:
            labels.append(default_label)
    return labels


# ---------------------------------------------------------------------
# ラベル変換・評価
# ---------------------------------------------------------------------


def gold_labels_to_binary(
    labels: List[str],
    host_speakers: List[str],
) -> List[int]:
    host_set = set(host_speakers)
    # 1 = HOST, 0 = GUEST
    return [1 if lab in host_set else 0 for lab in labels]


def evaluate_best_mapping(
    sys_labels: List[str],
    gold_binary: List[int],
) -> Tuple[float, Dict[str, int], List[List[int]]]:
    """
    sys_labels: システムの話者ラベル列（例: SPEAKER_00 / SPEAKER_01）
    gold_binary: 0=GUEST, 1=HOST

    戻り値:
        accuracy, mapping (sys_speaker -> {0,1}), confusion[gold][pred]
    """
    assert len(sys_labels) == len(gold_binary)
    sys_speakers = sorted(set(sys_labels))
    K = len(sys_speakers)
    idx = {spk: i for i, spk in enumerate(sys_speakers)}

    best_acc = -1.0
    best_mask = None
    best_counts: List[List[int]] = [[0, 0], [0, 0]]
    total = len(gold_binary)

    # mask のビットが 1 なら HOST, 0 なら GUEST
    # all-guest (0) / all-host (2^K-1) は除外
    for mask in range(1, (1 << K) - 1):
        counts = [[0, 0], [0, 0]]
        correct = 0
        for g, sl in zip(gold_binary, sys_labels):
            p = 1 if (mask & (1 << idx[sl])) else 0
            counts[g][p] += 1
            if g == p:
                correct += 1
        acc = correct / float(total)
        if acc > best_acc:
            best_acc = acc
            best_mask = mask
            best_counts = counts

    if best_mask is None:
        mapping = {spk: 0 for spk in sys_speakers}
        return 0.0, mapping, best_counts

    mapping = {
        sys_speakers[i]: (1 if (best_mask & (1 << i)) else 0)
        for i in range(K)
    }
    return best_acc, mapping, best_counts


# ---------------------------------------------------------------------
# 誤分類区間検出
# ---------------------------------------------------------------------


def find_error_intervals(
    times: np.ndarray,
    gold_binary: List[int],
    sys_labels: List[str],
    mapping: Dict[str, int],
    frame_hz: float,
    target_from: int,
    target_to: int,
    min_dur: float,
) -> List[Tuple[float, float, float]]:
    """
    gold_binary: 0=GUEST,1=HOST
    mapping: sys_speaker -> {0,1}
    target_from, target_to: 例) 0→1 (GUEST→HOST)
    """
    dt = 1.0 / frame_hz
    pred_roles = [mapping[sl] for sl in sys_labels]

    intervals: List[Tuple[float, float, float]] = []
    in_err = False
    start_t: float = 0.0

    for i, (g, p) in enumerate(zip(gold_binary, pred_roles)):
        is_err = (g == target_from) and (p == target_to)
        if is_err and not in_err:
            in_err = True
            start_t = float(times[i])
        elif not is_err and in_err:
            # 直前までエラーだったので閉じる
            end_t = float(times[i])  # times[i] が次フレームの開始
            dur = end_t - start_t
            if dur >= min_dur:
                intervals.append((start_t, end_t, dur))
            in_err = False

    if in_err:
        # 最後までエラーが続いていた場合
        end_t = float(times[-1] + dt)
        dur = end_t - start_t
        if dur >= min_dur:
            intervals.append((start_t, end_t, dur))

    return intervals


def print_error_intervals(
    label: str,
    intervals: List[Tuple[float, float, float]],
) -> None:
    print(f"=== Misclassified intervals ({label}) ===")
    if not intervals:
        print("  (none)")
        print()
        return

    total = sum(d for _, _, d in intervals)
    print(f"  count : {len(intervals)}")
    print(f"  total : {total:.1f} sec")
    print("  #   start[s]   end[s]   dur[s]")
    for i, (s, e, d) in enumerate(intervals, 1):
        print(f"  {i:2d}  {s:8.2f}  {e:8.2f}  {d:7.2f}")
    print()


# ---------------------------------------------------------------------
# レポート
# ---------------------------------------------------------------------


def print_report(
    frame_hz: float,
    times: np.ndarray,
    host_speakers: List[str],
    acc: float,
    mapping: Dict[str, int],
    confusion: List[List[int]],
) -> None:
    dt = 1.0 / frame_hz
    total_frames = len(times)
    total_sec = total_frames * dt

    guest_frames = confusion[0][0] + confusion[0][1]
    host_frames = confusion[1][0] + confusion[1][1]

    print("=== Diarization Evaluation (HOST vs GUEST) ===")
    print(f"- Frame step        : {dt:.3f} sec ({frame_hz:.1f} Hz)")
    print(f"- Total frames      : {total_frames}  ({total_sec:.1f} sec)")
    print(f"- Gold HOST speakers: {', '.join(host_speakers)}")
    print()

    print("Best system → role mapping (0=GUEST, 1=HOST):")
    for spk, role in mapping.items():
        role_name = "HOST" if role == 1 else "GUEST"
        print(f"  {spk:10s} → {role} ({role_name})")
    print()

    print(f"Overall accuracy    : {acc:.4f}")
    print()

    def frames_to_sec(fr: int) -> float:
        return fr * dt

    print("Confusion matrix (durations in seconds)")
    print("  gold\\pred   GUEST       HOST")
    print(
        "  GUEST     "
        f"{frames_to_sec(confusion[0][0]):7.1f}   {frames_to_sec(confusion[0][1]):7.1f}"
    )
    print(
        "  HOST      "
        f"{frames_to_sec(confusion[1][0]):7.1f}   {frames_to_sec(confusion[1][1]):7.1f}"
    )
    print()

    if guest_frames > 0:
        guest_acc = confusion[0][0] / float(guest_frames)
    else:
        guest_acc = 0.0
    if host_frames > 0:
        host_acc = confusion[1][1] / float(host_frames)
    else:
        host_acc = 0.0

    print("Per-role accuracy (by gold role)")
    print(
        f"  GUEST: {guest_acc:.4f}  "
        f"(correct {frames_to_sec(confusion[0][0]):.1f}s / total {frames_to_sec(guest_frames):.1f}s)"
    )
    print(
        f"  HOST : {host_acc:.4f}  "
        f"(correct {frames_to_sec(confusion[1][1]):.1f}s / total {frames_to_sec(host_frames):.1f}s)"
    )
    print()


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "人力トランスクリプト（gold）とシステム出力（system）を比較し、"
            "HOST vs GUEST の 2 クラスで 0.1 秒グリッド評価＋誤分類区間一覧を出すスクリプト。"
        )
    )
    parser.add_argument(
        "gold",
        type=Path,
        help="人力で修正したトランスクリプト（例: tmpXXX_rewrite.txt）",
    )
    parser.add_argument(
        "system",
        type=Path,
        help="システム出力のトランスクリプト（例: tmpYYY.txt）",
    )
    parser.add_argument(
        "--host-speakers",
        type=str,
        default="SPEAKER_00,SPEAKER_03",
        help=(
            "gold 側で HOST とみなす話者 ID をカンマ区切りで指定 "
            "(デフォルト: 'SPEAKER_00,SPEAKER_03')"
        ),
    )
    parser.add_argument(
        "--frame-hz",
        type=float,
        default=10.0,
        help="評価グリッドのフレームレート [Hz] (デフォルト: 10.0 → 0.1 秒刻み)",
    )
    parser.add_argument(
        "--min-error-dur",
        type=float,
        default=0.5,
        help="誤分類区間として列挙する最小長さ [sec] (デフォルト: 0.5)",
    )

    args = parser.parse_args()

    host_speakers = [
        s.strip() for s in args.host_speakers.split(",") if s.strip()
    ]
    if not host_speakers:
        raise SystemExit("ERROR: --host-speakers を 1 つ以上指定してください。")

    print(f"[INFO] Loading gold transcript   : {args.gold}")
    print(f"[INFO] Loading system transcript : {args.system}")
    gold_segments = parse_segments(args.gold)
    sys_segments = parse_segments(args.system)

    if not gold_segments:
        raise SystemExit("ERROR: gold 側でセグメントが 1 つも見つかりませんでした。")
    if not sys_segments:
        raise SystemExit("ERROR: system 側でセグメントが 1 つも見つかりませんでした。")

    print(
        f"[INFO] Gold segments : {len(gold_segments)} "
        f"(speakers={sorted({s.speaker for s in gold_segments})})"
    )
    print(
        f"[INFO] System segments: {len(sys_segments)} "
        f"(speakers={sorted({s.speaker for s in sys_segments})})"
    )

    times = build_time_grid(gold_segments, sys_segments, frame_hz=args.frame_hz)
    print(
        f"[INFO] Time grid: {len(times)} frames, "
        f"range=[{times[0]:.2f}, {times[-1]:.2f}] sec, "
        f"step={1.0/args.frame_hz:.3f} sec"
    )

    gold_labels = segments_to_labels(gold_segments, times)
    sys_labels = segments_to_labels(sys_segments, times)

    gold_binary = gold_labels_to_binary(gold_labels, host_speakers)
    acc, mapping, confusion = evaluate_best_mapping(sys_labels, gold_binary)
    print()
    print_report(
        frame_hz=args.frame_hz,
        times=times,
        host_speakers=host_speakers,
        acc=acc,
        mapping=mapping,
        confusion=confusion,
    )

    # 誤分類区間を抽出
    g2h = find_error_intervals(
        times,
        gold_binary,
        sys_labels,
        mapping,
        frame_hz=args.frame_hz,
        target_from=0,
        target_to=1,
        min_dur=args.min_error_dur,
    )
    h2g = find_error_intervals(
        times,
        gold_binary,
        sys_labels,
        mapping,
        frame_hz=args.frame_hz,
        target_from=1,
        target_to=0,
        min_dur=args.min_error_dur,
    )

    print_error_intervals("GUEST → HOST (false HOST)", g2h)
    print_error_intervals("HOST → GUEST (false GUEST)", h2g)


if __name__ == "__main__":
    main()
