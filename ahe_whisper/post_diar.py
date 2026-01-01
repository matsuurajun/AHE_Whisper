# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

LOGGER = logging.getLogger("ahe_whisper_post_diar")


def collect_speaker_stats(segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for seg in segments:
        spk = seg.get("speaker")
        if not spk:
            continue
        try:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)
        except (TypeError, ValueError):
            continue

        dur = end - start
        if dur <= 0.0:
            continue

        rec = stats.setdefault(spk, {"duration": 0.0, "count": 0.0})
        rec["duration"] += dur
        rec["count"] += 1.0

    return stats


def infer_effective_speakers(
    stats: Dict[str, Dict[str, float]],
    coverage_th: float = 0.97,
    max_speakers: int | None = None,
    min_speakers: int = 1,
) -> Tuple[List[str], List[str]]:
    """
    duration ベースで主要話者を選ぶ。
    - coverage_th: 累積durationがこの割合に達するまで keep に追加
    - max_speakers: keep の上限
    - min_speakers: coverage_th を満たしていても最低限残す話者数
    """
    if not stats:
        return [], []

    # duration 降順に並べる
    items = sorted(stats.items(), key=lambda kv: kv[1]["duration"], reverse=True)
    total = sum(v["duration"] for _, v in items) or 1.0

    keep: List[str] = []
    acc = 0.0
    for spk, rec in items:
        keep.append(spk)
        acc += rec["duration"]
        # coverage を満たし、かつ min_speakers 以上確保できているときだけ break
        if (acc / total) >= coverage_th and len(keep) >= max(1, min_speakers):
            break

    # max_speakers の上限を適用（ただし min_speakers は割らない）
    if max_speakers is not None and max_speakers > 0:
        if len(keep) > max_speakers:
            keep = keep[:max_speakers]
        # 念のため、安全側に揃える
        if len(keep) < min_speakers and len(items) >= min_speakers:
            keep = [spk for spk, _ in items[:min_speakers]]

    drop = [spk for spk, _ in items if spk not in keep]
    return keep, drop


def _nearest_keep_speaker(
    segments: List[Dict[str, Any]],
    idx: int,
    keep_spk: List[str],
) -> str | None:
    n = len(segments)
    target_idx = idx

    prev_seg = segments[target_idx - 1] if target_idx > 0 else None
    next_seg = segments[target_idx + 1] if target_idx + 1 < n else None

    candidates: List[Tuple[float, str]] = []

    for neighbor in (prev_seg, next_seg):
        if not neighbor:
            continue
        spk = neighbor.get("speaker")
        if spk not in keep_spk:
            continue

        try:
            s0 = float(segments[target_idx].get("start", 0.0) or 0.0)
            e0 = float(segments[target_idx].get("end", s0) or s0)
            s1 = float(neighbor.get("start", 0.0) or 0.0)
            e1 = float(neighbor.get("end", s1) or s1)
        except (TypeError, ValueError):
            continue

        gap = max(0.0, max(s0 - e1, s1 - e0))
        score = 1.0 / (1.0 + gap)
        candidates.append((score, spk))

    if not candidates:
        return None

    return max(candidates, key=lambda x: x[0])[1]


def reassign_dropped_speakers(
    segments: List[Dict[str, Any]],
    keep_spk: List[str],
    drop_spk: List[str],
) -> List[Dict[str, Any]]:
    if not segments or not drop_spk:
        return segments

    segs = sorted(segments, key=lambda s: float(s.get("start", 0.0) or 0.0))
    for idx, seg in enumerate(segs):
        spk = seg.get("speaker")
        if spk not in drop_spk:
            continue

        target = _nearest_keep_speaker(segs, idx, keep_spk)
        if target is None:
            # 周辺に keep スピーカーがいない場合は最長話者に吸収させる
            target = keep_spk[0]

        seg["speaker"] = target

    return segs


def merge_timeline(
    segments: List[Dict[str, Any]],
    max_gap: float = 0.3,
    min_len: float = 0.0,
) -> List[Dict[str, Any]]:
    if not segments:
        return segments

    segs = sorted(segments, key=lambda s: float(s.get("start", 0.0) or 0.0))
    merged: List[Dict[str, Any]] = []

    prev = segs[0].copy()
    for cur in segs[1:]:
        try:
            prev_start = float(prev.get("start", 0.0) or 0.0)
            prev_end = float(prev.get("end", prev_start) or prev_start)
            cur_start = float(cur.get("start", 0.0) or 0.0)
            cur_end = float(cur.get("end", cur_start) or cur_start)
        except (TypeError, ValueError):
            merged.append(prev)
            prev = cur.copy()
            continue

        if cur.get("speaker") == prev.get("speaker"):
            gap = cur_start - prev_end
            if 0.0 <= gap <= max_gap:
                prev["end"] = max(prev_end, cur_end)
                continue

        dur = prev_end - prev_start
        if dur >= min_len:
            merged.append(prev)
        else:
            merged.append(prev)

        prev = cur.copy()

    merged.append(prev)
    return merged

def smooth_speaker_islands(
    segments: List[Dict[str, Any]],
    max_island_sec: float = 2.0,
    merge_gap: float = 0.3,
    min_len: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    A-B-A 型の短い「話者の島」を前後の話者に吸収する。

    - segments は start 昇順であることを前提とする
    - max_island_sec 秒以下の中央セグメントだけを対象にする
    - 話者ラベルを書き換えたあと、merge_timeline で再度マージする
    """
    if not segments or len(segments) < 3 or max_island_sec <= 0.0:
        return segments

    segs = sorted(segments, key=lambda s: float(s.get("start", 0.0) or 0.0))
    updated = False
    num_smoothed = 0

    for idx in range(1, len(segs) - 1):
        prev_seg = segs[idx - 1]
        cur_seg = segs[idx]
        next_seg = segs[idx + 1]

        spk_prev = prev_seg.get("speaker")
        spk_cur = cur_seg.get("speaker")
        spk_next = next_seg.get("speaker")

        if not spk_prev or not spk_cur or not spk_next:
            continue

        # A-B-A パターンのみ対象
        if spk_prev == spk_next and spk_prev != spk_cur:
            try:
                s = float(cur_seg.get("start", 0.0) or 0.0)
                e = float(cur_seg.get("end", s) or s)
            except (TypeError, ValueError):
                continue

            dur = e - s
            if 0.0 < dur <= max_island_sec:
                # 中央の島を前後と同じ話者に張り替える
                new_seg = dict(cur_seg)
                new_seg["speaker"] = spk_prev
                segs[idx] = new_seg
                updated = True
                num_smoothed += 1

    if not updated:
        return segments

    LOGGER.info(
        "[POST-DIAR] smooth_speaker_islands: smoothed %d islands (max_island_sec=%.2f)",
        num_smoothed,
        max_island_sec,
    )
    return merge_timeline(segs, max_gap=merge_gap, min_len=min_len)


def smooth_high_switch_density_zones(
    segments: List[Dict[str, Any]],
    duration_sec: float,
    window_sec: float = 30.0,
    step_sec: float = 10.0,
    min_window_sec: float = 25.0,
    majority_th: float = 0.8,
    min_switch_density: float = 0.15,
    merge_gap: float = 0.3,
    min_len: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    30秒前後の窓で「スイッチ密度が異常に高いのに実質1話者が支配的なゾーン」を検出し、
    majority speaker に一括で寄せることでモノローグ崩壊型の誤りを抑制する。

    - segments は start 昇順であることを前提とする
    - window_sec, step_sec は秒単位
    - min_window_sec 未満の実効長しかない窓は無視する
    """
    if not segments or window_sec <= 0.0 or step_sec <= 0.0:
        return segments

    segs = sorted(segments, key=lambda s: float(s.get("start", 0.0) or 0.0))

    # 音声全体の終端（duration_sec とセグメント終端の最大値の大きい方）
    try:
        last_end = float(segs[-1].get("end", duration_sec) or duration_sec)
    except (TypeError, ValueError):
        last_end = duration_sec
    audio_end = max(duration_sec, last_end)

    zones: List[Dict[str, Any]] = []
    num_flagged = 0

    win_start = 0.0
    while win_start < audio_end:
        win_end = min(win_start + window_sec, audio_end)

        total_dur = 0.0
        per_spk: Dict[str, float] = {}
        last_spk_in_win: str | None = None
        switches = 0

        for seg in segs:
            try:
                s = float(seg.get("start", 0.0) or 0.0)
                e = float(seg.get("end", s) or s)
            except (TypeError, ValueError):
                continue

            if e <= win_start:
                continue
            if s >= win_end:
                break

            ov_start = max(s, win_start)
            ov_end = min(e, win_end)
            if ov_end <= ov_start:
                continue

            dur = ov_end - ov_start
            if dur <= 0.0:
                continue

            spk = seg.get("speaker")
            if not spk:
                continue

            total_dur += dur
            per_spk[spk] = per_spk.get(spk, 0.0) + dur

            if last_spk_in_win is None:
                last_spk_in_win = spk
            elif spk != last_spk_in_win:
                switches += 1
                last_spk_in_win = spk

        if total_dur >= min_window_sec and per_spk:
            maj_spk, maj_dur = max(per_spk.items(), key=lambda kv: kv[1])
            majority_ratio = maj_dur / max(total_dur, 1e-6)
            switch_density = switches / max(total_dur, 1e-6)

            if majority_ratio >= majority_th and switch_density >= min_switch_density:
                num_flagged += 1
                # 同じ話者・連続/重複する窓はマージ
                if zones and zones[-1]["speaker"] == maj_spk and win_start <= zones[-1]["end"]:
                    zones[-1]["end"] = max(zones[-1]["end"], win_end)
                else:
                    zones.append(
                        {
                            "start": win_start,
                            "end": win_end,
                            "speaker": maj_spk,
                        }
                    )

        win_start += step_sec

    if not zones:
        return segments

    # 検出されたゾーン内のセグメントを majority speaker 側に寄せる
    updated = 0
    for seg in segs:
        spk = seg.get("speaker")
        if not spk:
            continue

        try:
            s = float(seg.get("start", 0.0) or 0.0)
            e = float(seg.get("end", s) or s)
        except (TypeError, ValueError):
            continue

        if e <= s:
            continue

        center = 0.5 * (s + e)
        for zone in zones:
            z_start = float(zone.get("start", 0.0) or 0.0)
            z_end = float(zone.get("end", z_start) or z_start)
            z_spk = zone.get("speaker")
            if not z_spk:
                continue

            if z_start <= center < z_end:
                if spk != z_spk:
                    seg["speaker"] = z_spk
                    updated += 1
                break

    if not updated:
        return segs

    LOGGER.info(
        "[POST-DIAR] smooth_high_switch_density_zones: updated %d segments across %d zones "
        "(window=%.1fs, step=%.1fs, majority_th=%.2f, min_switch_density=%.3f)",
        updated,
        len(zones),
        window_sec,
        step_sec,
        majority_th,
        min_switch_density,
    )

    return merge_timeline(segs, max_gap=merge_gap, min_len=min_len)

def sanitize_speaker_timeline(
    segments: List[Dict[str, Any]],
    duration_sec: float,
    config: Any,
) -> List[Dict[str, Any]]:
    if not segments:
        return segments

    stats = collect_speaker_stats(segments)
    coverage_th = getattr(config, "coverage_threshold", 0.97)
    merge_gap = getattr(config, "merge_gap_sec", 0.3)
    min_len = getattr(config, "min_segment_len_sec", 0.0)
    max_island_sec = getattr(config, "max_island_sec", 0.0)
    max_speakers = getattr(config, "max_speakers", None)
    min_speakers = getattr(config, "min_speakers", 1)

    keep_spk, drop_spk = infer_effective_speakers(
        stats,
        coverage_th=coverage_th,
        max_speakers=max_speakers,
        min_speakers=min_speakers,
    )

    LOGGER.info(
        "[POST-DIAR] stats=%s, keep=%s, drop=%s",
        {k: round(v["duration"], 2) for k, v in stats.items()},
        keep_spk,
        drop_spk,
    )

    segs = segments
    if drop_spk:
        segs = reassign_dropped_speakers(segs, keep_spk, drop_spk)

    segs = merge_timeline(segs, max_gap=merge_gap, min_len=min_len)

    # A-B-A 型の短い島を前後の話者に吸収
    if max_island_sec > 0.0:
        segs = smooth_speaker_islands(
            segs,
            max_island_sec=max_island_sec,
            merge_gap=merge_gap,
            min_len=min_len,
        )

    # 30秒窓ベースの「高スイッチ密度ゾーン」を majority speaker に寄せる
    if getattr(config, "enable_high_switch_smoothing", False):
        window_sec = getattr(config, "high_switch_window_sec", 30.0)
        step_sec = getattr(config, "high_switch_step_sec", 10.0)
        min_window_sec = getattr(config, "high_switch_min_window_sec", 25.0)
        majority_th = getattr(config, "high_switch_majority_ratio", 0.8)
        min_switch_density = getattr(
            config,
            "high_switch_min_switch_density",
            0.15,
        )

        if window_sec > 0.0 and step_sec > 0.0:
            segs = smooth_high_switch_density_zones(
                segs,
                duration_sec=duration_sec,
                window_sec=window_sec,
                step_sec=step_sec,
                min_window_sec=min_window_sec,
                majority_th=majority_th,
                min_switch_density=min_switch_density,
                merge_gap=merge_gap,
                min_len=min_len,
            )

    # 時間範囲のサニティチェック
    fixed: List[Dict[str, Any]] = []
    for seg in segs:
        try:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)
        except (TypeError, ValueError):
            continue

        start = max(0.0, min(start, duration_sec))
        end = max(start, min(end, duration_sec))

        out = dict(seg)
        out["start"] = start
        out["end"] = end
        fixed.append(out)

    return fixed
