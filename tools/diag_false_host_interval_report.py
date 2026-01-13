#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_false_host_interval_report.py

Create a focused report for false HOST intervals using boundary_snap events.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

RUN_TS_RE = re.compile(r"^(\d{8}_\d{6})_")
PUNCT_TOKENS = ("?", "？", "。", ".")


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def derive_run_ts(run_dir: Path) -> Optional[str]:
    match = RUN_TS_RE.match(run_dir.name)
    if not match:
        return None
    return match.group(1)


def resolve_eval(run_dir: Path, base_dir: Path, eval_arg: Optional[str]) -> Path:
    if eval_arg:
        eval_path = resolve_path(eval_arg, base_dir)
    else:
        eval_path = run_dir / "eval.txt"
    if not eval_path.exists():
        raise FileNotFoundError(f"eval not found: {eval_path}")
    return eval_path


def resolve_snap(run_dir: Path, base_dir: Path, snap_arg: Optional[str]) -> Path:
    if snap_arg:
        snap_path = resolve_path(snap_arg, base_dir)
        if not snap_path.exists():
            raise FileNotFoundError(f"snap not found: {snap_path}")
        return snap_path

    run_ts = derive_run_ts(run_dir)
    if not run_ts:
        raise FileNotFoundError("cannot derive run timestamp (use --snap)")
    out_dir = run_dir.parent
    matches = sorted(out_dir.glob(f"{run_ts}_boundary_snap*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"no boundary_snap file found for run_ts={run_ts} in {out_dir}")
    preferred = out_dir / f"{run_ts}_boundary_snap.jsonl"
    if preferred.exists():
        return preferred
    return matches[0]


def load_thresholds(run_dir: Path) -> Dict[str, float]:
    run_config = run_dir / "run_config.json"
    defaults = {
        "vad_th": 0.35,
        "spk_margin_th": 0.1,
    }
    if not run_config.exists():
        return defaults
    try:
        data = json.loads(run_config.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return defaults
    aligner = data.get("aligner", {}) if isinstance(data, dict) else {}
    vad_th = aligner.get("silence_vad_th", defaults["vad_th"])
    spk_margin_th = aligner.get("switch_spk_margin_th", defaults["spk_margin_th"])
    try:
        defaults["vad_th"] = float(vad_th)
    except (TypeError, ValueError):
        pass
    try:
        defaults["spk_margin_th"] = float(spk_margin_th)
    except (TypeError, ValueError):
        pass
    return defaults


def parse_false_host_intervals(eval_path: Path) -> List[Tuple[float, float]]:
    lines = eval_path.read_text(encoding="utf-8", errors="replace").splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("=== Misclassified intervals (GUEST → HOST"):
            start_idx = i + 1
            break
    if start_idx is None:
        return []

    intervals: List[Tuple[float, float]] = []
    for line in lines[start_idx:]:
        if line.startswith("==="):
            break
        m = re.match(r"^\s*\d+\s+([0-9.]+)\s+([0-9.]+)\s+[0-9.]+", line)
        if not m:
            continue
        start = float(m.group(1))
        end = float(m.group(2))
        if end > start:
            intervals.append((start, end))
    return intervals


def _nearest_frame(frames: Iterable[Dict[str, Any]], target_t: float) -> Optional[Dict[str, Any]]:
    best = None
    best_dist = None
    for fr in frames:
        try:
            t = float(fr.get("t"))
        except (TypeError, ValueError):
            continue
        dist = abs(t - target_t)
        if best is None or dist < best_dist:
            best = fr
            best_dist = dist
    return best


def _frames_in_window(
    frames: Iterable[Dict[str, Any]],
    start_t: float,
    end_t: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fr in frames:
        try:
            t = float(fr.get("t"))
        except (TypeError, ValueError):
            continue
        if start_t <= t <= end_t:
            out.append(fr)
    return out


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    values_sorted = sorted(values)
    n = len(values_sorted)
    p50 = values_sorted[int(0.5 * (n - 1))]
    p95 = values_sorted[int(0.95 * (n - 1))]
    return {
        "count": float(n),
        "min": float(values_sorted[0]),
        "p50": float(p50),
        "p95": float(p95),
        "max": float(values_sorted[-1]),
        "mean": float(sum(values_sorted) / n),
    }


def _fmt_stats(label: str, stats: Dict[str, float]) -> str:
    if not stats:
        return f"{label}: NONE"
    return (
        f"{label}: count={int(stats['count'])} min={stats['min']:.3f} "
        f"p50={stats['p50']:.3f} p95={stats['p95']:.3f} "
        f"max={stats['max']:.3f} mean={stats['mean']:.3f}"
    )


def _ratio_below(frames: Iterable[Dict[str, Any]], key: str, th: float) -> Tuple[int, int]:
    total = 0
    low = 0
    for fr in frames:
        if key not in fr:
            continue
        try:
            val = float(fr.get(key))
        except (TypeError, ValueError):
            continue
        total += 1
        if val < th:
            low += 1
    return low, total


def _has_punct(text: str) -> bool:
    return any(tok in text for tok in PUNCT_TOKENS)


def _clip_text(text: str, width: int = 60) -> str:
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run output directory")
    ap.add_argument("--eval", default=None, help="Override eval.txt path")
    ap.add_argument("--snap", default=None, help="Override boundary_snap.jsonl path")
    ap.add_argument("--out", default=None, help="Output report path (default: run_dir/false_host_interval_report.txt)")
    ap.add_argument("--topk", type=int, default=10, help="Top-k low margin events to list")
    ap.add_argument("--pre-sec", type=float, default=0.5, help="Seconds before switch to analyze (default: 0.5)")
    ap.add_argument("--post-sec", type=float, default=0.5, help="Seconds after switch to analyze (default: 0.5)")
    ap.add_argument("--vad-th", type=float, default=None, help="Low VAD threshold (default: from run_config)")
    ap.add_argument("--p-margin-th", type=float, default=0.2, help="Low p_margin threshold (default: 0.2)")
    ap.add_argument("--spk-margin-th", type=float, default=None, help="Low spk_margin threshold (default: from run_config)")
    args = ap.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    run_dir = resolve_path(args.run, project_root)
    eval_path = resolve_eval(run_dir, project_root, args.eval)
    snap_path = resolve_snap(run_dir, project_root, args.snap)
    out_path = resolve_path(args.out, project_root) if args.out else run_dir / "false_host_interval_report.txt"

    intervals = parse_false_host_intervals(eval_path)
    if not intervals:
        raise RuntimeError("no false HOST intervals found in eval.txt")

    thresholds = load_thresholds(run_dir)
    vad_th = float(args.vad_th) if args.vad_th is not None else thresholds["vad_th"]
    spk_margin_th = (
        float(args.spk_margin_th) if args.spk_margin_th is not None else thresholds["spk_margin_th"]
    )
    p_margin_th = float(args.p_margin_th)

    events: List[Dict[str, Any]] = []
    with snap_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("event_type") != "switch":
                continue
            try:
                ev_time = float(obj.get("time"))
            except (TypeError, ValueError):
                continue
            in_interval = any(start <= ev_time <= end for start, end in intervals)
            frame = _nearest_frame(obj.get("frames", []), ev_time)
            dp_terms = obj.get("dp_cost_terms", {}) or {}
            words_text = str(obj.get("words_text") or "")
            frames = obj.get("frames", []) or []
            events.append(
                {
                    "time": ev_time,
                    "in_interval": in_interval,
                    "p_margin": frame.get("p_margin") if frame else None,
                    "spk_margin": frame.get("spk_margin") if frame else None,
                    "vad": frame.get("vad") if frame else None,
                    "cp": frame.get("cp") if frame else None,
                    "entropy": frame.get("p_entropy") if frame else None,
                    "uncertain_hit": bool(frame.get("uncertain_hit")) if frame else False,
                    "dp_total": obj.get("dp_cost_total"),
                    "dp_sw_pen": dp_terms.get("sw_pen"),
                    "dp_spk": dp_terms.get("spk"),
                    "dp_vad": dp_terms.get("vad"),
                    "dp_word": dp_terms.get("word"),
                    "cp_scale": dp_terms.get("cp_scale"),
                    "uncertain_scale": dp_terms.get("uncertain_scale"),
                    "lex_scale": dp_terms.get("lex_scale"),
                    "post_to_scale_path": dp_terms.get("post_to_scale_path"),
                    "words_text": words_text,
                    "has_punct": _has_punct(words_text),
                    "frames": frames,
                }
            )

    def collect(metric: str, subset: List[Dict[str, Any]]) -> List[float]:
        out: List[float] = []
        for ev in subset:
            val = ev.get(metric)
            if val is None:
                continue
            try:
                out.append(float(val))
            except (TypeError, ValueError):
                continue
        return out

    in_events = [e for e in events if e["in_interval"]]
    out_events = [e for e in events if not e["in_interval"]]
    pre_sec = float(args.pre_sec)
    post_sec = float(args.post_sec)

    def collect_frames(subset: List[Dict[str, Any]], rel_start: float, rel_end: float) -> List[Dict[str, Any]]:
        frames: List[Dict[str, Any]] = []
        for ev in subset:
            t0 = float(ev["time"])
            frames.extend(_frames_in_window(ev["frames"], t0 + rel_start, t0 + rel_end))
        return frames

    in_pre = collect_frames(in_events, -pre_sec, 0.0)
    in_post = collect_frames(in_events, 0.0, post_sec)
    out_pre = collect_frames(out_events, -pre_sec, 0.0)
    out_post = collect_frames(out_events, 0.0, post_sec)

    report_lines: List[str] = []
    report_lines.append("False HOST Interval Report")
    report_lines.append("=" * 28)
    report_lines.append(f"run_dir: {run_dir}")
    report_lines.append(f"eval: {eval_path}")
    report_lines.append(f"snap: {snap_path}")
    report_lines.append("")
    report_lines.append(f"false HOST intervals: {len(intervals)}")
    for idx, (start, end) in enumerate(intervals, start=1):
        report_lines.append(f"  {idx:02d}: {start:.2f} - {end:.2f} ({end - start:.2f}s)")
    report_lines.append("")
    report_lines.append(f"switch events total: {len(events)}")
    report_lines.append(f"  in intervals : {len(in_events)}")
    report_lines.append(f"  out intervals: {len(out_events)}")
    report_lines.append("")

    report_lines.append(f"window analysis: pre_sec={pre_sec:.2f} post_sec={post_sec:.2f}")
    report_lines.append(f"thresholds: vad<{vad_th:.2f} p_margin<{p_margin_th:.2f} spk_margin<{spk_margin_th:.2f}")
    report_lines.append("")

    for metric in ("p_margin", "spk_margin", "vad", "cp", "entropy"):
        report_lines.append(_fmt_stats(f"in/{metric}", _summarize(collect(metric, in_events))))
        report_lines.append(_fmt_stats(f"out/{metric}", _summarize(collect(metric, out_events))))
        report_lines.append("")

    for metric in ("p_margin", "spk_margin", "vad", "cp", "entropy"):
        report_lines.append(_fmt_stats(f"in/pre/{metric}", _summarize(collect(metric, in_pre))))
        report_lines.append(_fmt_stats(f"in/post/{metric}", _summarize(collect(metric, in_post))))
        report_lines.append(_fmt_stats(f"out/pre/{metric}", _summarize(collect(metric, out_pre))))
        report_lines.append(_fmt_stats(f"out/post/{metric}", _summarize(collect(metric, out_post))))
        report_lines.append("")

    low_pm_in_pre = _ratio_below(in_pre, "p_margin", p_margin_th)
    low_pm_in_post = _ratio_below(in_post, "p_margin", p_margin_th)
    low_pm_out_pre = _ratio_below(out_pre, "p_margin", p_margin_th)
    low_pm_out_post = _ratio_below(out_post, "p_margin", p_margin_th)
    report_lines.append(
        f"in/pre/low_p_margin(<{p_margin_th:.2f}): {low_pm_in_pre[0]}/{low_pm_in_pre[1]}"
    )
    report_lines.append(
        f"in/post/low_p_margin(<{p_margin_th:.2f}): {low_pm_in_post[0]}/{low_pm_in_post[1]}"
    )
    report_lines.append(
        f"out/pre/low_p_margin(<{p_margin_th:.2f}): {low_pm_out_pre[0]}/{low_pm_out_pre[1]}"
    )
    report_lines.append(
        f"out/post/low_p_margin(<{p_margin_th:.2f}): {low_pm_out_post[0]}/{low_pm_out_post[1]}"
    )
    report_lines.append("")

    low_sm_in_pre = _ratio_below(in_pre, "spk_margin", spk_margin_th)
    low_sm_in_post = _ratio_below(in_post, "spk_margin", spk_margin_th)
    low_sm_out_pre = _ratio_below(out_pre, "spk_margin", spk_margin_th)
    low_sm_out_post = _ratio_below(out_post, "spk_margin", spk_margin_th)
    report_lines.append(
        f"in/pre/low_spk_margin(<{spk_margin_th:.2f}): {low_sm_in_pre[0]}/{low_sm_in_pre[1]}"
    )
    report_lines.append(
        f"in/post/low_spk_margin(<{spk_margin_th:.2f}): {low_sm_in_post[0]}/{low_sm_in_post[1]}"
    )
    report_lines.append(
        f"out/pre/low_spk_margin(<{spk_margin_th:.2f}): {low_sm_out_pre[0]}/{low_sm_out_pre[1]}"
    )
    report_lines.append(
        f"out/post/low_spk_margin(<{spk_margin_th:.2f}): {low_sm_out_post[0]}/{low_sm_out_post[1]}"
    )
    report_lines.append("")

    low_vad_in_pre = _ratio_below(in_pre, "vad", vad_th)
    low_vad_in_post = _ratio_below(in_post, "vad", vad_th)
    low_vad_out_pre = _ratio_below(out_pre, "vad", vad_th)
    low_vad_out_post = _ratio_below(out_post, "vad", vad_th)
    report_lines.append(
        f"in/pre/low_vad(<{vad_th:.2f}): {low_vad_in_pre[0]}/{low_vad_in_pre[1]}"
    )
    report_lines.append(
        f"in/post/low_vad(<{vad_th:.2f}): {low_vad_in_post[0]}/{low_vad_in_post[1]}"
    )
    report_lines.append(
        f"out/pre/low_vad(<{vad_th:.2f}): {low_vad_out_pre[0]}/{low_vad_out_pre[1]}"
    )
    report_lines.append(
        f"out/post/low_vad(<{vad_th:.2f}): {low_vad_out_post[0]}/{low_vad_out_post[1]}"
    )
    report_lines.append("")

    for metric in ("dp_sw_pen", "dp_spk", "dp_vad", "dp_word", "cp_scale", "uncertain_scale"):
        report_lines.append(_fmt_stats(f"in/{metric}", _summarize(collect(metric, in_events))))
        report_lines.append(_fmt_stats(f"out/{metric}", _summarize(collect(metric, out_events))))
        report_lines.append("")

    if in_events:
        in_uncertain = sum(1 for ev in in_events if ev["uncertain_hit"])
        report_lines.append(f"in/uncertain_hit: {in_uncertain}/{len(in_events)}")
    if out_events:
        out_uncertain = sum(1 for ev in out_events if ev["uncertain_hit"])
        report_lines.append(f"out/uncertain_hit: {out_uncertain}/{len(out_events)}")
    report_lines.append("")

    if in_events:
        punct_count = sum(1 for ev in in_events if ev["has_punct"])
        report_lines.append(f"in/has_punct: {punct_count}/{len(in_events)}")
    if out_events:
        punct_count = sum(1 for ev in out_events if ev["has_punct"])
        report_lines.append(f"out/has_punct: {punct_count}/{len(out_events)}")
    report_lines.append("")

    topk = max(1, int(args.topk))
    in_sorted = sorted(
        [ev for ev in in_events if ev.get("p_margin") is not None],
        key=lambda ev: ev["p_margin"],
    )[:topk]
    report_lines.append(f"Top-{topk} lowest p_margin events in false HOST intervals")
    report_lines.append("-" * 56)
    for ev in in_sorted:
        report_lines.append(
            "t={:.3f} p_margin={:.3f} spk_margin={:.3f} vad={:.3f} cp={:.3f} "
            "dp_total={:.3f} sw_pen={:.3f} words={}".format(
                float(ev["time"]),
                float(ev["p_margin"]),
                float(ev["spk_margin"]) if ev["spk_margin"] is not None else -1.0,
                float(ev["vad"]) if ev["vad"] is not None else -1.0,
                float(ev["cp"]) if ev["cp"] is not None else -1.0,
                float(ev["dp_total"]) if ev["dp_total"] is not None else 0.0,
                float(ev["dp_sw_pen"]) if ev["dp_sw_pen"] is not None else 0.0,
                _clip_text(str(ev["words_text"])),
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
