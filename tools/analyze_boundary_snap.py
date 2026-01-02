#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _as_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        v2 = v.strip()
        if re.fullmatch(r"[+-]?\d+", v2):
            try:
                return int(v2)
            except ValueError:
                return None
    return None


def _first_key(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def _event_type(ev: Dict[str, Any]) -> str:
    v = _first_key(ev, ("event", "kind", "type", "event_type", "evt"))
    if isinstance(v, str) and v.strip():
        return v.strip()
    # fallback: try infer from flags
    if "switch" in ev:
        return "switch"
    return "UNKNOWN"


def _time_sec(ev: Dict[str, Any]) -> Optional[float]:
    return _as_float(_first_key(ev, ("time", "t", "sec", "timestamp", "ts")))


def _cp(ev: Dict[str, Any]) -> Optional[float]:
    return _as_float(_first_key(ev, ("cp", "cp_score", "cp_peak", "changepoint", "change_point")))


def _p_entropy(ev: Dict[str, Any]) -> Optional[float]:
    return _as_float(_first_key(ev, ("p_entropy", "entropy", "post_entropy", "posterior_entropy")))


def _p_margin(ev: Dict[str, Any]) -> Optional[float]:
    return _as_float(_first_key(ev, ("p_margin", "margin", "post_margin", "posterior_margin")))


def _path_spk(ev: Dict[str, Any]) -> Optional[int]:
    return _as_int(_first_key(ev, ("path_spk", "path", "path_speaker", "path_spk_id", "spk_path")))


def _p_argmax(ev: Dict[str, Any]) -> Optional[int]:
    return _as_int(_first_key(ev, ("p_argmax", "argmax", "posterior_argmax", "post_argmax")))


@dataclass(frozen=True)
class Row:
    idx: int
    t: float
    etype: str
    cp: Optional[float]
    entropy: Optional[float]
    margin: Optional[float]
    old_spk: Optional[int]
    path_spk: Optional[int]
    p_argmax: Optional[int]
    mismatch: Optional[bool]
    dp_cost_total: Optional[float]
    dp_cost_terms: Any
    raw: Dict[str, Any]


def _has_frames(ev: Dict[str, Any]) -> bool:
    fr = ev.get("frames")
    return isinstance(fr, list) and len(fr) > 0 and isinstance(fr[0], dict)


def _frame_time(fr: Dict[str, Any], ev: Dict[str, Any]) -> Optional[float]:
    return _as_float(fr.get("time")) or _time_sec(ev)


def _frame_at_event(ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not _has_frames(ev):
        return None
    frames = ev["frames"]
    target_t = _as_int(ev.get("frame"))
    if target_t is not None:
        for fr in frames:
            if _as_int(fr.get("t")) == target_t:
                return fr
    return frames[len(frames) // 2]


def _prev_frame(ev: Dict[str, Any], fr: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not _has_frames(ev):
        return None
    t = _as_int(fr.get("t"))
    if t is None:
        return None
    for x in ev["frames"]:
        if _as_int(x.get("t")) == t - 1:
            return x
    return None
 
 
def _debug_at_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    fr = _frame_at_event(ev)
    if fr is None:
        return {}
    return {
        "ent_norm": _as_float(fr.get("p_entropy_norm")),
        "unc_hit": fr.get("uncertain_hit"),
        "unc_scale": _as_float(fr.get("uncertain_scale")),
        "cp_scale": _as_float(fr.get("cp_scale")),
        "cp_floor": _as_float(fr.get("cp_floor")),
        "post_path": _as_float(fr.get("post_to_scale_path")),
        "post_arg": _as_float(fr.get("post_to_scale_argmax")),
        "post_conf": _as_float(fr.get("post_conf")),
        "dp_total": _as_float(fr.get("dp_cost_total")),
        "dp_terms": fr.get("dp_cost_terms"),
    }


def _fmt_dbg(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _summarize_dp_terms_version(raw: List[Dict[str, Any]]) -> Tuple[List[int], int, int]:
    versions: List[int] = []
    total_terms = 0
    missing_version = 0
    seen: Dict[int, bool] = {}
    for ev in raw:
        if not isinstance(ev, dict):
            continue
        has_terms = "dp_cost_terms" in ev
        frames = ev.get("frames")
        if not has_terms and isinstance(frames, list):
            for fr in frames:
                if isinstance(fr, dict) and "dp_cost_terms" in fr:
                    has_terms = True
                    break
        if has_terms:
            total_terms += 1
        v = _as_int(ev.get("dp_terms_version"))
        if v is not None:
            if v not in seen:
                seen[v] = True
                versions.append(v)
        elif has_terms:
            missing_version += 1
    versions.sort()
    return versions, total_terms, missing_version


def _term_value(terms: Any, key: str) -> Optional[float]:
    if not isinstance(terms, dict):
        return None
    if key in terms:
        return _as_float(terms.get(key))
    if key == "post_to_scale_path":
        return _as_float(terms.get("post_to_scale"))
    return None


def _dominant_term(terms: Any, keys: Iterable[str]) -> Tuple[Optional[str], Optional[float]]:
    best_key: Optional[str] = None
    best_val: Optional[float] = None
    best_mag: Optional[float] = None
    for k in keys:
        v = _term_value(terms, k)
        if v is None:
            continue
        mag = abs(v)
        if best_mag is None or mag > best_mag:
            best_mag = mag
            best_key = k
            best_val = v
    return best_key, best_val


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _bucket_stats(rows: List[Row], keys: Iterable[str]) -> Dict[str, Any]:
    total_vals: List[float] = []
    term_vals: Dict[str, List[float]] = {k: [] for k in keys}
    for r in rows:
        if r.dp_cost_total is not None:
            total_vals.append(float(r.dp_cost_total))
        terms = r.dp_cost_terms
        for k in keys:
            v = _term_value(terms, k)
            if v is not None:
                term_vals[k].append(float(v))
    return {
        "count": len(rows),
        "dp_cost_total": _mean(total_vals),
        "terms": {k: _mean(vs) for k, vs in term_vals.items()},
    }


def _switches_in_intervals(
    switches: List[Row],
    intervals: List[Tuple[float, float]],
    tol: float,
) -> List[Row]:
    out: List[Row] = []
    seen: Dict[int, bool] = {}
    for start, end in intervals:
        lo = start - tol
        hi = end + tol
        for r in switches:
            if r.t < lo or r.t > hi:
                continue
            if r.idx in seen:
                continue
            seen[r.idx] = True
            out.append(r)
    out.sort(key=lambda r: (r.t, r.idx))
    return out


def fmt_terms(v: Any, nd: int = 3, max_len: int = 60) -> str:
    if v is None:
        return "NA"
    if isinstance(v, dict):
        parts = []
        for k in sorted(v.keys()):
            x = v.get(k)
            if isinstance(x, (int, float)):
                parts.append(f"{k}={float(x):.{nd}f}")
            else:
                parts.append(f"{k}={x}")
        s = ",".join(parts)
    else:
        s = str(v)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _as_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        if v in (0, 1):
            return bool(v)
        return None
    if isinstance(v, float):
        if v in (0.0, 1.0):
            return bool(int(v))
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
    return None


def _parse_eval_misclassified_intervals(text: str) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = {"false_host": [], "false_guest": []}

    # Accept slight variations in arrows/spacing.
    header_map = {
        "false_host": re.compile(
            r"^===\s+Misclassified intervals\s*\(\s*GUEST\s*[→>-]+\s*HOST\s*\(false HOST\)\s*\)\s*===\s*$"
        ),
        "false_guest": re.compile(
            r"^===\s+Misclassified intervals\s*\(\s*HOST\s*[→>-]+\s*GUEST\s*\(false GUEST\)\s*\)\s*===\s*$"
        ),
    }

    lines = text.splitlines()
    cur: Optional[str] = None
    for line in lines:
        s = line.rstrip("\n")
        if cur is None:
            for k, pat in header_map.items():
                if pat.match(s.strip()):
                    cur = k
                    break
            continue

        # End current section on the next === header
        if s.strip().startswith("===") and "Misclassified intervals" in s:
            cur = None
            for k, pat in header_map.items():
                if pat.match(s.strip()):
                    cur = k
                    break
            continue
        if s.strip().startswith("===") and "Misclassified intervals" not in s:
            cur = None
            continue

        m = re.match(r"^\s*\d+\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$", s)
        if not m:
            continue
        a = float(m.group(1))
        b = float(m.group(2))
        if b > a:
            out[cur].append((a, b))

    # Sort and de-overlap defensively
    for k in list(out.keys()):
        xs = sorted(out[k])
        merged: List[Tuple[float, float]] = []
        for a, b in xs:
            if not merged:
                merged.append((a, b))
                continue
            pa, pb = merged[-1]
            if a <= pb:
                merged[-1] = (pa, max(pb, b))
            else:
                merged.append((a, b))
        out[k] = merged
    return out


def _load_eval_intervals(path: str) -> Dict[str, List[Tuple[float, float]]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return _parse_eval_misclassified_intervals(f.read())


def _interval_total_dur(xs: List[Tuple[float, float]]) -> float:
    return sum(max(0.0, b - a) for a, b in xs)


def _time_in_intervals(t: float, xs: List[Tuple[float, float]], tol: float) -> bool:
    for a, b in xs:
        if (a - tol) <= t <= (b + tol):
            return True
    return False


def _uncertain_hit_for_row(r: Row) -> Optional[bool]:
    fr = _frame_at_event(r.raw)
    if fr is None:
        return None
    return _as_bool(fr.get("uncertain_hit"))


def _count_overlaps(
    switches: List[Row],
    intervals: List[Tuple[float, float]],
    tol: float,
) -> Tuple[int, Dict[str, int]]:
    hit = 0
    by_unc = {"true": 0, "false": 0, "na": 0}
    for r in switches:
        if not _time_in_intervals(r.t, intervals, tol=tol):
            continue
        hit += 1
        u = _uncertain_hit_for_row(r)
        if u is True:
            by_unc["true"] += 1
        elif u is False:
            by_unc["false"] += 1
        else:
            by_unc["na"] += 1
    return hit, by_unc


def _intervals_with_any_switch(
    intervals: List[Tuple[float, float]],
    switches: List[Row],
    tol: float,
) -> int:
    n = 0
    for a, b in intervals:
        ok = False
        lo = a - tol
        hi = b + tol
        for r in switches:
            if lo <= r.t <= hi:
                ok = True
                break
        if ok:
            n += 1
    return n




def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise SystemExit(f"ERROR: JSON decode failed at {path}:{ln}: {e}") from e
            if not isinstance(obj, dict):
                continue
            out.append(obj)
    return out


def build_rows(events: List[Dict[str, Any]]) -> List[Row]:
    rows: List[Row] = []
    for i, ev in enumerate(events):
        et = _event_type(ev)

        # Prefer per-event frame record if present (your snap schema)
        fr = _frame_at_event(ev)
        if fr is not None:
            t = _frame_time(fr, ev)
            if t is None:
                continue
            cp = _as_float(fr.get("cp"))
            ent = _as_float(fr.get("p_entropy"))
            mar = _as_float(fr.get("p_margin"))
            path = _as_int(fr.get("path_spk"))
            argm = _as_int(fr.get("p_argmax"))
            pv = _prev_frame(ev, fr)
            old = _as_int(pv.get("path_spk")) if pv is not None else None
        else:
            # Fallback to flat schema
            t = _time_sec(ev)
            if t is None:
                continue
            cp = _cp(ev)
            ent = _p_entropy(ev)
            mar = _p_margin(ev)
            path = _path_spk(ev)
            argm = _p_argmax(ev)
            old = None

        mm: Optional[bool] = None
        if path is not None and argm is not None:
            mm = (path != argm)
        rows.append(
            Row(
                idx=i,
                t=float(t),
                etype=et,
                cp=cp,
                entropy=ent,
                margin=mar,
                old_spk=old,
                path_spk=path,
                p_argmax=argm,
                mismatch=mm,
                dp_cost_total=_as_float(ev.get("dp_cost_total")),
                dp_cost_terms=ev.get("dp_cost_terms"),
                raw=ev,
            )
        )
    rows.sort(key=lambda r: (r.t, r.idx))
    return rows


def is_switch(etype: str) -> bool:
    et = etype.lower()
    return "switch" in et


def is_cp_peak(etype: str) -> bool:
    et = etype.lower()
    return "cp_peak" in et or ("peak" in et and "cp" in et)


def fmt_opt(v: Any, nd: int = 6) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def neighbors(rows: List[Row], t0: float, window_sec: float) -> List[Row]:
    lo = t0 - window_sec
    hi = t0 + window_sec
    return [r for r in rows if lo <= r.t <= hi]


def summarize_counts(rows: List[Row]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.etype] = counts.get(r.etype, 0) + 1
    return counts


def find_low_cp_switches(
    rows: List[Row],
    cp_th: float,
    entropy_th: float,
    margin_th: Optional[float],
) -> List[Row]:
    out: List[Row] = []
    for r in rows:
        if not is_switch(r.etype):
            continue
        if r.cp is None:
            continue
        if r.cp >= cp_th:
            continue
        # Optional gates: high entropy and/or low margin help focus on "almost-equal posterior" cases
        if r.entropy is not None and r.entropy < entropy_th:
            continue
        if margin_th is not None and r.margin is not None and r.margin > margin_th:
            continue
        out.append(r)
    return out


def find_strong_cp_switches(rows: List[Row], cp_strong_th: float) -> List[Row]:
    out: List[Row] = []
    for r in rows:
        if not is_switch(r.etype):
            continue
        if r.cp is None:
            continue
        if r.cp >= cp_strong_th:
            out.append(r)
    return out


def find_stuck_candidates(
    rows: List[Row],
    strong_margin_th: float,
    cp_strong_th: float,
    max_delay_sec: float,
) -> List[Tuple[Row, Row, float]]:
    """
    Heuristic (schema-agnostic) stuck detection using only snap events:
      - pick an event where (p_argmax != path_spk) and margin is strong
      - find the next "strong cp switch" within max_delay_sec
      - if that switch's path_spk equals the earlier p_argmax (when available), treat as candidate
    Limit: if snap does not include non-switch frames, earliest mismatch time is under-sampled.
    """
    strong_switches = find_strong_cp_switches(rows, cp_strong_th=cp_strong_th)
    out: List[Tuple[Row, Row, float]] = []
    for r in rows:
        if r.mismatch is not True:
            continue
        if r.margin is None or r.margin < strong_margin_th:
            continue
        t0 = r.t
        best: Optional[Row] = None
        for s in strong_switches:
            if s.t <= t0:
                continue
            if s.t - t0 > max_delay_sec:
                break
            # If we can check "resolves toward argmax", prefer those.
            if r.p_argmax is not None and s.path_spk is not None:
                if s.path_spk != r.p_argmax:
                    continue
            best = s
            break
        if best is not None:
            out.append((r, best, best.t - t0))
    return out


SRT_TIME_RE = re.compile(
    r"^(?P<h1>\d{2}):(?P<m1>\d{2}):(?P<s1>\d{2})[.,](?P<ms1>\d{3})\s*-->\s*"
    r"(?P<h2>\d{2}):(?P<m2>\d{2}):(?P<s2>\d{2})[.,](?P<ms2>\d{3})\s*$"
)


def _hmsms_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600.0 + int(m) * 60.0 + int(s) + int(ms) / 1000.0


@dataclass(frozen=True)
class Seg:
    start: float
    end: float
    text: str


def load_srt_segments(path: str) -> List[Seg]:
    segs: List[Seg] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    i = 0
    n = len(lines)
    while i < n:
        # skip blank lines
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        # optional index line
        if lines[i].strip().isdigit():
            i += 1
            if i >= n:
                break
        # timing line
        m = SRT_TIME_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        start = _hmsms_to_sec(m.group("h1"), m.group("m1"), m.group("s1"), m.group("ms1"))
        end = _hmsms_to_sec(m.group("h2"), m.group("m2"), m.group("s2"), m.group("ms2"))
        i += 1
        # text lines until blank
        txt_lines: List[str] = []
        while i < n and lines[i].strip():
            txt_lines.append(lines[i])
            i += 1
        text = "\n".join(txt_lines).strip()
        segs.append(Seg(start=start, end=end, text=text))
    return segs


def excerpt_segments(segs: List[Seg], t0: float, window_sec: float) -> List[Seg]:
    lo = t0 - window_sec
    hi = t0 + window_sec
    out: List[Seg] = []
    for s in segs:
        if s.end < lo:
            continue
        if s.start > hi:
            continue
        out.append(s)
    return out


def print_header(title: str) -> None:
    print("")
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True, help="Path to boundary_snap_*.jsonl")
    ap.add_argument("--cp-th", type=float, default=0.1, help="Threshold for low-cp switch filter (default: 0.1)")
    ap.add_argument(
        "--cp-high-th",
        type=float,
        default=0.9,
        help="Threshold for high-cp bucket (default: 0.9)",
    )
    ap.add_argument(
        "--cp-low-th",
        type=float,
        default=0.3,
        help="Threshold for low-cp bucket (default: 0.3)",
    )
    ap.add_argument(
        "--entropy-th",
        type=float,
        default=0.68,
        help="Minimum entropy for low-cp switch filter (default: 0.68; ln2≈0.693)",
    )
    ap.add_argument(
        "--margin-th",
        type=float,
        default=None,
        help="Optional: require margin <= margin-th for low-cp switch filter (default: disabled)",
    )
    ap.add_argument("--context-sec", type=float, default=2.0, help="Neighbor window around each hit (default: 2.0)")
    ap.add_argument(
        "--transcript",
        default=None,
        help="Optional: .srt path to print excerpt around each hit (UNKNOWN for .txt format; srt supported)",
    )
    ap.add_argument("--out", default=None, help="Optional: write a TSV summary to this path")
    ap.add_argument(
        "--eval",
        default=None,
        help="Optional: path to eval_two_speaker_from_transcripts.py console output (or a text file containing those sections). Used to compute overlap between switch times and misclassified intervals.",
    )
    ap.add_argument(
        "--eval-tol-sec",
        type=float,
        default=0.0,
        help="Optional: tolerance (sec) when matching switch time to misclassified intervals (default: 0.0).",
    )
    ap.add_argument(
        "--stuck-strong-margin",
        type=float,
        default=0.5,
        help="Heuristic: strong margin threshold for stuck candidate seed (default: 0.5)",
    )
    ap.add_argument(
        "--stuck-cp-strong",
        type=float,
        default=0.9,
        help="Heuristic: strong cp threshold for the resolving switch (default: 0.9)",
    )
    ap.add_argument(
        "--stuck-max-delay",
        type=float,
        default=2.0,
        help="Heuristic: max delay (sec) between seed mismatch and resolving switch (default: 2.0)",
    )
    ap.add_argument(
        "--stuck-lookback-sec",
        type=float,
        default=1.0,
        help="Exact(stored frames): look back window before switch (default: 1.0)",
    )
    ap.add_argument(
        "--stuck-margin-th",
        type=float,
        default=0.6,
        help="Exact(stored frames): require p_margin >= this (default: 0.6)",
    )
    args = ap.parse_args(argv)

    snap_path = args.snap
    if not os.path.exists(snap_path):
        print(f"ERROR: --snap not found: {snap_path}", file=sys.stderr)
        return 2

    raw = load_jsonl(snap_path)
    versions, total_terms, missing_version = _summarize_dp_terms_version(raw)
    rows = build_rows(raw)
    if not rows:
        print("ERROR: no usable events with time field found in snap.", file=sys.stderr)
        return 2

    counts = summarize_counts(rows)
    switches = sum(1 for r in rows if is_switch(r.etype))
    cp_peaks = sum(1 for r in rows if is_cp_peak(r.etype))
    ln2 = math.log(2.0)
    all_switches = [r for r in rows if is_switch(r.etype)]

    print_header("boundary_snap summary")
    print(f"snap: {snap_path}")
    print(f"rows: {len(rows)}  switches: {switches}  cp_peaks: {cp_peaks}")
    if total_terms > 0:
        if not versions:
            print("WARNING: dp_terms_version missing (legacy snap schema).")
        elif missing_version > 0:
            print(f"WARNING: dp_terms_version missing in {missing_version}/{total_terms} events.")
        else:
            print(f"dp_terms_version: {','.join(str(v) for v in versions)}")
    print(f"entropy reference: ln2={ln2:.6f}")
    print("event type counts:")
    for k in sorted(counts.keys()):
        print(f"  {k}: {counts[k]}")

    low = find_low_cp_switches(
        rows,
        cp_th=args.cp_th,
        entropy_th=args.entropy_th,
        margin_th=args.margin_th,
    )

    print_header("low-cp switches (filtered)")
    print(
        "columns: time  cp  entropy  margin  path(old->new)  p_argmax  mismatch  etype  dp_total  dp_terms"
    )
    if not low:
        print("NONE")
    else:
        for r in low:
            path_arrow = fmt_opt(r.path_spk)
            if "->" not in str(path_arrow) and path_arrow not in ("NA", ""):
                # Assuming path_spk might be a string like "0->1" or similar in some cases,
                # but here Row handles it as int usually. Let's stick to diff's spirit.
                # Actually Row.path_spk is Optional[int].
                pass 
            
            # Use original diff's formatting for columns
            print(
                f"{r.t:9.3f}  {fmt_opt(r.cp):>9}  {fmt_opt(r.entropy):>9}  {fmt_opt(r.margin):>9}  "
                f"{fmt_opt(r.path_spk):>12}  {fmt_opt(r.p_argmax):>8}  {fmt_opt(r.mismatch):>8}  {r.etype:>7}  "
                f"{fmt_opt(r.dp_cost_total, nd=6):>9}  {fmt_terms(r.dp_cost_terms, nd=3, max_len=60)}"
            )
            
            neigh = neighbors(rows, r.t, args.context_sec)
            if neigh:
                print("  neighbors:")
                for n in neigh:
                    dbg_n = _debug_at_event(n.raw)
                    dt = n.t - r.t
                    print(
                        (
                            f"    dt={dt:+.3f}  t={n.t:.3f}  {n.etype}  cp={fmt_opt(n.cp)}  "
                            f"ent={fmt_opt(n.entropy)}  mar={fmt_opt(n.margin)}  "
                            f"path={fmt_opt(n.path_spk)}  arg={fmt_opt(n.p_argmax)}  mm={fmt_opt(n.mismatch)}  "
                            f"dp={fmt_opt(n.dp_cost_total, nd=6)}  terms={fmt_terms(n.dp_cost_terms, nd=2, max_len=40)}"
                        )
                    )

    cp_keys = (
        "vad",
        "spk",
        "word",
        "sw_pen",
        "cp_scale",
        "uncertain_scale",
        "lex_scale",
        "post_to_scale_path",
    )
    high_cp = [r for r in all_switches if r.cp is not None and r.cp >= args.cp_high_th]
    low_cp = [r for r in all_switches if r.cp is not None and r.cp <= args.cp_low_th]

    print_header("cp buckets (switches)")
    print(f"thresholds: high>={args.cp_high_th:.2f}  low<={args.cp_low_th:.2f}")
    for label, bucket in (("high", high_cp), ("low", low_cp)):
        stats = _bucket_stats(bucket, cp_keys)
        terms = stats.get("terms", {})
        print(
            f"{label}: count={stats['count']}  "
            f"dp_total={fmt_opt(stats.get('dp_cost_total'), nd=6)}  "
            f"sw_pen={fmt_opt(terms.get('sw_pen'), nd=6)}  "
            f"cp_scale={fmt_opt(terms.get('cp_scale'), nd=6)}  "
            f"uncertain_scale={fmt_opt(terms.get('uncertain_scale'), nd=6)}  "
            f"lex_scale={fmt_opt(terms.get('lex_scale'), nd=6)}  "
            f"post_to_scale={fmt_opt(terms.get('post_to_scale_path'), nd=6)}"
        )

    segs: List[Seg] = []
    if args.transcript:
        if not os.path.exists(args.transcript):
            print(f"WARNING: --transcript not found: {args.transcript}", file=sys.stderr)
        elif args.transcript.lower().endswith(".srt"):
            try:
                segs = load_srt_segments(args.transcript)
            except Exception as e:
                print(f"WARNING: failed to parse srt: {e}", file=sys.stderr)
        else:
            print("WARNING: transcript parsing supports .srt only (txt format is UNKNOWN).", file=sys.stderr)

    if segs and low:
        print_header("transcript excerpts (.srt only)")
        for r in low:
            print(f"[hit] t={r.t:.3f} cp={fmt_opt(r.cp)} ent={fmt_opt(r.entropy)} mar={fmt_opt(r.margin)}")
            ex = excerpt_segments(segs, r.t, args.context_sec)
            if not ex:
                print("  (no overlapping srt segments)")
                continue
            for s in ex:
                print(f"  {s.start:9.3f} -> {s.end:9.3f}")
                for line in s.text.splitlines():
                    print(f"    {line}")

    # Prefer exact stuck detection if frames are stored in switch events
    stuck_exact: List[Tuple[float, Row, Dict[str, Any], Dict[str, Any], int, int]] = []
    for r in rows:
        if not is_switch(r.etype):
            continue
        ev = r.raw
        if not _has_frames(ev):
            continue
        fr_sw = _frame_at_event(ev)
        if fr_sw is None:
            continue
        pv = _prev_frame(ev, fr_sw)
        if pv is None:
            continue
        old = _as_int(pv.get("path_spk"))
        new = _as_int(fr_sw.get("path_spk"))
        if old is None or new is None or old == new:
            continue
        t_sw = _as_float(fr_sw.get("time")) or r.t
        t0 = t_sw - args.stuck_lookback_sec
        cand: List[Dict[str, Any]] = []
        for fr in ev["frames"]:
            tt = _as_float(fr.get("time"))
            if tt is None or tt < t0 or tt >= t_sw:
                continue
            if _as_int(fr.get("path_spk")) != old:
                continue
            if _as_int(fr.get("p_argmax")) != new:
                continue
            mar = _as_float(fr.get("p_margin")) or 0.0
            if mar < args.stuck_margin_th:
                continue
            cand.append(fr)
        if not cand:
            continue
        fr_pre = max(cand, key=lambda x: _as_float(x.get("p_margin")) or 0.0)
        delay = t_sw - (_as_float(fr_pre.get("time")) or t_sw)
        stuck_exact.append((delay, r, fr_pre, fr_sw, old, new))
    stuck_exact.sort(key=lambda x: -x[0])

    # Fallback heuristic if frames are not available
    stuck_heur = find_stuck_candidates(
        rows,
        strong_margin_th=args.stuck_strong_margin,
        cp_strong_th=args.stuck_cp_strong,
        max_delay_sec=args.stuck_max_delay,
    )

    print_header("stuck candidates")
    if stuck_exact:
        print("exact(stored frames) columns: delay  pre_time(pre_cp,pre_margin)  switch_time(sw_cp,sw_margin)  old->new")
        for i, (delay, rr, pre, sw, old, new) in enumerate(stuck_exact, start=1):
            print(
                f"{i:02d} delay={delay:6.3f}s  "
                f"pre={fmt_opt(_as_float(pre.get('time'))):>9}("
                f"cp={fmt_opt(_as_float(pre.get('cp')))},m={fmt_opt(_as_float(pre.get('p_margin')))}"
                f")  "
                f"switch={fmt_opt(_as_float(sw.get('time'))):>9}("
                f"cp={fmt_opt(_as_float(sw.get('cp')))},m={fmt_opt(_as_float(sw.get('p_margin')))}"
                f")  {old}->{new}"
            )
    else:
        print("exact(stored frames): NONE")

    if stuck_heur:
        print("")
        print("fallback(heuristic) columns: seed_t  seed_margin  seed_cp  seed_path  seed_argmax  ->  switch_t  switch_cp  switch_path  delay")
        for seed, sw, delay in stuck_heur:
            print(
                f"{seed.t:9.3f}  {fmt_opt(seed.margin):>9}  {fmt_opt(seed.cp):>9}  "
                f"{fmt_opt(seed.path_spk):>8}  {fmt_opt(seed.p_argmax):>10}  ->  "
                f"{sw.t:9.3f}  {fmt_opt(sw.cp):>9}  {fmt_opt(sw.path_spk):>10}  {delay:7.3f}s"
            )
    else:
        print("")
        print("fallback(heuristic): NONE")

    if args.out:
        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(
                "\t".join(
                    [
                        "time",
                        "cp",
                        "entropy",
                        "margin",
                        "path_spk",
                        "p_argmax",
                        "mismatch",
                        "etype",
                        "ent_norm",
                        "unc_hit",
                        "unc_scale",
                        "cp_scale",
                        "cp_floor",
                        "post_path",
                        "post_arg",
                        "post_conf",
                    ]
                )
                + "\n"
            )
            for r in low:
                dbg = _debug_at_event(r.raw)
                f.write(
                    "\t".join(
                        [
                            f"{r.t:.6f}",
                            fmt_opt(r.cp),
                            fmt_opt(r.entropy),
                            fmt_opt(r.margin),
                            fmt_opt(r.path_spk),
                            fmt_opt(r.p_argmax),
                            fmt_opt(r.mismatch),
                            r.etype,
                            _fmt_dbg(dbg.get("ent_norm")),
                            _fmt_dbg(dbg.get("unc_hit")),
                            _fmt_dbg(dbg.get("unc_scale")),
                            _fmt_dbg(dbg.get("cp_scale")),
                            _fmt_dbg(dbg.get("cp_floor")),
                            _fmt_dbg(dbg.get("post_path")),
                            _fmt_dbg(dbg.get("post_arg")),
                            _fmt_dbg(dbg.get("post_conf")),
                        ]
                    )
                    + "\n"
                )
        print_header("wrote")
        print(out_path)

    if args.eval:
        eval_path = args.eval
        if not os.path.exists(eval_path):
            print(f"ERROR: --eval not found: {eval_path}", file=sys.stderr)
            return 2
        intervals = _load_eval_intervals(eval_path)

        low_switches = low
        dom_keys = (
            "vad",
            "spk",
            "word",
            "sw_pen",
            "cp_scale",
            "uncertain_scale",
            "lex_scale",
            "post_to_scale_path",
        )

        print_header("eval overlap")
        print(f"eval: {eval_path}")
        print(f"tolerance: {args.eval_tol_sec:.3f} sec")
        print(f"switches: all={len(all_switches)}  low-cp={len(low_switches)}")

        for k, label in (
            ("false_host", "GUEST -> HOST (false HOST)"),
            ("false_guest", "HOST -> GUEST (false GUEST)"),
        ):
            xs = intervals.get(k, [])
            total_d = _interval_total_dur(xs)
            hit_int_all = (
                _intervals_with_any_switch(xs, all_switches, tol=args.eval_tol_sec) if xs else 0
            )
            hit_all, by_unc_all = _count_overlaps(all_switches, xs, tol=args.eval_tol_sec)
            hit_low, by_unc_low = _count_overlaps(low_switches, xs, tol=args.eval_tol_sec)

            print("")
            print(
                f"{label}: intervals={len(xs)}  total_dur={total_d:.1f}s  hit_intervals(all_switch)={hit_int_all}/{len(xs) if xs else 0}"
            )
            print(
                "  all switches: "
                f"hit={hit_all}/{len(all_switches)}  "
                f"unc_true={by_unc_all['true']}  unc_false={by_unc_all['false']}  unc_na={by_unc_all['na']}"
            )
            print(
                "  low-cp switches: "
                f"hit={hit_low}/{len(low_switches)}  "
                f"unc_true={by_unc_low['true']}  unc_false={by_unc_low['false']}  unc_na={by_unc_low['na']}"
            )

            interval_switches = _switches_in_intervals(all_switches, xs, tol=args.eval_tol_sec)
            if interval_switches:
                dom_counts: Dict[str, int] = {}
                missing_terms = 0
                for r in interval_switches:
                    k, _ = _dominant_term(r.dp_cost_terms, dom_keys)
                    if k is None:
                        missing_terms += 1
                        continue
                    dom_counts[k] = dom_counts.get(k, 0) + 1
                print(
                    "  switches in intervals: "
                    f"{len(interval_switches)}  "
                    f"missing_terms={missing_terms}"
                )
                if dom_counts:
                    dom_list = ", ".join(f"{k}={dom_counts[k]}" for k in sorted(dom_counts.keys()))
                    print(f"  dominant terms: {dom_list}")
                sample = interval_switches[:10]
                print("  sample switches: time  cp  dp_total  dominant(term=value)")
                for r in sample:
                    dk, dv = _dominant_term(r.dp_cost_terms, dom_keys)
                    if dk is None:
                        dom = "NA"
                    else:
                        dom = f"{dk}={fmt_opt(dv, nd=6)}"
                    print(
                        f"    {r.t:9.3f}  {fmt_opt(r.cp):>8}  "
                        f"{fmt_opt(r.dp_cost_total, nd=6):>9}  {dom}"
                    )
            else:
                print("  switches in intervals: 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
