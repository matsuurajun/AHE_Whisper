#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_boundary_snap_eval.py

Run eval_two_speaker_from_transcripts.py and analyze_boundary_snap.py in sequence.
Defaults are tuned for the AHE-Whisper diagnostic workflow.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

RUN_TS_RE = re.compile(r"^(\d{8}_\d{6})_")


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def derive_run_ts(run_dir: Path) -> str | None:
    match = RUN_TS_RE.match(run_dir.name)
    if not match:
        return None
    return match.group(1)


def derive_safe_basename(run_dir: Path) -> str | None:
    match = RUN_TS_RE.match(run_dir.name)
    if not match:
        return None
    rest = run_dir.name[match.end():]
    if re.search(r"-\d+$", rest):
        return rest.rsplit("-", 1)[0]
    return rest


def resolve_transcript(
    run_dir: Path,
    base_dir: Path,
    transcript_arg: str | None,
) -> Path:
    if transcript_arg:
        transcript_path = resolve_path(transcript_arg, base_dir)
        if not transcript_path.exists():
            raise FileNotFoundError(f"transcript not found: {transcript_path}")
        return transcript_path

    safe_basename = derive_safe_basename(run_dir)
    if safe_basename:
        candidate = run_dir / f"{safe_basename}.txt"
        if candidate.exists():
            return candidate

    txt_candidates = sorted(
        p
        for p in run_dir.glob("*.txt")
        if p.name not in ("eval.txt", "analyze_boundary_snap.txt")
    )
    if len(txt_candidates) == 1:
        return txt_candidates[0]
    if not txt_candidates:
        raise FileNotFoundError("no transcript .txt found in run_dir (use --transcript)")
    raise FileNotFoundError(
        "multiple .txt candidates found (use --transcript): "
        + ", ".join(p.name for p in txt_candidates)
    )


def resolve_snap(
    run_dir: Path,
    base_dir: Path,
    snap_arg: str | None,
) -> Path:
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
        raise FileNotFoundError(
            f"no boundary_snap file found for run_ts={run_ts} in {out_dir} (use --snap)"
        )

    preferred = out_dir / f"{run_ts}_boundary_snap.jsonl"
    if preferred.exists():
        if len(matches) > 1:
            print(
                f"[INFO] multiple snap files found; using {preferred.name}",
                file=sys.stderr,
            )
        return preferred

    if len(matches) > 1:
        print(
            f"[INFO] multiple snap files found; using {matches[0].name}",
            file=sys.stderr,
        )
    return matches[0]


def run_and_tee(cmd: list[str], out_path: Path, cwd: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            out_file.write(line)
        return proc.wait()


def run_and_capture(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    chunks: list[str] = []
    for line in proc.stdout:
        sys.stdout.write(line)
        chunks.append(line)
    return proc.wait(), "".join(chunks)


def main(argv: list[str]) -> int:
    project_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        required=True,
        help="Run output directory (e.g. AHE_Whisper_output/20260105_094659_tmpd_hz738j)",
    )
    ap.add_argument(
        "--gold",
        default="data/gold/miyazaki_7m_rewrite.txt",
        help="Gold transcript path (default: data/gold/miyazaki_7m_rewrite.txt)",
    )
    ap.add_argument(
        "--transcript",
        default=None,
        help="Override system transcript path (.txt).",
    )
    ap.add_argument(
        "--snap",
        default=None,
        help="Override boundary_snap.jsonl path.",
    )
    ap.add_argument(
        "--eval-out",
        default=None,
        help="Override eval output path (default: <run_dir>/eval.txt)",
    )
    ap.add_argument(
        "--host-speakers",
        default="SPEAKER_00,SPEAKER_03",
        help="Comma-separated host speakers (default: SPEAKER_00,SPEAKER_03)",
    )
    ap.add_argument("--frame-hz", type=float, default=10.0)
    ap.add_argument("--min-error-dur", type=float, default=0.5)
    ap.add_argument("--eval-tol-sec", type=float, default=0.1)
    ap.add_argument("--stuck-lookback-sec", type=float, default=1.0)
    ap.add_argument("--stuck-margin-th", type=float, default=0.6)
    args = ap.parse_args(argv)

    run_dir = resolve_path(args.run, project_root)
    if not run_dir.exists():
        print(f"ERROR: --run not found: {run_dir}", file=sys.stderr)
        return 2
    if not run_dir.is_dir():
        print(f"ERROR: --run is not a directory: {run_dir}", file=sys.stderr)
        return 2

    gold_path = resolve_path(args.gold, project_root)
    if not gold_path.exists():
        print(f"ERROR: --gold not found: {gold_path}", file=sys.stderr)
        return 2

    try:
        transcript_path = resolve_transcript(run_dir, project_root, args.transcript)
        snap_path = resolve_snap(run_dir, project_root, args.snap)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    eval_out = resolve_path(args.eval_out, project_root) if args.eval_out else run_dir / "eval.txt"

    eval_script = project_root / "tools" / "eval_two_speaker_from_transcripts.py"
    analyze_script = project_root / "tools" / "analyze_boundary_snap.py"

    eval_cmd = [
        sys.executable,
        str(eval_script),
        str(gold_path),
        str(transcript_path),
        "--host-speakers",
        args.host_speakers,
        "--frame-hz",
        str(args.frame_hz),
        "--min-error-dur",
        str(args.min_error_dur),
    ]

    rc = run_and_tee(eval_cmd, eval_out, project_root)
    if rc != 0:
        print(f"ERROR: eval_two_speaker_from_transcripts.py failed (code {rc})", file=sys.stderr)
        return rc

    analyze_cmd = [
        sys.executable,
        str(analyze_script),
        "--snap",
        str(snap_path),
        "--eval",
        str(eval_out),
        "--eval-tol-sec",
        str(args.eval_tol_sec),
        "--stuck-lookback-sec",
        str(args.stuck_lookback_sec),
        "--stuck-margin-th",
        str(args.stuck_margin_th),
    ]

    rc, output = run_and_capture(analyze_cmd, project_root)
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    with eval_out.open("a", encoding="utf-8") as out_file:
        out_file.write("\n")
        out_file.write("\n=== analyze_boundary_snap.py output ===\n")
        out_file.write(output)
    if rc != 0:
        print(f"ERROR: analyze_boundary_snap.py failed (code {rc})", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
