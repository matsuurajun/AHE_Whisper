#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
from multiprocessing import get_context
from pathlib import Path
from threading import Thread

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ahe_whisper.config import AppConfig
from ahe_whisper.pipeline_worker import worker_process_loop


def _drain_logs(log_q) -> None:
    while True:
        msg = log_q.get()
        if msg is None:
            break


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--audio",
        required=True,
        help="Audio file path (.wav/.mp3/.m4a/.flac).",
    )
    ap.add_argument(
        "--preset",
        default="three_step",
        help="Aligner boundary_refine preset name (default: three_step).",
    )
    ap.add_argument(
        "--boundary-snap",
        action="store_true",
        help="Enable boundary_snap export via AHE_BOUNDARY_SNAP=1.",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    if args.boundary_snap:
        os.environ.setdefault("AHE_BOUNDARY_SNAP", "1")

    project_root = PROJECT_ROOT
    config = AppConfig()
    if args.preset:
        config.aligner.boundary_refine_preset = str(args.preset)

    ctx = get_context("spawn")
    job_q = ctx.Queue()
    result_q = ctx.Queue()
    log_q = ctx.Queue()

    log_thread = Thread(target=_drain_logs, args=(log_q,), daemon=True)
    log_thread.start()

    proc = ctx.Process(
        target=worker_process_loop,
        args=(job_q, result_q, log_q, str(project_root)),
    )
    proc.start()

    job_q.put((config.to_dict(), str(audio_path)))
    job_q.put(None)

    proc.join()

    log_q.put(None)
    log_thread.join(timeout=2)

    result = None
    if not result_q.empty():
        result = result_q.get()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
