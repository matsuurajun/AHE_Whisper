# -*- coding: utf-8 -*-
import sys
import os
import logging
import time
import traceback
import json
import io
from pathlib import Path
from datetime import datetime
from multiprocessing import Queue
from typing import Dict, Any, List, Tuple, Optional
import re

from dacite import from_dict, Config as DaciteConfig

from ahe_whisper.config import AppConfig
from ahe_whisper.pipeline import run as run_ai_pipeline
from ahe_whisper.exporter import Exporter
from ahe_whisper.utils import reset_metrics

def _merge_continuous_speaker_segments(segments: List, max_gap: float, min_dur: float) -> List:
    if not segments:
        return []

    # --- sort safely for both dict and tuple structures ---
    if isinstance(segments[0], dict):
        segments.sort(key=lambda x: float(x.get("start", 0.0)))
    else:
        segments.sort(key=lambda x: x[0])

    merged = []
    if isinstance(segments[0], dict):
        cs, ce, ck = float(segments[0]["start"]), float(segments[0]["end"]), segments[0].get("speaker", "SPEAKER_00")
        for seg in segments[1:]:
            ns, ne, nk = float(seg["start"]), float(seg["end"]), seg.get("speaker", "SPEAKER_00")
            if nk == ck and (ns - ce) <= max_gap:
                ce = max(ce, ne)
            else:
                if ce - cs >= min_dur:
                    merged.append({"start": float(cs), "end": float(ce), "speaker": ck})
                cs, ce, ck = ns, ne, nk
        if ce - cs >= min_dur:
            merged.append({"start": float(cs), "end": float(ce), "speaker": ck})
    else:
        cs, ce, ck = segments[0]
        for ns, ne, nk in segments[1:]:
            if nk == ck and (ns - ce) <= max_gap:
                ce = max(ce, ne)
            else:
                if ce - cs >= min_dur:
                    merged.append((cs, ce, ck))
                cs, ce, ck = ns, ne, nk
        if ce - cs >= min_dur:
            merged.append((cs, ce, ck))

    return merged

def worker_process_loop(job_q: Queue, result_q: Queue, log_q: Queue, project_root_str: str) -> None:
    project_root = Path(project_root_str)
    
    # === [AHE PATCH] ログバッファとリダイレクト用クラスの定義 ===
    import io
    LOG_BUFFER = io.StringIO()
    STDERR_TRACE_EVENTS: List[Dict[str, Any]] = []

    class BufferHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = self.format(record)
            LOG_BUFFER.write(msg + "\n")

    class QueueHandler(logging.Handler):
        def __init__(self, q: Queue) -> None:
            super().__init__()
            self.q = q
            self._ahe_queue_handler = True
        def emit(self, record: logging.LogRecord) -> None:
            self.q.put(self.format(record))

    class _AlignerTraceFilter(logging.Filter):
        def __init__(self) -> None:
            super().__init__()
            self._keep = os.environ.get("AHE_KEEP_ALIGNER_TRACE_LOGS", "0") in ("1", "true", "True", "yes", "YES", "on", "ON")
        def filter(self, record: logging.LogRecord) -> bool:
            if self._keep:
                return True
            try:
                msg = record.getMessage()
            except Exception:
                return True
            # pipeline.py が出す TRACE をデフォルトで抑止（必要時は env で opt-in）
            if msg.startswith("[TRACE-ALIGNER"):
                return False
            return True

    class _LogRedirect:
        _ALREADY_LOG_LINE_RES = (
            # Our own format (QueueHandler / BufferHandler)
            re.compile(r"^\d{2}:\d{2}:\d{2}-(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)-[^:]+:"),
            # Common logging formats from libraries (avoid stderr re-injection)
            re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:,\d{3})?\s*-\s*(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*-\s*[^:]+:"),
            re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:,\d{3})?\s*\[(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\]\s*[^:]+:"),
            # Block re-injection of our own prefixed output
            re.compile(r"^\[STD(?:ERR|OUT)\]"),
            # Block SENTINEL messages to prevent dual-logging
            re.compile(r"^\[SENTINEL\]"),
        )
        def __init__(
            self,
            logger: logging.Logger,
            level: int,
            orig_fd: Optional[int],
            stream_name: str,
            trace_events: Optional[List[Dict[str, Any]]],
        ) -> None:
            self._logger = logger
            self._level = level
            self._orig_fd = orig_fd
            self._stream_name = stream_name
            self._trace_events = trace_events
            self._collecting_traceback = False
            self._traceback_lines = []

        def write(self, msg: str) -> int:
            if not msg:
                return 0
            text = str(msg)
            # Some libraries write \r updates; treat them as separators.
            text = text.replace("\r", "\n")
            lines = text.split("\n")
            written = len(msg)
            
            for raw in lines:
                line = raw
                if not line or not line.strip():
                    continue
                check_line = line.lstrip()

                # 1) Block re-injection of already-formatted log lines (especially important for stderr redirect)
                if any(r.match(check_line) for r in self._ALREADY_LOG_LINE_RES):
                    continue
                # 2) Block re-injection of our own multi-line Performance Report when it leaks to stderr/stdout
                if check_line.startswith("--- Performance Report") or check_line.startswith("---- Performance Report"):
                    continue
                if check_line.startswith("- ") and ("RTF=" in check_line or "VAD Speech Ratio" in check_line or "Time breakdown" in check_line):
                    continue

                # Traceback capture (only for stderr)
                if self._stream_name == "stderr":
                    if check_line.startswith("Traceback"):
                        self._collecting_traceback = True
                        self._traceback_lines = [line]
                        continue
                    if self._collecting_traceback:
                        self._traceback_lines.append(line)
                        # Heuristic end of traceback: last line with exception type/message (no indent)
                        if check_line and not line.startswith(" "):
                            tb_text = "\n".join(self._traceback_lines)
                            self._collecting_traceback = False
                            self._traceback_lines = []
                            try:
                                if self._trace_events is not None:
                                    # 元々の trace_events は dict のリストなので合わせる
                                    self._trace_events.append({"text": tb_text, "origin": "TRACEBACK", "stack": tb_text})
                            except Exception:
                                pass
                            self._logger.log(self._level, f"[STDERR][TRACEBACK]\n{tb_text}")
                        continue

                # Best-effort origin capture
                origin = None
                try:
                    stack = traceback.extract_stack(limit=10)
                    for fr in reversed(stack):
                        if "pipeline_worker.py" not in fr.filename:
                            origin = f"{Path(fr.filename).name}:{fr.lineno}"
                            break
                except Exception:
                    origin = None

                prefix = "[STDERR]" if self._stream_name == "stderr" else "[STDOUT]"
                
                # stdout/stderr に流れてきた「デバッグ/トレース print」を INFO/ERROR に上げない
                # 例: [DEBUG-*], [TRACE-*] が performance.txt を汚すのを防ぐ
                if check_line.startswith("[DEBUG-") or check_line.startswith("[TRACE-"):
                    if origin:
                        self._logger.debug(f"{prefix}({origin}) {check_line}")
                    else:
                        self._logger.debug(f"{prefix} {check_line}")
                    continue

                # 特殊キーワードの処理 (Fetching/Downloading など)
                if "Fetching" in check_line or "Downloading" in check_line or "%|" in check_line:
                    self._logger.info(f"{prefix}({origin if origin else '?'}) {check_line}")
                elif 'warning' in check_line.lower():
                    self._logger.warning(f"{prefix}({origin if origin else '?'}) {check_line}")
                else:
                    if origin:
                        self._logger.log(self._level, f"{prefix}({origin}) {check_line}")
                    else:
                        self._logger.log(self._level, f"{prefix} {check_line}")
            
            return written
        def flush(self) -> None: pass
        def isatty(self) -> bool: return False
        def fileno(self) -> int:
            if self._orig_fd is not None: return self._orig_fd
            raise io.UnsupportedOperation("fileno")

    # === [AHE PATCH] ルートロガーへの統一と再入安全な設定 ===
    root = logging.getLogger()
    
    # ハンドラが既に設定されているかチェックするためのフラグ
    if not hasattr(root, "_ahe_handlers_set"):
        trace_filter = _AlignerTraceFilter()
        # QueueHandler
        q_handler = QueueHandler(log_q)
        q_handler.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s-%(name)s: %(message)s', datefmt='%H:%M:%S'))
        q_handler.setLevel(logging.INFO)
        q_handler.addFilter(trace_filter)
        root.addHandler(q_handler)
        
        # BufferHandler
        b_handler = BufferHandler()
        b_handler.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s-%(name)s: %(message)s', datefmt='%H:%M:%S'))
        b_handler.setLevel(logging.INFO)
        b_handler.addFilter(trace_filter)
        root.addHandler(b_handler)
        
        # Python standard warnings を捕捉
        logging.captureWarnings(True)
        # py.warnings は WARNING として GUI に出す（stderr 再捕捉で ERROR 化させない）
        logging.getLogger("py.warnings").setLevel(logging.WARNING)
        
        # うるさいライブラリを黙らせる
        logging.getLogger("numba").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        
        root.setLevel(logging.DEBUG)  # allow submodule DEBUG; handlers filter to INFO
        root._ahe_handlers_set = True

    # 既存の worker logger は propagate=True にしてハンドラを持たせない
    logger = logging.getLogger("ahe_whisper_worker")
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.propagate = True
    logger.setLevel(logging.INFO)

    original_stdout, original_stderr = sys.stdout, sys.stderr
    
    orig_stdout_fd: Optional[int] = None
    orig_stderr_fd: Optional[int] = None
    try:
        if hasattr(sys, '__stdout__') and sys.__stdout__ is not None and hasattr(sys.__stdout__, 'fileno'):
            orig_stdout_fd = sys.__stdout__.fileno()
        if hasattr(sys, '__stderr__') and sys.__stderr__ is not None and hasattr(sys.__stderr__, 'fileno'):
            orig_stderr_fd = sys.__stderr__.fileno()
    except (AttributeError, io.UnsupportedOperation):
        pass

    sys.stdout = _LogRedirect(logger, logging.INFO, orig_stdout_fd, "stdout", None)
    sys.stderr = _LogRedirect(logger, logging.ERROR, orig_stderr_fd if orig_stderr_fd is not None else orig_stdout_fd, "stderr", STDERR_TRACE_EVENTS)

    dacite_config = DaciteConfig(type_hooks={int: lambda data: int(round(data)) if isinstance(data, float) else data})

    def generate_rich_perf_report(step_times: List[Tuple[str, float]], result: Dict[str, Any]) -> List[str]:
        total_time = sum(t for _, t in step_times) or 1.0
        duration = result.get('duration_sec', 0.0)
        rtf = total_time / max(duration, 1.0)
        
        metrics = result.get('metrics', {})
        asr_coverage = metrics.get('asr.coverage_ratio', 0.0) * 100
        vad_speech = metrics.get('vad.speech_ratio', 0.0) * 100

        # diarizer-related metrics
        num_found = metrics.get("diarizer.num_speakers_found")
        num_eff = metrics.get("diarizer.num_speakers_effective")
        min_unmet = bool(metrics.get("diarizer.min_speakers_unmet", False))
        cluster_mass = metrics.get("diarizer.cluster_mass", None)

        report = [f"--- Performance Report (RTF={rtf:.3f}) ---"]
        report.append(f"- ASR Coverage             : {asr_coverage:6.2f}%")
        report.append(f"- VAD Speech Ratio         : {vad_speech:6.2f}%")

        if num_found is not None and num_eff is not None:
            line = f"- Speakers (found/effective): {int(num_found):3d} / {int(num_eff):3d}"
            if min_unmet:
                line += "  [MIN_UNMET]"
            report.append(line)

        if cluster_mass:
            try:
                mass_str = ", ".join(f"{float(m):.3f}" for m in cluster_mass)
            except Exception:
                mass_str = str(cluster_mass)
            report.append(f"- Cluster mass fractions   : [{mass_str}]")

        report.append("-" * 38)
        for name, step_time in step_times:
            report.append(f"- {name:<25}: {step_time:>7.2f}s ({(step_time / total_time * 100):.1f}%)")
        return report

    try:
        while True:
            # === [AHE PATCH] 複数ジョブ実行時にバッファを毎回リセット ===
            LOG_BUFFER.seek(0)
            LOG_BUFFER.truncate(0)
            STDERR_TRACE_EVENTS.clear()
            LOG_BUFFER.write(f"{datetime.now().isoformat()} - INFO - Worker start job\n")

            job = job_q.get()
            if job is None: 
                logger.info("Shutdown signal received.")
                break
            try:
                config_dict, audio_path = job
                config = from_dict(data_class=AppConfig, data=config_dict, config=dacite_config)
                
                logger.info(f"--- Job Start: {Path(audio_path).name} ---")
                if os.environ.get("AHE_TEST_PY_WARNINGS", "0") == "1":
                    import warnings
                    warnings.warn(
                        "AHE TEST: warnings.warn() should appear as WARNING (logger=py.warnings)",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                reset_metrics()
                step_times, t_start = [], time.perf_counter()
                #res = run_ai_pipeline(audio_path, config, project_root)
                #step_times.append(("Core AI Pipeline", time.perf_counter() - t_start))
                # --- Stage-by-stage measurement ---
                step_times = []
                t_total_start = time.perf_counter()
                
                # --- ASR ---
                t_asr = time.perf_counter()
                run_ai_pipeline(audio_path, config, project_root, stage="asr")
                step_times.append(("ASR (mlx_whisper)", time.perf_counter() - t_asr))
                
                # --- VAD ---
                t_vad = time.perf_counter()
                run_ai_pipeline(audio_path, config, project_root, stage="vad")
                step_times.append(("VAD (silero_vad)", time.perf_counter() - t_vad))
                
                # --- DIAR ---
                t_diar = time.perf_counter()
                run_ai_pipeline(audio_path, config, project_root, stage="diar")
                backend_name = getattr(config.embedding, "backend", "campplus")
                step_times.append(
                    (f"DIAR ({backend_name})", time.perf_counter() - t_diar)
                )
                
                # --- FULL PIPELINE (ALIGN + EXPORT) ---
                t_main = time.perf_counter()
                res = run_ai_pipeline(audio_path, config, project_root)
                step_times.append(("Core AI Pipeline (Align+Post)", time.perf_counter() - t_main))
                
                # --- ensure exportable structure ---
                if "speaker_segments" not in res or not res["speaker_segments"]:
                    logger.warning("[PIPELINE-WORKER] No speaker segments found. Creating fallback segment for export.")
                    dur = res.get("duration_sec", 0.0)
                    res["speaker_segments"] = [{"start": 0.0, "end": dur, "speaker": "SPEAKER_00"}]
                
                # reconstruct text if missing
                if "text" not in res or not isinstance(res["text"], str) or not res["text"].strip():
                    if "words" in res:
                        res["text"] = " ".join([w.get("word", "") for w in res["words"] if w.get("word")])
                        logger.info(f"[PIPELINE-WORKER] Reconstructed text length: {len(res['text'])}")
                    else:
                        res["text"] = ""

                t_post_start = time.perf_counter()
                is_fallback = res.get("is_fallback", False)
                min_duration = config.diarization.min_fallback_duration_sec if is_fallback else config.diarization.min_speaker_duration_sec
                merged = _merge_continuous_speaker_segments(res.get("speaker_segments", []), config.diarization.max_merge_gap_sec, min_duration)
                res["speaker_segments"] = merged
                step_times.append(("Post-processing", time.perf_counter() - t_post_start))
                
                # --- attach text to each segment for export ---
                if "speaker_segments" in res and "words" in res:
                    for seg in res["speaker_segments"]:
                        seg_words = [
                            w.get("word", w.get("text", ""))  # 安全に取得
                            for w in res.get("words", [])
                            if isinstance(w, dict) and seg["start"] <= w.get("start", 0) < seg["end"]
                        ]
                        seg["text"] = "".join(seg_words).strip()
                    valid_segs = [s for s in res["speaker_segments"] if s.get("text")]
                    logger.info(f"[PIPELINE-WORKER] Segments with text: {len(valid_segs)} / {len(res['speaker_segments'])}")
                    
                    # --- ensure backward-compatible format for exporter ---
                    if "speaker_segments" in res and isinstance(res["speaker_segments"], list):
                        segs = []
                        for seg in res["speaker_segments"]:
                            start = seg.get("start", 0.0)
                            end = seg.get("end", start)
                            spk = seg.get("speaker", 0)
                            segs.append((start, end, spk))
                        res["speaker_segments_raw"] = segs
                        logger.info(f"[PIPELINE-WORKER] Exporter-compatible speaker_segs_raw: {len(segs)} items")

                t_export_start = time.perf_counter()
                exporter = Exporter(config.output_dir, config.export)
                run_dir = exporter.save(res, Path(audio_path).stem)
                (run_dir / "run_config.json").write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
                step_times.append(("Exporting Files", time.perf_counter() - t_export_start))

                report = generate_rich_perf_report(step_times, res)
                logger.info("\n" + "\n".join(report))
                
                # === [AHE PATCH] ジョブ全体のGUIログ（レポートを含む）を保存 ===
                # report は前の logger.info で LOG_BUFFER に書き込まれているため、
                # 直接書き込みを廃止し、ログバッファ全体を出力することで重複を防ぐ
                full_log_path = run_dir / "performance.log"
                with open(full_log_path, "w", encoding="utf-8") as f:
                    f.write(LOG_BUFFER.getvalue())
                txt_log_path = run_dir / "performance.txt"
                with open(txt_log_path, "w", encoding="utf-8") as f:
                    f.write(LOG_BUFFER.getvalue())
                logger.info(f"[PIPELINE] log saved to {full_log_path}")
                logger.info(f"[PIPELINE] log saved to {txt_log_path}")

                # === [AHE TRACE] stderr の生出力が残っている場合、1例だけ出所を表示 ===
                if STDERR_TRACE_EVENTS:
                    ev = STDERR_TRACE_EVENTS[0]
                    logger.info(
                        "[STDERR-TRACE] raw stderr sample captured: "
                        f"origin={ev.get('origin','UNKNOWN')} text={ev.get('text','')}"
                    )
                    try:
                        trace_path = run_dir / "stderr_trace.log"
                        with open(trace_path, "w", encoding="utf-8") as tf:
                            tf.write(f"origin: {ev.get('origin','UNKNOWN')}\n")
                            tf.write(f"text: {ev.get('text','')}\n\n")
                            tf.write(ev.get('stack',''))
                    except Exception:
                        pass

                result_q.put({"success": True, "output_dir": str(run_dir)})
            except Exception as e:
                logger.error(f"Critical error in worker: {e}\n{traceback.format_exc()}")
                result_q.put({"success": False, "error": str(e)})
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
