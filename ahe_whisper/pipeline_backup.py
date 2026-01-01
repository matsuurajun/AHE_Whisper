# -*- coding: utf-8 -*-
import time
import os
import logging
from pathlib import Path
import numpy as np
import librosa
from typing import Dict, Any

import mlx_whisper

from ahe_whisper.config import AppConfig
from ahe_whisper.embedding import build_ecapa_session, warmup_ecapa, ecapa_embed_batched
from ahe_whisper.vad import VAD
from ahe_whisper.diarizer import Diarizer
from ahe_whisper.aligner import OverlapDPAligner
from ahe_whisper.utils import get_metrics, add_metric, calculate_coverage_metrics
from ahe_whisper.word_grouper import group_words_sudachi
from ahe_whisper.model_manager import ensure_model_available

LOGGER = logging.getLogger("ahe_whisper_worker")

def run(
    audio_path: str,
    config: AppConfig,
    project_root: Path
) -> Dict[str, Any]:
    
    t0 = time.perf_counter()
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration_sec = len(waveform) / sr
    LOGGER.info(f"Audio loaded: duration={duration_sec:.2f}s")
    
    asr_model_path = ensure_model_available('asr', project_root)
    vad_model_path = ensure_model_available('vad', project_root)
    ecapa_model_path = ensure_model_available('embedding', project_root)
    
    ecapa_sess = build_ecapa_session(ecapa_model_path, config.embedding)
    warmup_ecapa(ecapa_sess)
    
    LOGGER.info(f"[DEBUG] calling mlx_whisper.transcribe: "
            f"no_speech_threshold={config.transcription.no_speech_threshold}, "
            f"vad_filter={config.transcription.vad_filter if hasattr(config.transcription, 'vad_filter') else 'N/A'}")

    asr_result = mlx_whisper.transcribe(
        audio=waveform,
        path_or_hf_repo=str(asr_model_path),
        language=config.transcription.language,
        word_timestamps=True,
        no_speech_threshold=getattr(config.transcription, "no_speech_threshold", 0.65),
        condition_on_previous_text=False,
)
    LOGGER.info(f"[DEBUG] asr_result keys: {list(asr_result.keys()) if asr_result else 'EMPTY'}")
    
    if asr_result and "segments" in asr_result:
        total_words = sum(len(seg.get("words", [])) for seg in asr_result["segments"])
        LOGGER.info(f"[TRACE-ASR-RAW] segments={len(asr_result['segments'])}, total_words={total_words}")
        
    segments = asr_result.get("segments", []) or []
    words = []

    # --- ASR 結果の正規化（segments→words） ---
    for seg in segments:
        seg_start = float(seg.get("start", 0.0) or 0.0)
        seg_end   = float(seg.get("end",   seg_start) or seg_start)
        
        if seg_end <= seg_start:
            continue
        
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                w_start = float(w.get("start", seg_start) or seg_start)
                w_end   = float(w.get("end",   seg_end)   or seg_end)
                if w_end <= w_start:
                    continue
                words.append({
                    "word":  (w.get("word") or w.get("text") or "").strip(),
                    "start": w_start,
                    "end":   w_end,
                    "confidence": w.get("confidence"),
                    "avg_logprob": w.get("avg_logprob"),
                })
        else:
            # word_timestamps が無い場合はセグメント単位で補完
            text = (seg.get("text") or "").strip()
            words.append({
                "word":  text,
                "start": seg_start,
                "end":   seg_end,
                "confidence": seg.get("confidence"),
                "avg_logprob": seg.get("avg_logprob"),
            })

    # --- Debug 出力（Aligner に渡す直前ログ） ---
    if words:
        first_w = words[0]; last_w = words[-1]
        LOGGER.info("[DEBUG-ASR→ALIGN] words_len=%d, first=(%.2f,'%.20s'), last=(%.2f,'%.20s')",
                    len(words),
                    float(first_w.get("start", 0.0) or 0.0), str(first_w.get("word",""))[:20],
                    float(last_w.get("end", 0.0) or 0.0),   str(last_w.get("word",""))[:20])
    else:
        LOGGER.warning("[DEBUG-ASR→ALIGN] words_len=0 (flatten failed)")

    LOGGER.info(f"[DEBUG-ASR] segments={len(segments)}, words={len(words)}, dur={asr_result.get('duration', 'N/A')}s")

    vad = VAD(vad_model_path, config.vad)
    vad_probs, grid_times = vad.get_speech_probabilities(waveform, sr, config.aligner.grid_hz)
    add_metric("vad.grid_size", len(grid_times))
    calculate_coverage_metrics(words, vad_probs, duration_sec, config)

    if not words:
        LOGGER.warning("Whisper detected no words. Aborting diarization.")
        return {"words": [], "speaker_segments": [], "duration_sec": duration_sec, "metrics": get_metrics(), "is_fallback": True}
    
    asr_words = words

    win_len = int(config.embedding.embedding_win_sec * sr)
    hop_len = int(config.embedding.embedding_hop_sec * sr)
    audio_chunks = [waveform[i:i+win_len] for i in range(0, len(waveform), hop_len)]
    
    embeddings = ecapa_embed_batched(ecapa_sess, ecapa_model_path, audio_chunks, sr, config.embedding) if audio_chunks else np.zeros((0, config.embedding.embedding_dim))
    
    valid_embeddings_mask = np.sum(np.abs(embeddings), axis=1) > 1e-6
    
    is_fallback = False
    if not np.any(valid_embeddings_mask):
        LOGGER.warning("No valid embeddings extracted. Falling back to single speaker.")
        valid_words = [w for w in words if w.get('start') is not None and w.get('end') is not None]
        speaker_segments = [(valid_words[0]['start'], valid_words[-1]['end'], 0)] if valid_words else []
        is_fallback = True
    else:
        diarizer = Diarizer(config.diarization)
        speaker_centroids, labels = diarizer.cluster(embeddings[valid_embeddings_mask])
        add_metric("diarizer.num_speakers_found", len(speaker_centroids))
        
        spk_probs = diarizer.get_speaker_probabilities(embeddings, valid_embeddings_mask, speaker_centroids, grid_times, hop_len, sr)
        
        config.aligner.non_speech_th = 0.02
        aligner = OverlapDPAligner(config.aligner)
        
        # === DEBUG (AHE): pre-align check ===
        try:
            LOGGER.info(f"[TRACE-ALIGNER-PRECHECK] type(words)={type(words)}, "
                        f"len(words)={len(words) if hasattr(words,'__len__') else 'N/A'}")
            if isinstance(words, list):
                LOGGER.info(f"[TRACE-ALIGNER-PRECHECK] sample(0:3)={words[:3]}")
        except Exception as e:
            LOGGER.error(f"[TRACE-ALIGNER-PRECHECK] inspection failed: {e}")
        
        # === EXISTING LOG ===
        LOGGER.info(f"[TRACE-ALIGNER-IN] words={len(words)}, vad_probs={len(vad_probs)}, "
                    f"spk_probs={spk_probs.shape if hasattr(spk_probs, 'shape') else 'N/A'}, "
                    f"grid_times={len(grid_times)}")
        
        speaker_segments = aligner.align(words, vad_probs, spk_probs, grid_times)
        
        if speaker_segments and isinstance(speaker_segments[0], (list, tuple)) and len(speaker_segments[0]) == 3:
            result["speaker_segments"] = [
                {"start": s, "end": e, "speaker": f"SPEAKER_{spk:02d}"}
                for s, e, spk in speaker_segments
            ]
            LOGGER.info(f"[PIPELINE] Aligner produced {len(result['speaker_segments'])} speaker segments")
        else:
            LOGGER.warning(f"[PIPELINE] Unexpected speaker_segments structure: {type(speaker_segments)}")
        
        if speaker_segments and speaker_segments[-1][1] < duration_sec * 0.9:
            LOGGER.warning(f"[ALIGNER-FIX] alignment ended early at {speaker_segments[-1][1]:.1f}s (<90% of audio). Expanding fallback.")
            speaker_segments = [(0.0, duration_sec, 0)]
    
    words = group_words_sudachi(words)
    add_metric("asr.word_count", len(words))

    if not speaker_segments and words:
        LOGGER.warning("DP alignment yielded no segments. Falling back to a single speaker segment.")
        valid_words = [w for w in words if w.get('start') is not None and w.get('end') is not None]
        speaker_segments = [(valid_words[0]['start'], valid_words[-1]['end'], 0)] if valid_words else []
        is_fallback = True

    add_metric("pipeline.total_time_sec", time.perf_counter() - t0)
    
    return {
        "words": words,
        "speaker_segments": speaker_segments,
        "duration_sec": duration_sec,
        "metrics": get_metrics(),
        "is_fallback": is_fallback
    }
