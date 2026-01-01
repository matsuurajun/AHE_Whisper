# -*- coding: utf-8 -*-
import numpy as np
import threading
from typing import List, Dict, Any

_METRICS = threading.local()

def reset_metrics():
    _METRICS.data = {}

def add_metric(key: str, value: Any):
    if not hasattr(_METRICS, 'data'):
        reset_metrics()
    _METRICS.data[key] = value

def get_metrics() -> Dict[str, Any]:
    return getattr(_METRICS, 'data', {})

def calculate_coverage_metrics(words: List[Dict[str, Any]], vad_probs: np.ndarray, duration_sec: float, config: Any):
    if not hasattr(_METRICS, 'data'):
        reset_metrics()
    
    if duration_sec > 0.1:
        valid_words = [w for w in words if w.get('start') is not None and w.get('end') is not None]
        if valid_words:
            try:
                start_times = [float(w['start']) for w in valid_words]
                end_times = [float(w['end']) for w in valid_words]
                asr_coverage_sec = max(end_times) - min(start_times)
                _METRICS.data['asr.coverage_ratio'] = asr_coverage_sec / duration_sec
            except (ValueError, TypeError):
                 _METRICS.data['asr.coverage_ratio'] = 0.0
        else:
            _METRICS.data['asr.coverage_ratio'] = 0.0

        if vad_probs.size > 0:
            speech_frames = np.sum(vad_probs > config.diarization.vad_th_start)
            _METRICS.data['vad.speech_ratio'] = speech_frames / vad_probs.size
        else:
            _METRICS.data['vad.speech_ratio'] = 0.0
    else:
        _METRICS.data['asr.coverage_ratio'] = 0.0
        _METRICS.data['vad.speech_ratio'] = 0.0


def safe_l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return np.divide(x, np.maximum(norm, eps), out=np.zeros_like(x), where=norm > 0)

def safe_softmax(s: np.ndarray, tau: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    z = s / max(tau, eps)
    m = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - m)
    p = e / np.maximum(np.sum(e, axis=-1, keepdims=True), eps)
    return p
