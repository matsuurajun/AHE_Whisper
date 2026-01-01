# -*- coding: utf-8 -*-
import numpy as np
import librosa
import logging
from typing import Optional, Union

from ahe_whisper.frontend_spec import FeaturizerSpec, CMVNPolicy

RNG = np.random.default_rng(42)

LOGGER = logging.getLogger("ahe_whisper_worker")

def povey_window(window_len: int) -> np.ndarray:
    if window_len <= 1:
        return np.ones(window_len, dtype=np.float32)
    n = np.arange(window_len, dtype=np.float32)
    cos_term = np.cos(2.0 * np.pi * n / (window_len - 1))
    return (0.5 - 0.5 * cos_term) ** 0.85

class Featurizer:
    def __init__(self, spec: FeaturizerSpec) -> None:
        self.spec = spec

    def get_mel_spectrogram(self, waveform: np.ndarray, sr: int, cmvn: CMVNPolicy) -> Optional[np.ndarray]:
        if waveform.ndim > 1:
            waveform = librosa.to_mono(waveform)
        if sr != self.spec.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.spec.sample_rate)

        if self.spec.dither > 0.0:
            waveform = waveform + self.spec.dither * RNG.standard_normal(size=waveform.shape).astype(waveform.dtype)

        window_func: Union[str, np.ndarray]
        if self.spec.window_type == "povey":
            window_func = povey_window(self.spec.win_length)
        else:
            window_func = self.spec.window_type

        center = not self.spec.snip_edges
        win = self.spec.win_length
        n_fft = 1 << (win - 1).bit_length()

        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.spec.sample_rate,
            n_fft=n_fft,
            hop_length=self.spec.hop_length,
            win_length=win,
            n_mels=self.spec.num_mel_bins,
            window=window_func,
            center=center,
            pad_mode="reflect",
        )
        
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec = np.nan_to_num(log_mel_spec, neginf=-80.0, posinf=80.0)
        log_mel_spec = log_mel_spec.T

        if log_mel_spec.shape[0] < 20:
            return None

        if cmvn.is_global():
            log_mel_spec = (log_mel_spec - cmvn.global_mean) / (cmvn.global_std + 1e-8)
        else:
            log_mel_spec = log_mel_spec - np.mean(log_mel_spec, axis=0)
            
        return log_mel_spec.astype(np.float32)
