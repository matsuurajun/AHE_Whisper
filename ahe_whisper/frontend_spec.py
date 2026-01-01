# -*- coding: utf-8 -*-
import json
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any
from collections import deque
from threading import Lock

try:
    import yaml
except ImportError:
    yaml = None

LOGGER = logging.getLogger("ahe_whisper_worker")

class BoundedWarningCache:
    def __init__(self, maxsize: int = 128) -> None:
        self._warned = set()
        self._queue = deque(maxlen=max(1, maxsize))
        self._lock = Lock()

    def add_and_check(self, item: str) -> bool:
        with self._lock:
            if item in self._warned:
                return False
            if len(self._queue) >= self._queue.maxlen:
                oldest = self._queue.popleft()
                self._warned.discard(oldest)
            self._warned.add(item)
            self._queue.append(item)
            return True

_WARNING_CACHE = BoundedWarningCache()

DEFAULT_SR = 16000
DEFAULT_FRAME_LENGTH_MS = 25.0
DEFAULT_FRAME_SHIFT_MS = 10.0
DEFAULT_NUM_MELS = 80
DEFAULT_WINDOW_TYPE = "povey"
DEFAULT_SNIP_EDGES = True
DEFAULT_DITHER = 1.0

@dataclass(frozen=True)
class FeaturizerSpec:
    sample_rate: int
    frame_length_ms: float
    frame_shift_ms: float
    num_mel_bins: int
    window_type: str
    snip_edges: bool
    dither: float
    
    @property
    def hop_length(self) -> int: return max(1, int(round(self.sample_rate * (self.frame_shift_ms / 1000.0))))
    
    @property
    def win_length(self) -> int: return max(1, int(round(self.sample_rate * (self.frame_length_ms / 1000.0))))

def _find_config_yaml(start_dir: Path) -> Optional[Path]:
    p = start_dir.resolve()
    for _ in range(5):
        if (cand := p / "config.yaml").exists(): return cand
        if p.parent == p: break
        p = p.parent
    return None

def load_spec_from_config(config_yaml: Path) -> FeaturizerSpec:
    if yaml is None: raise RuntimeError("PyYAML is required.")
    with config_yaml.open("r", encoding="utf-8") as f: data = yaml.safe_load(f)
    frontend = data.get("frontend", {})
    fbank = frontend.get("fbank", {})
    return FeaturizerSpec(
        sample_rate=int(fbank.get("sample_rate", DEFAULT_SR)),
        frame_length_ms=float(fbank.get("frame_length", DEFAULT_FRAME_LENGTH_MS)),
        frame_shift_ms=float(fbank.get("frame_shift", DEFAULT_FRAME_SHIFT_MS)),
        num_mel_bins=int(fbank.get("num_mel_bins", DEFAULT_NUM_MELS)),
        window_type=str(fbank.get("window_type", DEFAULT_WINDOW_TYPE)),
        snip_edges=bool(fbank.get("snip_edges", DEFAULT_SNIP_EDGES)),
        dither=float(fbank.get("dither", DEFAULT_DITHER))
    )

def _get_default_spec() -> FeaturizerSpec:
    return FeaturizerSpec(DEFAULT_SR, DEFAULT_FRAME_LENGTH_MS, DEFAULT_FRAME_SHIFT_MS, DEFAULT_NUM_MELS, DEFAULT_WINDOW_TYPE, DEFAULT_SNIP_EDGES, DEFAULT_DITHER)

def load_spec_for_model(model_file: Path) -> Tuple[FeaturizerSpec, Optional[Path]]:
    try:
        model_dir = model_file.parent
        cfg = _find_config_yaml(model_dir)
        if cfg: return load_spec_from_config(cfg), cfg
        
        if _WARNING_CACHE.add_and_check(str(model_dir)):
            LOGGER.warning("config.yaml not found near model: %s; using default.", model_file)
        return _get_default_spec(), None
    except Exception as e:
        LOGGER.exception("Failed to load config.yaml near %s; using defaults. Error: %s", model_file, e)
        return _get_default_spec(), None

@dataclass(frozen=True)
class CMVNPolicy:
    mode: str
    global_mean: Optional[np.ndarray]
    global_std: Optional[np.ndarray]
    
    def is_global(self) -> bool: return self.mode == "cmvn_global"

def resolve_cmvn_policy(model_dir: Path, prefer_cmvn: bool = True) -> CMVNPolicy:
    if not prefer_cmvn: return CMVNPolicy(mode="cmn", global_mean=None, global_std=None)
    for fname in ("mean_std.json", "cmvn.json"):
        if (path := model_dir / fname).exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                mean = np.array(data["mean"], dtype=np.float32)
                std = np.array(data["std"], dtype=np.float32)
                if mean.ndim == 1 and std.ndim == 1:
                    LOGGER.info("Found global CMVN stats file: %s", path)
                    return CMVNPolicy(mode="cmvn_global", global_mean=mean, global_std=std)
            except Exception:
                LOGGER.warning(f"Failed to parse CMVN file: {path}", exc_info=True)
                continue
    return CMVNPolicy(mode="cmn", global_mean=None, global_std=None)
