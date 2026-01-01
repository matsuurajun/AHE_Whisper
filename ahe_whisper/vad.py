# -*- coding: utf-8 -*-
import numpy as np
import onnxruntime as ort
from pathlib import Path
import logging
from typing import Any, Tuple

LOGGER = logging.getLogger("ahe_whisper_worker")

class VAD:
    def __init__(self, model_path: Path, config: Any) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.config = config
        
        # モデルの入出力名を動的に取得
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # state入力が必要かどうかを判定
        self.uses_combined_state = 'state' in self.input_names
        self.reset_states()
        LOGGER.info(f"VAD interface detected: uses_combined_state={self.uses_combined_state}")

    def reset_states(self) -> None:
        # モデルの仕様に合わせて内部状態(h, c または state)を初期化
        try:
            if self.uses_combined_state:
                # v5 style: 'state' (2, 1, 128)
                # ★【修正】バッチサイズ(dim 1)は必ず1にする
                # モデルから取得したshape情報の2番目が動的(str)の場合でも1に固定
                state_input = next((i for i in self.session.get_inputs() if i.name == 'state'), None)
                if state_input and len(state_input.shape) == 3:
                     shape = list(state_input.shape)
                     # shape[0] (layer数) が整数の場合はそのまま、そうでなければ2
                     shape[0] = shape[0] if isinstance(shape[0], int) else 2
                     # shape[1] (バッチサイズ) は必ず 1 に固定
                     shape[1] = 1
                     # shape[2] (hidden_size) が整数の場合はそのまま、そうでなければ128
                     shape[2] = shape[2] if isinstance(shape[2], int) else 128
                     
                     self._state = np.zeros(shape, dtype=np.float32)
                else:
                     # フォールバック（標準的なSilero V5の形状）
                     self._state = np.zeros((2, 1, 128), dtype=np.float32)
            else:
                # v4 style: 'h', 'c'
                self._h = np.zeros((2, 1, 64), dtype=np.float32)
                self._c = np.zeros((2, 1, 64), dtype=np.float32)
        except Exception as e:
            LOGGER.warning(f"State initialization fallback used: {e}")
            # 安全策として最も一般的な形状で初期化
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, chunk: np.ndarray, sr: int) -> float:
        # ONNXモデルを実行して確率を返す
        if self.uses_combined_state:
            ort_inputs = {
                "input": chunk,
                "sr": np.array([sr], dtype=np.int64),
                "state": self._state,
            }
            out, state = self.session.run(None, ort_inputs)
            self._state = state
        else:
            ort_inputs = {
                "input": chunk,
                "sr": np.array([sr], dtype=np.int64),
                "h": self._h,
                "c": self._c,
            }
            out, h, c = self.session.run(None, ort_inputs)
            self._h, self._c = h, c
            
        return out[0][0]

    # --- 確率列を整形するメソッド（提言の実装） ---
    def refine_probs(self, probs: np.ndarray, kernel: int = 7, sharpness: float = 5.0) -> np.ndarray:
        """
        1. 移動平均でフレーム毎のバラつき（チャタリング）を抑制
        2. シグモイド関数で 0.5 を中心に 0/1 へと信号を分離（コントラスト強調）
        """
        if len(probs) < kernel:
            return probs
            
        # 1. 移動平均 (Smoothing)
        # カーネルサイズ7は約70ms-100msの平滑化に相当
        k = np.ones(kernel, dtype=np.float32) / float(kernel)
        p_smooth = np.convolve(probs, k, mode="same")
        
        # 2. 非線形圧縮 (Non-linear Compression)
        # x = 0 (確率0.5) を中心に、sharpnessの強さで 0.0 または 1.0 に押し切る
        x = (p_smooth - 0.5) * sharpness
        p_sharp = 1.0 / (1.0 + np.exp(-x))
        
        return p_sharp

    def get_speech_probabilities(self, waveform: np.ndarray, sr: int, grid_hz: int):
        window_size = getattr(self.config, "window_size_samples", 512)
        
        # --- 1. ロバスト正規化（入力レベルの最適化） ---
        if len(waveform) > 0:
            # 99.5%点の音量を基準にする（突発ノイズ無視）
            robust_max = np.percentile(np.abs(waveform), 99.5)
            if robust_max > 0:
                # 0.9はクリッピング防止のマージン
                scale_factor = 0.9 / (robust_max + 1e-8)
                waveform_norm = np.clip(waveform * scale_factor, -1.0, 1.0)
                LOGGER.info(f"[DEBUG-VAD] Applied robust normalization: scale_factor={scale_factor:.2f}")
            else:
                waveform_norm = waveform
        else:
            waveform_norm = waveform

        probs = []
        self.reset_states()
        
        # --- 2. 推論実行 ---
        for i in range(0, len(waveform_norm), window_size):
            chunk = waveform_norm[i: i + window_size]
            if len(chunk) < window_size:
                chunk = np.pad(chunk, (0, window_size - len(chunk)))
            
            # (1, window_size) の形状にして推論
            prob = self(chunk.reshape(1, -1), sr)
            probs.append(float(prob))

        probs = np.array(probs, dtype=np.float32)
        
        # --- 3. 確率列の整形（平滑化＋強調） ---
        # 提言の推奨値: kernel=7, sharpness=5.0
        probs_refined = self.refine_probs(probs, kernel=7, sharpness=5.0)
        
        if len(probs) > 0:
            LOGGER.info(f"[DEBUG-VAD] Probs stats (raw)   : mean={probs.mean():.3f}, std={probs.std():.3f}")
            LOGGER.info(f"[DEBUG-VAD] Probs stats (refined): mean={probs_refined.mean():.3f}, std={probs_refined.std():.3f}")

        # 時間軸の生成
        total_sec = len(waveform) / sr
        if total_sec <= 0:
             return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
             
        num_grid_points = int(total_sec * grid_hz)
        if num_grid_points <= 0:
             grid_times = np.array([0.0], dtype=np.float32)
             interp_probs = np.array([probs_refined[0]] if len(probs_refined)>0 else [0.0], dtype=np.float32)
             return interp_probs, grid_times

        grid_times = np.linspace(0.0, total_sec, num_grid_points, endpoint=False, dtype=np.float32)

        # リサンプリング
        frame_times = np.arange(len(probs_refined), dtype=np.float32) * (window_size / sr)

        if frame_times.size < 2:
            fill_value = probs_refined[0] if probs_refined.size > 0 else 0.0
            interp_probs = np.full_like(grid_times, fill_value, dtype=np.float32)
        else:
            interp_probs = np.interp(grid_times, frame_times, probs_refined).astype(np.float32)
        
        return interp_probs, grid_times