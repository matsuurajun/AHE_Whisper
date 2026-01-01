# -*- coding: utf-8 -*-
import hashlib
import logging
import os
import sys
import json
import time
import math
import bisect

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ahe_whisper.config import AlignerConfig

logger = logging.getLogger(__name__)

# --- AHE trace toggles ---
# - Enable aligner trace by: AHE_TRACE_ALIGNER=1
# - Extra arg tracing (small) can be enabled by: AHE_TRACE_ALIGNER_ARGS=1
# Default OFF (normal runs should not emit SENTINEL/TRACE)
_AHE_TRACE_ALIGNER: bool = os.environ.get("AHE_TRACE_ALIGNER", "0") in ("1", "true", "True")
_AHE_ALIGNER_FP: str = hex(abs(hash(__file__)) & 0xFFFFFFFFFFFF)[2:]
_AHE_SENTINEL_ONCE: set[str] = set()


def _emit_sentinel_entry(where: str) -> None:
    if not _AHE_TRACE_ALIGNER:
        return
    key = f"{where}:{os.getpid()}"
    if key in _AHE_SENTINEL_ONCE:
        return
    _AHE_SENTINEL_ONCE.add(key)

    module_file = globals().get("__file__", "unknown")
    try:
        fr = sys._getframe(2)  # caller of __init__/align
        caller = f"{fr.f_code.co_filename}:{fr.f_lineno} in {fr.f_code.co_name}"
    except Exception:
        caller = "unknown"

    msg = f"[SENTINEL][ALIGNER] {where} entered | module={module_file} fp={_AHE_ALIGNER_FP} | caller={caller}"
    try:
        logger.info(msg)
    except Exception:
        pass
    # stderr.write removed to avoid dual-logging via _LogRedirect

# --- length-aware DP / island smoothing パラメータ ---
# 短い孤立スイッチ (A-B-A) を「島」とみなす最大長 (秒)
_SWITCH_ISLAND_MIN_DUR = 1.0

# 「長く続いている話者からのスイッチ」を重く罰するためのパラメータ
# runlen_sec が 1 秒を超えたぶんだけ、最大 8 秒ぶんまでを見る
_SWITCH_LEN_THRESHOLD = 1.0   # これ以下の長さのランからのスイッチは従来どおり
_SWITCH_LEN_SCALE = 0.35      # 超過 1 秒あたりの追加係数（従来 0.2 から強化）
_SWITCH_LEN_CAP = 8.0         # 追加で見る最大秒数（従来 5.0 から拡大）


class OverlapDPAligner:
    def __init__(self, config: AlignerConfig) -> None:
        _emit_sentinel_entry("OverlapDPAligner.__init__")
        self.config = config  # 設定全体を保持
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.delta_switch = config.delta_switch
        self.non_speech_th = config.non_speech_th
        # TEN VAD を使う場合の重み（0.0 なら無効）
        self.ten_weight: float = getattr(config, "ten_weight", 0.0)
        # align() 呼び出しごとにセットされる TEN スコア列（[-1, 1]）
        self._ten_score: Optional[np.ndarray] = None

    def align(
        self,
        word_info: List[Dict[str, Any]],
        vad_probs: np.ndarray,
        spk_probs: np.ndarray,
        grid_times: np.ndarray,
        ten_probs: Optional[np.ndarray] = None,
    ) -> List[Tuple[float, float, int]]:
        _emit_sentinel_entry("OverlapDPAligner.align")

        # Optional: tiny arg summary for debugging (kept out of default noise)
        if (
            _AHE_TRACE_ALIGNER
            and os.environ.get("AHE_TRACE_ALIGNER_ARGS", "0") == "1"
            and logger.isEnabledFor(logging.DEBUG)
        ):
            try:
                wlen = len(word_info) if word_info is not None else 0
                logger.debug(
                    "[TRACE-ALIGNER-ARGS] word_info=%d vad=%s spk=%s grid=%s ten=%s",
                    wlen,
                    getattr(vad_probs, "shape", None),
                    getattr(spk_probs, "shape", None),
                    getattr(grid_times, "shape", None),
                    "None" if ten_probs is None else getattr(ten_probs, "shape", None),
                )
            except Exception:
                pass

        # === DEBUG (AHE): Aligner input summary ===
        if logger.isEnabledFor(logging.DEBUG):
            import inspect, sys
            logger.debug("[ALIGN-CALLER] %s:%d", inspect.stack()[1].filename, inspect.stack()[1].lineno)
            logger.debug("[ALIGN-MODULE] %s loaded_from %s", __name__, sys.modules[__name__].__file__)

        if _AHE_TRACE_ALIGNER and os.environ.get("AHE_TRACE_ALIGNER_ARGS", "0") == "1":
            try:
                total_words = len(word_info) if word_info is not None else 0
                logger.debug(
                    "[TRACE-ALIGNER-ENTRY] len(word_info)=%d type=%s sample=%s",
                    total_words,
                    type(word_info),
                    word_info[:3] if isinstance(word_info, list) else "N/A",
                )
            except Exception as e:
                logger.debug("[TRACE-ALIGNER-ENTRY] inspect failed: %s", e)

        if not word_info or len(grid_times) == 0:
            return []

        num_frames = len(grid_times)
        num_speakers = spk_probs.shape[1]

        if num_speakers == 0:
            return []

        # --- TEN VAD スコアの前処理（任意） ---
        self._ten_score = None
        if ten_probs is not None and self.ten_weight != 0.0:
            ten_arr = np.asarray(ten_probs, dtype=np.float32).reshape(-1)

            if ten_arr.shape[0] != num_frames:
                # 長さが違う場合は、簡易的に truncate / pad で合わせる
                if ten_arr.shape[0] > num_frames:
                    ten_arr = ten_arr[:num_frames]
                else:
                    pad = num_frames - ten_arr.shape[0]
                    ten_arr = np.pad(ten_arr, (0, pad), mode="edge")

            # p in [0,1] → s in [-1,1] へ線形マップ
            self._ten_score = 2.0 * (ten_arr - 0.5)
            self._ten_score = np.clip(self._ten_score, -1.0, 1.0)

        # フレーム間隔 (秒) を推定
        if num_frames >= 2:
            frame_dt = np.diff(grid_times, prepend=grid_times[0]).astype(np.float32)
            if frame_dt[0] <= 0.0 and num_frames > 1:
                frame_dt[0] = frame_dt[1]
        else:
            frame_dt = np.array([0.02], dtype=np.float32)  # 約 20ms 相当の適当な値

        # DP テーブル
        cost = np.full((num_frames, num_speakers), np.inf, dtype=np.float32)
        path = np.full((num_frames, num_speakers), -1, dtype=np.int32)
        # 各状態 (speaker) ごとの「直近ラン長 (秒)」を DP と一緒に更新
        runlen_time = np.zeros((num_frames, num_speakers), dtype=np.float32)
        runlen_time[0, :] = max(float(frame_dt[0]), 1e-3)

        # === [AHE PATCH v90.97] Smooth Aligner (optional pre-processing) ===
        if getattr(self, "config", None) and getattr(self.config, "use_smooth_aligner", False):
            alpha = getattr(self.config, "smooth_alpha", 0.25)
            gamma = getattr(self.config, "smooth_gamma", 1.2)
            logger.info("[SMOOTH-ALIGNER] Applying EMA(alpha=%.3f) + Peak(gamma=%.3f) to spk_probs", alpha, gamma)

            # --- Peak強調 ---
            spk_probs = np.power(spk_probs, gamma)
            spk_probs /= np.sum(spk_probs, axis=1, keepdims=True) + 1e-12

            # --- 時間平滑 (EMA) ---
            ema = np.empty_like(spk_probs)
            ema[0] = spk_probs[0]
            for t in range(1, len(spk_probs)):
                ema[t] = alpha * spk_probs[t] + (1 - alpha) * ema[t - 1]
            spk_probs = ema

            logger.info(
                "[SMOOTH-ALIGNER] Smoothed spk_probs: "
                "shape=%s, mean_max=%.3f",
                spk_probs.shape,
                np.mean(np.max(spk_probs, axis=1))
            )

        # === Posterior calibration / peakiness stats (optional) ===
        # spk_probs は (T, K) の確率行列を想定するが、念のため正規化して扱う。
        eps = 1e-12
        t_scale = float(getattr(self.config, "posterior_temperature", 1.0))
        u_mix = float(getattr(self.config, "posterior_uniform_mix", 0.0))
        a_w = float(getattr(self.config, "acoustic_weight", 1.0))

        spk_probs_norm = np.asarray(spk_probs, dtype=np.float32)
        spk_probs_clip = np.clip(spk_probs_norm, eps, None)
        spk_probs_norm = spk_probs_clip / (np.sum(spk_probs_clip, axis=1, keepdims=True) + eps)

        max_p_before = np.max(spk_probs_norm, axis=1)
        ent_before = -np.sum(spk_probs_norm * np.log(spk_probs_norm + eps), axis=1)
        mean_max_before = float(np.mean(max_p_before))
        mean_ent_before = float(np.mean(ent_before))

        spk_probs_cal = spk_probs_norm
        enabled = (t_scale != 1.0) or (u_mix != 0.0) or (a_w != 1.0)
        if enabled:
            if t_scale != 1.0:
                spk_probs_cal = np.power(spk_probs_cal, 1.0 / t_scale)
                spk_probs_cal = spk_probs_cal / (np.sum(spk_probs_cal, axis=1, keepdims=True) + eps)
            if a_w != 1.0:
                spk_probs_cal = np.power(spk_probs_cal, a_w)
                spk_probs_cal = spk_probs_cal / (np.sum(spk_probs_cal, axis=1, keepdims=True) + eps)
            if u_mix != 0.0:
                spk_probs_cal = (1.0 - u_mix) * spk_probs_cal + u_mix * (1.0 / float(num_speakers))
            spk_probs = spk_probs_cal

        max_p_after = np.max(spk_probs_cal, axis=1)
        ent_after = -np.sum(spk_probs_cal * np.log(spk_probs_cal + eps), axis=1)
        mean_max_after = float(np.mean(max_p_after))
        mean_ent_after = float(np.mean(ent_after))

        has_t = hasattr(self.config, "posterior_temperature")
        logger.info(
            "[POSTERIOR-CAL] has_posterior_temperature=%s T=%.3f lam=%.3f w=%.3f mean_max: %.4f -> %.4f, mean_entropy: %.4f -> %.4f",
            has_t,
            t_scale,
            u_mix,
            a_w,
            mean_max_before,
            mean_max_after,
            mean_ent_before,
            mean_ent_after,
        )

        # === Frame-DP emission mode (prob vs log-prob) ===
        # Purpose:
        # - Make "log-emission" unambiguous: frame DP uses spk_emit, while spk_probs remains for diagnostics/refine/cp/word-dp.
        # - Prove activation via a single INFO line showing emission stats (speech-only when possible).
        _env_mode_raw = os.environ.get("AHE_FRAME_EMISSION_MODE", "").strip()
        _env_mode = _env_mode_raw.lower()
        _cfg_mode = str(getattr(self.config, "frame_emission_mode", "")).strip().lower()
        # Env override takes precedence over config to enable clean A/B tests without editing files.
        frame_emission_mode = (_env_mode if _env_mode else (_cfg_mode if _cfg_mode else "prob"))
        if frame_emission_mode in ("log", "logp", "logprob", "log-prob", "log_prob"):
            spk_emit = np.log(np.clip(np.asarray(spk_probs, dtype=np.float32), eps, 1.0)).astype(np.float32)
            frame_emission_mode = "log"
        else:
            spk_emit = np.asarray(spk_probs, dtype=np.float32)
            frame_emission_mode = "prob"

        # One-line proof log (keep log hygiene: 1 line per align call)
        try:
            vad_1d = np.asarray(vad_probs, dtype=np.float32).reshape(-1)
            speech_th = float(
                max(
                    float(self.non_speech_th),
                    float(getattr(self.config, "silence_vad_th", self.non_speech_th)),
                )
            )
            speech_mask = (
                (vad_1d >= speech_th) if vad_1d.size == num_frames else np.ones(num_frames, dtype=bool)
            )
            x = spk_emit[speech_mask] if (spk_emit.ndim == 2 and speech_mask.size == num_frames) else spk_emit
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            if x.size > 0:
                q50 = float(np.quantile(x.astype(np.float64), 0.50))
                q95 = float(np.quantile(x.astype(np.float64), 0.95))
                logger.info(
                    "[FRAME-EMISSION] mode=%s (env=%s) | emit[min=%.4f p50=%.4f p95=%.4f max=%.4f] speech_th=%.3f",
                    frame_emission_mode,
                    _env_mode_raw if _env_mode_raw else "unset",
                    float(np.min(x)),
                    q50,
                    q95,
                    float(np.max(x)),
                    speech_th,
                )
        except Exception as exc:
            logger.warning("[FRAME-EMISSION] diagnostics failed: %s", exc)

        # === Posterior argmax / margin (for switch gating) ===
        # Use spk_probs (not spk_emit) so it works in both prob/log emission modes.
        try:
            p_argmax_t = np.argmax(spk_probs, axis=1).astype(np.int32)
            if num_speakers >= 2:
                top2 = np.partition(spk_probs, -2, axis=1)[:, -2:]
                p_margin_t = (top2[:, 1] - top2[:, 0]).astype(np.float32)

                # normalized entropy in [0,1] (1.0 is max-entropy)
                eps_ent = 1e-12
                p_ent = np.asarray(spk_probs, dtype=np.float32)
                p_ent = np.clip(p_ent, eps_ent, 1.0)
                p_ent = p_ent / (np.sum(p_ent, axis=1, keepdims=True) + eps_ent)
                ent = -np.sum(p_ent * np.log(p_ent + eps_ent), axis=1)
                denom = float(np.log(float(num_speakers)))
                p_entropy_norm_t = (ent / max(1e-12, denom)).astype(np.float32)
            else:
                p_margin_t = np.zeros((num_frames,), dtype=np.float32)
                p_entropy_norm_t = np.zeros((num_frames,), dtype=np.float32)
        except Exception:
            p_argmax_t = None
            p_margin_t = None
            p_entropy_norm_t = None

        switch_post_k = float(getattr(self.config, "switch_post_k", 0.0))
        switch_post_margin_th = float(getattr(self.config, "switch_post_margin_th", 0.5))
        if switch_post_k > 0.0:
            logger.info(
                "[SWITCH-POST-GATE] post_k=%.3f post_margin_th=%.3f",
                switch_post_k,
                switch_post_margin_th,
            )

        switch_uncertain_penalty_mult = float(getattr(self.config, "switch_uncertain_penalty_mult", 1.0))
        switch_uncertain_entropy_norm_th = float(
            getattr(self.config, "switch_uncertain_entropy_norm_th", 1.0)
        )
        switch_uncertain_margin_th = float(getattr(self.config, "switch_uncertain_margin_th", 0.0))
        switch_uncertain_use_speech_mask = bool(
            getattr(self.config, "switch_uncertain_use_speech_mask", False)
        )
        speech_th_for_uncertain = float(
            max(
                float(self.non_speech_th),
                float(getattr(self.config, "silence_vad_th", self.non_speech_th)),
            )
        )

        if (
            switch_uncertain_penalty_mult > 1.0
            and p_entropy_norm_t is not None
            and p_margin_t is not None
            and num_frames > 0
        ):
            try:
                vad_1d = np.asarray(vad_probs, dtype=np.float32).reshape(-1)
                if switch_uncertain_use_speech_mask:
                    speech_mask = (
                        (vad_1d >= speech_th_for_uncertain)
                        if vad_1d.size == num_frames
                        else np.ones((num_frames,), dtype=bool)
                    )
                else:
                    speech_mask = np.ones((num_frames,), dtype=bool)

                uncertain = (
                    speech_mask
                    & (p_entropy_norm_t >= switch_uncertain_entropy_norm_th)
                    & (p_margin_t <= switch_uncertain_margin_th)
                )
                unc_sec = (
                    float(np.sum(frame_dt[uncertain]))
                    if frame_dt.size == num_frames
                    else float(np.sum(uncertain.astype(np.float32))) / float(self.grid_hz)
                )
                logger.info(
                    "[SWITCH-UNCERTAIN] mult=%.2f ent_norm_th=%.3f margin_th=%.3f | use_speech_mask=%s speech_th=%.3f uncertain_sec=%.2f",
                    switch_uncertain_penalty_mult,
                    switch_uncertain_entropy_norm_th,
                    switch_uncertain_margin_th,
                    switch_uncertain_use_speech_mask,
                    speech_th_for_uncertain,
                    unc_sec,
                )
            except Exception as exc:
                logger.warning("[SWITCH-UNCERTAIN] diagnostics failed: %s", exc)

        # === Anti-dictatorship detector: posterior dominance diagnostics (log-only) ===
        # 目的: "embedding の独裁" を「観測可能」にする（DP の入力 posterior が支配的すぎないか）。
        # - mean_max: 各フレームの top1 posterior の平均
        # - top1-top2: 2位との差（マージン）
        # - entropy: 分布の鋭さ（低いほど独裁的）
        try:
            spk_stats = np.asarray(spk_probs, dtype=np.float32)
            if spk_stats.ndim == 2 and spk_stats.shape[0] == num_frames and spk_stats.shape[1] == num_speakers:
                spk_stats = np.clip(spk_stats, eps, None)
                spk_stats = spk_stats / (np.sum(spk_stats, axis=1, keepdims=True) + eps)

                # top1 / top2 / margin
                if num_speakers >= 2:
                    part = np.partition(spk_stats, kth=num_speakers - 2, axis=1)
                    top1 = part[:, -1]
                    top2 = part[:, -2]
                else:
                    top1 = spk_stats[:, 0]
                    top2 = np.zeros_like(top1)

                margin12 = top1 - top2
                ent = -np.sum(spk_stats * np.log(spk_stats + eps), axis=1)

                # 診断用の「発話フレーム」: silence_vad_th があればそれを優先、無ければ non_speech_th
                speech_th = float(
                    max(
                        float(self.non_speech_th),
                        float(getattr(self.config, "silence_vad_th", self.non_speech_th)),
                    )
                )
                vad_1d = np.asarray(vad_probs, dtype=np.float32).reshape(-1)
                speech_mask = (vad_1d >= speech_th) if vad_1d.size == num_frames else np.ones(num_frames, dtype=bool)

                def _q(x: np.ndarray, q: float) -> float:
                    if x.size == 0:
                        return float("nan")
                    return float(np.quantile(x.astype(np.float64), q))

                # speech-only stats（無音を混ぜると mean_max が不当に上がる/下がるので切り分け）
                top1_s = top1[speech_mask]
                margin_s = margin12[speech_mask]
                ent_s = ent[speech_mask]
                dt_s = frame_dt[speech_mask] if frame_dt.size == num_frames else np.array([], dtype=np.float32)

                speech_sec = float(np.sum(dt_s)) if dt_s.size > 0 else float(grid_times[-1] - grid_times[0]) if num_frames >= 2 else 0.0

                mean_max = float(np.mean(top1_s)) if top1_s.size > 0 else float(np.mean(top1))
                mean_margin = float(np.mean(margin_s)) if margin_s.size > 0 else float(np.mean(margin12))
                mean_ent = float(np.mean(ent_s)) if ent_s.size > 0 else float(np.mean(ent))

                # 独裁ゾーン検出（ログ用。DP の挙動は変えない）
                max_th = 0.95
                margin_th = 0.70
                ent_th = 0.30
                min_zone_sec = 0.50

                hi = (
                    speech_mask
                    & (top1 >= max_th)
                    & (margin12 >= margin_th)
                    & (ent <= ent_th)
                )

                zones = []
                in_zone = False
                z_start = 0
                z_sec = 0.0
                for t in range(num_frames):
                    dt = float(frame_dt[t]) if t < frame_dt.size else 0.0
                    if bool(hi[t]):
                        if not in_zone:
                            in_zone = True
                            z_start = t
                            z_sec = 0.0
                        z_sec += dt
                    else:
                        if in_zone:
                            zones.append((z_start, t - 1, z_sec))
                            in_zone = False
                if in_zone:
                    zones.append((z_start, num_frames - 1, z_sec))

                zones = [z for z in zones if z[2] >= min_zone_sec]
                zones_sorted = sorted(zones, key=lambda z: z[2], reverse=True)

                high_sec = float(np.sum(frame_dt[hi])) if frame_dt.size == num_frames else 0.0
                high_ratio = (high_sec / max(1e-6, speech_sec)) if speech_sec > 0.0 else 0.0
                longest_sec = float(zones_sorted[0][2]) if zones_sorted else 0.0

                top_zones_str = ""
                if zones_sorted:
                    parts = []
                    for s_idx, e_idx, sec in zones_sorted[:5]:
                        s_t = float(grid_times[s_idx])
                        e_t = float(grid_times[e_idx]) + float(frame_dt[e_idx]) if e_idx < frame_dt.size else float(grid_times[e_idx])
                        parts.append(f"{s_t:.2f}-{e_t:.2f}s({sec:.2f}s)")
                    top_zones_str = ", ".join(parts)

                log_fn = logger.warning if (high_ratio >= 0.50 and longest_sec >= 2.0) else logger.info
                log_fn(
                    "[DOMINANCE] speech_th=%.3f speech_sec=%.2f | "
                    "mean_max=%.3f(p95=%.3f,p99=%.3f) "
                    "mean_margin=%.3f(p95=%.3f) "
                    "mean_entropy=%.3f(p05=%.3f) | "
                    "hi_conf: sec=%.2f(%.1f%%) longest=%.2fs zones=%d | %s",
                    speech_th,
                    speech_sec,
                    mean_max,
                    _q(top1_s if top1_s.size > 0 else top1, 0.95),
                    _q(top1_s if top1_s.size > 0 else top1, 0.99),
                    mean_margin,
                    _q(margin_s if margin_s.size > 0 else margin12, 0.95),
                    mean_ent,
                    _q(ent_s if ent_s.size > 0 else ent, 0.05),
                    high_sec,
                    100.0 * high_ratio,
                    longest_sec,
                    len(zones),
                    top_zones_str,
                )

                logger.debug(
                    "[DOMINANCE-TRACE] first10: max=%s margin=%s ent=%s",
                    np.array2string(top1[:10], precision=3),
                    np.array2string(margin12[:10], precision=3),
                    np.array2string(ent[:10], precision=3),
                )
        except Exception as exc:
            logger.warning("[DOMINANCE] diagnostics failed: %s", exc)

        word_costs = self._precompute_word_costs(word_info, grid_times)
        # 句読点ベースのスイッチペナルティスケール（use_lexical_prior が False なら全て 1.0）
        lexical_scales = self._compute_lexical_switch_scales(word_info, grid_times)

        # === beta_t: time-varying speaker posterior weight (anti "embedding dictatorship") ===
        # DP の speaker 項を self.beta で固定せず、フレームごとに beta_t[t] を使う。
        # - posterior が極端に鋭い(=支配的)ほど beta を下げ、他の要素(語境界/VAD/スイッチ罰則)を通す
        # - さらに VAD の speech onset/offset 近傍では beta を追加で下げて境界を動かしやすくする
        beta_t = np.full((num_frames,), float(self.beta), dtype=np.float32)
        if bool(getattr(self.config, "use_beta_t", False)) and num_speakers > 0:
            eps = 1e-12
            strength = float(getattr(self.config, "beta_t_strength", 0.60))
            min_ratio = float(getattr(self.config, "beta_t_min_ratio", 0.25))
            max_ratio = float(getattr(self.config, "beta_t_max_ratio", 1.00))
            speech_th = float(
                getattr(
                    self.config,
                    "beta_t_speech_th",
                    getattr(self.config, "silence_vad_th", 0.35),
                )
            )
            edge_window_sec = float(getattr(self.config, "beta_t_edge_window_sec", 0.50))
            edge_ratio = float(getattr(self.config, "beta_t_edge_ratio", 0.60))

            strength = float(np.clip(strength, 0.0, 1.0))
            min_ratio = float(np.clip(min_ratio, 0.0, 1.0))
            max_ratio = float(np.clip(max_ratio, min_ratio, 1.0))
            edge_ratio = float(np.clip(edge_ratio, 0.0, 1.0))
            edge_window_sec = float(max(edge_window_sec, 0.0))

            # spk_probs を正規化してから、dominance 指標を計算
            p = np.asarray(spk_probs, dtype=np.float32)
            p = np.clip(p, eps, 1.0)
            p = p / (np.sum(p, axis=1, keepdims=True) + eps)

            if num_speakers > 1:
                ent = -np.sum(p * np.log(p + eps), axis=1)
                ent_norm = ent / float(np.log(float(num_speakers)))
                ent_norm = np.clip(ent_norm, 0.0, 1.0)

                top2 = np.partition(p, num_speakers - 2, axis=1)[:, -2:]
                top1 = np.max(top2, axis=1)
                top2v = np.min(top2, axis=1)
                margin = top1 - top2v
                # K=2 なら margin ∈ [0,1]。一般 K では最大 ~ (1 - 1/K) にスケール。
                denom = float(max(1e-6, 1.0 - 1.0 / float(num_speakers)))
                margin_norm = np.clip(margin / denom, 0.0, 1.0)

                dominance = (1.0 - ent_norm) * margin_norm  # ∈ [0,1]
            else:
                dominance = np.zeros((num_frames,), dtype=np.float32)

            ratio = 1.0 - strength * dominance
            ratio = np.clip(ratio, min_ratio, max_ratio).astype(np.float32)
            beta_t = (float(self.beta) * ratio).astype(np.float32)

            # VAD の speech onset/offset 近傍で追加で down-weight（境界を動かしやすくする）
            if edge_window_sec > 0.0 and edge_ratio < 1.0:
                speech = np.asarray(vad_probs, dtype=np.float32) >= float(speech_th)

                since_onset = np.full((num_frames,), np.inf, dtype=np.float32)
                run = np.inf
                for t in range(num_frames):
                    if bool(speech[t]):
                        if t == 0 or not bool(speech[t - 1]):
                            run = 0.0
                        else:
                            run = float(run) + float(frame_dt[t - 1])
                        since_onset[t] = float(run)
                    else:
                        run = np.inf

                until_offset = np.full((num_frames,), np.inf, dtype=np.float32)
                run = np.inf
                for t in range(num_frames - 1, -1, -1):
                    if bool(speech[t]):
                        if t == num_frames - 1 or not bool(speech[t + 1]):
                            run = 0.0
                        else:
                            run = float(run) + float(frame_dt[t])
                        until_offset[t] = float(run)
                    else:
                        run = np.inf

                edge = speech & ((since_onset <= edge_window_sec) | (until_offset <= edge_window_sec))
                beta_t[edge] *= float(edge_ratio)

                # 端の down-weight の後も min/max を維持
                beta_t = np.clip(beta_t, float(self.beta) * min_ratio, float(self.beta) * max_ratio)

            try:
                q05 = float(np.quantile(beta_t, 0.05))
                q95 = float(np.quantile(beta_t, 0.95))
            except Exception:
                q05 = float(np.min(beta_t))
                q95 = float(np.max(beta_t))

            logger.info(
                "[BETA-T] enabled=True base=%.3f strength=%.2f edge=%.2fs*%.2f | beta_t: mean=%.3f min=%.3f p05=%.3f p95=%.3f max=%.3f",
                float(self.beta),
                strength,
                edge_window_sec,
                edge_ratio,
                float(np.mean(beta_t)),
                float(np.min(beta_t)),
                q05,
                q95,
                float(np.max(beta_t)),
            )

        # ------------------------
        # Switch change-point (cp_t) from posterior movement (proxy for embedding change-point)
        # - cp_t in [0,1], larger => more boundary-like => reduce switch penalty locally
        # - Default OFF: switch_cp_k=0.0 (no behavior change)
        switch_cp_k = float(getattr(self.config, "switch_cp_k", 0.0))
        switch_cp_floor_uncertain = float(getattr(self.config, "switch_cp_floor_uncertain", 0.0))
        if switch_cp_floor_uncertain < 0.0:
            switch_cp_floor_uncertain = 0.0
        if switch_cp_floor_uncertain > 1.0:
            switch_cp_floor_uncertain = 1.0
        cp_t = None  # type: Optional[np.ndarray]
        if switch_cp_k > 0.0 and num_frames >= 2 and num_speakers > 0:
            eps = 1e-12
            p0 = spk_probs[:-1, :].astype(np.float32, copy=False)
            p1 = spk_probs[1:, :].astype(np.float32, copy=False)
            num = np.sum(p0 * p1, axis=1)
            den = (np.linalg.norm(p0, axis=1) * np.linalg.norm(p1, axis=1)) + eps
            d = 1.0 - (num / den)  # [0, 1] (practically), higher => bigger change
            d = np.clip(d, 0.0, 1.0).astype(np.float32)

            raw = np.zeros((num_frames,), dtype=np.float32)
            raw[1:] = d

            # smoothing (frames)
            win = int(getattr(self.config, "switch_cp_smooth_win", 5))
            win = max(1, win)
            if win > 1:
                kernel = np.ones((win,), dtype=np.float32) / float(win)
                smooth = np.convolve(raw, kernel, mode="same").astype(np.float32)
            else:
                smooth = raw

            # robust normalize to [0,1] using p05/p95
            p_lo = float(getattr(self.config, "switch_cp_norm_p_lo", 0.05))
            p_hi = float(getattr(self.config, "switch_cp_norm_p_hi", 0.95))
            p_lo = float(np.clip(p_lo, 0.0, 0.49))
            p_hi = float(np.clip(p_hi, 0.51, 1.0))
            try:
                qlo = float(np.quantile(smooth, p_lo))
                qhi = float(np.quantile(smooth, p_hi))
            except Exception:
                qlo = float(np.min(smooth))
                qhi = float(np.max(smooth))

            if qhi > qlo + 1e-9:
                cp = (smooth - qlo) / (qhi - qlo)
                cp = np.clip(cp, 0.0, 1.0).astype(np.float32)
            else:
                cp = np.zeros((num_frames,), dtype=np.float32)

            cp_t = cp

            # log only when enabled (keep log hygiene)
            try:
                c_mean = float(np.mean(cp))
                c_min = float(np.min(cp))
                c_max = float(np.max(cp))
                c_p95 = float(np.quantile(cp, 0.95))
            except Exception:
                c_mean = float(np.mean(cp))
                c_min = float(np.min(cp))
                c_max = float(np.max(cp))
                c_p95 = c_max
            logger.info(
                "[SWITCH-CP] enabled=True k=%.3f win=%d norm_p=(%.2f,%.2f) | cp_t: mean=%.3f min=%.3f p95=%.3f max=%.3f",
                switch_cp_k,
                win,
                p_lo,
                p_hi,
                c_mean,
                c_min,
                c_p95,
                c_max,
            )

        # === スコア最大化 DP ===
        eye = np.eye(num_speakers, dtype=np.float32)

        # 初期スコア (t=0)
        for k in range(num_speakers):
            cost[0, k] = -(
                self.alpha * vad_probs[0]
                + float(beta_t[0]) * spk_emit[0, k]
                - self.gamma * word_costs[0]
            )

        # ------------------------
        # Silence reset (runlen reset) to prevent stale-run of the same speaker label
        # Log only once per continuous silence zone to avoid spam
        runlen_reset_logged = False
        runlen_reset_count = 0  # サマリー用カウンタ
        silence_run_sec = 0.0
        # Fetching config once outside the loop for efficiency
        silence_vad_th = float(getattr(self.config, "silence_vad_th", 0.35))
        silence_reset_min_sec = float(getattr(self.config, "silence_reset_min_sec", 0.20))
        use_silence_reset = getattr(self.config, "use_silence_runlen_reset", False)

        # DP更新
        for t in range(1, num_frames):
            vad_score = self.alpha * vad_probs[t]
            word_score_t = -self.gamma * word_costs[t]
            dt = float(frame_dt[t]) if t < len(frame_dt) else float(frame_dt[-1])
            prev_runlen = runlen_time[t - 1]

            # --- 無音に基づくラン長リセット（オプション） ---
            if use_silence_reset:
                vad_t = float(vad_probs[t])
                if vad_t >= silence_vad_th:
                    silence_run_sec = 0.0
                    runlen_reset_logged = False
                else:
                    silence_run_sec += dt

                if silence_reset_min_sec > 0.0 and silence_run_sec >= silence_reset_min_sec:
                    if not runlen_reset_logged:
                        logger.debug(
                            "[RUNLEN-RESET] silence_run_sec=%.2fs >= %.2fs => reset (one-shot per zone)",
                            silence_run_sec,
                            silence_reset_min_sec,
                        )
                        runlen_reset_logged = True
                        runlen_reset_count += 1
                        prev_runlen = np.zeros_like(prev_runlen, dtype=np.float32)

            # 長さ依存のスイッチペナルティ係数（speaker j からの離脱コスト）
            excess = np.clip(prev_runlen - _SWITCH_LEN_THRESHOLD, 0.0, _SWITCH_LEN_CAP)
            penalty_factor = 1.0 + _SWITCH_LEN_SCALE * excess
            # TEN VAD による時間依存スケール：
            #   ten_score[t] > 0 (発話っぽい) ならペナルティを強く、
            #   ten_score[t] < 0 (静音寄り) なら弱くする。
            ten_scale = 1.0
            if self._ten_score is not None and self.ten_weight != 0.0:
                raw_scale = 1.0 + self.ten_weight * float(self._ten_score[t])  # [-1,1] → [1-w,1+w]
                ten_scale = float(np.clip(raw_scale, 0.2, 3.0))

            # 句読点に基づくスイッチペナルティスケール（lexical_scales[t] <= 1.0）
            lexical_scale_t = 1.0
            if lexical_scales is not None and 0 <= t < lexical_scales.shape[0]:
                lexical_scale_t = float(lexical_scales[t])

            # Local switch penalty modulation by change-point (boundary-like => easier to switch)
            cp_scale = 1.0
            if cp_t is not None:
                # scale = max(0, 1 - k*cp_t)
                cp_scale = 1.0 - (switch_cp_k * float(cp_t[t]))
                if cp_scale < 0.0:
                    cp_scale = 0.0

                # Guardrail: when posterior is highly uncertain (small p_margin), do NOT let cp_scale go to 0 too easily.
                # This targets short flip-backs (A->B->A) right after a true boundary when cp is high but margin is tiny.
                if switch_cp_floor_uncertain > 0.0 and p_margin_t is not None:
                    mar = float(p_margin_t[t])
                    if mar < 0.0:
                        mar = 0.0
                    if mar > 1.0:
                        mar = 1.0
                    floor = switch_cp_floor_uncertain * ((1.0 - mar) ** 2)
                    if floor > cp_scale:
                        cp_scale = floor

            # Uncertainty-aware boost: discourage switching when posterior is almost max-entropy and margin is tiny.
            # (This specifically targets low-cp switches where the posterior itself is ambiguous.)
            uncertain_scale = 1.0
            if (
                switch_uncertain_penalty_mult > 1.0
                and p_entropy_norm_t is not None
                and p_margin_t is not None
                and (
                    (not switch_uncertain_use_speech_mask)
                    or (float(vad_probs[t]) >= speech_th_for_uncertain)
                )
            ):
                if (
                    float(p_entropy_norm_t[t]) >= float(switch_uncertain_entropy_norm_th)
                    and float(p_margin_t[t]) <= float(switch_uncertain_margin_th)
                ):
                    uncertain_scale = float(switch_uncertain_penalty_mult)

            switch_penalties = (
                self.delta_switch
                * penalty_factor
                * ten_scale
                * lexical_scale_t
                * cp_scale
                * uncertain_scale
            )  # shape: (num_speakers,)

            # --- DEBUG: lexical prior が実際に使われたフレームだけログ ---
            if lexical_scale_t != 1.0:
                logger.debug(
                    "[LEXICAL-FRAME] "
                    "t=%d, time=%.2fs, "
                    "lexical_scale_t=%.3f",
                    t,
                    float(grid_times[t]),
                    lexical_scale_t
                )

            for k in range(num_speakers):
                spk_score = float(beta_t[t]) * spk_emit[t, k]
                base_score = vad_score + spk_score + word_score_t

                # If enabled: reduce switch penalty only when switching INTO the posterior-argmax speaker
                to_scale = 1.0
                if switch_post_k > 0.0 and p_argmax_t is not None and p_margin_t is not None:
                    if int(p_argmax_t[t]) == int(k):
                        denom = max(1e-6, 1.0 - float(switch_post_margin_th))
                        conf = (float(p_margin_t[t]) - float(switch_post_margin_th)) / denom
                        if conf > 0.0:
                            conf = min(1.0, conf)
                            to_scale = 1.0 - (switch_post_k * conf)
                            to_scale = max(0.0, to_scale)

                # j→k のとき j!=k にだけ「離脱コスト」を課す
                prev_scores = cost[t - 1, :] - (switch_penalties * to_scale) * (1.0 - eye[k])

                best_prev_k = int(np.argmax(prev_scores))
                max_prev_score = prev_scores[best_prev_k]

                cost[t, k] = max_prev_score + base_score
                path[t, k] = best_prev_k

                # ラン長 (秒) を更新：同じ話者なら継続、違えばリセット
                if best_prev_k == k:
                    runlen_time[t, k] = prev_runlen[k] + dt
                else:
                    runlen_time[t, k] = dt

        # バックトラッキング (最大スコアパス)
        final_path = np.zeros(num_frames, dtype=np.int32)
        bt_path = np.zeros(num_frames, dtype=np.int32)
        refined_path = np.zeros(num_frames, dtype=np.int32)
        if num_frames > 0:
            final_path[-1] = int(np.argmax(cost[-1, :]))
            for t in range(num_frames - 2, -1, -1):
                final_path[t] = path[t + 1, final_path[t + 1]]

            bt_path = final_path.copy()
            refined_path = bt_path

            # --- posterior に基づくローカル境界巻き戻し（オプション） ---
            if getattr(self.config, "use_posterior_refine", False):
                refined_path = self._refine_speaker_path_with_posteriors(
                    bt_path, spk_probs, grid_times
                )
            final_path = refined_path

            # --- A-B-A 型の短い「話者の島」を潰す post-fix ---
            final_path = self._smooth_speaker_path(final_path, grid_times)

            # --- Word-level Viterbi DP (optional): override frame path inside ASR word spans ---
            if bool(getattr(self.config, "enable_word_dp", False)):
                try:
                    final_path = self._apply_word_dp_override(
                        final_path=final_path,
                        word_info=word_info,
                        spk_probs=spk_probs,
                        grid_times=grid_times,
                        beta_t=beta_t,
                        lexical_scales=lexical_scales,
                    )
                except Exception:
                    logger.exception("[WORD-DP] failed; falling back to frame DP path")

            # === Anti-dictatorship: (1) DP path vs posterior argmax mismatch, (2) term scale diagnostics ===
            try:
                spk_arr = np.asarray(spk_probs, dtype=np.float32)
                if (
                    spk_arr.ndim == 2
                    and spk_arr.shape[0] == num_frames
                    and spk_arr.shape[1] == num_speakers
                    and num_speakers > 0
                ):
                    post_argmax = np.argmax(spk_arr, axis=1).astype(np.int32)

                    vad_1d = np.asarray(vad_probs, dtype=np.float32).reshape(-1)
                    speech_th = float(
                        max(
                            float(self.non_speech_th),
                            float(getattr(self.config, "silence_vad_th", self.non_speech_th)),
                        )
                    )
                    speech_mask = (
                        (vad_1d >= speech_th) if vad_1d.size == num_frames else np.ones(num_frames, dtype=bool)
                    )

                    def _mismatch_rate(dp_path_1d: np.ndarray) -> Tuple[float, float]:
                        if dp_path_1d.size == 0:
                            return float("nan"), float("nan")
                        mism = (dp_path_1d.astype(np.int32) != post_argmax)
                        all_r = float(np.mean(mism))
                        if speech_mask.size == dp_path_1d.size and np.any(speech_mask):
                            speech_r = float(np.mean(mism[speech_mask]))
                        else:
                            speech_r = float("nan")
                        return all_r, speech_r

                    bt_all, bt_speech = _mismatch_rate(bt_path)
                    rf_all, rf_speech = _mismatch_rate(refined_path)
                    fn_all, fn_speech = _mismatch_rate(final_path)

                    dur = float(grid_times[-1] - grid_times[0]) if num_frames > 1 else 0.0
                    sw = int(np.sum(final_path[1:] != final_path[:-1])) if final_path.size > 1 else 0
                    sw_per_min = (60.0 * float(sw) / dur) if dur > 0 else float("nan")

                    logger.info(
                        "[DICTATORSHIP-MISMATCH] dp!=argmax: bt=%.3f(speech=%.3f) refine=%.3f(speech=%.3f) final=%.3f(speech=%.3f) | switches=%d (%.2f/min) speech_th=%.2f",
                        bt_all,
                        bt_speech,
                        rf_all,
                        rf_speech,
                        fn_all,
                        fn_speech,
                        sw,
                        sw_per_min,
                        speech_th,
                    )

                    # Term scales: compare speaker node margin vs effective switch penalties (on actual switches)
                    beta_1d = np.asarray(beta_t, dtype=np.float32).reshape(-1)
                    if beta_1d.size == num_frames:
                        spk_term = beta_1d[:, None] * spk_arr  # (T,K)
                        if num_speakers >= 2:
                            top1 = np.max(spk_term, axis=1)
                            top2 = np.min(spk_term, axis=1) if num_speakers == 2 else np.partition(spk_term, kth=num_speakers - 2, axis=1)[:, -2]
                        else:
                            top1 = spk_term[:, 0]
                            top2 = np.zeros_like(top1)
                        margin = (top1 - top2)
                    else:
                        top1 = np.asarray([], dtype=np.float32)
                        margin = np.asarray([], dtype=np.float32)

                    vad_term = (self.alpha * vad_1d) if vad_1d.size == num_frames else np.asarray([], dtype=np.float32)
                    word_term = (-self.gamma * np.asarray(word_costs, dtype=np.float32).reshape(-1)) if np.asarray(word_costs).size == num_frames else np.asarray([], dtype=np.float32)

                    sw_mask = (final_path[1:] != final_path[:-1]) if final_path.size > 1 else np.asarray([], dtype=bool)
                    sw_pen = np.asarray([], dtype=np.float32)
                    if sw_mask.size > 0 and np.any(sw_mask) and runlen_time.shape[0] >= num_frames:
                        prev_spk = final_path[:-1].astype(np.int32)
                        rl = runlen_time[:-1, :][np.arange(num_frames - 1), prev_spk]
                        excess = np.clip(rl - _SWITCH_LEN_THRESHOLD, 0.0, _SWITCH_LEN_CAP)
                        pen_fac = 1.0 + _SWITCH_LEN_SCALE * excess

                        ten_scale_vec = np.ones(num_frames - 1, dtype=np.float32)
                        if self._ten_score is not None and self.ten_weight != 0.0:
                            ts = np.asarray(self._ten_score, dtype=np.float32).reshape(-1)
                            if ts.size == num_frames:
                                raw = 1.0 + float(self.ten_weight) * ts[1:]
                                ten_scale_vec = np.clip(raw, 0.2, 3.0).astype(np.float32)

                        lex_vec = np.ones(num_frames - 1, dtype=np.float32)
                        if lexical_scales is not None:
                            ls = np.asarray(lexical_scales, dtype=np.float32).reshape(-1)
                            if ls.size == num_frames:
                                lex_vec = ls[1:]

                        eff_sw_pen = float(self.delta_switch) * pen_fac * ten_scale_vec * lex_vec
                        sw_pen = eff_sw_pen[sw_mask].astype(np.float32)

                    def _q(x: np.ndarray, qv: float) -> float:
                        if x.size == 0:
                            return float("nan")
                        return float(np.quantile(x.astype(np.float64), qv))

                    # speech-only views where applicable
                    top1_s = top1[speech_mask] if top1.size == num_frames else np.asarray([], dtype=np.float32)
                    margin_s = margin[speech_mask] if margin.size == num_frames else np.asarray([], dtype=np.float32)
                    vad_s = vad_term[speech_mask] if vad_term.size == num_frames else np.asarray([], dtype=np.float32)
                    word_s = np.abs(word_term[speech_mask]) if word_term.size == num_frames else np.asarray([], dtype=np.float32)

                    logger.info(
                        "[DICTATORSHIP-SCALES] spk_top1(mean=%.3f p95=%.3f) spk_margin(p50=%.3f p95=%.3f) vad(alpha*p)(mean=%.3f p95=%.3f) word(|gamma*w|)(p50=%.3f p95=%.3f) switch_pen(on_switch p50=%.3f p95=%.3f)",
                        float(np.mean(top1_s)) if top1_s.size else float("nan"),
                        _q(top1_s, 0.95),
                        _q(margin_s, 0.50),
                        _q(margin_s, 0.95),
                        float(np.mean(vad_s)) if vad_s.size else float("nan"),
                        _q(vad_s, 0.95),
                        _q(word_s, 0.50),
                        _q(word_s, 0.95),
                        _q(sw_pen, 0.50),
                        _q(sw_pen, 0.95),
                    )
            except Exception as exc:
                logger.warning("[DICTATORSHIP] dp/posterior diagnostics failed: %s", exc)

        # === RUNLEN-RESET サマリー ===
        if use_silence_reset and runlen_reset_count > 0:
            logger.info("[RUNLEN-RESET-SUMMARY] %d silence zones triggered runlen reset", runlen_reset_count)

        # === セグメント生成 ===
        segments: List[Tuple[float, float, int]] = []
        if num_frames > 0:
            # 無音区間を検出（VAD確率が低い箇所）
            silence_threshold = 0.3  # VAD確率がこれ以下を無音とみなす
            min_silence_duration = 1.0  # 最小無音区間長（秒）
            max_segment_duration = 30.0  # 最大セグメント長（秒）

            # 無音区間の検出
            is_silence = vad_probs < silence_threshold
            silence_starts: List[int] = []
            silence_ends: List[int] = []

            in_silence = False
            silence_start_idx = 0

            for t in range(len(is_silence)):
                if is_silence[t] and not in_silence:
                    # 無音区間開始
                    silence_start_idx = t
                    in_silence = True
                elif not is_silence[t] and in_silence:
                    # 無音区間終了
                    silence_duration = grid_times[t] - grid_times[silence_start_idx]
                    if silence_duration >= min_silence_duration:
                        silence_starts.append(silence_start_idx)
                        silence_ends.append(t)
                    in_silence = False

            # セグメント生成
            segment_start_time = float(grid_times[0])
            current_speaker = int(final_path[0])

            # logger.debug("[SEGMENT-DEBUG] Detected %d silence periods", len(silence_starts))

            # 話者変化点と無音区間を使ってセグメントを分割
            for t in range(1, num_frames):
                current_time = float(grid_times[t])
                segment_duration = current_time - segment_start_time

                # セグメント分割条件
                should_split = False
                split_reason = ""

                # 1. 話者が変わった
                if int(final_path[t]) != current_speaker:
                    should_split = True
                    split_reason = "speaker_change"

                # 2. 無音区間の中央付近
                for i, (s_start, s_end) in enumerate(zip(silence_starts, silence_ends)):
                    if s_start <= t <= s_end and segment_duration > 5.0:  # 5秒以上のセグメント
                        should_split = True
                        split_reason = f"silence_{i}"
                        break

                # 3. セグメントが長すぎる
                if segment_duration >= max_segment_duration:
                    should_split = True
                    split_reason = "max_duration"

                if should_split:
                    segments.append((segment_start_time, current_time, current_speaker))
                    logger.info(
                        "[SEGMENT] %.2f-%.2fs, speaker=%d, reason=%s",
                        segment_start_time,
                        current_time,
                        current_speaker,
                        split_reason
                    )

                    # 次のセグメント開始
                    segment_start_time = current_time
                    current_speaker = int(final_path[t])

            # 最後のセグメント
            end_time = float(grid_times[-1])
            segments.append((segment_start_time, end_time, current_speaker))
            logger.info("[SEGMENT-FINAL] %.2f-%.2fs, speaker=%d", segment_start_time, end_time, current_speaker)

        # === フォールバック: セグメントが少なすぎる場合 ===
        if len(segments) <= 1 and num_frames > 0:
            logger.info("[FALLBACK] Too few segments, creating artificial splits")
            segments = []
            segment_duration = 20.0  # 20秒ごとに分割

            # 話者ごとの平均確率を計算
            avg_spk_probs = np.mean(spk_probs, axis=0)
            dominant_speaker = int(np.argmax(avg_spk_probs))

            total_duration = float(grid_times[-1] - grid_times[0])
            num_segments = max(1, int(total_duration / segment_duration))

            for i in range(num_segments):
                start_time = float(grid_times[0] + i * segment_duration)
                end_time = float(min(grid_times[-1], start_time + segment_duration))

                # この区間の話者を決定
                start_idx = int(np.searchsorted(grid_times, start_time))
                end_idx = int(np.searchsorted(grid_times, end_time))
                if end_idx > start_idx:
                    segment_path = final_path[start_idx:end_idx]
                    # 最頻値を話者とする
                    speaker = int(np.median(segment_path))
                else:
                    speaker = dominant_speaker

                segments.append((start_time, end_time, speaker))
                logger.info("[FALLBACK-SEGMENT] %.2f-%.2fs, speaker=%d", start_time, end_time, speaker)

        # --- Boundary snapshot exporter (disabled by default) ---
        self._maybe_export_boundary_snap(
            grid_times=grid_times,
            spk_probs=spk_probs,
            spk_emit=spk_emit,
            vad_probs=vad_probs,
            final_path=final_path,
            beta_t=beta_t,
            lexical_scales=lexical_scales,
            cp_t=cp_t,
            word_info=word_info,
            frame_emission_mode=frame_emission_mode,
        )

        logger.info(
            "[ALIGNER-RESULT] Generated %d segments covering %.1fs",
            len(segments),
            (segments[-1][1] if segments else 0.0),
        )
        return segments

    def _refine_speaker_path_with_posteriors(
        self,
        path: np.ndarray,
        spk_probs: np.ndarray,
        grid_times: np.ndarray,
    ) -> np.ndarray:
        """
        DP が出した frame-wise path に対して、
        A→B 切替の直前区間で「B の posterior が A を上回っている」フレームまで
        境界を巻き戻すローカル realign。
        """
        if path.size == 0:
            return path
        T = path.size
        if spk_probs.shape[0] != T:
            return path

        max_rb_sec = getattr(self.config, "posterior_max_rollback_sec", 0.0)
        margin = getattr(self.config, "posterior_margin", 0.0)
        min_run_sec = getattr(self.config, "posterior_min_run_sec", 0.0)
        if max_rb_sec <= 0.0 or margin <= 0.0 or min_run_sec <= 0.0:
            return path

        # フレームレートは config.grid_hz を優先
        frame_hz = float(getattr(self.config, "grid_hz", 50) or 50.0)
        max_rb_frames = max(1, int(max_rb_sec * frame_hz + 0.5))
        min_run_frames = max(1, int(min_run_sec * frame_hz + 0.5))

        refined = path.copy()
        t = 1
        while t < T:
            prev_label = int(refined[t - 1])
            cur_label = int(refined[t])
            if cur_label == prev_label:
                t += 1
                continue

            a = prev_label  # 旧話者
            b = cur_label   # 新話者

            # 新話者 b のラン長（t から先）
            run_len = 1
            tt = t + 1
            while tt < T and int(refined[tt]) == b:
                run_len += 1
                tt += 1
            if run_len < min_run_frames:
                t += run_len
                continue

            start = max(0, t - max_rb_frames)
            best_t = None

            # t-1, t-2, ... と遡りながら、
            # refined[tt] が a のままで、
            # spk_probs[tt, b] - spk_probs[tt, a] > margin を満たす最初の位置を探す。
            for tt in range(t - 1, start - 1, -1):
                if int(refined[tt]) != a:
                    # 直前の A ランを越えて巻き戻さない
                    break
                if spk_probs[tt, b] - spk_probs[tt, a] > margin:
                    best_t = tt

            if best_t is not None and best_t < t:
                logger.info(
                    "[POST-REFINE] t=%d->%d time_old=%.2fs time_new=%.2fs a=%d b=%d margin=%.3f delta=%.3f",
                    t,
                    best_t,
                    float(grid_times[t]),
                    float(grid_times[best_t]),
                    a,
                    b,
                    float(margin),
                    float(spk_probs[best_t, b] - spk_probs[best_t, a]),
                )
                refined[best_t:t] = b

            t += run_len

        return refined

    def _smooth_speaker_path(self, path: np.ndarray, grid_times: np.ndarray) -> np.ndarray:
        """
        frame-wise speaker path に対して、
        A (長い) - B (短い) - A (長い) のような「話者 B の短い島」を
        近傍 A で塗りつぶしてジッターを軽減する。
        時間・セグメントの数はここではいじらず、あくまでラベル列だけを修正する。
        """
        if path.size == 0 or grid_times.size == 0:
            return path

        runs: List[Tuple[int, int, int]] = []
        start_idx = 0
        current_spk = int(path[0])

        # 同一話者の連続ランを抽出
        for i in range(1, len(path)):
            spk = int(path[i])
            if spk != current_spk:
                runs.append((start_idx, i - 1, current_spk))
                start_idx = i
                current_spk = spk
        runs.append((start_idx, len(path) - 1, current_spk))

        if len(runs) < 3:
            return path  # A-B-A がそもそも存在しない

        smoothed = path.copy()

        for idx in range(1, len(runs) - 1):
            prev_start, prev_end, prev_spk = runs[idx - 1]
            mid_start, mid_end, mid_spk = runs[idx]
            next_start, next_end, next_spk = runs[idx + 1]

            # A-B-A のみ対象
            if prev_spk != next_spk:
                continue
            if mid_spk == prev_spk:
                continue

            # 真ん中 B ランの長さ (秒)
            start_time = float(grid_times[mid_start])
            end_time = float(grid_times[min(mid_end, len(grid_times) - 1)])
            mid_dur = max(0.0, end_time - start_time)

            if mid_dur <= _SWITCH_ISLAND_MIN_DUR:
                # B ランが十分短ければ、両側の A に合わせる
                smoothed[mid_start:mid_end + 1] = prev_spk

        return smoothed

    def _compute_lexical_switch_scales(
        self,
        word_info: Optional[List[Dict[str, Any]]],
        grid_times: np.ndarray,
    ) -> np.ndarray:
        """
        各フレーム t ごとの「スイッチペナルティ倍率」を返す。
        use_lexical_prior が False のときは全て 1.0。
        """
        num_frames = grid_times.shape[0]
        scales = np.ones(num_frames, dtype=np.float32)

        if not getattr(self.config, "use_lexical_prior", False):
            return scales

        window_sec = float(getattr(self.config, "lexical_window_sec", 0.0) or 0.0)
        if window_sec <= 0.0:
            return scales

        q_scale = float(getattr(self.config, "lexical_question_switch_scale", 1.0) or 1.0)
        p_scale = float(getattr(self.config, "lexical_period_switch_scale", 1.0) or 1.0)

        if not word_info:
            return scales

        for word in word_info:
            text = str(word.get("word") or "")
            if not text:
                continue

            if text.endswith("？") or text.endswith("?"):
                scale = q_scale
            elif text.endswith("。") or text.endswith("."):
                scale = p_scale
            else:
                continue

            if scale >= 1.0:
                continue

            try:
                w_end = float(word.get("end"))
            except (TypeError, ValueError):
                continue

            start = w_end
            stop = w_end + window_sec
            start_idx = int(np.searchsorted(grid_times, start, side="left"))
            stop_idx = int(np.searchsorted(grid_times, stop, side="right"))
            if stop_idx <= start_idx:
                continue

            scales[start_idx:stop_idx] = np.minimum(scales[start_idx:stop_idx], scale)

        # --- DEBUG: lexical prior がどの程度効いたかをサマリ表示 ---
        try:
            min_scale = float(scales.min()) if scales.size > 0 else 1.0
            max_scale = float(scales.max()) if scales.size > 0 else 1.0
            logger.info(
                "[LEXICAL-SCALES] "
                f"use_lexical_prior={getattr(self.config, 'use_lexical_prior', False)}, "
                f"window_sec={window_sec:.2f}, "
                f"q_scale={q_scale:.2f}, p_scale={p_scale:.2f}, "
                f"min_scale={min_scale:.3f}, max_scale={max_scale:.3f}"
            )
        except Exception as e:
            logger.warning(f"[LEXICAL-SCALES] debug failed: {e}")

        return scales

    def _precompute_word_costs(self, word_info: List[Dict[str, Any]], grid_times: np.ndarray) -> np.ndarray:
        word_costs = np.ones_like(grid_times, dtype=np.float32)
        if not word_info:
            return word_costs

        for word in word_info:
            w_start = word.get("start")
            w_end = word.get("end")

            # MLX-Whisper fallback: use 'confidence' or 'avg_logprob' if 'prob' missing
            prob_raw = word.get("prob")
            if prob_raw is None:
                prob_raw = word.get("confidence", None)
            if prob_raw is None:
                prob_raw = word.get("avg_logprob", None)
            if prob_raw is None:
                prob_raw = 0.8  # safe fallback

            try:
                w_start = float(w_start)
                w_end = float(w_end)
                prob = float(prob_raw)
            except (TypeError, ValueError):
                continue

            if w_end <= w_start:
                continue

            start_idx = int(np.searchsorted(grid_times, w_start, side="left"))
            end_idx = int(np.searchsorted(grid_times, w_end, side="right"))
            clamped_prob = max(0.0, min(1.0, prob))
            word_costs[start_idx:end_idx] = 1.0 - clamped_prob

        return word_costs

    def _apply_word_dp_override(
        self,
        final_path: np.ndarray,
        word_info: Optional[List[Dict[str, Any]]],
        spk_probs: np.ndarray,
        grid_times: np.ndarray,
        beta_t: np.ndarray,
        lexical_scales: np.ndarray,
    ) -> np.ndarray:
        if final_path.size == 0:
            return final_path
        if not word_info:
            return final_path

        p = np.asarray(spk_probs, dtype=np.float32)
        if p.ndim != 2 or p.shape[0] != final_path.shape[0]:
            return final_path

        num_frames, num_speakers = p.shape
        if num_speakers <= 1:
            return final_path

        eps = 1e-8
        p = np.clip(p, eps, 1.0)
        p = p / (np.sum(p, axis=1, keepdims=True) + eps)

        beta_arr = np.asarray(beta_t, dtype=np.float32).reshape(-1)
        if beta_arr.shape[0] != num_frames:
            beta_arr = np.full((num_frames,), float(self.beta), dtype=np.float32)

        spans: List[Tuple[int, int, int, float, float]] = []  # (orig_word_idx, i0, i1, start_sec, end_sec)
        for wi, w in enumerate(word_info):
            s = w.get("start")
            e = w.get("end")
            try:
                s_f = float(s)
                e_f = float(e)
            except (TypeError, ValueError):
                continue
            if not (e_f > s_f):
                continue

            i0 = int(np.searchsorted(grid_times, s_f, side="left"))
            i1 = int(np.searchsorted(grid_times, e_f, side="right"))
            i0 = max(0, min(i0, num_frames - 1))
            i1 = max(i0 + 1, min(i1, num_frames))

            if (i1 - i0) < int(getattr(self.config, "word_dp_min_frames", 1)):
                continue

            spans.append((wi, i0, i1, s_f, e_f))

        if not spans:
            return final_path

        spans.sort(key=lambda x: x[1])
        W = len(spans)
        emissions = np.zeros((W, num_speakers), dtype=np.float32)
        # Use log-prob emissions (additive domain) for word-level Viterbi.
        # Take mean(log p) over frames inside the word span to avoid length bias.
        eps = 1e-6
        for j, (_wi, i0, i1, _s_f, _e_f) in enumerate(spans):
            p_seg = np.clip(p[i0:i1, :], eps, 1.0)
            logp_mean = np.mean(np.log(p_seg), axis=0)
            b_mean = float(np.mean(beta_arr[i0:i1]))
            emissions[j, :] = b_mean * logp_mean

        base_pen = float(self.delta_switch) * float(getattr(self.config, "word_dp_switch_penalty_scale", 1.0))
        use_lex = bool(getattr(self.config, "word_dp_use_lexical_scale", True))
        gap_sec = float(getattr(self.config, "word_dp_gap_sec", 0.0))
        gap_scale = float(getattr(self.config, "word_dp_gap_switch_scale", 1.0))
        penalties = np.full((W,), base_pen, dtype=np.float32)
        penalties[0] = 0.0
        gap_hits = 0

        for j in range(1, W):
            _wi, i0, _i1, s_f, _e_f = spans[j]
            pen = base_pen
            if use_lex and lexical_scales is not None and getattr(lexical_scales, "shape", None) == (num_frames,):
                idx = max(0, min(int(i0), num_frames - 1))
                pen *= float(lexical_scales[idx])
            if gap_sec > 0.0 and gap_scale != 1.0:
                prev_e = spans[j - 1][4]
                gap = float(s_f - prev_e)
                if gap >= gap_sec:
                    pen *= gap_scale
                    gap_hits += 1
            penalties[j] = pen

        word_path = self._word_dp_viterbi(emissions, penalties)

        out = np.array(final_path, copy=True)
        for j, (_wi, i0, i1, _s_f, _e_f) in enumerate(spans):
            out[i0:i1] = int(word_path[j])

        try:
            diff_frames = int(np.sum(out != final_path))
            switches = int(np.sum(word_path[1:] != word_path[:-1])) if W > 1 else 0
            logger.info(
                "[WORD-DP] applied: words=%d switches=%d diff_frames=%d / %d (pen=%.4f, use_lex=%s, gap_hits=%d)",
                W,
                switches,
                diff_frames,
                int(num_frames),
                float(base_pen),
                str(use_lex),
                int(gap_hits),
            )
        except Exception:
            pass

        return out

    def _word_dp_viterbi(self, emissions: np.ndarray, penalties: np.ndarray) -> np.ndarray:
        W, K = emissions.shape
        if W == 0:
            return np.zeros((0,), dtype=np.int32)

        dp = np.full((W, K), -1.0e30, dtype=np.float32)
        bp = np.zeros((W, K), dtype=np.int32)
        dp[0, :] = emissions[0, :]

        for w in range(1, W):
            pen = float(penalties[w])
            prev = dp[w - 1, :]
            for k in range(K):
                cand = prev.copy()
                if pen > 0.0:
                    for pk in range(K):
                        if pk != k:
                            cand[pk] -= pen
                best_pk = int(np.argmax(cand))
                dp[w, k] = emissions[w, k] + float(cand[best_pk])
                bp[w, k] = best_pk

        path = np.zeros((W,), dtype=np.int32)
        path[-1] = int(np.argmax(dp[-1, :]))
        for w in range(W - 2, -1, -1):
            path[w] = bp[w + 1, path[w + 1]]
        return path

    def _maybe_export_boundary_snap(
        self,
        *,
        grid_times: np.ndarray,
        spk_probs: np.ndarray,
        spk_emit: np.ndarray,
        vad_probs: np.ndarray,
        final_path: np.ndarray,
        beta_t: Optional[np.ndarray],
        lexical_scales: Optional[np.ndarray],
        cp_t: Optional[np.ndarray],
        word_info: List[Dict[str, Any]],
        frame_emission_mode: str,
    ) -> None:
        # Enable via env only (simple & explicit)
        enabled = os.environ.get("AHE_BOUNDARY_SNAP", "0").strip().lower() in ("1", "true", "yes")
        if not enabled:
            return

        out_dir = os.environ.get("AHE_BOUNDARY_SNAP_DIR", "").strip()
        if not out_dir:
            # Default to the same output directory as config if env var is empty
            out_dir = "AHE_Whisper_output"
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            logger.warning("[BOUNDARY-SNAP] cannot create out_dir=%s: %s", out_dir, exc)
            return

        T = int(final_path.size)
        if T <= 2 or spk_probs.ndim != 2 or spk_probs.shape[0] != T:
            logger.warning("[BOUNDARY-SNAP] invalid shapes: T=%d spk_probs=%s", T, getattr(spk_probs, "shape", None))
            return

        # --- derived signals for "real" transition penalty inspection (debug only) ---
        K = int(spk_probs.shape[1])
        lnK = math.log(max(2, K))

        runlen_sec = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            if int(final_path[t]) == int(final_path[t - 1]):
                runlen_sec[t] = runlen_sec[t - 1] + float(grid_times[t] - grid_times[t - 1])

        ten_score_t = getattr(self, "_ten_score", None)
        ten_weight = float(self.ten_weight)

        # Tunables
        window_sec = float(os.environ.get("AHE_BOUNDARY_SNAP_WINDOW_SEC", "1.0"))
        max_events = int(os.environ.get("AHE_BOUNDARY_SNAP_MAX_EVENTS", "30"))
        mode = os.environ.get("AHE_BOUNDARY_SNAP_MODE", "switch+cp").strip().lower()
        cp_th = float(os.environ.get("AHE_BOUNDARY_SNAP_CP_TH", "0.90"))
        cp_topk = int(os.environ.get("AHE_BOUNDARY_SNAP_CP_TOPK", "10"))

        # Config-derived knobs (recorded into snap for debugging; exporter itself must not change decoding)
        switch_cp_k = float(getattr(self.config, "switch_cp_k", 0.0))
        switch_cp_floor_uncertain = float(getattr(self.config, "switch_cp_floor_uncertain", 0.0))
        switch_post_k = float(getattr(self.config, "switch_post_k", 0.0))
        switch_post_margin_th = float(getattr(self.config, "switch_post_margin_th", 0.5))
        switch_uncertain_penalty_mult = float(getattr(self.config, "switch_uncertain_penalty_mult", 1.0))
        switch_uncertain_entropy_norm_th = float(getattr(self.config, "switch_uncertain_entropy_norm_th", 1.0))
        switch_uncertain_margin_th = float(getattr(self.config, "switch_uncertain_margin_th", 0.0))
        speech_th_for_uncertain = float(
            max(
                float(self.non_speech_th),
                float(getattr(self.config, "silence_vad_th", self.non_speech_th)),
            )
        )
        switch_uncertain_use_speech_mask = bool(
            getattr(self.config, "switch_uncertain_use_speech_mask", False)
        )

        # dt/win
        try:
            dt = float(np.median(np.diff(grid_times.astype(np.float64))))
        except Exception:
            dt = float(grid_times[1] - grid_times[0]) if grid_times.size >= 2 else 0.1
        if dt <= 0.0:
            dt = 0.1
        win = int(round(max(0.0, window_sec) / dt))

        # events: switches and/or cp peaks
        events: List[Tuple[int, str]] = []
        if "switch" in mode:
            sw = np.where(final_path[1:] != final_path[:-1])[0]
            for j in sw:
                events.append((int(j) + 1, "switch"))

        if cp_t is not None and ("cp" in mode or "peak" in mode):
            cp = np.asarray(cp_t, dtype=np.float32).reshape(-1)
            if cp.size == T:
                peaks: List[int] = []
                for i in range(1, T - 1):
                    if cp[i] >= cp_th and cp[i] > cp[i - 1] and cp[i] >= cp[i + 1]:
                        peaks.append(i)
                peaks = sorted(peaks, key=lambda i: float(cp[i]), reverse=True)[:max(0, cp_topk)]
                seen = {e[0] for e in events}
                for i in peaks:
                    if i not in seen:
                        events.append((int(i), "cp_peak"))

        if not events:
            logger.info("[BOUNDARY-SNAP] enabled=True but no events found (mode=%s)", mode)
            return

        events = sorted(events, key=lambda x: x[0])[:max(1, max_events)]

        # --------------------------
        # Near-boundary words export
        # --------------------------
        # Controls (keep it lightweight by default)
        # - window_sec: time window around event center to collect words
        # - max_words: cap count per event
        word_window_sec = float(os.environ.get("AHE_BOUNDARY_SNAP_WORD_WINDOW_SEC", "1.5"))
        max_words = int(os.environ.get("AHE_BOUNDARY_SNAP_MAX_WORDS", "20"))

        words_sorted: List[Tuple[float, float, str]] = []
        if word_info:
            for w in word_info:
                s = w.get("start", None)
                e = w.get("end", None)
                if s is None and e is None:
                    continue
                # try common keys for surface form
                txt = (
                    w.get("text", None)
                    or w.get("word", None)
                    or w.get("token", None)
                    or w.get("surface", None)
                    or ""
                )
                try:
                    ss = float(s) if s is not None else float(e)
                    ee = float(e) if e is not None else float(s)
                except Exception:
                    continue
                if ee < ss:
                    ss, ee = ee, ss
                words_sorted.append((ss, ee, str(txt)))
            words_sorted.sort(key=lambda x: x[0])

        words_starts = [w[0] for w in words_sorted]  # for bisect

        # Precompute DP word costs so boundary_snap can report the "real" per-frame DP added score.
        word_costs = self._precompute_word_costs(word_info, grid_times)

        def _runlen_end(path, t_end: int) -> int:
            # run length (in frames) ending at t_end for the state at t_end
            s = int(path[t_end])
            r = 1
            tt = t_end - 1
            while tt >= 0 and int(path[tt]) == s:
                r += 1
                tt -= 1
            return r

        def _calc_dp_terms(
            t: int,
            cur_state: int,
            prev_state: int,
            vad_v,
            lex_scale_v,
            cp_scale_v: float,
            uncertain_scale_v: float,
            post_to_scale_v,
        ):
            vad_score = float(self.alpha) * float(vad_v) if vad_v is not None else 0.0
            spk_score = float(beta_t[t]) * float(spk_emit[t, cur_state])
            word_score = -float(self.gamma) * float(word_costs[t]) if word_costs is not None else 0.0

            sw_pen_applied = 0.0
            sw_pen_base = 0.0
            sw_pen_factor = 1.0
            sw_ten_scale = 1.0
            sw_runlen = 0

            if t > 0 and prev_state != cur_state:
                sw_runlen = _runlen_end(final_path, t - 1)
                excess = max(0, min(_SWITCH_LEN_CAP, sw_runlen - _SWITCH_LEN_THRESHOLD))
                sw_pen_factor = 1.0 + _SWITCH_LEN_SCALE * float(excess)

                if self._ten_score is not None:
                    sw_ten_scale = 1.0 + float(self.ten_weight) * float(self._ten_score[t])

                lex_v = float(lex_scale_v) if lex_scale_v is not None else 1.0
                to_v = float(post_to_scale_v) if post_to_scale_v is not None else 1.0

                sw_pen_base = (
                    float(self.delta_switch)
                    * float(sw_pen_factor)
                    * float(sw_ten_scale)
                    * float(lex_v)
                    * float(cp_scale_v)
                    * float(uncertain_scale_v)
                )
                sw_pen_applied = sw_pen_base * to_v

            dp_add_total = vad_score + spk_score + word_score - sw_pen_applied
            return (
                dp_add_total,
                vad_score,
                spk_score,
                word_score,
                sw_pen_applied,
                sw_pen_base,
                sw_pen_factor,
                sw_ten_scale,
                sw_runlen,
            )

        def _collect_words(te: float) -> List[Dict[str, Any]]:
            if not words_sorted or max_words <= 0 or word_window_sec <= 0.0:
                return []
            lo_t = te - word_window_sec
            hi_t = te + word_window_sec
            # start from first word with start >= lo_t, but step back a bit to include overlaps
            j = bisect.bisect_left(words_starts, lo_t)
            j = max(0, j - 3)
            out: List[Dict[str, Any]] = []
            while j < len(words_sorted):
                ws, we, wt = words_sorted[j]
                if ws > hi_t:
                    break
                # overlap condition: [ws,we] intersects [lo_t,hi_t]
                if we >= lo_t and ws <= hi_t:
                    out.append(
                        {
                            "start": float(ws),
                            "end": float(we),
                            "text": wt,
                        }
                    )
                    if len(out) >= max_words:
                        break
                j += 1
            return out

        run_tag = os.environ.get("AHE_RUN_ID", "").strip() or time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"boundary_snap_{run_tag}_{os.getpid()}.jsonl")

        def _entropy(p: np.ndarray) -> float:
            x = np.clip(p.astype(np.float64), 1e-12, 1.0)
            return float(-np.sum(x * np.log(x)))

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                for idx, etype in events:
                    idx = int(idx)
                    # event-level DP terms (captured from dt=0 frame in window)
                    dp_evt_total = None
                    dp_evt_vad = None
                    dp_evt_spk = None
                    dp_evt_word = None
                    dp_evt_sw_pen = None
                    dp_evt_sw_pen_base = None
                    dp_evt_sw_pen_factor = None
                    dp_evt_sw_ten_scale = None
                    dp_evt_sw_runlen = None
                    dp_evt_cp_scale = None
                    dp_evt_uncertain_scale = None
                    dp_evt_lex_scale = None
                    dp_evt_post_to_scale = None

                    lo = max(0, idx - win)
                    hi = min(T, idx + win + 1)
                    frames: List[Dict[str, Any]] = []
                    for t in range(lo, hi):
                        p = np.asarray(spk_probs[t], dtype=np.float32)
                        e = np.asarray(spk_emit[t], dtype=np.float32)

                        k1 = int(np.argmax(p))
                        tmp = p.copy()
                        tmp[k1] = -1.0
                        k2 = int(np.argmax(tmp)) if p.size >= 2 else k1
                        tmp[k2] = -1.0
                        k3 = int(np.argmax(tmp)) if p.size >= 3 else k2

                        v1 = float(p[k1])
                        v2 = float(p[k2])
                        v3 = float(p[k3]) if p.size >= 3 else None

                        p_margin = float(v1 - v2)
                        p_entropy = _entropy(p)
                        denom = float(np.log(float(max(2, int(p.size)))))
                        p_entropy_norm = float(p_entropy / denom) if denom > 0.0 else 0.0

                        vad_v = (
                            float(vad_probs[t])
                            if vad_probs is not None and vad_probs.shape == (T,)
                            else None
                        )
                        lex_scale_v = (
                            float(lexical_scales[t])
                            if lexical_scales is not None and getattr(lexical_scales, "shape", None) == (T,)
                            else None
                        )
                        cp_v = (
                            float(cp_t[t])
                            if cp_t is not None and getattr(cp_t, "shape", None) == (T,)
                            else None
                        )

                        cp_scale = None
                        cp_floor = None
                        if cp_v is not None:
                            raw = 1.0 - switch_cp_k * float(cp_v)
                            raw = float(max(0.0, raw))
                            cp_scale = raw
                            if switch_cp_floor_uncertain > 0.0:
                                mar01 = float(np.clip(p_margin, 0.0, 1.0))
                                floor = float(switch_cp_floor_uncertain * ((1.0 - mar01) ** 2))
                                cp_floor = floor
                                cp_scale = float(max(cp_scale, floor))

                        uncertain_scale = 1.0
                        if (
                            switch_uncertain_penalty_mult > 1.0
                            and (not switch_uncertain_use_speech_mask or (vad_v is not None and float(vad_v) >= speech_th_for_uncertain))
                        ):
                            if (
                                float(p_entropy_norm) >= float(switch_uncertain_entropy_norm_th)
                                and float(p_margin) <= float(switch_uncertain_margin_th)
                            ):
                                uncertain_scale = float(switch_uncertain_penalty_mult)
                        uncertain_hit = bool(uncertain_scale > 1.0)

                        # Re-calculation based on DP terms (requested diff logic)
                        post_to_scale_path = 1.0
                        post_to_scale_argmax = 1.0
                        post_to_scale_conf = 1.0

                        if bool(getattr(self.config, "switch_post_gate", False)):
                            k_path = int(final_path[t])
                            k_arg = int(p_argmax)
                            # Note: using the formula from user's diff. 
                            # If switch_post_gate is False, this block is skipped.
                            post_to_scale_path = (
                                (1.0 + post_k * max(0.0, (post_margin_th - p_margin)))
                                if k_path == k_arg
                                else 1.0
                            )
                            post_to_scale_argmax = (
                                (1.0 + post_k * max(0.0, (post_margin_th - p_margin)))
                                if k_arg == k_arg
                                else 1.0
                            )
                            post_to_scale_conf = post_to_scale_path

                        prev_state = int(final_path[t - 1]) if t > 0 else int(final_path[t])
                        cur_state = int(final_path[t])
                        (
                            dp_add_total,
                            dp_add_vad,
                            dp_add_spk,
                            dp_add_word,
                            dp_sw_pen,
                            dp_sw_pen_base,
                            dp_sw_pen_factor,
                            dp_sw_ten_scale,
                            dp_sw_runlen_frames,
                        ) = _calc_dp_terms(
                            t,
                            cur_state,
                            prev_state,
                            vad_v,
                            lex_scale_v,
                            cp_scale,
                            uncertain_scale,
                            post_to_scale_path,
                        )

                        if t == idx:
                            dp_evt_total = dp_add_total
                            dp_evt_vad = dp_add_vad
                            dp_evt_spk = dp_add_spk
                            dp_evt_word = dp_add_word
                            dp_evt_sw_pen = dp_sw_pen
                            dp_evt_sw_pen_base = dp_sw_pen_base
                            dp_evt_sw_pen_factor = dp_sw_pen_factor
                            dp_evt_sw_ten_scale = dp_sw_ten_scale
                            dp_evt_sw_runlen = dp_sw_runlen_frames
                            dp_evt_cp_scale = cp_scale
                            dp_evt_uncertain_scale = uncertain_scale
                            dp_evt_lex_scale = lex_scale_v
                            dp_evt_post_to_scale = post_to_scale_path

                        fr = {
                            "t": float(grid_times[t]),
                            "i": int(t),
                            "path_spk": int(final_path[t]),
                            "vad": vad_v,
                            "p_top1": v1,
                            "p_top2": v2,
                            "p_top3": v3,
                            "p_top1_spk": k1,
                            "p_top2_spk": k2,
                            "p_top3_spk": int(k3) if p.size >= 3 else None,
                            "p_margin": p_margin,
                            "p_entropy": p_entropy,
                            "p_entropy_norm": p_entropy_norm,
                            "p_argmax": k1,
                            "emit_argmax": int(np.argmax(e)) if e.size > 0 else 0,
                            "beta": float(beta_t[t]) if beta_t is not None and getattr(beta_t, "shape", None) == (T,) else None,
                            "lex_scale": lex_scale_v,
                            "cp": cp_v,
                            "cp_scale": cp_scale,
                            "cp_floor": cp_floor,
                            "uncertain_scale": float(uncertain_scale),
                            "uncertain_hit": uncertain_hit,
                            "post_to_scale_argmax": float(post_to_scale_argmax) if post_to_scale_argmax is not None else None,
                            "post_to_scale_path": float(post_to_scale_path) if post_to_scale_path is not None else None,
                            "post_to_scale_conf": float(post_to_scale_conf) if post_to_scale_conf is not None else None,
                            "dp_cost_total": float(dp_add_total),
                            "dp_cost_terms": {
                                "vad": float(dp_add_vad),
                                "spk": float(dp_add_spk),
                                "word": float(dp_add_word),
                                "sw_pen": float(dp_sw_pen),
                                "sw_pen_base": float(dp_sw_pen_base),
                                "sw_pen_factor": float(dp_sw_pen_factor),
                                "sw_ten_scale": float(dp_sw_ten_scale),
                                "cp_scale": float(cp_scale) if cp_scale is not None else None,
                                "uncertain_scale": float(uncertain_scale),
                                "lex_scale": float(lex_scale_v) if lex_scale_v is not None else None,
                                "post_to_scale_path": float(post_to_scale_path) if post_to_scale_path is not None else None,
                                "runlen": int(dp_sw_runlen_frames),
                            },
                        }
                        frames.append(fr)

                    words_meta = _collect_words(float(grid_times[idx])) if 0 <= idx < grid_times.size else []
                    words = [w.get("text", "") for w in words_meta]
                    words_text = "".join(words)

                    path_old = int(final_path[idx - 1]) if etype == "switch" and idx > 0 else None
                    path_new = int(final_path[idx]) if etype == "switch" and 0 <= idx < T else None

                    rec = {
                        "v": 2,
                        "event_type": etype,
                        "frame": int(idx),
                        "time": float(grid_times[idx]) if 0 <= idx < grid_times.size else None,
                        "window_sec": float(window_sec),
                        "window_frames": int(win),
                        "lo": int(lo),
                        "hi": int(hi),
                        "frame_emission_mode": str(frame_emission_mode),
                        "num_speakers": int(spk_probs.shape[1]),
                        "path_old": path_old,
                        "path_new": path_new,
                        "dp_cost_total": float(dp_evt_total) if dp_evt_total is not None else None,
                        "dp_cost_terms": {
                            "vad": float(dp_evt_vad),
                            "spk": float(dp_evt_spk),
                            "word": float(dp_evt_word),
                            "sw_pen": float(dp_evt_sw_pen),
                            "sw_pen_base": float(dp_evt_sw_pen_base),
                            "sw_pen_factor": float(dp_evt_sw_pen_factor),
                            "sw_ten_scale": float(dp_evt_sw_ten_scale),
                            "cp_scale": float(dp_evt_cp_scale) if dp_evt_cp_scale is not None else None,
                            "uncertain_scale": float(dp_evt_uncertain_scale),
                            "lex_scale": float(dp_evt_lex_scale) if dp_evt_lex_scale is not None else None,
                            "post_to_scale_path": float(dp_evt_post_to_scale) if dp_evt_post_to_scale is not None else None,
                            "runlen": int(dp_evt_sw_runlen),
                        } if dp_evt_total is not None else None,
                        "switch_knobs": {
                            "switch_cp_k": float(switch_cp_k),
                            "switch_cp_floor_uncertain": float(switch_cp_floor_uncertain),
                            "switch_post_k": float(switch_post_k),
                            "switch_post_margin_th": float(switch_post_margin_th),
                            "switch_uncertain_penalty_mult": float(switch_uncertain_penalty_mult),
                            "switch_uncertain_entropy_norm_th": float(switch_uncertain_entropy_norm_th),
                            "switch_uncertain_margin_th": float(switch_uncertain_margin_th),
                            "switch_uncertain_use_speech_mask": bool(switch_uncertain_use_speech_mask),
                            "speech_th_for_uncertain": float(speech_th_for_uncertain),
                        },
                        "word_window_sec": float(word_window_sec),
                        "max_words": int(max_words),
                        # near-boundary words for debugging "巻き込み"
                        "words": words,
                        "words_text": words_text,
                        "words_meta": words_meta,
                        "frames": frames,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("[BOUNDARY-SNAP] export failed: %s", exc)
            return

        logger.info(
            "[BOUNDARY-SNAP] enabled=True mode=%s events=%d win=%.2fs dt=%.3f out=%s",
            mode,
            len(events),
            window_sec,
            dt,
            out_path,
        )
