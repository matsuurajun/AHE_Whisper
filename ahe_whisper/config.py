# -*- coding: utf-8 -*-
from dataclasses import dataclass, field, asdict, replace, fields
from typing import List, Optional, Dict

@dataclass
class TranscriptionConfig:
    model_name: str = "mlx-community/whisper-large-v3-turbo"
    language: Optional[str] = "ja"
    no_speech_threshold: float = 0.4
    logprob_threshold: float = -1.0

@dataclass
class EmbeddingConfig:
    # 使用する埋め込みバックエンド
    # - "campplus"   : 3D-Speaker Cam++ (既存既定)
    # - "titanet"    : sherpa-onnx nemo_en_titanet_large
    # - "speakernet" : sherpa-onnx nemo_en_speakerverification_speakernet
    # （将来的に "ecapa512" / "resnet293" 等も拡張可能）
    backend: str = "titanet"

    embedding_dim: int = 192
    batch_cap: int = 16
    bucket_step: int = 80
    prefer_coreml_ep: bool = False
    intra_threads: Optional[int] = None
    inter_threads: int = 1
    embedding_win_sec: float = 1.5
    embedding_hop_sec: float = 0.75
    # VAD の平均スコアがこの値以上のチャンクだけを
    # 「話者クラスタリングに使う embedding」として残す
    min_chunk_speech_prob: float = 0.4
    # 時間方向スムージング用カーネル
    # 0 → スムージング無効
    # 3,5 など奇数 → 前後チャンクを含む移動平均
    smooth_embeddings_kernel: int = 3

@dataclass
class DiarizationConfig:
    min_speakers: int = 2
    max_speakers: int = 2
    # GUI: Auto/2/3/4. Auto is None.
    target_speakers: Optional[int] = 2
    # VBx-Lite resegmentation
    enable_vbx_resegmentation: bool = True
    vbx_p_stay_speech: float = 0.998
    vbx_p_stay_silence: float = 0.9995
    vbx_speech_th: float = 0.35
    vbx_out_hard_mix: float = 0.20
    vbx_min_run_sec: float = 1.0
    engine: str = "soft-em-adc"
    vad_th_start: float = 0.5
    vad_th_end: float = 0.3
    em_tau_schedule: List[float] = field(default_factory=lambda: [10.0, 5.0, 3.0])
    min_speaker_duration_sec: float = 1.5
    # A-B-A 型の真ん中の「話者島」を吸収する最大長（秒）
    max_island_sec: float = 3.0
    island_conf_margin: float = 0.05
    min_fallback_duration_sec: float = 1.0
    min_speech_sec: float = 0.3
    max_merge_gap_sec: float = 1.5
    # --- NEW: cluster post-processing knobs ---
    # クラスタが担当する embedding 比率がこれ未満なら「小さすぎる」とみなして候補から外す
    min_cluster_mass: float = 0.05
    # セントロイド同士のコサイン類似度がこれ以上なら「同一話者」とみなしてマージ
    centroid_merge_sim: float = 0.90

    # --- NEW: 30秒窓ベースの「高スイッチ密度ゾーン潰し」 ---
    # True にすると、post_diar 側の smooth_high_switch_density_zones を有効化する
    enable_high_switch_smoothing: bool = False
    # スライディング窓の長さ（秒）
    high_switch_window_sec: float = 30.0
    # 窓をどれくらいずつずらすか（秒）
    high_switch_step_sec: float = 10.0
    # 実効的にこの長さ未満しか発話が無い窓はスキップ
    high_switch_min_window_sec: float = 25.0
    # 「この割合以上を1人がしゃべっていたら、実質モノローグ」とみなす閾値
    # まずはやや緩めに 0.65 くらいから試す
    high_switch_majority_ratio: float = 0.8
    # 1 秒あたりのスイッチ回数（switches / total_dur）がこの値以上なら
    # 「スイッチが多すぎる」＝異常とみなす
    high_switch_min_switch_density: float = 0.20

    def __post_init__(self) -> None:
        if not (0.0 <= self.vad_th_start <= 1.0):
            raise ValueError(f"vad_th_start must be in [0, 1], got {self.vad_th_start}")
        if not (0.0 <= self.vad_th_end <= 1.0):
            raise ValueError(f"vad_th_end must be in [0, 1], got {self.vad_th_end}")
        if self.vad_th_start < self.vad_th_end:
            raise ValueError(f"vad_th_start ({self.vad_th_start}) must be >= vad_th_end ({self.vad_th_end}).")
        if self.min_speakers > self.max_speakers:
            raise ValueError(f"min_speakers ({self.min_speakers}) cannot be greater than max_speakers ({self.max_speakers}).")
        if self.target_speakers is not None and self.target_speakers < 1:
            raise ValueError(f"target_speakers must be >= 1 or None, got {self.target_speakers}")
        for name in ("vbx_p_stay_speech", "vbx_p_stay_silence", "vbx_speech_th", "vbx_out_hard_mix"):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.vbx_min_run_sec < 0.0:
            raise ValueError(f"vbx_min_run_sec must be >= 0, got {self.vbx_min_run_sec}")
        if not (0.0 < self.min_cluster_mass <= 1.0):
            raise ValueError(f"min_cluster_mass must be in (0, 1], got {self.min_cluster_mass}")
        if not (0.0 < self.centroid_merge_sim <= 1.0):
            raise ValueError(f"centroid_merge_sim must be in (0, 1], got {self.centroid_merge_sim}")

        if not (0.0 <= self.high_switch_majority_ratio <= 1.0):
            raise ValueError(
                f"high_switch_majority_ratio must be in [0, 1], got {self.high_switch_majority_ratio}"
            )
        if self.high_switch_window_sec <= 0.0:
            raise ValueError(f"high_switch_window_sec must be > 0, got {self.high_switch_window_sec}")
        if self.high_switch_step_sec <= 0.0:
            raise ValueError(f"high_switch_step_sec must be > 0, got {self.high_switch_step_sec}")
        if self.high_switch_min_window_sec <= 0.0:
            raise ValueError(f"high_switch_min_window_sec must be > 0, got {self.high_switch_min_window_sec}")

@dataclass
class VadConfig:
    window_size_samples: int = 512

@dataclass
class BoundaryRefineParams:
    max_candidates: int = 40
    merge_sec: float = 0.5
    merge_max_window_sec: float = 2.0
    window_sec: float = 1.0
    candidate_use_switch: bool = True
    candidate_use_lexical: bool = True
    lexical_tokens: str = "?？。."
    short_win_sec: float = 0.5
    short_hop_sec: float = 0.25
    use_local_dp: bool = True
    local_dp_switch_scale: float = 0.3
    local_dp_min_frames: int = 3
    multi_scale_enable: bool = False
    multi_scale_win_secs: List[float] = field(default_factory=lambda: [0.8, 0.4, 0.2])
    multi_scale_hop_secs: List[float] = field(default_factory=lambda: [0.4, 0.2, 0.1])
    multi_scale_vote_mode: str = "weighted"
    multi_scale_vote_margin_th: float = 0.10
    multi_scale_vote_weight_exp: float = 1.0
    multi_scale_vote_min: int = 2
    multi_scale_micro_avg_offsets_sec: List[float] = field(
        default_factory=lambda: [-0.1, 0.0, 0.1]
    )
    split_window_enable: bool = False
    split_window_center_pad_sec: float = 0.0

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {self.max_candidates}")
        if self.merge_sec < 0.0:
            raise ValueError(f"merge_sec must be >= 0, got {self.merge_sec}")
        if self.merge_max_window_sec <= 0.0:
            raise ValueError(f"merge_max_window_sec must be > 0, got {self.merge_max_window_sec}")
        if self.window_sec <= 0.0:
            raise ValueError(f"window_sec must be > 0, got {self.window_sec}")
        if self.short_win_sec <= 0.0:
            raise ValueError(f"short_win_sec must be > 0, got {self.short_win_sec}")
        if self.short_hop_sec <= 0.0:
            raise ValueError(f"short_hop_sec must be > 0, got {self.short_hop_sec}")
        if self.local_dp_switch_scale < 0.0:
            raise ValueError(f"local_dp_switch_scale must be >= 0, got {self.local_dp_switch_scale}")
        if self.local_dp_min_frames < 1:
            raise ValueError(f"local_dp_min_frames must be >= 1, got {self.local_dp_min_frames}")
        if not self.multi_scale_win_secs:
            raise ValueError("multi_scale_win_secs must not be empty")
        if not self.multi_scale_hop_secs:
            raise ValueError("multi_scale_hop_secs must not be empty")
        if len(self.multi_scale_win_secs) != len(self.multi_scale_hop_secs):
            raise ValueError(
                "multi_scale_win_secs and multi_scale_hop_secs must have the same length"
            )
        if any(sec <= 0.0 for sec in self.multi_scale_win_secs):
            raise ValueError("multi_scale_win_secs must be > 0")
        if any(sec <= 0.0 for sec in self.multi_scale_hop_secs):
            raise ValueError("multi_scale_hop_secs must be > 0")
        if self.multi_scale_vote_mode not in ("weighted", "approval"):
            raise ValueError(
                f"multi_scale_vote_mode must be 'weighted' or 'approval', got {self.multi_scale_vote_mode}"
            )
        if self.multi_scale_vote_margin_th < 0.0:
            raise ValueError(
                f"multi_scale_vote_margin_th must be >= 0, got {self.multi_scale_vote_margin_th}"
            )
        if self.multi_scale_vote_weight_exp <= 0.0:
            raise ValueError(
                f"multi_scale_vote_weight_exp must be > 0, got {self.multi_scale_vote_weight_exp}"
            )
        if self.multi_scale_vote_min < 1:
            raise ValueError(
                f"multi_scale_vote_min must be >= 1, got {self.multi_scale_vote_min}"
            )
        if self.split_window_center_pad_sec < 0.0:
            raise ValueError(
                f"split_window_center_pad_sec must be >= 0, got {self.split_window_center_pad_sec}"
            )

_BOUNDARY_REFINE_PRESETS: Dict[str, BoundaryRefineParams] = {
    "default": BoundaryRefineParams(),
    "conservative": BoundaryRefineParams(
        max_candidates=20,
        merge_sec=0.4,
        merge_max_window_sec=1.5,
        window_sec=0.8,
        short_win_sec=0.4,
        short_hop_sec=0.2,
        local_dp_switch_scale=0.5,
    ),
    "aggressive": BoundaryRefineParams(
        max_candidates=60,
        merge_sec=0.6,
        merge_max_window_sec=2.5,
        window_sec=1.2,
        short_win_sec=0.6,
        short_hop_sec=0.3,
        local_dp_switch_scale=0.2,
    ),
    "split_window": BoundaryRefineParams(
        split_window_enable=True,
    ),
    "three_step": BoundaryRefineParams(
        multi_scale_enable=True,
        multi_scale_vote_margin_th=0.2,
        multi_scale_vote_min=3,
    ),
    "three_step_split": BoundaryRefineParams(
        multi_scale_enable=True,
        split_window_enable=True,
    ),
}

@dataclass
class AlignerConfig:
    # VAD / spk_probs / word_cost の重み
    alpha: float = 0.6    # VAD 重み（そのまま維持）
    beta: float = 0.8    # 話者確率 spk_probs の重み
    gamma: float = 0.5    # word_cost の重み

    delta_switch: float = 1.5
    non_speech_th: float = 0.02
    grid_hz: int = 50

    use_smooth_aligner: bool = False
    smooth_alpha: float = 0.55
    smooth_gamma: float = 1.4

    # --- NEW: ラン長依存スイッチペナルティ ---
    # 同じ話者が長く続いているほど、その話者から別話者へのスイッチを重くする係数。
    # 0.0 のままなら従来どおり「ラン長に依存しない一定ペナルティ」になる。
    runlen_switch_gamma: float = 0.0
    # ラン長として考慮する最大秒数（これ以上は同じ扱い。長大ターンでも暴れないようにサチュレート）
    runlen_max_sec: float = 20.0
    # --- NEW: 短時間での再スイッチ抑制（振動対策） ---
    # 直前のスイッチからこの秒数以内の再スイッチを重くする。
    switch_hysteresis_sec: float = 6.0
    # 再スイッチ抑制の倍率（1.0なら無効）。
    switch_hysteresis_mult: float = 5.0

    # --- NEW: TEN VAD 由来スコアの重み ---
    # DP 内のスイッチペナルティを、
    #   scale = 1 + ten_weight * ten_score[t]
    # という時間依存スケールで補正するための係数。
    # 0.0 のままなら TEN 特徴量は一切使わない（従来どおり）。
    ten_weight: float = 0.0

    # --- NEW: Lexical-based switch prior ---
    # True のとき、Whisper の単語末尾（? / 。 など）をもとに
    # スイッチペナルティを時間依存でスケールする。
    use_lexical_prior: bool = True
    # 句読点の終端から、この秒数ぶんの範囲でスイッチペナルティを緩和する。
    lexical_window_sec: float = 0.3
    # 質問文末（? / ？）直後でのスイッチペナルティ倍率（0.0〜1.0、1.0=無効）。
    lexical_question_switch_scale: float = 1.0
    # 平叙文末（。 / .）直後でのスイッチペナルティ倍率（0.0〜1.0、1.0=無効）。
    lexical_period_switch_scale: float = 1.0

    # --- NEW: embedding change-point based switch modulation ---
    # change-point が強いほどスイッチ遷移を通しやすくするための係数。
    # 0.0 のままなら無効（従来どおり）。
    # B（cp有効）: switch_cp_k=0.5, switch_cp_smooth_win=5
    switch_cp_k: float = 1.0
    switch_cp_smooth_win: int = 25
    # --- NEW: cp-scale floor under posterior uncertainty ---
    # cp が強い (cp_t が大きい) ほど switch_penalty を下げる設計だが、posterior が曖昧 (p_margin が小さい) 場合に
    # cp_scale が 0 になりすぎると「境界直後の短い反転 (A->B->A)」が起きやすくなる。
    # switch_cp_floor_uncertain > 0 のとき、cp_scale の下限を
    #   floor = switch_cp_floor_uncertain * (1 - clamp(p_margin,0,1))^2
    # で与える（=曖昧なほど下限が上がる）。0.0 なら従来どおり。
    switch_cp_floor_uncertain: float = 1.0

    # --- NEW: posterior-margin gate for switch penalty (anti-dictatorship switch) ---
    # そのフレームの posterior argmax 話者へ「侵入」するとき、
    # 2位とのマージンが switch_post_margin_th 以上であれば、スイッチ罰則を
    #   scale = max(0, 1 - switch_post_k * (margin - th) / (1 - th))
    # のように緩和する。
    switch_post_k: float = 2.0
    switch_post_margin_th: float = 0.5

    # --- NEW: uncertainty-aware switch penalty boost ---
    # posterior が「ほぼ最大エントロピー」かつマージンが小さい（=極めて曖昧）フレームでは、
    # スイッチを安易に通すと短い誤スイッチ（low-cp switch）が残りやすい。
    # そのため、そのようなフレームではスイッチ罰則を倍率で増やして「その場でのスイッチ」を抑制する。
    # - entropy_norm = entropy / log(K) in [0,1]（K=話者数）
    # - margin は top1-top2 in [0,1]
    # 互換性: mult=1.0 なら従来と同じ。
    switch_uncertain_penalty_mult: float = 1.5
    switch_uncertain_entropy_norm_th: float = 0.99
    switch_uncertain_margin_th: float = 0.10

    # Uncertainty boost を VAD(speech_mask) でゲートするか。
    # False: VAD が境界で落ちるケースでも uncertain 判定を有効にする（境界スイッチ遅延の診断/改善向け）。
    switch_uncertain_use_speech_mask: bool = False

    # --- NEW: speaker-margin based switch suppression ---
    # spk_margin が小さい（話者が判別できない）フレームではスイッチ罰則を強化する。
    # switch_spk_margin_mult=1.0 なら無効。
    switch_spk_margin_th: float = 0.10
    switch_spk_margin_mult: float = 3.0

    # --- NEW: Frame-level DP emission mode ---
    # "prob" (既定) or "log" / "logp". 
    # "log" にするとフレームDPの speaker 項を対数ドメインで扱い、スイッチ罰則（delta_switch）との整合を高める。
    frame_emission_mode: str = "prob"

    # --- NEW: Word-level Viterbi DP (単語DP) ---
    # ASR の単語区間で spk_probs/beta_t を集約し、単語列上で Viterbi を回して
    # フレームパスを局所的に上書きする（境界での "巻き込み" 対策の最小実装）。
    enable_word_dp: bool = True
    # 単語DPで使うスイッチ罰則の倍率（= delta_switch * word_dp_switch_penalty_scale）。
    word_dp_switch_penalty_scale: float = 0.3
    # 句読点由来の lexical_scales を単語DPの遷移罰則へ反映する
    word_dp_use_lexical_scale: bool = True
    # 単語区間が短すぎる場合の上書きを抑止するための最小フレーム数
    word_dp_min_frames: int = 1
    # 単語間の無音ギャップがこの秒数以上なら、スイッチ罰則を緩める
    word_dp_gap_sec: float = 0.25
    # ギャップ緩和時にかける倍率（1.0 未満でスイッチしやすく）
    word_dp_gap_switch_scale: float = 0.7

    # --- beta_t: time-varying speaker posterior weight (anti "embedding dictatorship") ---
    # 既定値は無効（互換性維持）。有効化は use_beta_t=True。
    use_beta_t: bool = False
    beta_t_strength: float = 0.60      # dominance に応じた down-weight 強度 (0..1)
    beta_t_min_ratio: float = 0.25     # beta_t / beta の下限
    beta_t_max_ratio: float = 1.00     # beta_t / beta の上限
    beta_t_speech_th: float = 0.35     # edge 検出に使う VAD 閾値
    beta_t_edge_window_sec: float = 0.50  # speech onset/offset の影響範囲
    beta_t_edge_ratio: float = 0.85    # edge 近傍での追加倍率 (0..1)

    # --- NEW: Titanet posterior calibration (peakiness control) ---
    # posterior_temperature > 1.0 で分布をフラット化（softmax 温度スケーリング相当）。
    # 入力 spk_probs は確率 (行和=1) 前提で、p^(1/T) により温度を適用する。
    posterior_temperature: float = 2.5
    # 一様分布との混合率 λ（0.0=無効）。p'=(1-λ)p+λ/K。
    posterior_uniform_mix: float = 0.08
    # 話者 posterior の“音響証拠”の強さを調整する係数（p^w で近似、1.0=無効）。
    # w < 1.0 でフラット化、w > 1.0 でピーク強調。
    acoustic_weight: float = 1.0

    # --- NEW: Silence-based run-length reset ---
    # True のとき、無音が一定時間以上続いたあとでラン長をリセットして、
    # 次の発話頭でのスイッチをしやすくする。
    use_silence_runlen_reset: bool = True
    # 無音とみなす VAD 閾値（vad_probs < silence_vad_th）。
    silence_vad_th: float = 0.35
    # 無音がこの秒数以上続いたときにラン長をリセットする。
    # 0.0 のままならリセットは行わない。
    silence_reset_min_sec: float = 0.20

    # --- NEW: posterior ベースの境界巻き戻し ---
    # True のとき、DP 最終パスに対して posteriors を用いたローカル realign を行う。
    use_posterior_refine: bool = True
    # 巻き戻し最大長（秒）
    posterior_max_rollback_sec: float = 1.5
    # 「新話者 posterior - 旧話者 posterior」がこの margin を超えたら巻き戻し候補とみなす
    posterior_margin: float = 0.015
    # 新話者側ラン長がこの秒数未満のときは巻き戻しを行わない（ノイズ拡大防止）
    posterior_min_run_sec: float = 0.5

    # --- NEW: boundary refinement (short-window + local DP) ---
    boundary_refine_enable: bool = True
    boundary_refine_preset: str = "three_step"
    boundary_refine_backend_required: Optional[str] = None
    boundary_refine_params: BoundaryRefineParams = field(
        default_factory=lambda: BoundaryRefineParams(
            max_candidates=30,
            window_sec=0.8,
            short_win_sec=0.4,
            short_hop_sec=0.2,
            local_dp_switch_scale=0.6,
        )
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.non_speech_th <= 1.0):
            raise ValueError(f"non_speech_th must be in [0, 1], got {self.non_speech_th}")
        if self.grid_hz < 1:
            raise ValueError(f"grid_hz must be >= 1, got {self.grid_hz}")
        if self.runlen_switch_gamma < 0.0:
            raise ValueError(f"runlen_switch_gamma must be >= 0, got {self.runlen_switch_gamma}")
        if self.runlen_max_sec <= 0.0:
            raise ValueError(f"runlen_max_sec must be > 0, got {self.runlen_max_sec}")
        if self.switch_hysteresis_sec < 0.0:
            raise ValueError(f"switch_hysteresis_sec must be >= 0, got {self.switch_hysteresis_sec}")
        if self.switch_hysteresis_mult < 1.0:
            raise ValueError(
                f"switch_hysteresis_mult must be >= 1.0, got {self.switch_hysteresis_mult}"
            )
        if self.ten_weight < 0.0:
            raise ValueError(f"ten_weight must be >= 0, got {self.ten_weight}")
        if self.lexical_window_sec < 0.0:
            raise ValueError(f"lexical_window_sec must be >= 0, got {self.lexical_window_sec}")
        if not (0.0 < self.lexical_question_switch_scale <= 1.0):
            raise ValueError(
                f"lexical_question_switch_scale must be in (0, 1], "
                f"got {self.lexical_question_switch_scale}"
            )
        if not (0.0 < self.lexical_period_switch_scale <= 1.0):
            raise ValueError(
                f"lexical_period_switch_scale must be in (0, 1], "
                f"got {self.lexical_period_switch_scale}"
            )

        if self.switch_uncertain_penalty_mult < 1.0:
            raise ValueError(
                f"switch_uncertain_penalty_mult must be >= 1.0, got {self.switch_uncertain_penalty_mult}"
            )
        if not (0.0 <= self.switch_uncertain_entropy_norm_th <= 1.0):
            raise ValueError(
                f"switch_uncertain_entropy_norm_th must be in [0, 1], got {self.switch_uncertain_entropy_norm_th}"
            )
        if not (0.0 <= self.switch_uncertain_margin_th <= 1.0):
            raise ValueError(
                f"switch_uncertain_margin_th must be in [0, 1], got {self.switch_uncertain_margin_th}"
            )

        if not isinstance(self.switch_uncertain_use_speech_mask, bool):
            raise TypeError(
                f"switch_uncertain_use_speech_mask must be bool, got {type(self.switch_uncertain_use_speech_mask).__name__}"
            )
        if self.switch_spk_margin_th < 0.0:
            raise ValueError(
                f"switch_spk_margin_th must be >= 0, got {self.switch_spk_margin_th}"
            )
        if self.switch_spk_margin_mult < 1.0:
            raise ValueError(
                f"switch_spk_margin_mult must be >= 1.0, got {self.switch_spk_margin_mult}"
            )
        if self.switch_cp_k < 0.0:
            raise ValueError(f"switch_cp_k must be >= 0, got {self.switch_cp_k}")
        if self.switch_cp_smooth_win < 0:
            raise ValueError(f"switch_cp_smooth_win must be >= 0, got {self.switch_cp_smooth_win}")
        if not (0.0 <= self.switch_cp_floor_uncertain <= 1.0):
            raise ValueError(
                f"switch_cp_floor_uncertain must be in [0, 1], got {self.switch_cp_floor_uncertain}"
            )
        if self.switch_post_k < 0.0:
            raise ValueError(f"switch_post_k must be >= 0, got {self.switch_post_k}")
        if not (0.0 <= self.switch_post_margin_th <= 1.0):
            raise ValueError(f"switch_post_margin_th must be in [0, 1], got {self.switch_post_margin_th}")
        if self.frame_emission_mode.lower() not in ("prob", "log", "logp", "logprob", "log-prob", "log_prob"):
            raise ValueError(f"frame_emission_mode must be 'prob' or 'log', got {self.frame_emission_mode}")
        if self.posterior_temperature <= 0.0:
            raise ValueError(
                f"posterior_temperature must be > 0, got {self.posterior_temperature}"
            )
        if not (0.0 <= self.posterior_uniform_mix < 1.0):
            raise ValueError(
                f"posterior_uniform_mix must be in [0, 1), got {self.posterior_uniform_mix}"
            )
        if self.acoustic_weight <= 0.0:
            raise ValueError(f"acoustic_weight must be > 0, got {self.acoustic_weight}")
        if not (0.0 <= self.silence_vad_th <= 1.0):
            raise ValueError(f"silence_vad_th must be in [0, 1], got {self.silence_vad_th}")
        if self.silence_reset_min_sec < 0.0:
            raise ValueError(f"silence_reset_min_sec must be >= 0, got {self.silence_reset_min_sec}")
        if self.posterior_max_rollback_sec < 0.0:
            raise ValueError(
                f"posterior_max_rollback_sec must be >= 0, got {self.posterior_max_rollback_sec}"
            )
        if self.posterior_margin < 0.0:
            raise ValueError(
                f"posterior_margin must be >= 0, got {self.posterior_margin}"
            )
        if self.posterior_min_run_sec < 0.0:
            raise ValueError(
                f"posterior_min_run_sec must be >= 0, got {self.posterior_min_run_sec}"
            )

        if self.word_dp_switch_penalty_scale < 0.0:
            raise ValueError(
                f"word_dp_switch_penalty_scale must be >= 0, got {self.word_dp_switch_penalty_scale}"
            )
        if self.word_dp_min_frames < 1:
            raise ValueError(f"word_dp_min_frames must be >= 1, got {self.word_dp_min_frames}")
        if self.word_dp_gap_sec < 0.0:
            raise ValueError(f"word_dp_gap_sec must be >= 0, got {self.word_dp_gap_sec}")
        if self.word_dp_gap_switch_scale < 0.0:
            raise ValueError(
                f"word_dp_gap_switch_scale must be >= 0, got {self.word_dp_gap_switch_scale}"
            )
        if not isinstance(self.boundary_refine_enable, bool):
            raise TypeError(
                f"boundary_refine_enable must be bool, got {type(self.boundary_refine_enable).__name__}"
            )
        if not isinstance(self.boundary_refine_preset, str):
            raise TypeError(
                f"boundary_refine_preset must be str, got {type(self.boundary_refine_preset).__name__}"
            )

    @classmethod
    def with_boundary_refine(cls, preset: str = "default", **overrides: float) -> "AlignerConfig":
        cfg = cls()
        cfg.boundary_refine_enable = True
        cfg.boundary_refine_preset = preset
        if overrides:
            cfg.boundary_refine_params = BoundaryRefineParams(**overrides)
        return cfg

    def resolve_boundary_refine_params(self) -> BoundaryRefineParams:
        preset_name = (self.boundary_refine_preset or "default").lower()
        base = _BOUNDARY_REFINE_PRESETS.get(preset_name, _BOUNDARY_REFINE_PRESETS["default"])
        params = replace(base)
        overrides = getattr(self, "boundary_refine_params", None)
        if overrides is None:
            return params
        default_params = _BOUNDARY_REFINE_PRESETS["default"]
        override_values: Dict[str, object] = {}
        for f in fields(BoundaryRefineParams):
            override_val = getattr(overrides, f.name)
            default_val = getattr(default_params, f.name)
            if override_val != default_val:
                override_values[f.name] = override_val
        if not override_values:
            return params
        return replace(params, **override_values)

@dataclass
class ExportConfig:
    output_formats: List[str] = field(default_factory=lambda: ["json", "srt", "txt"])
    srt_max_line_width: int = 42
    # 句読点付与（XLM-R punctuator）を有効にするかどうか
    enable_punctuation: bool = True
    # プロジェクトのルートディレクトリ
    # None の場合はカレントディレクトリ(Path.cwd())を使う
    project_root: Optional[str] = None

    # --- NEW: Exporter boundary snap (segment -> word boundary) ---
    # True のとき、セグメント境界を近傍の単語境界へスナップする（max 秒以内）。
    enable_boundary_snap: bool = True
    boundary_snap_max_sec: float = 0.5
    # True のとき、Sudachi による形態素グルーピングをスナップ候補に使う
    boundary_snap_use_sudachi: bool = True

@dataclass
class AppConfig:
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    aligner: AlignerConfig = field(default_factory=AlignerConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    output_dir: str = "AHE_Whisper_output"

    def to_dict(self) -> dict:
        return asdict(self)
