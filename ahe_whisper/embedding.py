# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import onnxruntime as ort

from ahe_whisper.utils import safe_l2_normalize
from ahe_whisper.features import Featurizer
from ahe_whisper.frontend_spec import load_spec_for_model, resolve_cmvn_policy
from ahe_whisper.config import EmbeddingConfig

LOGGER = logging.getLogger("ahe_whisper_worker")

try:
    # sherpa-onnx (Titanet-L / Speakernet 用)
    import sherpa_onnx  # type: ignore
except Exception:  # ImportError 等を包括
    sherpa_onnx = None  # type: ignore[assignment]

# =========================================================
# 共通ヘルパー
# =========================================================

def _apply_cmvn_and_scale(feat: np.ndarray) -> np.ndarray:
    """
    【重要修正】
    Instance CMVN (平均0, 分散1) を適用し、録音環境によるバイアスを除去する。
    """
    # 時間方向(axis=0)の平均と標準偏差を計算
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    # ゼロ除算を防ぐための微小値
    return (feat - mean) / (std + 1e-6)


# =========================================================
# Wespeaker ResNet293-LM (VoxCeleb) 用ラッパー
# =========================================================

def build_resnet293_session(model_path: Path, config: EmbeddingConfig) -> ort.InferenceSession:
    """
    Wespeaker / wespeaker-voxceleb-resnet293-LM の ONNX セッションを構築。
    CoreMLExecutionProvider が使えればそれを優先。
    """
    sess_options = ort.SessionOptions()
    if getattr(config, "intra_threads", None) is not None:
        sess_options.intra_op_num_threads = int(config.intra_threads)
    if getattr(config, "inter_threads", None) is not None:
        sess_options.inter_op_num_threads = int(config.inter_threads)

    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if getattr(config, "prefer_coreml_ep", False) and "CoreMLExecutionProvider" in available:
        providers.insert(0, "CoreMLExecutionProvider")

    LOGGER.info(
        "[ResNet293] Building InferenceSession: model=%s, providers=%s, intra=%s, inter=%s",
        str(model_path),
        providers,
        getattr(config, "intra_threads", None),
        getattr(config, "inter_threads", None),
    )
    sess = ort.InferenceSession(str(model_path), sess_options, providers=providers)
    LOGGER.info(
        "[ResNet293] Session built. Inputs=%s, Outputs=%s",
        sess.get_inputs(),
        sess.get_outputs(),
    )
    return sess


def _detect_resnet293_input_layout(session: ort.InferenceSession, feat_dim: int) -> str:
    """
    Wespeaker ResNet293 ONNX の入力テンソルレイアウトを推定する。
    """
    in0 = session.get_inputs()[0]
    shape = list(in0.shape)
    ndim = len(shape)

    if ndim == 3:
        # (B, T, F) or (B, F, T)
        _, d1, d2 = shape
        if d2 in (-1, feat_dim):
            return "BTF"  # (B, T, F)
        if d1 in (-1, feat_dim):
            return "BFT"  # (B, F, T)
        return "BTF"

    if ndim == 4:
        # (B, 1, F, T) or (B, 1, T, F)
        _, _, d1, d2 = shape
        if d1 in (-1, feat_dim):
            return "B1FT"
        if d2 in (-1, feat_dim):
            return "B1TF"
        return "B1FT"

    raise RuntimeError(f"[ResNet293] Unsupported input rank: {shape}")


def _run_resnet293_single(
    session: ort.InferenceSession,
    layout: str,
    input_name: str,
    feat: np.ndarray,
) -> np.ndarray:
    """
    単一チャンク分の特徴量 (T, F) から 1 本の埋め込みベクトルを得る。
    """
    x = feat.astype(np.float32, copy=False)

    if layout == "BTF":
        x = x[None, :, :]          # (1, T, F)
    elif layout == "BFT":
        x = x.T[None, :, :]        # (1, F, T)
    elif layout == "B1FT":
        x = x.T[None, None, :, :]  # (1, 1, F, T)
    elif layout == "B1TF":
        x = x[None, None, :, :]    # (1, 1, T, F)
    else:
        raise RuntimeError(f"[ResNet293] Unknown layout: {layout}")

    y = session.run(None, {input_name: x})[0]
    emb = np.squeeze(y, axis=0)
    return emb.reshape(-1)


def resnet293_embed_batched(
    session: ort.InferenceSession,
    model_path: Path,
    chunks: List[np.ndarray],
    sr: int,
    config: EmbeddingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wespeaker ResNet293-LM (VoxCeleb) 向けの埋め込み抽出ラッパー。
    """
    num_chunks = len(chunks)
    if num_chunks == 0:
        emb_dim0 = getattr(config, "embedding_dim", 256)
        return np.zeros((0, emb_dim0), np.float32), np.zeros((0,), bool)

    # --- frontend 設定 ---
    spec_info = load_spec_for_model(model_path)
    if isinstance(spec_info, tuple):
        spec, default_cmvn = spec_info
    else:
        spec = spec_info
        default_cmvn = None

    cmvn_policy = resolve_cmvn_policy(
        spec,
        getattr(config, "cmvn_policy", default_cmvn),
    )
    featurizer = Featurizer(spec)
    min_frames = int(max(getattr(config, "min_frames", 1), 1))

    features: List[Optional[np.ndarray]] = []
    valid_indices: List[int] = []

    for idx, chunk in enumerate(chunks):
        if chunk is None or len(chunk) == 0:
            LOGGER.debug("[ResNet293] Skip chunk=%d: empty", idx)
            features.append(None)
            continue
        
        # 【重要修正 1】 音声スケール補正 (librosa [-1,1] -> WeSpeaker [-32768,32767])
        chunk = chunk * 32768.0

        feat = featurizer.get_mel_spectrogram(chunk, sr, cmvn_policy)
        if feat is None or feat.shape[0] < min_frames:
            LOGGER.debug("[ResNet293] Skip chunk=%d: too short", idx)
            features.append(None)
            continue
        
        # 【重要修正 2】 Instance CMVN 適用
        feat = _apply_cmvn_and_scale(feat)

        feat = feat.astype(np.float32, copy=False)
        features.append(feat)
        valid_indices.append(idx)

    if not valid_indices:
        LOGGER.warning("[ResNet293] No valid chunks. num_chunks=%d", num_chunks)
        emb_dim0 = getattr(config, "embedding_dim", 256)
        return np.zeros((num_chunks, emb_dim0), np.float32), np.zeros(num_chunks, bool)

    # --- 推論準備 ---
    first_feat = next(f for f in features if f is not None)
    feat_dim = int(first_feat.shape[1])
    layout = _detect_resnet293_input_layout(session, feat_dim)
    in_name = session.get_inputs()[0].name

    out0 = session.get_outputs()[0]
    emb_dim = 0
    if out0.shape and out0.shape[-1] not in (-1, None):
        emb_dim = int(out0.shape[-1])

    if not emb_dim:
        LOGGER.info("[ResNet293] Probing emb_dim via dummy forward...")
        emb_probe = _run_resnet293_single(session, layout, in_name, first_feat)
        emb_dim = int(emb_probe.shape[-1])

    setattr(config, "embedding_dim", emb_dim)
    buffer = np.zeros((len(valid_indices), emb_dim), dtype=np.float32)

    # --- 推論 ---
    for bi, orig_idx in enumerate(valid_indices):
        feat = features[orig_idx]
        if feat is None: continue
        
        emb = _run_resnet293_single(session, layout, in_name, feat)
        if emb.shape[-1] != emb_dim:
            LOGGER.error("[ResNet293] emb_dim mismatch: got=%d, expected=%d", emb.shape[-1], emb_dim)
            continue
        buffer[bi, :] = emb

    # --- 結果格納 ---
    final_embeddings = np.zeros((num_chunks, emb_dim), dtype=np.float32)
    for i, orig_idx in enumerate(valid_indices):
        final_embeddings[orig_idx, :] = buffer[i]

    final_embeddings = safe_l2_normalize(final_embeddings.astype(np.float32))
    valid_mask = np.zeros(num_chunks, dtype=bool)
    valid_mask[valid_indices] = True

    LOGGER.info("[ResNet293] Done: valid=%d/%d, dim=%d", len(valid_indices), num_chunks, emb_dim)
    return final_embeddings, valid_mask


# =========================================================
# Wespeaker ECAPA-TDNN512-LM 用ラッパー
# =========================================================

def build_ecapa512_session(model_path: Path, config: EmbeddingConfig) -> ort.InferenceSession:
    """
    Wespeaker ECAPA-TDNN512-LM の ONNX セッションを構築。
    """
    sess_options = ort.SessionOptions()
    if getattr(config, "intra_threads", None) is not None:
        sess_options.intra_op_num_threads = int(config.intra_threads)
    if getattr(config, "inter_threads", None) is not None:
        sess_options.inter_op_num_threads = int(config.inter_threads)

    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if getattr(config, "prefer_coreml_ep", False) and "CoreMLExecutionProvider" in available:
        providers.insert(0, "CoreMLExecutionProvider")

    LOGGER.info(
        "[ECAPA512] Building InferenceSession: model=%s, providers=%s",
        str(model_path), providers
    )
    sess = ort.InferenceSession(str(model_path), sess_options, providers=providers)
    return sess


def ecapa512_embed_batched(
    session: ort.InferenceSession,
    model_path: Path,
    chunks: List[np.ndarray],
    sr: int,
    config: EmbeddingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wespeaker ECAPA-TDNN512-LM 向けの埋め込み抽出ラッパー。
    """
    num_chunks = len(chunks)
    if num_chunks == 0:
        emb_dim0 = getattr(config, "embedding_dim", 192)
        return np.zeros((0, emb_dim0), np.float32), np.zeros((0,), bool)

    # --- frontend 設定 ---
    spec_info = load_spec_for_model(model_path)
    if isinstance(spec_info, tuple):
        spec, default_cmvn = spec_info
    else:
        spec = spec_info
        default_cmvn = None

    cmvn_policy = resolve_cmvn_policy(spec, getattr(config, "cmvn_policy", default_cmvn))
    featurizer = Featurizer(spec)
    min_frames = int(max(getattr(config, "min_frames", 1), 1))

    features: List[Optional[np.ndarray]] = []
    valid_indices: List[int] = []

    for idx, chunk in enumerate(chunks):
        if chunk is None or len(chunk) == 0:
            LOGGER.debug("[ECAPA512] Skip chunk=%d: empty", idx)
            features.append(None)
            continue
        
        # 【重要修正 1】 音声スケール補正
        chunk = chunk * 32768.0

        feat = featurizer.get_mel_spectrogram(chunk, sr, cmvn_policy)
        if feat is None or feat.shape[0] < min_frames:
            LOGGER.debug("[ECAPA512] Skip chunk=%d: too short", idx)
            features.append(None)
            continue
        
        # 【重要修正 2】 Instance CMVN 適用
        feat = _apply_cmvn_and_scale(feat)

        feat = feat.astype(np.float32, copy=False)
        features.append(feat)
        valid_indices.append(idx)

    if not valid_indices:
        LOGGER.warning("[ECAPA512] No valid chunks. num_chunks=%d", num_chunks)
        emb_dim0 = getattr(config, "embedding_dim", 192)
        return np.zeros((num_chunks, emb_dim0), np.float32), np.zeros(num_chunks, bool)

    # --- 推論準備 ---
    first_feat = next(f for f in features if f is not None)
    feat_dim = int(first_feat.shape[1])
    
    # ResNet293のレイアウト検出ロジックを流用
    layout = _detect_resnet293_input_layout(session, feat_dim)
    in_name = session.get_inputs()[0].name

    emb_dim = getattr(config, "embedding_dim", 0)
    if not emb_dim:
        out0 = session.get_outputs()[0]
        if out0.shape and out0.shape[-1] not in (-1, None):
            emb_dim = int(out0.shape[-1])
        else:
            LOGGER.info("[ECAPA512] Probing emb_dim via dummy forward...")
            emb_probe = _run_resnet293_single(session, layout, in_name, first_feat)
            emb_dim = int(emb_probe.shape[-1])

    buffer = np.zeros((len(valid_indices), emb_dim), dtype=np.float32)

    # --- 推論 ---
    for bi, orig_idx in enumerate(valid_indices):
        feat = features[orig_idx]
        if feat is None: continue
        
        emb = _run_resnet293_single(session, layout, in_name, feat)
        if emb.shape[-1] != emb_dim:
            LOGGER.warning("[ECAPA512] emb_dim mismatch: got=%d, expected=%d", emb.shape[-1], emb_dim)
            emb = emb[:emb_dim]
        buffer[bi, :] = emb

    # --- 結果格納 ---
    final_embeddings = np.zeros((num_chunks, emb_dim), dtype=np.float32)
    for i, orig_idx in enumerate(valid_indices):
        final_embeddings[orig_idx, :] = buffer[i]

    final_embeddings = safe_l2_normalize(final_embeddings.astype(np.float32))
    valid_mask = np.zeros(num_chunks, dtype=bool)
    valid_mask[valid_indices] = True

    LOGGER.info("[ECAPA512] Done: valid=%d/%d, dim=%d", len(valid_indices), num_chunks, emb_dim)
    return final_embeddings, valid_mask


# =========================================================
# 3D-Speaker / ERes2NetV2 / Cam++ 用ラッパー
# =========================================================

def build_er2v2_session(model_path: Path, config: EmbeddingConfig) -> ort.InferenceSession:
    """
    ERes2NetV2 / Cam++ 用の ONNXRuntime セッションを構築する。
    """
    sess_options = ort.SessionOptions()
    if config.intra_threads is not None:
        sess_options.intra_op_num_threads = int(config.intra_threads)
    if config.inter_threads is not None:
        sess_options.inter_op_num_threads = int(config.inter_threads)

    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if config.prefer_coreml_ep and "CoreMLExecutionProvider" in available:
        providers.insert(0, "CoreMLExecutionProvider")

    LOGGER.info(
        "[CampPlus] Building InferenceSession: model=%s, providers=%s",
        str(model_path), providers
    )
    session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
    return session


def warmup_er2v2(session: ort.InferenceSession) -> None:
    try:
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(1, 200, 80).astype(np.float32)
        _ = session.run(None, {input_name: dummy})
        LOGGER.info("[CampPlus] Warmup done.")
    except Exception as e:
        LOGGER.warning("[CampPlus] Warmup failed: %s", e)


def er2v2_embed_batched(
    session: ort.InferenceSession,
    model_path: Path,
    audio_chunks: List[np.ndarray],
    sr: int,
    config: EmbeddingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cam++ / ERes2Net 向けの埋め込み抽出ラッパー。
    """
    num_chunks = len(audio_chunks)
    emb_dim = int(config.embedding_dim)

    if num_chunks == 0:
        return (np.zeros((0, emb_dim), dtype=np.float32), np.zeros(0, dtype=bool))

    spec, _ = load_spec_for_model(model_path)
    cmvn_policy = resolve_cmvn_policy(model_path.parent)
    featurizer = Featurizer(spec)
    min_frames = getattr(config, "min_frames", 40)

    LOGGER.info("[CampPlus] Starting extraction: chunks=%d, sr=%d", num_chunks, sr)

    features = []
    valid_indices = []
    for idx, chunk in enumerate(audio_chunks):
        if chunk is None or len(chunk) == 0:
            features.append(None)
            continue
        
        # 【重要修正 1】 音声スケール補正
        chunk = chunk * 32768.0

        feat = featurizer.get_mel_spectrogram(chunk, sr, cmvn_policy)
        if feat is None or feat.shape[0] < min_frames:
            LOGGER.debug("[CampPlus] Skip chunk=%d: too short", idx)
            features.append(None)
            continue
        
        # 【重要修正 2】 Instance CMVN 適用
        feat = _apply_cmvn_and_scale(feat)

        feat = feat.astype(np.float32, copy=False)
        features.append(feat)
        valid_indices.append(idx)

    final_embeddings = np.zeros((num_chunks, emb_dim), dtype=np.float32)
    if not valid_indices:
        LOGGER.warning("[CampPlus] No valid chunks.")
        valid_mask = np.zeros(num_chunks, dtype=bool)
        return final_embeddings, valid_mask

    valid_features = [features[i] for i in valid_indices]
    feat_lens = [f.shape[0] for f in valid_features]

    # mel_dim 決定
    mel_dim = int(valid_features[0].shape[1])
    for k in ("num_mels", "n_mels", "mel_bins", "feature_dim", "dim"):
        if hasattr(spec, k):
            try:
                v = int(getattr(spec, k))
                if v > 0: mel_dim = v; break
            except: pass
    if mel_dim <= 0: mel_dim = 80

    order = np.argsort(feat_lens)
    buffer = np.zeros((len(valid_features), emb_dim), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    batch_cap = int(getattr(config, "batch_cap", 16))
    bucket_step = int(getattr(config, "bucket_step", 40))

    batch_start = 0
    while batch_start < len(order):
        curr_len = feat_lens[order[batch_start]]
        bucket_max = ((curr_len // bucket_step) + 1) * bucket_step
        batch_end = batch_start
        while (batch_end < len(order) and feat_lens[order[batch_end]] <= bucket_max 
               and (batch_end - batch_start) < batch_cap):
            batch_end += 1

        idxs = order[batch_start:batch_end]
        max_len = max(feat_lens[i] for i in idxs)
        batch_size = len(idxs)
        batch_input = np.zeros((batch_size, max_len, mel_dim), dtype=np.float32)

        for bi, fi in enumerate(idxs):
            feat = valid_features[fi]
            t, f = feat.shape
            use_cols = min(mel_dim, f)
            batch_input[bi, :t, :use_cols] = feat[:, :use_cols]
            if t < max_len:
                last_frame = batch_input[bi, t - 1 : t, :use_cols]
                repeat = max_len - t
                batch_input[bi, t:, :use_cols] = last_frame.repeat(repeat, axis=0)

        try:
            outputs = session.run([output_name], {input_name: batch_input})[0]
        except Exception as e:
            LOGGER.error("[CampPlus] Batch failed: %s", e)
            batch_start = batch_end
            continue

        if outputs.ndim == 3: outputs = outputs.mean(axis=1)

        for bi, fi in enumerate(idxs):
            if bi < outputs.shape[0]: buffer[fi, :] = outputs[bi]
        batch_start = batch_end

    for i, orig_idx in enumerate(valid_indices):
        final_embeddings[orig_idx, :] = buffer[i]

    final_embeddings = safe_l2_normalize(final_embeddings.astype(np.float32))
    LOGGER.info("[CampPlus] Done: valid=%d/%d, dim=%d", len(valid_indices), num_chunks, emb_dim)
    
    valid_mask = np.zeros(num_chunks, dtype=bool)
    valid_mask[valid_indices] = True
    return final_embeddings, valid_mask


# =========================================================
# sherpa-onnx SpeakerEmbeddingExtractor (Titanet-L / Speakernet 共通)
# =========================================================

def _require_sherpa_onnx():
    """
    sherpa-onnx がインストールされているか確認し、なければ ImportError を投げる。
    """
    if sherpa_onnx is None:
        raise ImportError(
            "sherpa-onnx is required for 'titanet' / 'speakernet' embedding backend. "
            "Install it via `pip install sherpa-onnx`."
        )
    return sherpa_onnx


def build_sherpa_speaker_extractor(
    model_path: Path,
    config: EmbeddingConfig,
    provider: str = "cpu",
):
    """
    sherpa-onnx の SpeakerEmbeddingExtractor を構築する。

    Titanet-L / Speakernet はどちらも同じクラスを使うため、ここでは
    モデルファイルの違いだけを引数で受け取る。
    """
    so = _require_sherpa_onnx()

    num_threads = config.intra_threads if config.intra_threads is not None else 4
    cfg = so.SpeakerEmbeddingExtractorConfig(
        model=str(model_path),
        num_threads=num_threads,
        provider=provider,
        debug=0,
    )
    if not cfg.validate():
        raise RuntimeError(
            f"Invalid SpeakerEmbeddingExtractorConfig for model={model_path}"
        )

    extractor = so.SpeakerEmbeddingExtractor(cfg)
    # embedding_dim が未設定なら、Extractor 側の dim を採用
    if getattr(config, "embedding_dim", 0) in (0, None):
        config.embedding_dim = int(extractor.dim)

    LOGGER.info(
        "[sherpa] SpeakerEmbeddingExtractor created: model=%s, dim=%d, provider=%s, threads=%d",
        model_path,
        extractor.dim,
        provider,
        num_threads,
    )
    return extractor


def sherpa_embed_batched(
    extractor,
    audio_chunks: List[np.ndarray],
    sr: int,
    config: EmbeddingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    sherpa-onnx SpeakerEmbeddingExtractor を用いて、生波形チャンクから
    1 チャンク 1 ベクトルの埋め込みを得る。

    - Titanet-L / Speakernet 共通実装
    - sr は 16kHz を想定（違う場合は警告だけ出してそのまま渡す）
    """
    _require_sherpa_onnx()

    num_chunks = len(audio_chunks)
    emb_dim = int(getattr(config, "embedding_dim", getattr(extractor, "dim", 192)))

    if num_chunks == 0:
        return (
            np.zeros((0, emb_dim), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )

    if sr != 16000:
        LOGGER.warning(
            "[sherpa] Expected sr=16000 for Titanet/Speakernet, but got sr=%d", sr
        )

    embeddings = np.zeros((num_chunks, emb_dim), dtype=np.float32)
    valid_mask = np.zeros(num_chunks, dtype=bool)

    for idx, chunk in enumerate(audio_chunks):
        if chunk is None or len(chunk) == 0:
            LOGGER.debug("[sherpa] Skip chunk=%d: empty", idx)
            continue

        # sherpa-onnx は float32 波形を期待（内部で特徴量計算）
        wav = chunk.astype(np.float32, copy=False)

        try:
            stream = extractor.create_stream()
            stream.accept_waveform(sr, wav)
            stream.input_finished()
            emb_list = extractor.compute(stream)
        except Exception as e:
            LOGGER.warning("[sherpa] Failed on chunk=%d: %s", idx, e)
            continue

        emb = np.asarray(emb_list, dtype=np.float32).reshape(-1)
        if emb.size < emb_dim:
            # 予想より小さい場合は尻を 0 埋め
            tmp = np.zeros((emb_dim,), dtype=np.float32)
            tmp[: emb.size] = emb
            emb = tmp
        elif emb.size > emb_dim:
            emb = emb[:emb_dim]

        embeddings[idx, :] = emb
        valid_mask[idx] = True

    # L2 正規化（ゼロベクトルはそのまま）
    embeddings = safe_l2_normalize(embeddings.astype(np.float32))

    LOGGER.info(
        "[sherpa] Done: valid=%d/%d, dim=%d",
        int(valid_mask.sum()),
        num_chunks,
        emb_dim,
    )
    return embeddings, valid_mask
