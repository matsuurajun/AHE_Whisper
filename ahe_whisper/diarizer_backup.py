# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from typing import Tuple
import logging

from ahe_whisper.utils import safe_softmax, safe_l2_normalize
from ahe_whisper.config import DiarizationConfig

LOGGER = logging.getLogger(__name__)
RNG = np.random.default_rng(42)

class Diarizer:
    def __init__(self, config: DiarizationConfig) -> None:
        self.config = config

    def _postprocess_clusters(
        self,
        embeddings: np.ndarray,
        centroids: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        KMeans+EM の結果（centroids, labels）に対して、
        - 空クラスタの除去
        - cluster_mass によるフィルタ
        - centroid 類似度によるマージ
        を行う。

        その際、min_speakers を「最終クラスタ数のソフト下限」として尊重し、
        min_speakers を下回るようなフィルタ / マージは行わない。
        """
        if embeddings.shape[0] == 0 or centroids.shape[0] == 0:
            return centroids, labels

        num_frames = embeddings.shape[0]
        k = centroids.shape[0]

        # --- empty cluster を除去 ---
        counts = np.bincount(labels, minlength=k)
        nonempty = counts > 0
        if not np.any(nonempty):
            # 念のためのフォールバック：全部空ならそのまま返す
            LOGGER.warning("[DIAR-MERGE] all clusters empty in postprocess; returning raw centroids.")
            return centroids, labels

        idx_nonempty = np.where(nonempty)[0]
        if len(idx_nonempty) < k:
            LOGGER.info(
                "[DIAR-MERGE] dropping %d empty clusters (original=%d, nonempty=%d)",
                k - len(idx_nonempty),
                k,
                len(idx_nonempty),
            )

        # 再マッピング
        remap_nonempty = {old: new for new, old in enumerate(idx_nonempty)}
        labels = np.array([remap_nonempty[l] for l in labels], dtype=int)
        centroids = centroids[idx_nonempty]
        counts = counts[idx_nonempty]

        num_found = centroids.shape[0]
        min_speakers = max(1, int(getattr(self.config, "min_speakers", 1)))
        max_speakers = max(min_speakers, int(getattr(self.config, "max_speakers", num_found)))

        # --- cluster mass の計算 ---
        mass = counts.astype(np.float32) / float(max(1, num_frames))

        min_cluster_mass = float(getattr(self.config, "min_cluster_mass", 0.0))
        centroid_merge_sim = float(getattr(self.config, "centroid_merge_sim", 0.0))

        LOGGER.info(
            "[DIAR-MERGE] clusters: original=%d, min_speakers=%d, max_speakers=%d",
            num_found,
            min_speakers,
            max_speakers,
        )

        # --- 質量しきい値でクラスタ削除（ただし min_speakers を下回らない範囲） ---
        if min_cluster_mass > 0.0 and num_found > min_speakers:
            keep = mass >= min_cluster_mass
            kept = int(keep.sum())
            LOGGER.info(
                "[DIAR-MERGE] mass filter: threshold=%.3f -> kept=%d",
                min_cluster_mass,
                kept,
            )

            # min_speakers を割るようならフィルタをスキップ
            if kept >= min_speakers:
                idx_keep = np.where(keep)[0]

                # まず centroid 側だけを絞る
                centroids = centroids[idx_keep]

                # 削除されたクラスタに属していたフレームは、
                # 「残ったクラスタの中で最も近い centroid」に振り直す
                sims = embeddings @ centroids.T
                labels = np.argmax(sims, axis=1)

                # counts / mass / num_found を再計算
                counts = np.bincount(labels, minlength=centroids.shape[0])
                mass = counts.astype(np.float32) / float(max(1, num_frames))
                num_found = centroids.shape[0]

                LOGGER.info(
                    "[DIAR-MERGE] clusters: after_mass=%d (min_speakers=%d)",
                    num_found,
                    min_speakers,
                )
            else:
                LOGGER.info(
                    "[DIAR-MERGE] skipping mass filter because kept=%d < min_speakers=%d",
                    kept,
                    min_speakers,
                )

        # --- centroid 類似度でマージ（min_speakers を割らない範囲で） ---
        if centroid_merge_sim > 0.0 and num_found > min_speakers:
            while True:
                sims = centroids @ centroids.T  # L2 正規化済み前提
                np.fill_diagonal(sims, -1.0)
                flat_idx = int(np.argmax(sims))
                i, j = divmod(flat_idx, sims.shape[1])
                max_sim = float(sims[i, j])

                if max_sim < centroid_merge_sim or centroids.shape[0] <= min_speakers:
                    break

                LOGGER.info(
                    "[DIAR-MERGE] merging clusters (%d, %d) with sim=%.3f (threshold=%.2f)",
                    i,
                    j,
                    max_sim,
                    centroid_merge_sim,
                )

                # j を i にマージ（counts で重み付け平均）
                total = counts[i] + counts[j]
                if total > 0:
                    merged = (centroids[i] * counts[i] + centroids[j] * counts[j]) / float(total)
                else:
                    merged = 0.5 * (centroids[i] + centroids[j])
                merged = safe_l2_normalize(merged.reshape(1, -1))[0]

                # centroids / counts を更新（j を削除）
                mask = np.ones(centroids.shape[0], dtype=bool)
                mask[j] = False
                centroids[i] = merged
                centroids = centroids[mask]
                counts[i] = total
                counts = counts[mask]

                # ラベルも再マッピング
                remap = {}
                new_idx = 0
                for old_idx in range(mask.shape[0]):
                    if mask[old_idx]:
                        remap[old_idx] = new_idx
                        new_idx += 1
                # マージされた j は i に吸収
                remap[j] = remap[i]
                labels = np.array([remap[l] for l in labels], dtype=int)

            LOGGER.info(
                "[DIAR-MERGE] clusters: after_merge=%d (min_speakers=%d)",
                centroids.shape[0],
                min_speakers,
            )

        # --- 最終セーフガード ---
        if centroids.shape[0] == 0:
            LOGGER.warning("[DIAR-MERGE] all clusters removed; falling back to single centroid.")
            mean_centroid = np.mean(embeddings, axis=0, keepdims=True)
            centroids = safe_l2_normalize(mean_centroid)
            labels = np.zeros(num_frames, dtype=int)

        return centroids, labels

    def cluster(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # [PATCH v90.90] CRITICAL ATTRIBUTE/INDEX ERROR FIX:
        # Safely handle empty embeddings array without referencing self.config.
        if embeddings.shape[0] == 0:
            embedding_dim = embeddings.shape[1] if (embeddings.ndim > 1 and embeddings.shape[1] > 0) else 192
            zero_centroid = np.zeros((1, embedding_dim), dtype=np.float32)
            return safe_l2_normalize(zero_centroid), np.array([], dtype=int)
            
        if embeddings.shape[0] < self.config.min_speakers:
            labels = np.zeros(embeddings.shape[0], dtype=int)
            centroids = np.mean(embeddings, axis=0, keepdims=True)
            return safe_l2_normalize(centroids), labels

        k_min = self.config.min_speakers
        k_max = min(self.config.max_speakers, embeddings.shape[0])
        
        k = max(k_min, min(k_max, int(np.sqrt(embeddings.shape[0]/2)) if embeddings.shape[0] > 8 else 2))

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
        
        attractors = safe_l2_normalize(kmeans.cluster_centers_)
        
        responsibilities = np.full((embeddings.shape[0], k), 1.0 / k, dtype=np.float32)

        for tau in self.config.em_tau_schedule:
            similarities = embeddings @ attractors.T
            responsibilities = safe_softmax(similarities, tau * 0.75)  # ER2V2: similarity 高め → tau を下げて硬さUP
            
            new_attractors = np.zeros_like(attractors)
            for i in range(k):
                r_i = responsibilities[:, i]
                if np.sum(r_i) > 1e-6:
                    weighted_sum = np.sum(embeddings * r_i[:, np.newaxis], axis=0)
                    new_attractors[i] = safe_l2_normalize(weighted_sum)
                else:
                    random_idx = RNG.choice(embeddings.shape[0])
                    new_attractors[i] = safe_l2_normalize(embeddings[random_idx])

            attractors = new_attractors
        
        final_labels = np.argmax(responsibilities, axis=1)

        # --- [NEW] クラスタ後処理（mass filter / merge / min_speakers ソフト下限） ---
        attractors, final_labels = self._postprocess_clusters(
            embeddings, attractors, final_labels
        )
        return attractors, final_labels

    def get_speaker_probabilities(
        self,
        embeddings: np.ndarray,
        valid_mask: np.ndarray,
        centroids: np.ndarray,
        grid_times: np.ndarray,
        hop_len: int,
        sr: int
    ) -> np.ndarray:
        
        # === DEBUG (AHE): diarizer input sanity ===
        try:
            print(f"[DEBUG-DIAR] grid_len={len(grid_times)}, "
                  f"embeddings.shape={getattr(embeddings, 'shape', None)}, "
                  f"valid_mask.sum={int(valid_mask.sum()) if valid_mask is not None else 'N/A'}, "
                  f"centroids.shape={getattr(centroids, 'shape', None)}, "
                  f"hop_len={hop_len}, sr={sr}")
        except Exception as e:
            print(f"[DEBUG-DIAR] debug header failed: {e}")
        
        num_speakers = len(centroids)
        embedding_times = np.arange(len(embeddings)) * (hop_len / sr)
        
        similarities = embeddings @ centroids.T
        
        spk_probs = np.zeros((len(grid_times), num_speakers), dtype=np.float32)
        
        valid_times = embedding_times[valid_mask]
        valid_similarities = similarities[valid_mask, :]
        
        # === DEBUG (AHE): timeline & similarities ===
        try:
            print(f"[DEBUG-DIAR] valid_times_len={len(valid_times)}, "
                  f"valid_sims.shape={getattr(valid_similarities, 'shape', None)}")
            if len(valid_times) >= 2:
                print(f"[DEBUG-DIAR] valid_times.head="
                      f"{[round(float(t),2) for t in valid_times[:5]]} ...")
        except Exception as e:
            print(f"[DEBUG-DIAR] debug valid* failed: {e}")
        
        # === DEBUG (AHE): early-exit condition check ===
        try:
            if len(valid_times) < 2:
                print(f"[DEBUG-DIAR] early-exit: len(valid_times)={len(valid_times)} -> "
                      f"{'copy-single or zeros then softmax' if len(valid_times)==1 else 'zeros then softmax'}")
        except Exception as e:
            print(f"[DEBUG-DIAR] debug early-exit failed: {e}")

        if len(valid_times) < 2:
            if len(valid_times) == 1:
                spk_probs[:, :] = valid_similarities[0, :]
            return safe_softmax(spk_probs)
        
        # === [PATCH v91-CONTRAST] Apply contrast normalization before interpolation ===
        # 行方向（1フレーム内での話者差）の std を平均して「コントラストの低さ」を判定する
        row_std = np.std(valid_similarities, axis=1)
        var = float(np.mean(row_std))
        if var < 0.05:
            print(
                "[DEBUG-DIAR] low per-frame variance "
                f"(mean row-std={var:.4f}) → applying contrast normalization (pre-interp)"
            )
            # Z-score normalization per frame
            valid_similarities = (valid_similarities - np.mean(valid_similarities, axis=1, keepdims=True)) / \
                                 (np.std(valid_similarities, axis=1, keepdims=True) + 1e-6)
            # Moderate contrast enhancement
            valid_similarities = np.tanh(valid_similarities * 3.0)
            print(f"[DEBUG-DIAR] AFTER norm: std={np.std(valid_similarities):.4f}, "
                  f"min={np.min(valid_similarities):.3f}, max={np.max(valid_similarities):.3f}")
            
        # === DIAR v90.96: Local-τ or Global-τ → interpolate once → return ===
        T, K = valid_similarities.shape

        # 低コントラストなら事前正規化 + tanh(×3)
        global_std = float(np.mean(np.std(valid_similarities, axis=1)))
        if global_std < 0.05:
            LOGGER.info(
                "[DEBUG-DIAR] low variance (mean row-std=%.4f) → z-score + tanh*3",
                global_std,
            )
            valid_similarities = (valid_similarities - np.mean(valid_similarities, axis=1, keepdims=True)) / \
                                 (np.std(valid_similarities, axis=1, keepdims=True) + 1e-6)
            valid_similarities = np.tanh(valid_similarities * 3.0)
            LOGGER.info("[DEBUG-DIAR] AFTER norm: std=%.4f, min=%.3f, max=%.3f",
                        float(np.std(valid_similarities)),
                        float(np.min(valid_similarities)), float(np.max(valid_similarities)))

            # --- Local-τ (時間局所stdからτ(t)を生成) ---
            win = 41  # ≈0.8s @50Hz
            if win % 2 == 0:
                win += 1
            half = win // 2

            loc_std = np.empty(T, dtype=np.float32)
            pad = np.pad(valid_similarities, ((half, half), (0, 0)), mode="edge")
            for t in range(T):
                sl = pad[t:t+win]
                loc_std[t] = np.std(sl, axis=1).mean()

            tau_min, tau_max = 0.35, 0.65
            std_lo, std_hi = 0.05, 0.20
            tau_t = tau_max - (np.clip(loc_std, std_lo, std_hi) - std_lo) * (tau_max - tau_min) / (std_hi - std_lo)
            tau_t = tau_t.astype(np.float32)  # (T,)

            logits = valid_similarities / tau_t[:, None]
            logits -= logits.max(axis=1, keepdims=True)
            spk_valid = np.exp(logits)
            spk_valid /= np.sum(spk_valid, axis=1, keepdims=True) + 1e-12

            # 事後シャープ + EMA平滑（ワンショット実験 gamma=1.55 / alpha=0.30）
            gamma = 1.20  # ER2V2: embedding分布がシャープ → 過度な硬さ抑える
            spk_valid = np.power(spk_valid, gamma)
            spk_valid /= np.sum(spk_valid, axis=1, keepdims=True) + 1e-12

            alpha = 0.30
            ema = np.empty_like(spk_valid)
            ema[0] = spk_valid[0]
            for t in range(1, T):
                ema[t] = alpha * spk_valid[t] + (1.0 - alpha) * ema[t-1]
            spk_valid = ema

            LOGGER.info("[DEBUG-DIAR-TAU] Local τ: min=%.2f, max=%.2f, mean=%.2f, std(valid_sims)=%.4f",
                        float(tau_t.min()), float(tau_t.max()), float(tau_t.mean()),
                        float(valid_similarities.std()))
        else:
            # --- Global-τ（簡易で高速なパス） ---
            tau_used = 0.28  # ER2V2 推奨: 0.25–0.30
            LOGGER.info("[DEBUG-DIAR-TAU] Global τ used = %.2f (std=%.4f)", tau_used, global_std)

            logits = valid_similarities / tau_used
            logits -= logits.max(axis=1, keepdims=True)
            spk_valid = np.exp(logits)
            spk_valid /= np.sum(spk_valid, axis=1, keepdims=True) + 1e-12

            gamma = 1.15  # ER2V2：過シャープ化防止
            spk_valid = np.power(spk_valid, gamma)
            spk_valid /= np.sum(spk_valid, axis=1, keepdims=True) + 1e-12

        # === ここで spk_valid は (T, K). これを grid_times に一度だけ補間 ===
        spk_probs = np.zeros((len(grid_times), K), dtype=np.float32)
        for k in range(K):
            fn = interp1d(valid_times, spk_valid[:, k], kind='linear', bounds_error=False, fill_value="extrapolate")
            spk_probs[:, k] = fn(grid_times)

        # 数値安定＆最終正規化（行方向で確率和=1）
        spk_probs = np.clip(spk_probs, 1e-9, 1.0)
        spk_probs /= np.sum(spk_probs, axis=1, keepdims=True)

        # --- diagnostics & degeneracy check ---
        self.last_valid_sims = valid_similarities.copy()

        mm = float(np.mean(np.max(spk_probs, axis=1)))
        ent = float(-np.mean(np.sum(spk_probs * np.log(spk_probs + 1e-12), axis=1)))
        LOGGER.info(
            "[SPK-PROBS] final(pre-check): mean_max=%.3f, mean_entropy=%.3f, shape=%s",
            mm,
            ent,
            str(spk_probs.shape),
        )

        # === [PATCH v91-FAILSAFE] almost-uniform な分布のときは「ハードクラスタ」にフォールバック ===
        min_mean_max = float(getattr(self.config, "min_probs_mean_max", 0.55))
        max_mean_entropy = float(getattr(self.config, "max_probs_mean_entropy", 0.67))

        if K >= 2 and (mm < min_mean_max or ent > max_mean_entropy):
            LOGGER.warning(
                "[SPK-PROBS] degenerate distribution detected "
                "(mean_max=%.3f, entropy=%.3f, K=%d). "
                "Falling back to hard cluster-based probabilities.",
                mm,
                ent,
                K,
            )

            # valid_similarities 上で argmax を取り、ほぼ one-hot な分布を生成
            hard_labels = np.argmax(valid_similarities, axis=1)  # (T,)
            T_valid = valid_similarities.shape[0]
            eps = 0.02
            spk_valid_hard = np.full((T_valid, K), eps / K, dtype=np.float32)
            spk_valid_hard[np.arange(T_valid), hard_labels] = 1.0 - eps + eps / K

            # grid_times へ再補間
            spk_probs = np.zeros((len(grid_times), K), dtype=np.float32)
            for k in range(K):
                fn = interp1d(
                    valid_times,
                    spk_valid_hard[:, k],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                spk_probs[:, k] = fn(grid_times)

            spk_probs = np.clip(spk_probs, 1e-9, 1.0)
            spk_probs /= np.sum(spk_probs, axis=1, keepdims=True)

            mm = float(np.mean(np.max(spk_probs, axis=1)))
            ent = float(-np.mean(np.sum(spk_probs * np.log(spk_probs + 1e-12), axis=1)))
            LOGGER.info(
                "[SPK-PROBS] fallback(hard-cluster): mean_max=%.3f, mean_entropy=%.3f, shape=%s",
                mm,
                ent,
                str(spk_probs.shape),
            )

        # --- 外部診断用に保存（pipeline などから覗けるように） ---
        self.last_probs = spk_probs.copy()

        return spk_probs
