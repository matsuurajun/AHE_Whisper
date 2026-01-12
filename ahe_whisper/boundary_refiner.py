# -*- coding: utf-8 -*-
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np

from ahe_whisper.config import AlignerConfig, BoundaryRefineParams

LOGGER = logging.getLogger(__name__)


@dataclass
class BoundaryCandidate:
    center_time: float
    window_start: float
    window_end: float
    reason: str


@dataclass
class ShortWindowResult:
    candidate_idx: int
    embeddings: np.ndarray
    grid_indices: np.ndarray
    grid_times: np.ndarray
    spk_probs_local: np.ndarray
    stats: Dict[str, float]


@dataclass
class BoundaryRefineResult:
    original_path: np.ndarray
    refined_path: np.ndarray
    applied_windows: List[BoundaryCandidate]
    skipped_windows: List[BoundaryCandidate]


def _resolve_params(config: AlignerConfig) -> BoundaryRefineParams:
    if hasattr(config, "resolve_boundary_refine_params"):
        return config.resolve_boundary_refine_params()
    return getattr(config, "boundary_refine_params", BoundaryRefineParams())


def _merge_candidates(
    candidates: List[BoundaryCandidate],
    merge_sec: float,
    merge_max_window_sec: float,
) -> List[BoundaryCandidate]:
    if not candidates:
        return []
    merged: List[BoundaryCandidate] = []
    cur = candidates[0]
    for cand in candidates[1:]:
        if cand.window_start <= cur.window_end + merge_sec:
            new_start = min(cur.window_start, cand.window_start)
            new_end = max(cur.window_end, cand.window_end)
            if (new_end - new_start) <= merge_max_window_sec:
                cur = BoundaryCandidate(
                    center_time=0.5 * (cur.center_time + cand.center_time),
                    window_start=new_start,
                    window_end=new_end,
                    reason=f"{cur.reason}+{cand.reason}",
                )
                continue
        merged.append(cur)
        cur = cand
    merged.append(cur)
    return merged


def extract_boundary_candidates(
    frame_path: np.ndarray,
    grid_times: np.ndarray,
    word_info: List[dict],
    config: AlignerConfig,
    switch_events: Optional[List[Dict[str, float]]] = None,
) -> List[BoundaryCandidate]:
    params = _resolve_params(config)
    candidates: List[BoundaryCandidate] = []
    window_half = float(params.window_sec) * 0.5
    t_min = float(grid_times[0]) if grid_times.size > 0 else 0.0
    t_max = float(grid_times[-1]) if grid_times.size > 0 else 0.0

    if params.candidate_use_switch and frame_path.size > 1:
        if switch_events:
            switch_times = [float(ev.get("time", 0.0)) for ev in switch_events]
        else:
            switch_idxs = np.where(frame_path[1:] != frame_path[:-1])[0]
            switch_times = [float(grid_times[int(i) + 1]) for i in switch_idxs]
        for t in switch_times:
            start = max(t_min, t - window_half)
            end = min(t_max, t + window_half)
            candidates.append(
                BoundaryCandidate(center_time=t, window_start=start, window_end=end, reason="switch")
            )

    if params.candidate_use_lexical and word_info:
        tokens = str(params.lexical_tokens or "")
        for w in word_info:
            text = str(w.get("word") or w.get("text") or "")
            if not text:
                continue
            if not any(text.endswith(tok) for tok in tokens):
                continue
            try:
                t = float(w.get("end"))
            except (TypeError, ValueError):
                continue
            start = max(t_min, t - window_half)
            end = min(t_max, t + window_half)
            candidates.append(
                BoundaryCandidate(center_time=t, window_start=start, window_end=end, reason="lexical")
            )

    if not candidates:
        return []

    candidates.sort(key=lambda c: c.window_start)
    merged = _merge_candidates(candidates, float(params.merge_sec), float(params.merge_max_window_sec))
    if len(merged) > params.max_candidates:
        LOGGER.info(
            "[BOUNDARY-REFINE] candidate cap reached: %d -> %d",
            len(merged),
            params.max_candidates,
        )
        merged = merged[: params.max_candidates]
    return merged


def compute_short_window_embeddings(
    waveform: np.ndarray,
    sr: int,
    candidates: List[BoundaryCandidate],
    win_sec: float,
    hop_sec: float,
    extractor: Callable[[List[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
    config: AlignerConfig,
    *,
    diarizer,
    speaker_centroids: np.ndarray,
    grid_times: np.ndarray,
) -> List[ShortWindowResult]:
    if not callable(extractor):
        raise ValueError("extractor must be callable")
    if diarizer is None or speaker_centroids is None or grid_times is None:
        raise ValueError("diarizer, speaker_centroids, and grid_times are required")

    win_len = int(max(1, win_sec * sr))
    hop_len = int(max(1, hop_sec * sr))

    chunk_meta: List[Tuple[int, int]] = []
    all_chunks: List[np.ndarray] = []

    for c_idx, cand in enumerate(candidates):
        start_s = int(max(0.0, cand.window_start) * sr)
        end_s = int(min(float(len(waveform)) / float(sr), cand.window_end) * sr)
        if end_s <= start_s:
            continue
        offset = start_s
        local_idx = 0
        while offset < end_s:
            chunk = waveform[offset : offset + win_len]
            if chunk.size == 0:
                break
            all_chunks.append(chunk)
            chunk_meta.append((c_idx, local_idx))
            local_idx += 1
            offset += hop_len

    if not all_chunks:
        return []

    embeddings_all, valid_mask_all = extractor(all_chunks)
    if embeddings_all.shape[0] != len(all_chunks):
        LOGGER.warning(
            "[BOUNDARY-REFINE] embeddings count mismatch: %d vs %d",
            embeddings_all.shape[0],
            len(all_chunks),
        )
        return []

    results: List[ShortWindowResult] = []
    for c_idx, cand in enumerate(candidates):
        idxs = [i for i, meta in enumerate(chunk_meta) if meta[0] == c_idx]
        if not idxs:
            continue

        emb = embeddings_all[idxs]
        valid_mask = valid_mask_all[idxs]

        start_idx = int(np.searchsorted(grid_times, cand.window_start, side="left"))
        end_idx = int(np.searchsorted(grid_times, cand.window_end, side="right") - 1)
        if end_idx < start_idx:
            continue
        local_grid_indices = np.arange(start_idx, end_idx + 1, dtype=np.int32)
        local_grid_times = np.asarray(grid_times[local_grid_indices], dtype=np.float32)
        local_grid_times = local_grid_times - float(cand.window_start)

        spk_probs_local = diarizer.get_speaker_probabilities(
            emb,
            valid_mask,
            speaker_centroids,
            local_grid_times,
            hop_len,
            sr,
        )

        if spk_probs_local.size == 0:
            continue

        max_per_row = np.max(spk_probs_local, axis=1)
        mean_max = float(np.mean(max_per_row)) if max_per_row.size > 0 else 0.0
        p50 = float(np.quantile(max_per_row, 0.5)) if max_per_row.size > 0 else 0.0
        p95 = float(np.quantile(max_per_row, 0.95)) if max_per_row.size > 0 else 0.0
        ent = float(
            -np.mean(
                np.sum(spk_probs_local * np.log(np.clip(spk_probs_local, 1e-9, 1.0)), axis=1)
            )
        )
        stats = {
            "mean_max": mean_max,
            "p50_max": p50,
            "p95_max": p95,
            "mean_entropy": ent,
        }
        LOGGER.info(
            "[BOUNDARY-REFINE] short spk_probs stats: idx=%d mean_max=%.3f p50=%.3f p95=%.3f mean_entropy=%.3f",
            c_idx,
            mean_max,
            p50,
            p95,
            ent,
        )

        results.append(
            ShortWindowResult(
                candidate_idx=c_idx,
                embeddings=emb,
                grid_indices=local_grid_indices,
                grid_times=grid_times[local_grid_indices],
                spk_probs_local=spk_probs_local,
                stats=stats,
            )
        )

    return results


def _compute_margin(spk_probs: np.ndarray) -> np.ndarray:
    if spk_probs.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if spk_probs.shape[1] < 2:
        return np.ones((spk_probs.shape[0],), dtype=np.float32)
    top2 = np.partition(spk_probs, -2, axis=1)
    return (top2[:, -1] - top2[:, -2]).astype(np.float32)


def _build_span_candidates(
    candidates: List[BoundaryCandidate],
    *,
    offset_sec: float,
    split_window: bool,
    center_pad_sec: float,
    t_min: float,
    t_max: float,
) -> Tuple[List[BoundaryCandidate], List[int]]:
    span_candidates: List[BoundaryCandidate] = []
    index_map: List[int] = []
    for idx, cand in enumerate(candidates):
        spans: List[Tuple[str, float, float]] = []
        if split_window:
            left_start = cand.window_start
            left_end = min(cand.center_time - center_pad_sec, cand.window_end)
            if left_end > left_start:
                spans.append(("L", left_start, left_end))
            right_start = max(cand.center_time + center_pad_sec, cand.window_start)
            right_end = cand.window_end
            if right_end > right_start:
                spans.append(("R", right_start, right_end))
        else:
            spans.append(("C", cand.window_start, cand.window_end))

        for label, start, end in spans:
            span_start = max(t_min, start + offset_sec)
            span_end = min(t_max, end + offset_sec)
            if span_end <= span_start:
                continue
            span_candidates.append(
                BoundaryCandidate(
                    center_time=cand.center_time,
                    window_start=span_start,
                    window_end=span_end,
                    reason=f"{cand.reason}|{label}",
                )
            )
            index_map.append(idx)
    return span_candidates, index_map


def _accumulate_short_results(
    results: List[ShortWindowResult],
    index_map: List[int],
    cand_meta: List[Optional[Dict[str, np.ndarray]]],
    sum_map: List[Optional[np.ndarray]],
    count_map: List[Optional[np.ndarray]],
) -> None:
    for res in results:
        orig_idx = index_map[res.candidate_idx]
        meta = cand_meta[orig_idx]
        if meta is None:
            continue
        start_idx = int(meta["start_idx"])
        local_positions = res.grid_indices - start_idx
        if local_positions.size == 0:
            continue
        valid_mask = (local_positions >= 0) & (local_positions < meta["length"])
        if not np.any(valid_mask):
            continue
        local_positions = local_positions[valid_mask]
        sum_map[orig_idx][local_positions] += res.spk_probs_local[valid_mask]
        count_map[orig_idx][local_positions] += 1.0


def compute_short_window_multi_scale(
    waveform: np.ndarray,
    sr: int,
    candidates: List[BoundaryCandidate],
    extractor: Callable[[List[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
    config: AlignerConfig,
    *,
    diarizer,
    speaker_centroids: np.ndarray,
    grid_times: np.ndarray,
    spk_probs_global: np.ndarray,
    params: BoundaryRefineParams,
) -> List[ShortWindowResult]:
    if grid_times.size == 0 or spk_probs_global.size == 0:
        return []

    if params.multi_scale_enable:
        win_secs = list(params.multi_scale_win_secs)
        hop_secs = list(params.multi_scale_hop_secs)
    else:
        win_secs = [float(params.short_win_sec)]
        hop_secs = [float(params.short_hop_sec)]

    if len(win_secs) != len(hop_secs):
        raise ValueError("multi_scale_win_secs and multi_scale_hop_secs length mismatch")

    num_scales = len(win_secs)
    t_min = float(grid_times[0])
    t_max = float(grid_times[-1])

    cand_meta: List[Optional[Dict[str, np.ndarray]]] = []
    for cand in candidates:
        start_idx = int(np.searchsorted(grid_times, cand.window_start, side="left"))
        end_idx = int(np.searchsorted(grid_times, cand.window_end, side="right") - 1)
        if end_idx < start_idx:
            cand_meta.append(None)
            continue
        fallback = np.asarray(spk_probs_global[start_idx:end_idx + 1], dtype=np.float32)
        cand_meta.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "length": int(end_idx - start_idx + 1),
                "fallback": fallback,
                "grid_indices": np.arange(start_idx, end_idx + 1, dtype=np.int32),
            }
        )

    min_win = min(win_secs) if win_secs else 0.0
    scale_data: List[List[Optional[Dict[str, np.ndarray]]]] = []

    for scale_idx, (win_sec, hop_sec) in enumerate(zip(win_secs, hop_secs)):
        offsets = [0.0]
        if params.multi_scale_enable and win_sec == min_win:
            offsets = list(params.multi_scale_micro_avg_offsets_sec or [0.0])

        sum_map: List[Optional[np.ndarray]] = [None] * len(candidates)
        count_map: List[Optional[np.ndarray]] = [None] * len(candidates)
        for idx, meta in enumerate(cand_meta):
            if meta is None:
                continue
            sum_map[idx] = np.zeros_like(meta["fallback"], dtype=np.float32)
            count_map[idx] = np.zeros((meta["length"],), dtype=np.float32)

        for offset_sec in offsets:
            span_candidates, index_map = _build_span_candidates(
                candidates,
                offset_sec=float(offset_sec),
                split_window=bool(params.split_window_enable),
                center_pad_sec=float(params.split_window_center_pad_sec),
                t_min=t_min,
                t_max=t_max,
            )
            if not span_candidates:
                continue
            span_results = compute_short_window_embeddings(
                waveform=waveform,
                sr=sr,
                candidates=span_candidates,
                win_sec=float(win_sec),
                hop_sec=float(hop_sec),
                extractor=extractor,
                config=config,
                diarizer=diarizer,
                speaker_centroids=speaker_centroids,
                grid_times=grid_times,
            )
            _accumulate_short_results(span_results, index_map, cand_meta, sum_map, count_map)

        per_cand_data: List[Optional[Dict[str, np.ndarray]]] = []
        for idx, meta in enumerate(cand_meta):
            if meta is None:
                per_cand_data.append(None)
                continue
            mask = count_map[idx] > 0.0
            probs = np.asarray(meta["fallback"], dtype=np.float32).copy()
            if np.any(mask):
                probs[mask] = sum_map[idx][mask] / count_map[idx][mask][:, None]
            per_cand_data.append({"probs": probs, "mask": mask})
        scale_data.append(per_cand_data)

        LOGGER.info(
            "[BOUNDARY-REFINE] multi-scale scale=%d win=%.2f hop=%.2f offsets=%d",
            scale_idx,
            float(win_sec),
            float(hop_sec),
            len(offsets),
        )

    short_results: List[ShortWindowResult] = []
    vote_mode = str(params.multi_scale_vote_mode or "weighted").lower()
    weight_exp = float(params.multi_scale_vote_weight_exp)
    margin_th = float(params.multi_scale_vote_margin_th)
    effective_vote_min = max(1, min(int(params.multi_scale_vote_min), num_scales))
    for cand_idx, meta in enumerate(cand_meta):
        if meta is None:
            continue
        fallback = np.asarray(meta["fallback"], dtype=np.float32)
        window_len = int(meta["length"])
        final_probs = fallback.copy()
        valid_votes = np.zeros((window_len,), dtype=np.int32)

        if vote_mode == "approval":
            votes = np.full((num_scales, window_len), -1, dtype=np.int32)
            for scale_idx in range(num_scales):
                data = scale_data[scale_idx][cand_idx]
                if data is None:
                    continue
                probs = data["probs"]
                mask = data["mask"]
                margins = _compute_margin(probs)
                use_mask = mask & (margins >= margin_th)
                if np.any(use_mask):
                    votes[scale_idx, use_mask] = np.argmax(probs[use_mask], axis=1)
            valid_votes = np.sum(votes >= 0, axis=0).astype(np.int32)
            for i in range(window_len):
                if valid_votes[i] < effective_vote_min:
                    continue
                frame_votes = votes[:, i]
                frame_votes = frame_votes[frame_votes >= 0]
                if frame_votes.size == 0:
                    continue
                counts = np.bincount(frame_votes, minlength=fallback.shape[1])
                winner = int(np.argmax(counts))
                if counts[winner] / float(frame_votes.size) < (2.0 / 3.0):
                    continue
                sum_probs = np.zeros((fallback.shape[1],), dtype=np.float32)
                used = 0
                for scale_idx in range(num_scales):
                    if votes[scale_idx, i] == winner:
                        sum_probs += scale_data[scale_idx][cand_idx]["probs"][i]
                        used += 1
                if used > 0:
                    final_probs[i] = sum_probs / float(used)
        else:
            weighted_sum = np.zeros_like(fallback, dtype=np.float32)
            weight_sum = np.zeros((window_len,), dtype=np.float32)
            for scale_idx in range(num_scales):
                data = scale_data[scale_idx][cand_idx]
                if data is None:
                    continue
                probs = data["probs"]
                mask = data["mask"]
                margins = _compute_margin(probs)
                use_mask = mask & (margins >= margin_th)
                if not np.any(use_mask):
                    continue
                if weight_exp == 1.0:
                    weights = margins
                else:
                    weights = np.power(margins, weight_exp)
                weights = np.where(use_mask, weights, 0.0).astype(np.float32)
                weighted_sum += probs * weights[:, None]
                weight_sum += weights
                valid_votes += use_mask.astype(np.int32)
            valid_mask = (weight_sum > 0.0) & (valid_votes >= effective_vote_min)
            if np.any(valid_mask):
                final_probs[valid_mask] = weighted_sum[valid_mask] / weight_sum[valid_mask][:, None]

        max_per_row = np.max(final_probs, axis=1)
        mean_max = float(np.mean(max_per_row)) if max_per_row.size > 0 else 0.0
        p50 = float(np.quantile(max_per_row, 0.5)) if max_per_row.size > 0 else 0.0
        p95 = float(np.quantile(max_per_row, 0.95)) if max_per_row.size > 0 else 0.0
        ent = float(
            -np.mean(np.sum(final_probs * np.log(np.clip(final_probs, 1e-9, 1.0)), axis=1))
        )
        stats = {
            "mean_max": mean_max,
            "p50_max": p50,
            "p95_max": p95,
            "mean_entropy": ent,
            "valid_votes": int(np.sum(valid_votes >= effective_vote_min)),
        }
        LOGGER.info(
            "[BOUNDARY-REFINE] multi-scale candidate idx=%d mode=%s valid_frames=%d/%d",
            cand_idx,
            vote_mode,
            stats["valid_votes"],
            window_len,
        )

        short_results.append(
            ShortWindowResult(
                candidate_idx=cand_idx,
                embeddings=np.empty((0,)),
                grid_indices=meta["grid_indices"],
                grid_times=grid_times[meta["grid_indices"]],
                spk_probs_local=final_probs,
                stats=stats,
            )
        )

    return short_results


def run_boundary_refine(
    *,
    aligner,
    frame_path: np.ndarray,
    grid_times: np.ndarray,
    spk_probs_global: np.ndarray,
    vad_probs: np.ndarray,
    word_info: List[dict],
    waveform: np.ndarray,
    sr: int,
    diarizer,
    speaker_centroids: np.ndarray,
    embed_fn: Callable[[List[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
    backend: str,
    config: AlignerConfig,
    switch_events: Optional[List[Dict[str, float]]] = None,
) -> BoundaryRefineResult:
    params = _resolve_params(config)
    if not getattr(config, "boundary_refine_enable", False):
        return BoundaryRefineResult(
            original_path=frame_path,
            refined_path=frame_path,
            applied_windows=[],
            skipped_windows=[],
        )

    required_backend = getattr(config, "boundary_refine_backend_required", None)
    if required_backend and str(backend).lower() != str(required_backend).lower():
        LOGGER.info(
            "[BOUNDARY-REFINE] backend mismatch: required=%s actual=%s -> skip",
            required_backend,
            backend,
        )
        return BoundaryRefineResult(
            original_path=frame_path,
            refined_path=frame_path,
            applied_windows=[],
            skipped_windows=[],
        )

    candidates = extract_boundary_candidates(
        frame_path=frame_path,
        grid_times=grid_times,
        word_info=word_info,
        config=config,
        switch_events=switch_events,
    )
    if not candidates:
        return BoundaryRefineResult(
            original_path=frame_path,
            refined_path=frame_path,
            applied_windows=[],
            skipped_windows=[],
        )

    use_multi_scale = bool(getattr(params, "multi_scale_enable", False))
    use_split = bool(getattr(params, "split_window_enable", False))
    if use_split:
        LOGGER.info(
            "[BOUNDARY-REFINE] split-window enabled: pad_sec=%.2f",
            float(params.split_window_center_pad_sec),
        )
    if use_multi_scale:
        LOGGER.info(
            "[BOUNDARY-REFINE] multi-scale enabled: scales=%d win=%s hop=%s mode=%s",
            len(params.multi_scale_win_secs),
            params.multi_scale_win_secs,
            params.multi_scale_hop_secs,
            params.multi_scale_vote_mode,
        )

    if use_multi_scale or use_split:
        short_results = compute_short_window_multi_scale(
            waveform=waveform,
            sr=sr,
            candidates=candidates,
            extractor=embed_fn,
            config=config,
            diarizer=diarizer,
            speaker_centroids=speaker_centroids,
            grid_times=grid_times,
            spk_probs_global=spk_probs_global,
            params=params,
        )
    else:
        short_results = compute_short_window_embeddings(
            waveform=waveform,
            sr=sr,
            candidates=candidates,
            win_sec=float(params.short_win_sec),
            hop_sec=float(params.short_hop_sec),
            extractor=embed_fn,
            config=config,
            diarizer=diarizer,
            speaker_centroids=speaker_centroids,
            grid_times=grid_times,
        )
    short_map = {res.candidate_idx: res for res in short_results}

    refined = np.asarray(frame_path, dtype=np.int32).copy()
    applied: List[BoundaryCandidate] = []
    skipped: List[BoundaryCandidate] = []

    for idx, cand in enumerate(candidates):
        if not params.use_local_dp:
            skipped.append(cand)
            continue
        res = short_map.get(idx)
        if res is None or res.spk_probs_local.size == 0:
            skipped.append(cand)
            continue

        start_idx = int(np.searchsorted(grid_times, cand.window_start, side="left"))
        end_idx = int(np.searchsorted(grid_times, cand.window_end, side="right") - 1)
        if end_idx <= start_idx:
            skipped.append(cand)
            continue

        window_len = int(end_idx - start_idx + 1)
        if window_len < int(params.local_dp_min_frames):
            skipped.append(cand)
            continue

        fixed_start = int(refined[start_idx])
        fixed_end = int(refined[end_idx])
        refined = aligner.local_dp_refine(
            base_path=refined,
            word_info=word_info,
            vad_probs=vad_probs,
            spk_probs_global=spk_probs_global,
            spk_probs_local=res.spk_probs_local,
            local_grid_indices=res.grid_indices,
            grid_times=grid_times,
            start_idx=start_idx,
            end_idx=end_idx,
            fixed_start_speaker=fixed_start,
            fixed_end_speaker=fixed_end,
        )
        applied.append(cand)

    return BoundaryRefineResult(
        original_path=frame_path,
        refined_path=refined,
        applied_windows=applied,
        skipped_windows=skipped,
    )
