# AHE-Whisper — Changelog


## [v90.99] — 2025-12-28

**Advanced Aligner Edition**

🧩 *Core*
* **Word-level Viterbi DP** (`enable_word_dp`): Runs speaker optimization at word boundaries instead of frame-level, reducing "word spillover" errors at speaker transitions.
* **Switch-CP Gating** (`switch_cp_k`): Uses posterior change-point detection to reduce switch penalty at natural embedding boundaries.
* **CP-Floor Guard** (`switch_cp_floor_uncertain`): Prevents excessive penalty reduction when posterior is uncertain, avoiding A→B→A flip-backs.
* **Posterior-Margin Gate** (`switch_post_k`): Reduces switch penalty when transitioning into a speaker with high posterior confidence.
* **Lexical Prior** (`use_lexical_prior`): Punctuation-aware switch penalty — lower after `？` (question) or `。` (period).
* **Beta_t Time-varying Weight** (`use_beta_t`): Anti-dictatorship mechanism that reduces embedding influence when one speaker dominates excessively.
* **Posterior Temperature** (`posterior_temperature`): Softmax-style calibration to control posterior peakiness.
* **Silence-based Run-length Reset** (`use_silence_runlen_reset`): Resets run-length after silence to ease speaker switching.
* **Posterior-based Boundary Rollback** (`use_posterior_refine`): Local realignment using posteriors to refine DP boundaries.

🧪 *Diagnostics*
* Added **Boundary Snap Export** (`AHE_BOUNDARY_SNAP=1`): Exports detailed frame-level diagnostics around speaker switches for debugging.
* Added `tools/analyze_boundary_snap.py`: Analyzes boundary snap JSONL files to detect low-CP switches and stuck candidates.

📦 *Config*
* New `AlignerConfig` fields: `switch_cp_floor_uncertain`, `switch_post_k`, `switch_post_margin_th`, `enable_word_dp`
* Updated defaults: `delta_switch=1.2`, `switch_cp_k=1.5`, `switch_post_k=2.0`


## [v90.98] — 2025-12-08

🧩 *Core*
* Added configurable **speaker embedding backends**:
  * `titanet` (default): sherpa-onnx `nemo_en_titanet_large.onnx`
  * `campplus`: 3D-Speaker `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`
  * `speakernet` (experimental): sherpa-onnx `nemo_en_speakerverification_speakernet.onnx`
* Switched default embedding backend from ECAPA-TDNN-512 to **Titanet-L**, based on internal evaluations on three two-speaker Japanese interview datasets (see `EVAL_embedding_backend.md`).

🧪 *Eval*
* Documented A/B results between Titanet and CampPlus in `EVAL_embedding_backend.md`.


## [v90.97]— 2025-11-12
 * Added Smooth Aligner (α=0.40, γ=1.4) — fully stable across 3 runs (Jaccard=0.99)

## [v90.96] — 2025-11-10

**Experimental Branch — `feature/v90.96_local-tau-smooth-align`**

🧩 *Core*
* Introduced **local τ (temperature) scheduler** for dynamic softmax scaling based on similarity variance.
* Added **post-sharping (γ=1.3)** and entropy monitoring for stable speaker probability contrast.
* Integrated **probability smoothing (EMA / local window)** to reduce speaker-flutter and over-segmentation.
* Implemented **temporary storage & diagnostic release** (`last_probs`) for memory-efficient debugging.

⚙️ *Perf*
* Improved diarization precision while maintaining RTF ≈ 0.19.
* Aligner now exhibits natural speaker transitions comparable to v34.02, with enhanced stability.

📦 *Infra*
* Added early-memory release hook (`del self.last_probs`) to limit peak memory footprint.
* Branch: `feature/v90.96_local-tau-smooth-align`


## [v90.90] — 2025-11-08

**Definitive Release**

* Consolidated all modules into unified `setup_ahe_whisper.py`
* Introduced hybrid VAD-ASR inference with OverlapDPAligner
* Integrated WeSpeaker ECAPA-TDNN512 (ONNX) for embeddings
* Optimized MLX Whisper inference (RTF ≈ 0.27 on M4 24 GB)
* Improved deterministic session cache handling
* Finalized GUI layer (niceGUI v9.5)
* Added `.gitignore` for model and output exclusion
* Verified reproducibility across M-series Macs

---

## [v83.12] — 2025-09-14

* Added adaptive energy normalization for Silero-VAD
* Enhanced diarization stability on multi-speaker audio
* Introduced internal logging pipeline (`TRACE-ALIGNER` mode)

---

## [v75.02] — 2025-07-03

* Transitioned to ONNXRuntime for embedding inference
* Added batch segment alignment with DP post-processing

---

## [v71.00] — 2025-05-11

* Implemented MLX-based WhisperKit backend
* Introduced `prefetch_models.py` for offline model setup

---

## [v34.00] — 2024-11-20

* Initial integration of VAD × ASR hybrid pipeline
* Prototype GUI (v3.0) and config generator

---

### Legend

🧩 *Core* = major architecture or engine changes
⚙️ *Perf* = speed or memory optimization
🧠 *UX/UI* = interface or usability update
📦 *Infra* = environment or build-related change
