# AHE-Whisper

**Adaptive Hybrid Engine for Whisper — v90.99 (Advanced Aligner Edition)**
Author: Matsuura Jun
Date: December 2025

---

## 🧭 Overview

**AHE-Whisper** is an Apple-Silicon-optimized, end-to-end transcription and speaker-diarization engine.
It integrates multiple ASR, VAD, and embedding pipelines — each modular and swappable — to achieve **maximum speed, accuracy, and reproducibility** for broadcast production workflows.

---

## ⚙️ Core Components

| Module        | Role                            | Model                                  |
| ------------- | ------------------------------- | -------------------------------------- |
| **ASR**       | Speech recognition              | Whisper Large-v3-Turbo (MLX-optimized) |
| **VAD**       | Voice activity detection        | Silero-VAD (ONNX)                      |
| **Embedding** | Speaker representation          | Titanet-L (sherpa-onnx, default), CampPlus (optional) |
| **Aligner**   | OverlapDPAligner + Word-DP      | Custom (see below)                     |
| **Clusterer** | Attractor-based deep clustering | Soft-EM ADC                            |
| **Frontend**  | niceGUI-based web UI            | Python 3.12 / MLX stack                |

---

## 🎯 Aligner Features

The **OverlapDPAligner** is the core of speaker boundary detection. Key features:

### Word-level DP (`enable_word_dp`)
Runs Viterbi optimization at word boundaries instead of frame-level, reducing "word spillover" errors.

### Switch-CP Gating (`switch_cp_k`)
Uses posterior change-point (embedding movement) to reduce switch penalty at natural boundaries.

### Posterior-Margin Gate (`switch_post_k`)
When switching into a speaker with high posterior confidence, switch penalty is reduced.

### Lexical Prior (`use_lexical_prior`)
Punctuation-aware switch: lowers penalty after `？` (question) or `。` (period).

### Beta_t Time-varying Weight (`use_beta_t`)
Anti-dictatorship mechanism: reduces speaker posterior weight when one embedding dominates excessively.

### Posterior Temperature (`posterior_temperature`)
Softmax-style calibration to control posterior peakiness (higher = flatter distribution).

### Boundary Snap Export
Environment-variable controlled diagnostic export (`AHE_BOUNDARY_SNAP=1`) for debugging alignment issues.

---

## 🔊 Embedding Backends

Supports multiple ONNX speaker embedding backends:

- `titanet` (default): sherpa-onnx `nemo_en_titanet_large.onnx`
- `campplus`: 3D-Speaker `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`
- `speakernet` (experimental): sherpa-onnx `nemo_en_speakerverification_speakernet.onnx`

Switch via `EmbeddingConfig.backend` in the config.

---

## 🧩 Directory Structure

```
AHE-Whisper/
├── ahe_whisper/               # Core engine modules
│   ├── aligner.py             # OverlapDPAligner + Word-DP
│   ├── config.py              # All configuration dataclasses
│   ├── diarizer.py            # Speaker clustering
│   ├── embedding.py           # Speaker embedding backends
│   ├── exporter.py            # SRT/JSON/TXT export
│   ├── pipeline.py            # Main transcription pipeline
│   └── ...
├── tools/                     # Diagnostics & evaluation
│   ├── analyze_boundary_snap.py  # Boundary event analyzer
│   └── ...
├── models/                    # (ignored) local model cache
├── gui.py                     # GUI launcher
├── main.py                    # CLI entrypoint
├── pyproject.toml             # Dependencies (uv)
└── README.md                  # ← You are here
```

---

## 🚀 Setup (Apple Silicon)

Using **uv** (recommended):

```bash
cd AHE-Whisper
uv sync
uv run python main.py
```

Or with venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python main.py
```

---

## 🧠 Key Features

* 🔹 MLX-accelerated Whisper inference (M-series optimized)
* 🔹 Adaptive Hybrid Engine: automatic VAD × ASR integration
* 🔹 OverlapDPAligner with Word-level DP for precise speaker boundaries
* 🔹 Change-point & posterior-margin aware switch penalty
* 🔹 Lexical-based switch prior (punctuation-aware)
* 🔹 Speaker-aware transcription export (`.srt`, `.json`, `.txt`)
* 🔹 Boundary snap diagnostic export for debugging
* 🔹 Offline execution (no external API dependency)

---

## ⚙️ Configuration

All configs are dataclasses in `ahe_whisper/config.py`:

| Config              | Purpose                                |
| ------------------- | -------------------------------------- |
| `TranscriptionConfig` | Whisper model, language, thresholds  |
| `EmbeddingConfig`   | Embedding backend, dimensions          |
| `DiarizationConfig` | Clustering parameters, VAD thresholds  |
| `AlignerConfig`     | DP weights, switch penalties, word-DP  |
| `ExportConfig`      | Output formats, boundary snap          |

---

## 📄 License

All model files follow their respective upstream licenses (OpenAI, ONNX-Community, WeSpeaker).
Custom code © 2025 Matsuura Jun.
This repository is intended for internal R&D use and not for model redistribution.

---

## 🏷️ Version

Current: **v90.99 — Advanced Aligner Edition**
Date: 2025-12-28

> Focus: Word-level DP, switch-CP gating, posterior-margin aware switching, lexical priors
