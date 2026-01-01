# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AHE-Whisper v90.99** is an Apple Silicon-optimized end-to-end transcription and speaker-diarization engine for broadcast production workflows. The system is designed for maximum speed, accuracy, and reproducibility, particularly optimized for Japanese language processing.

**Core Technology Stack:**
- ASR: MLX-Whisper (Large-v3-Turbo, Apple Silicon optimized)
- VAD: Silero-VAD (ONNX)
- Embeddings: Titanet-L (default), CampPlus, or Speakernet
- Clustering: Soft-EM Attractor-based Deep Clustering
- UI: niceGUI web interface with multiprocess worker architecture

## Development Commands

### Environment Setup
```bash
# Using uv (recommended for reproducible builds)
uv sync

# Or using venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Running the Application
```bash
# Launch web UI (recommended)
uv run python gui.py

# Or with activated venv
python gui.py

# Pre-download all models for offline use
python prefetch_models.py
```

### Diagnostic Tools
```bash
# Run diagnostic scripts from tools/ directory
python tools/test_nemo_titanet_large.py
python tools/diagnose_embeddings_v90.py
python tools/analyze_boundary_snap.py

# Enable boundary snap diagnostics (environment variable)
AHE_BOUNDARY_SNAP=1 python gui.py

# Enable aligner tracing
AHE_TRACE_ALIGNER=1 python gui.py
```

**Note:** This project does not have a formal test suite. Testing is done through diagnostic scripts in `tools/` and evaluation scripts that compare output against reference datasets in `date/`.

## Architecture

### Core Pipeline Flow
The transcription pipeline (`ahe_whisper/pipeline.py`) orchestrates these sequential steps:

1. **Audio Loading** → librosa (16 kHz mono)
2. **VAD** → Silero-VAD produces frame-level speech probability
3. **ASR** → MLX-Whisper generates word-level transcription with timestamps
4. **Embedding Extraction** → Titanet/CampPlus extracts speaker embeddings (1.5s windows, 0.75s hop)
5. **Clustering** → Soft-EM ADC assigns speaker labels to embedding frames
6. **Alignment** → OverlapDPAligner optimizes speaker boundaries using Viterbi DP
7. **Post-processing** → Island removal, high-density smoothing, segment merging
8. **Export** → SRT/JSON/TXT with optional punctuation restoration

### Key Modules

**`ahe_whisper/pipeline.py` (884 lines)**
- `AHEPipeline` class: Main orchestration logic
- Manages end-to-end flow from audio input to final export
- Handles metrics calculation and coverage analysis

**`ahe_whisper/aligner.py` (1,614 lines)**
- `OverlapDPAligner`: Core speaker boundary detection engine
- Implements word-level Viterbi DP optimization
- Advanced features:
  - Switch-CP gating: Uses embedding change-points to reduce switch penalty
  - Posterior-margin gate: Anti-dictatorship mechanism for high-confidence speakers
  - Lexical prior: Punctuation-aware switching (lower penalty after `？` or `。`)
  - Beta_t weighting: Time-varying speaker posterior weight
  - Silence-based run-length reset
  - Posterior refinement: Local realignment after DP

**`ahe_whisper/embedding.py` (644 lines)**
- Three backend implementations: `TitanetBackend`, `CampplusBackend`, `SpeakernetBackend`
- All use ONNX runtime for inference
- Configurable via `EmbeddingConfig.backend` in config

**`ahe_whisper/diarizer.py` (472 lines)**
- Soft-EM attractor-based clustering
- KMeans initialization with mass filtering and centroid merging
- Handles 2-3 speakers (configurable via `min_speakers`/`max_speakers`)

**`ahe_whisper/config.py` (318 lines)**
- All configuration is defined as nested dataclasses:
  - `TranscriptionConfig` - Whisper model settings
  - `EmbeddingConfig` - Backend selection and embedding parameters
  - `DiarizationConfig` - Clustering and post-processing
  - `AlignerConfig` - 30+ DP parameters for boundary optimization
  - `VadConfig` - Silero-VAD settings
  - `ExportConfig` - Output formats and boundary snap diagnostics
- Includes validation in `__post_init__()` methods

**`ahe_whisper/post_diar.py` (469 lines)**
- Post-processing: island removal, high-density smoothing
- Merges short A-B-A segments into single speaker regions
- Optional 30-second sliding window analysis for switch density anomalies

**`ahe_whisper/exporter.py` (693 lines)**
- Exports to SRT, JSON, TXT formats
- Optional Japanese punctuation restoration via XLM-Roberta
- Boundary snap diagnostic export when `AHE_BOUNDARY_SNAP=1`

**`ahe_whisper/pipeline_worker.py` (440 lines)**
- Multiprocess worker loop for GUI integration
- Communicates via queues: `job_q`, `result_q`, `log_q`
- Isolates heavy ML computation from UI thread

**`gui.py` (140 lines)**
- niceGUI-based web interface
- Spawns background worker process on startup
- Real-time log streaming and result display

### Configuration Architecture

All pipeline behavior is controlled through dataclass configurations in `config.py`. Key patterns:

- **Immutable dataclasses**: All configs are frozen after creation
- **Validation**: `__post_init__()` methods validate parameter ranges
- **Nested structure**: `AppConfig` contains all sub-configs
- **Serialization**: Uses `dacite` for YAML ↔ dataclass conversion

Example of critical aligner parameters:
```python
@dataclass
class AlignerConfig:
    delta_switch: float = 1.2          # Base speaker switch penalty
    switch_cp_k: float = 1.5            # Change-point gating strength
    switch_post_k: float = 2.0          # Posterior-margin gating strength
    posterior_temperature: float = 2.0  # Softmax calibration
    enable_word_dp: bool = True         # Word-level DP optimization
    use_beta_t: bool = True             # Anti-dictatorship mechanism
    use_lexical_prior: bool = True      # Punctuation-aware switching
```

### Multiprocess Architecture

The GUI uses a producer-consumer pattern with multiprocessing:

```
┌──────────────┐         ┌──────────────┐
│   Main GUI   │  job_q  │    Worker    │
│   Process    ├────────>│   Process    │
│  (nicegui)   │<────────┤  (pipeline)  │
│              │ result_q│              │
│              │<────────┤              │
│              │  log_q  │              │
└──────────────┘         └──────────────┘
```

- **Job Queue**: Sends audio file paths and configs to worker
- **Result Queue**: Returns transcription results to GUI
- **Log Queue**: Streams real-time logging to browser console
- **Thread safety**: Log buffer uses `deque(maxlen=5000)` with background thread polling

### Embedding Backend Switching

The system supports three embedding backends (configured in `EmbeddingConfig`):

1. **`titanet`** (default): `sherpa-onnx` with `nemo_en_titanet_large.onnx`
   - 192-dimensional embeddings
   - Highest accuracy in Japanese interview evaluations

2. **`campplus`**: `3D-Speaker` with `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`
   - Multilingual (ZH-EN)
   - Original default before v90.98

3. **`speakernet`**: `sherpa-onnx` with `nemo_en_speakerverification_speakernet.onnx`
   - Experimental

All backends implement the same interface in `embedding.py` and are initialized lazily.

## Important Implementation Details

### Word-level DP vs Frame-level DP

The aligner runs Viterbi DP optimization at **word boundaries** instead of audio frames when `enable_word_dp=True` (default). This significantly reduces "word spillover" where a single word is split between two speakers.

Implementation: `aligner.py:OverlapDPAligner._run_word_dp()` post-processes frame-level DP output by re-optimizing transitions at word boundaries using aggregated costs.

### Anti-Dictatorship Mechanism (Beta_t)

When one speaker's embedding dominates excessively (high posterior confidence over long duration), the `beta_t` weight reduces the influence of speaker posteriors in the DP cost function. This prevents the aligner from getting "stuck" on a single speaker.

Configured via `AlignerConfig.use_beta_t`, `beta_t_min`, `beta_t_decay_factor`.

Implementation: `aligner.py` line ~800-900, calculates time-varying weight based on embedding dominance.

### Boundary Snap Diagnostics

Set `AHE_BOUNDARY_SNAP=1` environment variable to export detailed frame-level diagnostics around speaker switches. This creates JSONL files in the output directory with:
- Frame-level VAD scores
- Speaker posteriors
- DP costs and transitions
- Change-point metrics

Analyze with: `python tools/analyze_boundary_snap.py output_dir/boundary_snap.jsonl`

### Model Caching

Models are cached in `models/` directory (gitignored). The `model_manager.py` handles automatic downloading via HuggingFace Hub.

To pre-download all models: `python prefetch_models.py`

Models include:
- `mlx-community/whisper-large-v3-turbo` (Whisper ASR)
- `k2-fsa/sherpa-onnx-*` (Titanet, Speakernet)
- `iic/SenseVoiceSmall` (VAD alternative)
- `3D-Speaker` CampPlus models
- XLM-Roberta punctuation models

### Japanese Language Support

The system is optimized for Japanese:
- Default language: `ja` in `TranscriptionConfig`
- Sudachi tokenizer for morphological analysis
- Punctuation restoration tuned for Japanese punctuation: `。` (period), `、` (comma), `？` (question)
- Lexical prior in aligner recognizes `？` and `。` as natural speaker switch points

## Common Development Workflows

### Adding a New Embedding Backend

1. Create new backend class in `ahe_whisper/embedding.py` implementing the same interface as `TitanetBackend`
2. Add backend initialization in `_get_embedding_backend()` factory function
3. Add model download logic in `prefetch_models.py`
4. Update `EmbeddingConfig.backend` documentation in `config.py`
5. Test with diagnostic script: `python tools/diagnose_embeddings_v90.py`

### Tuning Aligner Parameters

1. Edit default values in `AlignerConfig` dataclass (`config.py`)
2. Run with diagnostic: `AHE_BOUNDARY_SNAP=1 python gui.py`
3. Analyze boundaries: `python tools/analyze_boundary_snap.py AHE-Whisper-output/*/boundary_snap.jsonl`
4. Compare against reference outputs in `date/` directory
5. Document changes in `CHANGELOG.md`

### Debugging Multiprocess Issues

The GUI uses multiprocessing which can make debugging tricky:
- Worker process logs go to `log_q` → check browser console
- Add print statements in `pipeline_worker.py:worker_process_loop()`
- For synchronous debugging, run pipeline directly:
  ```python
  from ahe_whisper.pipeline import AHEPipeline
  from ahe_whisper.config import AppConfig

  config = AppConfig()  # Load defaults
  pipeline = AHEPipeline(config)
  results = pipeline.run("path/to/audio.wav")
  ```

## Output Structure

Results are stored in `AHE-Whisper-output/YYYYMMDD_HHMMSS_<filename>/`:
- `transcript.srt` - SRT subtitle format with speaker labels
- `transcript.json` - Detailed JSON with word-level timing
- `transcript.txt` - Plain text transcript
- `metadata.json` - Pipeline configuration and metrics
- `boundary_snap.jsonl` - Frame-level diagnostics (if enabled)

## Key Dependencies

- **Python 3.12.4** (specified in `.python-version` and `pyproject.toml`)
- **uv** for package management (reproducible builds via `uv.lock`)
- **MLX** framework for Apple Silicon acceleration
- **ONNX Runtime** (1.17.0-1.19.0) for embedding inference
- **sherpa-onnx** (1.12.18+) for speaker verification models

## Performance Characteristics

- **Real-time Factor**: ~0.19-0.27 on M4 24GB
- **Memory**: ~2-4GB peak during inference
- **Batch Processing**: Embedding extraction uses batching with `batch_cap=16`
- **Offline Capable**: All models can be pre-fetched for air-gapped execution

## Notes for Future Development

- The aligner (`aligner.py`) is the most complex component with 1,614 lines. Modifications should be thoroughly tested with boundary snap diagnostics.
- All configs use dataclass validation - never bypass `__post_init__()` checks.
- When adding features, follow the existing pattern: config param → pipeline logic → exporter output.
- The project uses Japanese and English mixed comments - preserve this pattern for consistency.
- Version history is meticulously tracked in `CHANGELOG.md` - maintain this for all significant changes.
