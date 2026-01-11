# AGENTS.md

This repository is operated with Codex CLI (and other coding agents). Treat this file as the persistent contract.

> Language policy (user-facing): respond in the user's language by default (Japanese if the user is writing Japanese; English if the user is writing English).  
> 内部メモやコードコメントは英語でも可。ユーザー向けの最終説明は原則ユーザー言語に合わせる。

---

## 0) Prime directive (source of truth)
- The source of truth is the filesystem + git history, not chat history.
- Prefer short sessions. If context grows, start a new conversation and resume from `git status` + `git diff`.

## 1) Project intent (one-liner)
- Maintain and improve an offline/local-first Japanese ASR + diarization pipeline (AHE-Whisper) with reproducible runs and measurable quality/performance.

## 2) Authority / precedence (conflicts)
When instructions conflict, follow this order:
1) Explicit user instructions in the current conversation (including scope constraints)
2) This `AGENTS.md`
3) The filesystem + git history (what actually exists in the repo)
4) Supplemental docs (e.g., `CLAUDE.md`, `.agent/PLANS.md`, `README.md`)
If something is ambiguous and could change behavior/outputs materially, ask the user rather than guessing.

## 3) Scope guardrails
### Allowed
- Read/write files inside this repo.
- Create new small utility scripts, docs, or config files when it improves reproducibility.

### Disallowed (unless explicitly requested)
- Editing anything outside the repo (home dir, system files, global config, `~/.codex`, etc.).
- Downloading large models / datasets, or changing model binaries.
- Destructive ops (delete/overwrite large directories) without showing the exact plan + diff.

### Exception clause: `prefetch_models` and model availability
There is a deliberate tension between “don’t download large models” and “the project needs models to run”.
- If the user explicitly asks to run/verify something that requires models, it is allowed to run the repo’s model prefetch helper (commonly named `prefetch_models.py` or `tools/prefetch_models.py`) **only after**:
  - explaining what will be fetched (roughly), where it will be stored, and that binaries must not be committed; and
  - getting an explicit “go ahead” from the user if the download is large or could be slow/expensive.
- If models are already present locally, prefer reusing them; do not re-download “just in case”.

### Repo-specific notes
- SSOT config path: `ahe_whisper/config.py` (NOT a repo-root `config.py`).
- Keep long outputs/logs out of chat: write them to files under the output/run directory and reference paths.

## 4) Repo conventions (paths & artifacts)
- Evaluation / reference data directory: `data/`.
- Output root directory: `AHE_Whisper_output/` (underscore).
- Outputs should be organized per run (e.g., `AHE_Whisper_output/<run_id>_YYYYMMDD_HHMMSS/`).
- Do NOT paste huge logs into chat. Save and reference them:
  - `command | tee -a AHE_Whisper_output/<run_dir>/run.log`
  - Save structured debug artifacts (JSON/plots) inside the run directory.

## 5) Workflow (agent loop)
### 5.1 Trivial vs non-trivial
- Trivial (typos, small doc edits, obvious single-line fixes): you may skip a formal plan, but still keep diffs minimal and do quick verification if behavior could change.
- Non-trivial (logic changes, refactors, anything touching pipeline behavior/metrics): follow the full loop below.

### 5.2 Gather context (minimum)
- Check working tree first:
  - `git status`
  - `git diff` (or Codex `/diff`)
- Open only the files needed to complete the task.
- If the task is about behavior/metrics, locate the latest run artifacts (logs, performance files, JSONs) under `AHE_Whisper_output/` and read them from disk.

### 5.3 Plan (keep it short)
State:
- What you will change (files + functions)
- Why
- How you will verify (exact commands)
- Any risks / unknowns
If key assumptions are uncertain, ask the user.

### 5.4 Act (small, reviewable steps)
- Prefer minimal diffs.
- Keep changes localized.
- If touching core pipeline logic, add/adjust diagnostics so regressions remain observable.

### 5.5 Verify (required for code changes)
- If there is an existing project command, use it.
- Otherwise use best-effort verification:
  - Run a minimal repro command
  - Validate outputs exist and basic invariants hold (no empty outputs, timestamps monotonic where expected, etc.)

### 5.6 Summarize (required)
Provide:
- What changed
- Why it changed
- How it was verified (exact commands)
- Any risks / remaining unknowns
If you modified code, show the diff (or instruct to use `/diff`).

## 6) Git usage (commits & history)
- Committing is generally safe and encouraged for traceability.
- Do not push unless the user explicitly requests it.
- Agents can and should inspect git history when useful (`git log`, `git show`, `git blame`), but the authoritative state is always what exists locally in the working tree + history.
- Prefer small commits with concrete messages. If experimenting, use a branch.

## 7) Resumption protocol (new conversation template)
Use this when starting a new chat to avoid long-context degradation:

```text
We are resuming from repo state (chat history is intentionally discarded).

Please:
1) run `git status` and inspect `git diff` (or use `/diff`)
2) summarize the current working tree changes
3) continue the task: <TASK>

Verification (exact commands to run): <HOW TO VERIFY>
Notes / constraints: <OPTIONAL>
```

## 8) What to write in “Verification: <HOW TO VERIFY>”
Write copy/paste-able commands that would convince you the change worked.

Examples (adapt to the repo’s actual commands/paths):
- If UI / basic startup:
  - `uv run python gui.py`
- If boundary diagnostics are relevant:
  - `AHE_BOUNDARY_SNAP=1 uv run python gui.py`
- If evaluating diarization output against a gold transcript:
  - `uv run python tools/eval_two_speaker_from_transcripts.py data/gold/<gold>.txt AHE_Whisper_output/<run_dir>/<out>.txt --host-speakers ...`

## 9) Optional: MICRO_SNAPSHOT (only when explicitly requested by the user)
Append exactly this template when the user asks:

### MICRO_SNAPSHOT
- Goal:
- Scope (allowed files):
- Latest run:
  - Command:
  - Key metrics:
  - Observations:
  - Conclusion:
- Delta (what changed since previous snapshot):
- Open questions (<=3):
- Next actions (<=3):

---

## Appendix A: Supplemental docs (references)

If anything there conflicts with Sections 0–9 above, Sections 0–9 win.

- `CLAUDE.md` — architecture notes and diagnostic procedures
- `.agent/PLANS.md` — ExecPlan format and rules
