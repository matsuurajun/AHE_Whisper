# -*- coding: utf-8 -*-
import io
import json
import logging
import secrets
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, List

from nicegui import ui

from ahe_whisper.config import AppConfig

LOGGER = logging.getLogger(__name__)

ALLOWED_AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.flac'}
MAX_UPLOAD_BYTES = 1_000_000_000

AHE_LOGO_SVG = """
<svg width="100" height="100" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="#4A5568" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M2 17L12 22L22 17" stroke="#4A5568" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M2 12L12 17L22 12" stroke="#4A5568" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

@dataclass
class AppState:
    upload_button: Optional[ui.upload] = None
    start_button: Optional[ui.button] = None
    status_label: Optional[ui.label] = None
    log_area: Optional[ui.log] = None
    result_link_container: Optional[ui.row] = None
    settings_container: Optional[ui.expansion] = None
    
    input_ready: bool = False
    
    audio_file_path: Optional[Path] = None
    config: AppConfig = field(default_factory=AppConfig)
    temp_files: List[Path] = field(default_factory=list)

    def cleanup_temp_files(self) -> None:
        for path in self.temp_files:
            try:
                if path.exists():
                    path.unlink()
            except Exception as e:
                LOGGER.warning(f"Failed to clean up temp file {path}: {e}")
        self.temp_files.clear()

def handle_file_upload(e: Any, state: AppState) -> None:
    if not e.content:
        return
    
    suffix = Path(e.name).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTS:
        ui.notify(f'Unsupported file type: {suffix}', color='negative')
        return

    size = None
    try:
        if hasattr(e.content, 'seek') and hasattr(e.content, 'tell'):
            current_pos = e.content.tell()
            e.content.seek(0, io.SEEK_END)
            size = e.content.tell()
            e.content.seek(current_pos)
    except (OSError, io.UnsupportedOperation):
        pass

    if size is not None and size > MAX_UPLOAD_BYTES:
        ui.notify('File too large (limit 1GB).', color='negative')
        return

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            CHUNK_SIZE = 4 * 1024 * 1024
            while True:
                chunk = e.content.read(CHUNK_SIZE)
                if not chunk:
                    break
                temp_file.write(chunk)
            
            tmp_path = Path(temp_file.name)
            state.audio_file_path = tmp_path
            state.temp_files.append(tmp_path)
    except Exception as ex:
        ui.notify(f'Upload failed: {ex}', color='negative')
        LOGGER.error(f"File upload error: {ex}", exc_info=True)
        return
    
    if state.status_label:
        state.status_label.set_text(f"File '{e.name}' uploaded. Ready to process.")
    if state.result_link_container:
        state.result_link_container.clear()
        
    state.input_ready = True
    LOGGER.info(f"File '{e.name}' processed, input_ready set to True.")


def start_processing(state: AppState, job_q: Any) -> None:
    state.input_ready = False
    
    if not state.audio_file_path or not state.audio_file_path.exists():
        ui.notify("Uploaded file not found. Please upload again.", color='negative')
        LOGGER.warning(f"Start processing blocked: invalid audio_file_path: {state.audio_file_path}")
        state.input_ready = True
        return
        
    if state.upload_button: state.upload_button.disable()
    if state.status_label: state.status_label.set_text("Processing... See logs for details.")
    if state.log_area: state.log_area.clear()
    if state.result_link_container: state.result_link_container.clear()

    _normalize_config_for_run(state.config)
    job = (state.config.to_dict(), str(state.audio_file_path))
    job_q.put(job)


def _normalize_config_for_run(cfg: AppConfig) -> None:
    target = getattr(cfg.diarization, "target_speakers", None)
    if target is not None:
        t = int(target)
        if t < 1:
            t = 1
        cfg.diarization.min_speakers = t
        cfg.diarization.max_speakers = t

    if cfg.diarization.min_speakers > cfg.diarization.max_speakers:
        cfg.diarization.max_speakers = int(cfg.diarization.min_speakers)

def update_ui_with_results(state: AppState, result_q: Any) -> None:
    if not result_q.empty():
        result = result_q.get()
        if result.get("success"):
            output_dir = result.get("output_dir")
            if state.status_label:
                state.status_label.set_text(f"Processing complete! Results in '{output_dir}'.")
            if state.result_link_container:
                with state.result_link_container:
                    ui.label(f"Output: {Path(output_dir).resolve()}")
                    path_for_js = json.dumps(str(Path(output_dir).resolve()))
                    ui.button(icon="content_copy", on_click=lambda: ui.run_javascript(f"navigator.clipboard.writeText({path_for_js})")).props("dense flat")
            state.input_ready = False
        else:
            error_msg = result.get("error", "Unknown error")
            if state.status_label:
                state.status_label.set_text(f"An error occurred: {error_msg}")
            ui.notify(f"Error: {error_msg}", color='negative', multi_line=True)
            state.input_ready = True
        
        if state.upload_button: state.upload_button.enable()
        state.cleanup_temp_files()

def get_default_config(state: AppState) -> None:
    cfg = state.config
    if state.settings_container:
        with state.settings_container:
            with ui.card().classes('w-full'):
                ui.label('Transcription').classes('text-lg font-semibold')
                ui.input('Model Name', value=cfg.transcription.model_name).bind_value(cfg.transcription, 'model_name')
                ui.input('Language', value=cfg.transcription.language).bind_value(cfg.transcription, 'language')
                ui.number('No Speech Threshold', value=cfg.transcription.no_speech_threshold, min=0.0, max=1.0, step=0.05, format='%.2f').bind_value(cfg.transcription, 'no_speech_threshold').tooltip('Higher values are more lenient to speech.')
                ui.number('Log-Prob Threshold', value=cfg.transcription.logprob_threshold, min=-3.0, max=0.0, step=0.1, format='%.1f').bind_value(cfg.transcription, 'logprob_threshold').tooltip('Higher values require more confidence.')

                ui.separator().classes('my-4')
                
                ui.label('Diarization').classes('text-lg font-semibold')
                ui.number('Min Speakers', value=cfg.diarization.min_speakers, min=1, format='%.0f').bind_value(cfg.diarization, 'min_speakers')
                ui.number('Max Speakers', value=cfg.diarization.max_speakers, min=1, format='%.0f').bind_value(cfg.diarization, 'max_speakers')
                options = {"Auto": None, "2": 2, "3": 3, "4": 4}
                target = getattr(cfg.diarization, "target_speakers", None)
                default_label = "Auto" if target is None else str(int(target))
                if default_label not in options:
                    default_label = "Auto"
                sel = ui.select('Target Speakers', options=list(options.keys()), value=default_label)

                def _apply_target(e):
                    target_value = options.get(e.value, None)
                    cfg.diarization.target_speakers = target_value
                    if target_value is not None:
                        cfg.diarization.min_speakers = int(target_value)
                        cfg.diarization.max_speakers = int(target_value)

                sel.on_value_change(_apply_target)
                ui.number('VAD Start Threshold', value=cfg.diarization.vad_th_start, min=0.0, max=1.0, step=0.05, format='%.2f').bind_value(cfg.diarization, 'vad_th_start')
                ui.number('VAD End Threshold', value=cfg.diarization.vad_th_end, min=0.0, max=1.0, step=0.05, format='%.2f').bind_value(cfg.diarization, 'vad_th_end')

                ui.separator().classes('my-4')
                ui.label('VBx-Lite Resegmentation').classes('text-lg font-semibold')
                ui.switch('Enable VBx', value=getattr(cfg.diarization, "enable_vbx_resegmentation", True)).bind_value(cfg.diarization, 'enable_vbx_resegmentation')
                ui.number('p_stay (speech)', value=getattr(cfg.diarization, "vbx_p_stay_speech", 0.995),
                          min=0.90, max=0.9999, step=0.0005, format='%.4f').bind_value(cfg.diarization, 'vbx_p_stay_speech')
                ui.number('p_stay (silence)', value=getattr(cfg.diarization, "vbx_p_stay_silence", 0.999),
                          min=0.90, max=0.9999, step=0.0005, format='%.4f').bind_value(cfg.diarization, 'vbx_p_stay_silence')
                ui.number('speech_th', value=getattr(cfg.diarization, "vbx_speech_th", 0.35),
                          min=0.0, max=1.0, step=0.05, format='%.2f').bind_value(cfg.diarization, 'vbx_speech_th')
                ui.number('out_hard_mix', value=getattr(cfg.diarization, "vbx_out_hard_mix", 0.20),
                          min=0.0, max=1.0, step=0.05, format='%.2f').bind_value(cfg.diarization, 'vbx_out_hard_mix')
                ui.number('min_run_sec', value=getattr(cfg.diarization, "vbx_min_run_sec", 1.0),
                          min=0.0, max=5.0, step=0.1, format='%.1f').bind_value(cfg.diarization, 'vbx_min_run_sec')

                ui.separator().classes('my-4')
                ui.label('Aligner (Speaker Switch)').classes('text-lg font-semibold')
                
                # 声の重要度 (beta)
                ui.number('Voice Weight (Beta)', value=cfg.aligner.beta, min=0.0, max=2.0, step=0.1, format='%.1f') \
                    .bind_value(cfg.aligner, 'beta') \
                    .tooltip('Higher = Trust voice vectors more (0.8 recommended)')
                
                # 文字のつながり重要度 (gamma)
                ui.number('Text Continuity (Gamma)', value=cfg.aligner.gamma, min=0.0, max=2.0, step=0.1, format='%.1f') \
                    .bind_value(cfg.aligner, 'gamma') \
                    .tooltip('Lower = Allow cutting sentences (0.5 recommended)')
                
                # 切り替えコスト (delta_switch)
                ui.number('Switch Cost (Delta)', value=cfg.aligner.delta_switch, min=0.0, max=1.0, step=0.01, format='%.2f') \
                    .bind_value(cfg.aligner, 'delta_switch') \
                    .tooltip('0.0 = Switch immediately if voice changes')

def get_persistent_secret() -> str:
    secret_dir = Path.home() / ".ahe_whisper"
    secret_dir.mkdir(mode=0o700, exist_ok=True)
    secret_file = secret_dir / "secret.key"
    
    if not secret_file.exists():
        secret_file.write_text(secrets.token_hex(32))
        try:
            secret_file.chmod(0o600)
        except OSError:
            pass
            
    return secret_file.read_text()
