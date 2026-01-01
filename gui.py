# -*- coding: utf-8 -*-
import sys
from collections import deque
from pathlib import Path
from multiprocessing import Process, Queue, freeze_support
from threading import Thread
from typing import Optional, Deque, List

from nicegui import ui, app

project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ahe_whisper.gui_logic import (
    AHE_LOGO_SVG, AppState, get_default_config,
    handle_file_upload, start_processing, update_ui_with_results,
    get_persistent_secret
)
from ahe_whisper.pipeline_worker import worker_process_loop

job_q: Optional[Queue] = None
result_q: Optional[Queue] = None
log_q: Optional[Queue] = None
worker_process: Optional[Process] = None
log_thread: Optional[Thread] = None
log_buffer: Optional[Deque[str]] = None

async def app_startup() -> None:
    print("Application starting up...")
    global job_q, result_q, log_q, worker_process, log_thread, log_buffer
    
    job_q = Queue()
    result_q = Queue()
    log_q = Queue()
    
    worker_args = (job_q, result_q, log_q, str(project_root))
    worker_process = Process(target=worker_process_loop, args=worker_args)
    worker_process.start()
    
    log_buffer = deque(maxlen=5000)

    def read_log_queue() -> None:
        while True:
            try:
                message = log_q.get()
                if message is None: break
                if log_buffer is not None:
                    log_buffer.append(message)
            except (EOFError, BrokenPipeError):
                break
    
    log_thread = Thread(target=read_log_queue, daemon=True)
    log_thread.start()
    
    print("Startup complete. AI Worker is running.")

async def app_shutdown() -> None:
    print("Shutting down...")
    global job_q, result_q, log_q, worker_process, log_thread
    if job_q: job_q.put(None)
    if worker_process and worker_process.is_alive():
        worker_process.join(timeout=5)
        if worker_process.is_alive(): worker_process.terminate()
    if log_q: log_q.put(None)
    if log_thread and log_thread.is_alive(): log_thread.join(timeout=2)
    print("Shutdown complete.")

app.on_startup(app_startup)
app.on_shutdown(app_shutdown)

@ui.page('/')
def main_page() -> None:
    app_state = AppState()
    
    ui.add_head_html('<style>body {background-color: #f0f4f8;}</style>')
    with ui.row().classes('w-full justify-center'):
        with ui.card().classes('w-full max-w-4xl mt-8 p-6 rounded-2xl shadow-lg'):
            with ui.row().classes('w-full items-center'):
                ui.html(AHE_LOGO_SVG).classes('w-16 h-16')
                ui.label('AHE-Whisper').classes('text-4xl font-bold text-gray-700 ml-4')
                
                # これを追加：右端にスペースを空けてボタンを配置
                ui.space() 
                ui.button('終了', on_click=app.shutdown, color='red').props('icon=logout')
            
            ui.separator().classes('my-6')
            app_state.settings_container = ui.expansion('⚙️ Settings').classes('w-full')
            
            with ui.row().classes('w-full items-center gap-4 mt-4'):
                app_state.upload_button = ui.upload(
                    on_upload=lambda e: handle_file_upload(e, app_state), auto_upload=True,
                    label="1. Upload Audio (Max 1GB)"
                ).props('accept=".wav,.mp3,.m4a,.flac"').classes('flex-grow')
                
                (
                    ui.button(
                        '2. Start Processing', on_click=lambda: start_processing(app_state, job_q)
                    )
                    .props('color=primary icon=play_arrow')
                    .classes('w-48')
                    .bind_enabled_from(app_state, 'input_ready')
                )

            app_state.status_label = ui.label('Upload an audio file to begin.').classes('mt-4 text-gray-600')
            app_state.result_link_container = ui.row()

            ui.label("Logs").classes('text-xl font-semibold mt-6 mb-2 text-gray-700')
            app_state.log_area = ui.log().classes('w-full h-64 bg-gray-800 text-white rounded-lg p-2')
    
    def update_logs_from_buffer() -> None:
        global log_buffer
        if log_buffer is None: return
        log_batch: List[str] = []
        while log_buffer:
            try:
                log_batch.append(log_buffer.popleft())
            except IndexError:
                break
        if log_batch and app_state.log_area:
            app_state.log_area.push('\n'.join(log_batch))

    ui.timer(0.2, update_logs_from_buffer)
    ui.timer(0.2, lambda: update_ui_with_results(app_state, result_q))
    
    app.on_disconnect(lambda _client: app_state.cleanup_temp_files())
    
    get_default_config(app_state)

if __name__ in {"__main__", "__mp_main__"}:
    freeze_support()
    ui.run(
        title="AHE-Whisper",
        storage_secret=get_persistent_secret(),
        reload=False
    )
