# -*- coding: utf-8 -*-
from pathlib import Path
import os
import logging
from huggingface_hub import hf_hub_download, snapshot_download
from typing import Dict, Any

from ahe_whisper.constants import MODELS

LOGGER = logging.getLogger("ahe_whisper_worker")

def ensure_model_available(model_key: str, project_root: Path) -> Path:
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}")
    
    model_info = MODELS[model_key]
    repo_id = model_info["repo_id"]
    revision = model_info.get("revision")
    local_only = bool(model_info.get("local_only", False))
    
    default_local_root = project_root / "models"
    local_root = Path(os.environ.get("AHE_MODEL_DIR", default_local_root))
    
    search_dir = local_root / repo_id
    
    local_model_is_complete = False
    if search_dir.is_dir():
        if model_info.get("use_snapshot"):
            if (search_dir / "config.json").exists():
                local_model_is_complete = True
        else:
            if all((search_dir / f).exists() for f in model_info["required_files"]):
                local_model_is_complete = True

    if local_model_is_complete:
        LOGGER.info(f"Using local model for '{model_key}' from: {search_dir}")
        primary_file = model_info.get("primary_file")
        return search_dir / primary_file if primary_file else search_dir

    # --- local_only モデルは HF から取得しない ---
    if local_only:
        raise FileNotFoundError(
            "Local-only model '{key}' not found or incomplete in '{dir}'. "
            "Please download the required files manually into this directory: {files}".format(
                key=model_key,
                dir=str(search_dir),
                files=", ".join(model_info.get("required_files", [])),
            )
        )

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise FileNotFoundError(
            f"Offline mode: model '{repo_id}' (revision: {revision}) not found or incomplete in '{local_root}'."
        )

    LOGGER.info(f"Local model for '{model_key}' not found or incomplete. Downloading...")
    
    try:
        if model_info.get("use_snapshot"):
            snapshot_download(
                repo_id=repo_id, 
                revision=revision, 
                local_dir=search_dir, 
                local_dir_use_symlinks=False, 
                max_workers=8
            )
            primary_file_path = search_dir
        else:
            for filename in model_info["required_files"]:
                hf_hub_download(
                    repo_id=repo_id,
                    revision=revision,
                    filename=filename,
                    local_dir=search_dir,
                    local_dir_use_symlinks=False
                )
            primary_file_path = search_dir / model_info["primary_file"]

        LOGGER.info(f"Successfully downloaded all files for '{model_key}'.")
        return primary_file_path

    except Exception as e:
        raise RuntimeError(f"Failed to download model '{repo_id}' (revision: {revision}): {e}")
