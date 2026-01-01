#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AHE-Whisper Model Prefetcher v90.90
"""
Downloads all necessary AI models from Hugging Face Hub for offline use.
This script ensures that the AHE-Whisper application can run without an
internet connection by pre-caching all required model files.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    from ahe_whisper.constants import MODELS
except ImportError:
    print("Error: Required packages are not found. Please set up your virtual environment and run:")
    print("`pip install -r requirements.txt` first.")
    sys.exit(1)

DEFAULT_MODELS_DIR = "models"
LOGGER = logging.getLogger("ahe_whisper.prefetch")

def _init_logger() -> None:
    """Initializes the logger if no handlers are configured."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main() -> int:
    """
    Main function to prefetch all models defined in constants.py.

    Returns:
        int: 0 on success, 1 on failure.
    """
    _init_logger()
    LOGGER.info("--- AHE-Whisper Model Prefetcher ---")

    env_dir = os.environ.get("AHE_MODEL_DIR", DEFAULT_MODELS_DIR)
    models_root = Path(env_dir)

    try:
        models_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        LOGGER.error(f"Failed to create models directory '{models_root}': {e}")
        return 1

    LOGGER.info(f"Models will be downloaded to: {models_root.resolve()}")
    
    all_successful = True
    for name, info in MODELS.items():
        repo_id: str = info["repo_id"]
        revision: str = info.get("revision")
        LOGGER.info(f"Processing '{name}' model ({repo_id} at revision {revision or 'latest'})...")
        
        try:
            local_dir = models_root / repo_id
            if info.get("use_snapshot"):
                snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir, local_dir_use_symlinks=False, max_workers=8)
            else:
                for filename in info["required_files"]:
                    LOGGER.info(f"  - Downloading {filename}...")
                    hf_hub_download(
                        repo_id=repo_id,
                        revision=revision,
                        filename=filename,
                        local_dir=local_dir,
                        local_dir_use_symlinks=False,
                    )
            LOGGER.info(f"✓ Successfully downloaded all files for '{name}' model.")
        except Exception as e:
            LOGGER.error(f"❌ Failed to download files for '{name}' model: {e}", exc_info=True)
            all_successful = False
            
    if all_successful:
        LOGGER.info("\n" + "="*50)
        LOGGER.info("✅ All models have been successfully downloaded!")
        LOGGER.info("You can now run AHE-Whisper in a fully reproducible, offline mode.")
        LOGGER.info("="*50)
        return 0
    else:
        LOGGER.error("\n" + "="*50)
        LOGGER.error("❌ Some models failed to download. Please check the errors above.")
        LOGGER.error("="*50)
        return 1

if __name__ == "__main__":
    sys.exit(main())
