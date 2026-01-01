# ahe_whisper/logger.py
import logging
import sys

# 共通ロガー設定
LOGGER = logging.getLogger("AHE-Whisper")
LOGGER.setLevel(logging.INFO)

# 既存のハンドラがある場合は一度クリア
if LOGGER.hasHandlers():
    LOGGER.handlers.clear()

# コンソール出力ハンドラ
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s-%(levelname)s-%(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
