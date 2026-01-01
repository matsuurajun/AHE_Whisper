#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_pipeline_patch_v2.py
AHE-Whisper pipeline.py に対して安全に result["speaker_segments"] 修正を挿入します。
"""

import re
from pathlib import Path
import shutil
import sys

project_root = Path(__file__).resolve().parents[1]
pipeline_path = project_root / "ahe_whisper" / "pipeline.py"
backup_path = pipeline_path.with_name("pipeline_backup.py")

print(f"🔧 Target: {pipeline_path}")

if not pipeline_path.exists():
    print("❌ pipeline.py が見つかりません。スクリプトのパスを確認してください。")
    sys.exit(1)

# バックアップ
shutil.copy2(pipeline_path, backup_path)
print(f"📦 バックアップ作成: {backup_path.name}")

code = pipeline_path.read_text(encoding="utf-8").splitlines()

patched = []
inserted = False
for i, line in enumerate(code):
    patched.append(line)
    # result = { ... } の定義を検出
    if re.search(r'^\s*result\s*=\s*\{', line):
        # すぐ次の空行の後にブロックを挿入
        patched.append("")
        patched.extend([
            "    # --- Aligner output validation and mapping ---",
            "    if speaker_segments and isinstance(speaker_segments[0], (list, tuple)) and len(speaker_segments[0]) == 3:",
            "        result[\"speaker_segments\"] = [",
            "            {\"start\": s, \"end\": e, \"speaker\": f\"SPEAKER_{spk:02d}\"}",
            "            for s, e, spk in speaker_segments",
            "        ]",
            "        LOGGER.info(f\"[PIPELINE] Aligner produced {len(result['speaker_segments'])} speaker segments\")",
            "    else:",
            "        LOGGER.warning(f\"[PIPELINE] Unexpected speaker_segments structure: {type(speaker_segments)}\")",
            ""
        ])
        inserted = True

if not inserted:
    print("⚠️ result 定義が見つかりませんでした。自動挿入できません。手動修正が必要です。")
else:
    pipeline_path.write_text("\n".join(patched), encoding="utf-8")
    print("✅ パッチを適用しました。pipeline.py が更新されました。")

# 構文チェック
import subprocess
print("🔍 構文を確認中...")
proc = subprocess.run(
    ["python3", "-m", "py_compile", str(pipeline_path)],
    capture_output=True, text=True
)
if proc.returncode == 0:
    print("✅ 構文チェック成功。エラーはありません。")
else:
    print("❌ 構文エラーがあります:")
    print(proc.stderr)
