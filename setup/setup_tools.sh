#!/usr/bin/env bash
# 前処理(00)と主観評価可視化(99)用の軽量 venv。
set -euo pipefail
source "$(dirname "$0")/../config.sh"

uv venv --clear --python 3.11 "$VENV_DIR/tools"
uv pip install --python "$VENV_DIR/tools/bin/python" soundfile scipy numpy praatio
echo "[setup] tools env -> $VENV_DIR/tools"
