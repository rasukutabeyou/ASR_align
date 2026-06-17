#!/usr/bin/env bash
# 前処理(00)と主観評価可視化(99)用の軽量 venv。
set -euo pipefail
source "$(dirname "$0")/../config.sh"

python3 -m venv "$VENV_DIR/tools"
source "$VENV_DIR/tools/bin/activate"
pip install -U pip wheel
pip install soundfile scipy numpy praatio
deactivate
echo "[setup] tools env -> $VENV_DIR/tools"
