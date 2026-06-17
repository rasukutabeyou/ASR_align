#!/usr/bin/env bash
# Vosk(オフライン認識)用 venv + 日本語モデルDL。
set -euo pipefail
source "$(dirname "$0")/../config.sh"

python3 -m venv "$VENV_DIR/vosk"
source "$VENV_DIR/vosk/bin/activate"
pip install -U pip wheel
pip install vosk soundfile scipy numpy
deactivate

# 日本語モデル(大: 高精度 ~1GB)。小型は vosk-model-small-ja-0.22。
MODEL_DIR="$PROJ_ROOT/models"
mkdir -p "$MODEL_DIR"
if [ ! -d "$MODEL_DIR/vosk-model-ja-0.22" ]; then
    cd "$MODEL_DIR"
    curl -LO https://alphacephei.com/vosk/models/vosk-model-ja-0.22.zip
    unzip -q vosk-model-ja-0.22.zip && rm vosk-model-ja-0.22.zip
fi
echo "[setup] vosk env -> $VENV_DIR/vosk"
echo "[setup] vosk model -> $MODEL_DIR/vosk-model-ja-0.22"
