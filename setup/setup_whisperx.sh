#!/usr/bin/env bash
# WhisperX + faster-whisper(共通転写も兼ねる)用 venv。
# このenvで 01_transcribe_whisper.py と 02_align_whisperx.py を実行する。
set -euo pipefail
source "$(dirname "$0")/../config.sh"

# RTX 50系(Blackwell)は cu128 以降が必要。GPUに合わせて TORCH_INDEX を変える。
#   例) export TORCH_INDEX=https://download.pytorch.org/whl/cu128
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

python3 -m venv "$VENV_DIR/whisperx"
source "$VENV_DIR/whisperx/bin/activate"
pip install -U pip wheel
pip install torch torchaudio --index-url "$TORCH_INDEX"
pip install whisperx faster-whisper
pip install soundfile scipy numpy
deactivate
echo "[setup] whisperx env -> $VENV_DIR/whisperx"
