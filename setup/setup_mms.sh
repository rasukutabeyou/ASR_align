#!/usr/bin/env bash
# MMS Alignment(torchaudio MMS_FA + uroman)用 venv。
set -euo pipefail
source "$(dirname "$0")/../config.sh"

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

python3 -m venv "$VENV_DIR/mms"
source "$VENV_DIR/mms/bin/activate"
pip install -U pip wheel
pip install torch torchaudio --index-url "$TORCH_INDEX"
pip install uroman fugashi unidic-lite
pip install soundfile scipy numpy
deactivate
echo "[setup] mms env -> $VENV_DIR/mms"
