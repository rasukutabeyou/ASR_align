#!/usr/bin/env bash
# MMS Alignment(torchaudio MMS_FA + uroman)用 venv。
set -euo pipefail
source "$(dirname "$0")/../config.sh"

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

uv venv --clear --python 3.11 "$VENV_DIR/mms"
PY="$VENV_DIR/mms/bin/python"
uv pip install --python "$PY" torch torchaudio --index-url "$TORCH_INDEX"
uv pip install --python "$PY" uroman fugashi unidic-lite
uv pip install --python "$PY" soundfile scipy numpy
echo "[setup] mms env -> $VENV_DIR/mms"
