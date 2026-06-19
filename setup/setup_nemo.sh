#!/usr/bin/env bash
# NeMo ASR(日本語: ReazonSpeech NeMo v2)用 venv。
set -euo pipefail
source "$(dirname "$0")/../config.sh"

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

uv venv --clear --python 3.11 "$VENV_DIR/nemo"
PY="$VENV_DIR/nemo/bin/python"
uv pip install --python "$PY" torch torchaudio --index-url "$TORCH_INDEX"
uv pip install --python "$PY" "nemo_toolkit[asr]"
# ReazonSpeech の NeMo インターフェース
uv pip install --python "$PY" Cython
uv pip install --python "$PY" "reazonspeech-nemo-asr @ git+https://github.com/reazon-research/ReazonSpeech#subdirectory=pkg/nemo-asr"
uv pip install --python "$PY" soundfile scipy numpy
echo "[setup] nemo env -> $VENV_DIR/nemo"
