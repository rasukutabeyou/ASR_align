#!/usr/bin/env bash
# NeMo ASR(日本語: ReazonSpeech NeMo v2)用 venv。
set -euo pipefail
source "$(dirname "$0")/../config.sh"

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

python3 -m venv "$VENV_DIR/nemo"
source "$VENV_DIR/nemo/bin/activate"
pip install -U pip wheel
pip install torch torchaudio --index-url "$TORCH_INDEX"
pip install "nemo_toolkit[asr]"
# ReazonSpeech の NeMo インターフェース
pip install Cython
pip install "reazonspeech-nemo-asr @ git+https://github.com/reazon-research/ReazonSpeech#subdirectory=pkg/nemo-asr"
pip install soundfile scipy numpy
deactivate
echo "[setup] nemo env -> $VENV_DIR/nemo"
