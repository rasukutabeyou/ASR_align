#!/usr/bin/env bash
# MFA(Montreal Forced Aligner)用 env。
#
# 【重要】MFA は Kaldi バイナリに依存し、pip 単体での導入は現実的でない。
# venv/pip 方針でも MFA だけは conda(miniforge)が事実上必須。
# ここでは miniforge があればそれを使い、無ければ導入を促す。
#
# fugashi(分かち書き)と praatio(TextGrid変換)は MFA env に入れる。
set -euo pipefail
source "$(dirname "$0")/../config.sh"

if ! command -v conda >/dev/null 2>&1; then
    cat <<'EOF'
[setup] conda が見つかりません。MFA は Kaldi 依存のため conda が必要です。
  miniforge を導入してください:
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
  その後この스크립트を再実行してください。
EOF
    exit 1
fi

# MFA 本体 + 日本語テキスト正規化に必要な spacy / sudachi トークナイザ。
# (これが無いと align 時に「Please install Japanese support」で失敗する)
conda create -y -n mfa -c conda-forge \
    montreal-forced-aligner spacy sudachipy sudachidict-core
# 後処理用(同 env に pip)
conda run -n mfa pip install fugashi unidic-lite praatio soundfile scipy numpy

# 日本語の事前学習モデル/辞書を直接DL(models/mfa へ)。
# `mfa model download` は GitHub API(60req/h)律速で失敗しやすいため、
# リリースアセットを直リンクで取得し、03_align_mfa.sh から直接参照する。
MFA_MODEL_DIR="$PROJ_ROOT/models/mfa"
mkdir -p "$MFA_MODEL_DIR"
MFA_REL=https://github.com/MontrealCorpusTools/mfa-models/releases/download
if [ ! -f "$MFA_MODEL_DIR/japanese_mfa_acoustic.zip" ]; then
    curl -sL -o "$MFA_MODEL_DIR/japanese_mfa_acoustic.zip" \
        "$MFA_REL/acoustic-japanese_mfa-v3.0.0/japanese_mfa.zip"
fi
if [ ! -f "$MFA_MODEL_DIR/japanese_mfa.dict" ]; then
    curl -sL -o "$MFA_MODEL_DIR/japanese_mfa.dict" \
        "$MFA_REL/dictionary-japanese_mfa-v3.0.0/japanese_mfa.dict"
fi

echo "[setup] mfa env (conda) 作成完了。 'conda activate mfa' で利用。"
echo "[setup] mfa model -> $MFA_MODEL_DIR (japanese_mfa acoustic/dict)"
