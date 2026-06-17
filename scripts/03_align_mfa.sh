#!/usr/bin/env bash
# ステップ3b: MFA Forced Alignment 実行。
# 事前に 03_prep_mfa.py でコーパス(wav + .lab)を用意しておくこと。
#
# 使い方: bash 03_align_mfa.sh <corpus_dir> <output_dir>
set -euo pipefail

CORPUS="$1"
OUTPUT="$2"

# 日本語の事前学習モデル/辞書(初回のみDL。~/Documents/MFA にキャッシュ)
mfa model download acoustic  japanese_mfa || true
mfa model download dictionary japanese_mfa || true

# 検証(辞書に無い語の確認。失敗してもalignは続行)
mfa validate "$CORPUS" japanese_mfa japanese_mfa --clean || true

# アライメント実行
mfa align --clean --overwrite \
    "$CORPUS" japanese_mfa japanese_mfa "$OUTPUT"

echo "[03_align_mfa] TextGrid -> $OUTPUT"
