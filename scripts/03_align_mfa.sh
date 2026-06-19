#!/usr/bin/env bash
# ステップ3b: MFA Forced Alignment 実行。
# 事前に 03_prep_mfa.py でコーパス(wav + .lab)を用意しておくこと。
#
# 使い方: bash 03_align_mfa.sh <corpus_dir> <output_dir>
set -euo pipefail

CORPUS="$1"
OUTPUT="$2"

# 日本語の事前学習モデル/辞書。
# `mfa model download` は GitHub API(60req/h)を使い律速・障害点になるため、
# リリースアセットを直接DLしたローカルファイルを参照する(setup で models/mfa に配置)。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MFA_DICT="${MFA_DICT:-$PROJ_ROOT/models/mfa/japanese_mfa.dict}"
MFA_ACOUSTIC="${MFA_ACOUSTIC:-$PROJ_ROOT/models/mfa/japanese_mfa_acoustic.zip}"

# 検証(辞書に無い語の確認。失敗してもalignは続行)
mfa validate "$CORPUS" "$MFA_DICT" "$MFA_ACOUSTIC" --clean || true

# アライメント実行
mfa align --clean --overwrite \
    "$CORPUS" "$MFA_DICT" "$MFA_ACOUSTIC" "$OUTPUT"

echo "[03_align_mfa] TextGrid -> $OUTPUT"
