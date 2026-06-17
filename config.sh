# 共通設定。各スクリプト/セットアップから source して使う。
# 例:  source config.sh

# 入力音声(本番データ)。別ファイルを試す場合はここを変える。
export AUDIO="${AUDIO:-/autofs/diamond3/share/users/sakai/VAP_data/Tabidachi_data/wav/101_1_2_zoom.wav}"

# 言語
export LANG_CODE="${LANG_CODE:-ja}"

# プロジェクトルート(このファイルの場所)
export PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 作業ディレクトリ
export WORK_DIR="${PROJ_ROOT}/data/work"   # 前処理済みの16k mono wav
export OUT_DIR="${PROJ_ROOT}/data/out"     # 各手法の統一JSON出力
export VENV_DIR="${PROJ_ROOT}/.venvs"      # 手法ごとの venv

# 処理対象チャンネル(話者分離)。0=L, 1=R
export CHANNELS="${CHANNELS:-0 1}"

# Whisper転写モデル(共通転写)
export WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"

mkdir -p "$WORK_DIR" "$OUT_DIR" "$VENV_DIR"

# 入力音声 stem(例: 101_1_2_zoom)
export STEM="$(basename "$AUDIO" .wav)"
