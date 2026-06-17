#!/usr/bin/env bash
# Oracle timestamp 5手法比較パイプライン一括実行。
#
# 前提: setup/ の各セットアップを済ませ、.venvs/* と models/ が存在すること。
# 使い方:
#   bash run_all.sh                 # config.sh の AUDIO / CHANNELS で全手法
#   METHODS="whisperx vosk" bash run_all.sh   # 一部手法のみ
#   AUDIO=/path/other.wav bash run_all.sh
set -uo pipefail
source "$(dirname "$0")/config.sh"

METHODS="${METHODS:-whisperx mfa mms vosk nemo}"
SC="$PROJ_ROOT/scripts"
VOSK_MODEL="${VOSK_MODEL:-$PROJ_ROOT/models/vosk-model-ja-0.22}"

py() { "$VENV_DIR/$1/bin/python" "${@:2}"; }   # py <env> <script...>

echo "=== 入力: $AUDIO  channels: $CHANNELS  methods: $METHODS ==="

# --- ステップ0: 前処理(チャンネル分離 + 16k) -----------------------------
py tools "$SC/00_preprocess.py" --audio "$AUDIO" --work-dir "$WORK_DIR" \
    --channels $CHANNELS

for CH in $CHANNELS; do
    WAV="$WORK_DIR/${STEM}_ch${CH}.wav"
    [ -f "$WAV" ] || { echo "skip ch$CH (wav無し)"; continue; }
    echo ""; echo "############ channel $CH ############"

    # --- ステップ1: 共通転写(Whisper) ----------------------------------
    py whisperx "$SC/01_transcribe_whisper.py" --wav "$WAV" --out-dir "$OUT_DIR" \
        --channel "$CH" --model "$WHISPER_MODEL" --language "$LANG_CODE"
    SEG="$OUT_DIR/whisper_segments_ch${CH}.json"
    TXT="$OUT_DIR/transcript_ch${CH}.txt"

    for M in $METHODS; do
        OUT="$OUT_DIR/${M}_alignment_ch${CH}.json"
        case "$M" in
        whisperx)
            py whisperx "$SC/02_align_whisperx.py" --wav "$WAV" --segments "$SEG" \
                --out "$OUT" --channel "$CH" --language "$LANG_CODE" ;;
        mms)
            py mms "$SC/04_align_mms.py" --wav "$WAV" --transcript "$TXT" \
                --out "$OUT" --channel "$CH" --language "$LANG_CODE" ;;
        vosk)
            py vosk "$SC/05_recognize_vosk.py" --wav "$WAV" --model "$VOSK_MODEL" \
                --out "$OUT" --channel "$CH" --language "$LANG_CODE" ;;
        nemo)
            py nemo "$SC/06_recognize_nemo.py" --wav "$WAV" --out "$OUT" \
                --channel "$CH" --language "$LANG_CODE" ;;
        mfa)
            CORPUS="$WORK_DIR/mfa_corpus_ch${CH}"; MFAOUT="$WORK_DIR/mfa_out_ch${CH}"
            rm -rf "$CORPUS" "$MFAOUT"; mkdir -p "$CORPUS"
            conda run -n mfa python "$SC/03_prep_mfa.py" --wav "$WAV" \
                --transcript "$TXT" --corpus "$CORPUS" --channel "$CH"
            conda run -n mfa bash "$SC/03_align_mfa.sh" "$CORPUS" "$MFAOUT"
            conda run -n mfa python "$SC/03_textgrid_to_json.py" \
                --textgrid "$MFAOUT/ch${CH}.TextGrid" --wav "$WAV" \
                --out "$OUT" --channel "$CH" --language "$LANG_CODE" ;;
        esac || echo "!! $M ch$CH 失敗(継続)"
    done

    # --- 主観評価用 TextGrid / CSV --------------------------------------
    py tools "$SC/99_inspect.py" --out-dir "$OUT_DIR" --channel "$CH" --wav "$WAV"
done

echo ""; echo "=== 完了。data/out/ の compare_ch*.TextGrid を Praat で確認 ==="
