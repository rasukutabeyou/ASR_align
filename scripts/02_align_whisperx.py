#!/usr/bin/env python3
"""ステップ2: WhisperX によるアライメント。

ステップ1の共通転写(whisper_segments_ch{N}.json)を入力に、
WhisperX の wav2vec2 ベース forced alignment で単語レベル timestamp を得る。
日本語の既定アライメントモデルは jonatasgrosman/wav2vec2-large-xlsr-53-japanese。

出力: whisperx_alignment_ch{N}.json(統一スキーマ)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.audio_utils import read_16k_mono, duration_sec
from common.schema import Alignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--segments", required=True, help="whisper_segments_ch{N}.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", type=int, required=True)
    ap.add_argument("--language", default="ja")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import whisperx

    with open(args.segments, encoding="utf-8") as f:
        segs = json.load(f)["segments"]

    audio = read_16k_mono(args.wav)  # np.ndarray を渡すので ffmpeg 不要
    model_a, metadata = whisperx.load_align_model(
        language_code=args.language, device=args.device)
    result = whisperx.align(
        segs, model_a, metadata, audio, args.device,
        return_char_alignments=False)

    align = Alignment(
        method="whisperx", audio=os.path.basename(args.wav), channel=args.channel,
        language=args.language, sample_rate=16000,
        model="whisperx/wav2vec2-large-xlsr-53-japanese",
        duration=duration_sec(args.wav), granularity="word")

    for seg in result["segments"]:
        for w in seg.get("words", []):
            # アライメント不能語は start/end が欠落することがある
            if "start" not in w or "end" not in w:
                continue
            align.add(w["word"], w["start"], w["end"], w.get("score"))

    align.save(args.out)


if __name__ == "__main__":
    main()
