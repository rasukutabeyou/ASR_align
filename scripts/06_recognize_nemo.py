#!/usr/bin/env python3
"""ステップ6: NeMo ASR による認識 + timestamp。

日本語の NeMo モデルとして ReazonSpeech NeMo v2(RNN-T)を使う。
reazonspeech.nemo.asr の transcribe はサブワード単位の時刻を返すため、
連続サブワードから span(start/end)を構成する(granularity="subword")。
NeMo/日本語は単語境界が曖昧なため、サブワード粒度で出力し主観比較する。

出力: nemo_alignment_ch{N}.json(統一スキーマ)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.audio_utils import duration_sec
from common.schema import Alignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", type=int, required=True)
    ap.add_argument("--language", default="ja")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path

    model = load_model(device=args.device)
    audio = audio_from_path(args.wav)
    ret = transcribe(model, audio)

    dur = duration_sec(args.wav)
    align = Alignment(
        method="nemo", audio=os.path.basename(args.wav), channel=args.channel,
        language=args.language, sample_rate=16000,
        model="reazonspeech-nemo-v2", duration=dur, granularity="subword")

    subs = list(getattr(ret, "subwords", []))
    for i, sw in enumerate(subs):
        token = getattr(sw, "token", None) or getattr(sw, "text", "")
        start = float(getattr(sw, "seconds", 0.0))
        # 終了時刻は次サブワード開始(最後は音声長)で近似
        end = float(subs[i + 1].seconds) if i + 1 < len(subs) else dur
        if not str(token).strip():
            continue
        align.add(str(token), start, max(end, start), None)

    # サブワードが取れない場合はセグメント単位にフォールバック
    if not align.words:
        for seg in getattr(ret, "segments", []):
            align.add(seg.text, float(seg.start_seconds), float(seg.end_seconds))
        align.granularity = "segment"

    align.save(args.out)


if __name__ == "__main__":
    main()
