#!/usr/bin/env python3
"""ステップ5: Vosk によるオフライン認識 + 単語 timestamp。

Vosk は CPU 動作。日本語モデル(vosk-model-ja-0.22)を使う。
16k mono PCM をストリーミングで食わせ、SetWords(True) で単語ごとの
start/end/conf を得る。Vosk は ASR と timestamp を同時に出す(共通転写は使わない)。

出力: vosk_alignment_ch{N}.json(統一スキーマ)
"""
import argparse
import json
import os
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.audio_utils import duration_sec
from common.schema import Alignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="16k mono PCM wav")
    ap.add_argument("--model", required=True, help="Voskモデルのディレクトリ")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", type=int, required=True)
    ap.add_argument("--language", default="ja")
    args = ap.parse_args()

    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)

    wf = wave.open(args.wav, "rb")
    assert wf.getnchannels() == 1 and wf.getframerate() == 16000 \
        and wf.getsampwidth() == 2, "16k/mono/16bit PCM が必要(00_preprocess を実行)"

    model = Model(args.model)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    results.append(json.loads(rec.FinalResult()))

    align = Alignment(
        method="vosk", audio=os.path.basename(args.wav), channel=args.channel,
        language=args.language, sample_rate=16000,
        model=os.path.basename(args.model.rstrip("/")),
        duration=duration_sec(args.wav), granularity="word")

    for r in results:
        for w in r.get("result", []):
            align.add(w["word"], w["start"], w["end"], w.get("conf"))

    align.save(args.out)


if __name__ == "__main__":
    main()
