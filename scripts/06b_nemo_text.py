#!/usr/bin/env python3
"""NeMo(ReazonSpeech)認識結果を「共通転写」として書き出す。

NeMo の認識テキストを入力に、他手法(WhisperX/MMS/MFA)で強制アライメントして
比較するための前段。以下を出力する:

    transcript_nemo_ch{N}.txt        … 認識テキスト全文(MMS/MFA 用)
    nemo_segments_ch{N}.json         … whisper_segments 互換 {"segments":[{text,start,end}]}
                                       (WhisperX のアンカーに使う)

セグメントは NeMo の segment 出力(start_seconds/end_seconds/text)をそのまま使う。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.audio_utils import duration_sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--txt-out", required=True)
    ap.add_argument("--seg-out", required=True)
    ap.add_argument("--channel", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path

    model = load_model(device=args.device)
    audio = audio_from_path(args.wav)
    ret = transcribe(model, audio)
    dur = duration_sec(args.wav)

    segments = []
    for seg in getattr(ret, "segments", []):
        text = str(getattr(seg, "text", "")).strip()
        if not text:
            continue
        start = float(getattr(seg, "start_seconds", 0.0))
        end = float(getattr(seg, "end_seconds", start))
        end = min(max(end, start + 0.01), dur)
        segments.append({"text": text, "start": start, "end": end})

    # フルテキスト(セグメント連結。無ければ ret.text)
    full = "".join(s["text"] for s in segments) or str(getattr(ret, "text", "")).strip()

    with open(args.txt_out, "w", encoding="utf-8") as f:
        f.write(full + "\n")
    with open(args.seg_out, "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, ensure_ascii=False, indent=1)

    print(f"[06b_nemo_text] ch{args.channel}: {len(segments)} segments, "
          f"{len(full)} chars -> {args.txt_out}")


if __name__ == "__main__":
    main()
