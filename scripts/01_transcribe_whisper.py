#!/usr/bin/env python3
"""ステップ1: 共通転写(Whisper)。

faster-whisper(large-v3)で各チャンネルを転写する。
出力は WhisperX / MFA / MMS の共通入力として使う:

  - transcript_ch{N}.txt        : プレーンテキスト(全文)
  - whisper_segments_ch{N}.json : セグメント区間付き [{start,end,text}]
                                   (WhisperX のアライメント入力に使用)

whisperx 環境で実行する(faster-whisper を含むため)。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.audio_utils import read_16k_mono


def transcribe(wav_path, model_name, language, device, compute_type):
    from faster_whisper import WhisperModel

    audio = read_16k_mono(wav_path)  # 16k mono float32(ffmpeg不要)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    # vad_filter で長尺の無音を除去。word_timestamps は WhisperX 側で付けるので False。
    segments, info = model.transcribe(
        audio,
        language=language,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
    )
    segs = []
    for s in segments:
        text = s.text.strip()
        if not text:
            continue
        segs.append({"start": float(s.start), "end": float(s.end), "text": text})
    return segs, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--channel", type=int, required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default="ja")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    segs, info = transcribe(args.wav, args.model, args.language,
                            args.device, args.compute_type)

    txt = "".join(s["text"] for s in segs)
    base = os.path.join(args.out_dir, f"transcript_ch{args.channel}")
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(base.replace("transcript", "whisper_segments") + ".json", "w",
              encoding="utf-8") as f:
        json.dump({"language": info.language, "segments": segs},
                  f, ensure_ascii=False, indent=2)
    print(f"[01_transcribe] ch{args.channel}: {len(segs)} segments, "
          f"{len(txt)} chars -> {base}.txt")


if __name__ == "__main__":
    main()
