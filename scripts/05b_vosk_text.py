#!/usr/bin/env python3
"""Vosk 認識結果を「共通転写」として書き出す(他手法で強制アライメントする前段)。

vosk_alignment_ch{N}.json(語+時刻)を読み、
    transcript_vosk_ch{N}.txt   … 認識テキスト全文(MMS/MFA 用)
    vosk_segments_ch{N}.json    … whisper_segments 互換 {"segments":[{text,start,end}]}
                                  (WhisperX のアンカー用。無音ギャップで分割)
を出力する。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.audio_utils import duration_sec

GAP = 0.4       # これ以上の無音で区切る
MAXDUR = 18.0   # 1セグメントの最大長(WhisperX の forward を抑える)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vosk-json", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--txt-out", required=True)
    ap.add_argument("--seg-out", required=True)
    ap.add_argument("--channel", type=int, required=True)
    args = ap.parse_args()

    words = json.load(open(args.vosk_json, encoding="utf-8"))["words"]
    dur = duration_sec(args.wav)

    segs = []
    cur = []
    for w in words:
        s, e = float(w["start"]), float(w["end"])
        tok = str(w["word"]).strip()
        if not tok:
            continue
        if cur:
            gap = s - cur[-1]["end"]
            seg_len = e - cur[0]["start"]
            if gap > GAP or seg_len > MAXDUR:
                segs.append(cur)
                cur = []
        cur.append({"word": tok, "start": s, "end": e})
    if cur:
        segs.append(cur)

    segments = []
    for g in segs:
        text = "".join(x["word"] for x in g)
        if not text:
            continue
        st = float(g[0]["start"])
        en = min(float(g[-1]["end"]), dur)
        segments.append({"text": text, "start": st, "end": max(en, st + 0.01)})

    full = "".join(s["text"] for s in segments)
    with open(args.txt_out, "w", encoding="utf-8") as f:
        f.write(full + "\n")
    with open(args.seg_out, "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, ensure_ascii=False, indent=1)

    print(f"[05b_vosk_text] ch{args.channel}: {len(segments)} segments, "
          f"{len(full)} chars -> {args.txt_out}")


if __name__ == "__main__":
    main()
