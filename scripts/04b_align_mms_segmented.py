#!/usr/bin/env python3
"""ステップ4(セグメント版): MMS Alignment を「セグメント窓ごと」に実行する。

04_align_mms.py は音声全体(~21分)を一括 forward + forced_align するため、
入力によっては forced_align が巨大メモリを一括確保して OOM 死することがある。
本スクリプトは whisper_segments 互換の {segments:[{text,start,end}]} を受け取り、
各セグメントの音声窓だけを切り出して MMS forced alignment する。
→ 1回の確保が小さく堅牢。語の絶対時刻はセグメント開始時刻を足して復元する。

出力: 統一スキーマ(granularity="word")。
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.audio_utils import duration_sec
from common.schema import Alignment


def ja_words(text):
    from fugashi import Tagger
    tagger = Tagger()
    out = []
    for m in tagger(text):
        s = m.surface.strip()
        if not s:
            continue
        pos = getattr(m.feature, "pos1", "")
        if pos == "補助記号" or s in "、。「」『』（）()【】・…":
            continue
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--segments", required=True, help="{segments:[{text,start,end}]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", type=int, required=True)
    ap.add_argument("--language", default="ja")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pad", type=float, default=0.1, help="窓の前後パディング秒")
    args = ap.parse_args()

    import torch
    import torchaudio
    from torchaudio.pipelines import MMS_FA as bundle
    from torchaudio.functional import forced_align, merge_tokens
    from uroman import Uroman

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = bundle.get_model().to(device)
    DICT = bundle.get_dict()
    uroman = Uroman()

    segments = json.load(open(args.segments, encoding="utf-8"))["segments"]
    waveform, sr = torchaudio.load(args.wav)
    assert sr == bundle.sample_rate, f"sr={sr}"
    waveform = waveform[0:1]
    total = waveform.size(1)
    dur = duration_sec(args.wav)

    align = Alignment(
        method="mms", audio=os.path.basename(args.wav), channel=args.channel,
        language=args.language, sample_rate=16000,
        model="MMS_FA(uroman, segmented)", duration=dur, granularity="word")

    n_seg_ok = 0
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        words = ja_words(text)
        targets, word_idx, kept = [], [], []
        for w in words:
            roman = re.sub(r"[^a-z']", "", uroman.romanize_string(w, lcode="jpn").lower())
            ids = [DICT[c] for c in roman if c in DICT]
            if not ids:
                continue
            kept.append(w)
            for tid in ids:
                targets.append(tid)
                word_idx.append(len(kept) - 1)
        if not targets:
            continue

        s0 = max(0.0, float(seg["start"]) - args.pad)
        s1 = min(dur, float(seg["end"]) + args.pad)
        i0, i1 = int(s0 * sr), int(s1 * sr)
        i1 = min(i1, total)
        if i1 - i0 < sr * 0.05:
            continue
        wav_seg = waveform[:, i0:i1].to(device)

        try:
            with torch.inference_mode():
                emission, _ = model(wav_seg)
                # targets が frame 数を超えるとアライメント不能 → スキップ
                if len(targets) >= emission.size(1):
                    continue
                targets_t = torch.tensor([targets], dtype=torch.int32, device=device)
                aligned, scores = forced_align(emission, targets_t, blank=0)
                spans = merge_tokens(aligned[0], scores[0].exp())
        except Exception as ex:
            print(f"  [skip] seg {seg['start']:.1f}-{seg['end']:.1f}s: {ex}")
            continue

        ratio = wav_seg.size(1) / emission.size(1) / sr
        agg = {i: [None, None, []] for i in range(len(kept))}
        for span, wi in zip(spans, word_idx):
            st = s0 + span.start * ratio
            en = s0 + span.end * ratio
            a = agg[wi]
            a[0] = st if a[0] is None else min(a[0], st)
            a[1] = en if a[1] is None else max(a[1], en)
            a[2].append(float(span.score))
        for i in range(len(kept)):
            st, en, sc = agg[i]
            if st is None:
                continue
            align.add(kept[i], st, en, sum(sc) / len(sc) if sc else None)
        n_seg_ok += 1

    align.save(args.out)
    print(f"[04b_mms_seg] ch{args.channel}: {n_seg_ok}/{len(segments)} seg, "
          f"{len(align.words)} words -> {args.out}")


if __name__ == "__main__":
    main()
