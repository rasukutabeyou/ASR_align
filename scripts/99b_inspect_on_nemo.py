#!/usr/bin/env python3
"""主観評価支援(NeMoテキスト版): NeMo の認識テキストを共通入力として
WhisperX / MFA / MMS で強制アライメントした結果を1つの TextGrid に束ねる。

tier:
    nemo            … NeMo 認識(subword, 時刻はNeMo自身。テキストの出所=参照)
    whisperx_nemo   … NeMo テキストを WhisperX(wav2vec2)で強制align
    mfa_nemo        … 同 MFA
    mms_nemo        … 同 MMS

出力: compare_on_nemo_ch{N}.TextGrid / .csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.schema import load
from common.audio_utils import duration_sec

# tier名 -> 入力JSONファイル(out-dir 相対のテンプレ)
TIERS = [
    ("nemo", "nemo_alignment_ch{ch}.json"),
    ("whisperx_nemo", "whisperx_on_nemo_alignment_ch{ch}.json"),
    ("mfa_nemo", "mfa_on_nemo_alignment_ch{ch}.json"),
    ("mms_nemo", "mms_on_nemo_alignment_ch{ch}.json"),
]


def sanitize(words, xmax):
    out = []
    prev_end = 0.0
    for w in sorted(words, key=lambda x: x["start"]):
        s = max(float(w["start"]), prev_end)
        e = min(float(w["end"]), xmax)
        if e <= s:
            e = min(s + 0.01, xmax)
        if e <= s:
            continue
        out.append((s, e, w["word"]))
        prev_end = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--channel", type=int, required=True)
    ap.add_argument("--wav", required=True)
    args = ap.parse_args()

    from praatio import textgrid as tgio
    from praatio.utilities.constants import Interval

    xmax = duration_sec(args.wav)
    tg = tgio.Textgrid()
    tg.minTimestamp, tg.maxTimestamp = 0.0, xmax

    print(f"\n=== channel {args.channel} (dur={xmax:.1f}s) — NeMoテキスト共通 ===")
    print(f"{'tier':14s} {'n_words':>8s} {'granularity':>12s}  {'coverage':>9s}")
    for tier_name, tmpl in TIERS:
        path = os.path.join(args.out_dir, tmpl.format(ch=args.channel))
        if not os.path.exists(path):
            print(f"{tier_name:14s} {'--':>8s}  (未生成)")
            continue
        d = load(path)
        ents = sanitize(d["words"], xmax)
        covered = sum(e - s for s, e, _ in ents)
        print(f"{tier_name:14s} {len(d['words']):>8d} {d.get('granularity',''):>12s}"
              f"  {covered/xmax*100:>7.1f}%")
        tier = tgio.IntervalTier(tier_name,
                                 [Interval(s, e, lab) for s, e, lab in ents],
                                 0.0, xmax)
        tg.addTier(tier)

    tg_path = os.path.join(args.out_dir, f"compare_on_nemo_ch{args.channel}.TextGrid")
    tg.save(tg_path, format="long_textgrid", includeBlankSpaces=True)
    print(f"\n[99b_inspect] TextGrid -> {tg_path}")
    print(f"           Praat で {os.path.basename(args.wav)} と一緒に開いて確認")

    csv_path = os.path.join(args.out_dir, f"compare_on_nemo_ch{args.channel}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["tier", "idx", "start", "end", "dur", "word", "conf"])
        for tier_name, tmpl in TIERS:
            path = os.path.join(args.out_dir, tmpl.format(ch=args.channel))
            if not os.path.exists(path):
                continue
            for i, w in enumerate(load(path)["words"]):
                wr.writerow([tier_name, i, w["start"], w["end"],
                             round(w["end"] - w["start"], 3),
                             w["word"], w.get("conf", "")])
    print(f"[99b_inspect] CSV -> {csv_path}")


if __name__ == "__main__":
    main()
