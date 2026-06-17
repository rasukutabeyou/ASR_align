#!/usr/bin/env python3
"""主観評価支援: 各手法の出力を1つの Praat TextGrid に束ねる。

チャンネルごとに、5手法を別 tier として並べた TextGrid を作る。
Praat で wav と一緒に開けば、同一時間軸上で全手法の単語境界を重ねて
目視・聴取確認できる(発話開始/終了、フィラー、言い直し等の確認に最適)。

併せて、手法別の語数・カバレッジを表示し、結合CSVも出す。

使い方:
    python 99_inspect.py --out-dir data/out --channel 0 \
        --wav data/work/101_1_2_zoom_ch0.wav
"""
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.schema import load
from common.audio_utils import duration_sec

METHODS = ["whisperx", "mfa", "mms", "vosk", "nemo"]


def sanitize(words, xmax):
    """praatio 用に重なり/逸脱を補正した (start, end, label) を返す。"""
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

    print(f"\n=== channel {args.channel} (dur={xmax:.1f}s) ===")
    print(f"{'method':10s} {'n_words':>8s} {'granularity':>12s}  {'coverage':>9s}")
    for m in METHODS:
        path = os.path.join(args.out_dir, f"{m}_alignment_ch{args.channel}.json")
        if not os.path.exists(path):
            print(f"{m:10s} {'--':>8s}  (未生成)")
            continue
        d = load(path)
        ents = sanitize(d["words"], xmax)
        covered = sum(e - s for s, e, _ in ents)
        print(f"{m:10s} {len(d['words']):>8d} {d.get('granularity',''):>12s}"
              f"  {covered/xmax*100:>7.1f}%")
        tier = tgio.IntervalTier(m, [Interval(s, e, lab) for s, e, lab in ents],
                                 0.0, xmax)
        tg.addTier(tier)

    tg_path = os.path.join(args.out_dir, f"compare_ch{args.channel}.TextGrid")
    tg.save(tg_path, format="long_textgrid", includeBlankSpaces=True)
    print(f"\n[99_inspect] TextGrid -> {tg_path}")
    print(f"           Praat で {os.path.basename(args.wav)} と一緒に開いて確認")

    # 結合CSV
    csv_path = os.path.join(args.out_dir, f"compare_ch{args.channel}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "idx", "start", "end", "dur", "word", "conf"])
        for m in METHODS:
            path = os.path.join(args.out_dir, f"{m}_alignment_ch{args.channel}.json")
            if not os.path.exists(path):
                continue
            for i, w in enumerate(load(path)["words"]):
                wr.writerow([m, i, w["start"], w["end"],
                             round(w["end"] - w["start"], 3),
                             w["word"], w.get("conf", "")])
    print(f"[99_inspect] CSV -> {csv_path}")


if __name__ == "__main__":
    main()
