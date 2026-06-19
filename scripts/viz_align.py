#!/usr/bin/env python3
"""アライメントの timestamp 精度を目視確認するQC画像を生成する。

スペクトログラム(上段)と、各手法の単語区間を箱+ラベルで並べたレーン(下段)を
同一時間軸で重ねて描く。箱の端(語境界)が音声エネルギーの立ち上がり/減衰に
合っているかを目で確認できる(Praat の TextGrid 表示の簡易版)。

使い方:
  python viz_align.py --wav data/work/101_1_2_zoom_ch0.wav --out-dir data/out \
      --channel 0 --t0 0 --t1 30 \
      --methods whisperx,mfa,mms,vosk,nemo \
      --png data/out/viz_ch0_0-30_whisper.png --title "ch0 Whisperテキスト版"
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.audio_utils import read_16k_mono

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

SR = 16000


def load_words(out_dir, method, ch):
    p = os.path.join(out_dir, f"{method}_alignment_ch{ch}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))["words"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--channel", type=int, required=True)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=30.0)
    ap.add_argument("--methods", required=True, help="カンマ区切りの手法キー")
    ap.add_argument("--png", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    t0, t1 = args.t0, args.t1

    wav = read_16k_mono(args.wav)
    i0, i1 = int(t0 * SR), int(min(t1, len(wav) / SR) * SR)
    seg = wav[i0:i1]

    # スペクトログラム
    f, tt, Sxx = signal.spectrogram(seg, fs=SR, nperseg=400, noverlap=320)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    n = len(methods)
    fig_w = max(10, (t1 - t0) * 0.7)
    fig, (ax_s, ax_l) = plt.subplots(
        2, 1, figsize=(fig_w, 2.6 + 0.62 * n), sharex=True,
        gridspec_kw={"height_ratios": [3, max(1.2, 0.62 * n)]})

    ax_s.pcolormesh(tt + t0, f, Sxx_db, shading="auto", cmap="magma",
                    vmin=np.percentile(Sxx_db, 35), vmax=np.percentile(Sxx_db, 99))
    ax_s.set_ylabel("Hz")
    ttl = f"ch{args.channel}  {t0:.0f}–{t1:.0f}s"
    if args.title:
        ttl += f"   [{args.title}]"
    ax_s.set_title(ttl)
    ax_s.set_ylim(0, 8000)

    colors = ["#1f77b4", "#ff7f0e"]
    for li, m in enumerate(methods):
        y = n - 1 - li  # 上から順に並べる
        ws = load_words(args.out_dir, m, args.channel)
        label = m
        if ws is None:
            ax_l.text(t0 + 0.1, y + 0.5, f"{m} (未生成)", va="center",
                      fontsize=9, color="red")
            continue
        cnt = 0
        for k, w in enumerate(ws):
            s, e = float(w["start"]), float(w["end"])
            if e < t0 or s > t1:
                continue
            cnt += 1
            ss, ee = max(s, t0), min(e, t1)
            ax_l.add_patch(Rectangle((ss, y + 0.12), ee - ss, 0.76,
                           facecolor=colors[k % 2], edgecolor="black",
                           linewidth=0.4, alpha=0.45))
            # 境界線(start)
            ax_l.plot([s, s], [y + 0.08, y + 0.92], color="black", lw=0.4)
            if ee - ss > 0.04:
                ax_l.text((ss + ee) / 2, y + 0.5, str(w["word"]),
                          ha="center", va="center", fontsize=7)
        ax_l.text(t0 - (t1 - t0) * 0.005, y + 0.5, f"{label}",
                  ha="right", va="center", fontsize=9, fontweight="bold")

    ax_l.set_ylim(0, n)
    ax_l.set_yticks([])
    ax_l.set_xlim(t0, t1)
    ax_l.set_xlabel("time [s]")
    # 1秒グリッド
    import math
    for x in range(int(math.ceil(t0)), int(t1) + 1):
        ax_s.axvline(x, color="white", lw=0.3, alpha=0.25)
        ax_l.axvline(x, color="gray", lw=0.3, alpha=0.4)
    ax_l.set_xticks(range(int(math.ceil(t0)), int(t1) + 1))

    fig.tight_layout()
    fig.savefig(args.png, dpi=130, bbox_inches="tight")
    print(f"[viz_align] -> {args.png}  ({n}手法, {t0:.0f}-{t1:.0f}s)")


if __name__ == "__main__":
    main()
