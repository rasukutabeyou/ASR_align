#!/usr/bin/env python3
"""ステップ0: 前処理。

ステレオ対話音声を話者(L/R)ごとに分離し、16kHz mono wav を生成する。
以降の全手法はこの per-channel wav を入力にする。

使い方:
    python 00_preprocess.py --audio /path/to.wav --work-dir data/work --channels 0 1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.audio_utils import split_and_resample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--channels", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.audio))[0]
    out_prefix = os.path.join(args.work_dir, stem)
    paths = split_and_resample(args.audio, out_prefix, channels=tuple(args.channels))
    print("[00_preprocess] 完了:", paths)


if __name__ == "__main__":
    main()
