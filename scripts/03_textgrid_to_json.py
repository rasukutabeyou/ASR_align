#!/usr/bin/env python3
"""ステップ3c: MFA の TextGrid を統一スキーマ JSON に変換。

MFA は "words" tier と "phones" tier を出力する。words tier を採用。
空区間(無音)は除外する。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.audio_utils import duration_sec
from common.schema import Alignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--textgrid", required=True, help="MFA出力 ch{N}.TextGrid")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", type=int, required=True)
    ap.add_argument("--language", default="ja")
    args = ap.parse_args()

    from praatio import textgrid as tgio

    tg = tgio.openTextgrid(args.textgrid, includeEmptyIntervals=False)
    # tier 名は MFA バージョンで "words" / "ch{N} - words" 等の揺れがある
    word_tier = None
    for name in tg.tierNames:
        if "word" in name.lower():
            word_tier = tg.getTier(name)
            break
    if word_tier is None:
        raise SystemExit(f"words tier が見つかりません: {tg.tierNames}")

    align = Alignment(
        method="mfa", audio=os.path.basename(args.wav), channel=args.channel,
        language=args.language, sample_rate=16000, model="japanese_mfa",
        duration=duration_sec(args.wav), granularity="word")

    for start, end, label in word_tier.entries:
        label = label.strip()
        if not label or label in ("", "<eps>", "sil", "sp", "spn"):
            continue
        align.add(label, start, end, None)

    align.save(args.out)


if __name__ == "__main__":
    main()
