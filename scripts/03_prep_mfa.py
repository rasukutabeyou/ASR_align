#!/usr/bin/env python3
"""ステップ3a: MFA 用コーパス準備。

MFA の日本語辞書は単語単位なので、共通転写を形態素分割(fugashi/MeCab)し、
スペース区切りの .lab を作る。音声は同名 .wav をコーパスdirに配置する。

出力:
    <corpus>/ch{N}.wav
    <corpus>/ch{N}.lab   (例: "今日 は 良い 天気 です")
"""
import argparse
import os
import shutil


def tokenize_ja(text: str) -> str:
    """fugashi(UniDic)で分かち書き。記号は除去。"""
    from fugashi import Tagger
    tagger = Tagger()
    toks = []
    for m in tagger(text):
        surf = m.surface.strip()
        if not surf:
            continue
        pos = m.feature.pos1 if hasattr(m.feature, "pos1") else ""
        if pos == "補助記号" or surf in "、。「」『』（）()【】・…":
            continue
        toks.append(surf)
    return " ".join(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--transcript", required=True, help="transcript_ch{N}.txt")
    ap.add_argument("--corpus", required=True, help="MFAコーパス出力dir")
    ap.add_argument("--channel", type=int, required=True)
    args = ap.parse_args()

    os.makedirs(args.corpus, exist_ok=True)
    with open(args.transcript, encoding="utf-8") as f:
        text = f.read().strip()

    lab = tokenize_ja(text)
    name = f"ch{args.channel}"
    shutil.copy(args.wav, os.path.join(args.corpus, name + ".wav"))
    with open(os.path.join(args.corpus, name + ".lab"), "w", encoding="utf-8") as f:
        f.write(lab + "\n")
    print(f"[03_prep_mfa] {len(lab.split())} tokens -> {args.corpus}/{name}.lab")


if __name__ == "__main__":
    main()
