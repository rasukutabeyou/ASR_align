#!/usr/bin/env python3
"""ステップ4: MMS Alignment(torchaudio MMS_FA)。

torchaudio の多言語 forced alignment(MMS_FA)を使う。MMS_FA はローマ字系の
文字 CTC のため、日本語は uroman でローマ字化してからアライメントする。
(torchaudio 公式 "Forced alignment for multilingual data" の手順に準拠)

単語境界は形態素分割(fugashi)で決め、各単語をローマ字化したトークン列に
word index を持たせて、フレーム span を単語単位へ集約する。

出力: mms_alignment_ch{N}.json(統一スキーマ)
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.audio_utils import duration_sec
from common.schema import Alignment


def ja_words(text: str):
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
    ap.add_argument("--transcript", required=True, help="transcript_ch{N}.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", type=int, required=True)
    ap.add_argument("--language", default="ja")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    import torchaudio
    from torchaudio.pipelines import MMS_FA as bundle
    from torchaudio.functional import forced_align, merge_tokens
    from uroman import Uroman

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = bundle.get_model().to(device)
    DICT = bundle.get_dict()  # {char: id}

    uroman = Uroman()

    with open(args.transcript, encoding="utf-8") as f:
        text = f.read().strip()
    words = ja_words(text)

    # 各単語をローマ字化 → 小文字 → 辞書に無い文字を除去 → トークン化
    targets, word_idx = [], []   # flat token id 列 と 各tokenの単語index
    kept_words = []
    for wi, w in enumerate(words):
        roman = uroman.romanize_string(w, lcode="jpn").lower()
        roman = re.sub(r"[^a-z']", "", roman)  # 辞書文字に絞る
        ids = [DICT[c] for c in roman if c in DICT]
        if not ids:
            continue  # アライメント不能語(記号のみ等)はスキップ
        kept_words.append(w)
        for tid in ids:
            targets.append(tid)
            word_idx.append(len(kept_words) - 1)

    if not targets:
        raise SystemExit("ローマ字化後トークンが空です")

    # 波形読み込み(16k mono)。MMS は 16k 入力。
    waveform, sr = torchaudio.load(args.wav)
    assert sr == bundle.sample_rate, f"sr={sr}, expected {bundle.sample_rate}"
    waveform = waveform[0:1].to(device)

    with torch.inference_mode():
        emission, _ = model(waveform)
        targets_t = torch.tensor([targets], dtype=torch.int32, device=device)
        aligned, scores = forced_align(emission, targets_t, blank=0)
        spans = merge_tokens(aligned[0], scores[0].exp())

    # フレーム→秒の換算
    ratio = waveform.size(1) / emission.size(1) / sr

    # token span は targets と 1:1。単語ごとに min(start)/max(end)/mean(score)。
    n_words = len(kept_words)
    agg = {i: [None, None, []] for i in range(n_words)}
    for span, wi in zip(spans, word_idx):
        st, en = span.start * ratio, span.end * ratio
        a = agg[wi]
        a[0] = st if a[0] is None else min(a[0], st)
        a[1] = en if a[1] is None else max(a[1], en)
        a[2].append(float(span.score))

    align = Alignment(
        method="mms", audio=os.path.basename(args.wav), channel=args.channel,
        language=args.language, sample_rate=16000, model="MMS_FA(uroman)",
        duration=duration_sec(args.wav), granularity="word")
    for i in range(n_words):
        st, en, sc = agg[i]
        if st is None:
            continue
        align.add(kept_words[i], st, en, sum(sc) / len(sc) if sc else None)

    align.save(args.out)


if __name__ == "__main__":
    main()
