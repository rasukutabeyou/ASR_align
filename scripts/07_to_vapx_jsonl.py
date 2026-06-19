#!/usr/bin/env python3
"""ステップ7: 強制アライメント結果(統一スキーマJSON) → vapx 言語入力(ASR JSONL)。

Oracle 実験用。``common/schema.py`` の統一スキーマで書かれた単語レベル
timestamp(``vosk_alignment_ch{N}.json`` / ``mms_alignment_ch{N}.json`` 等)を、
vapx の ``extract_lang_features.py`` がそのまま消費できる
**ストリーミング ASR JSONL 契約**へ変換する。

vapx 側の取り込み口
-------------------
vapx の言語パイプラインは2段:

    run_*_asr.py  --(JSONL)-->  extract_lang_features.py  --(.npz)-->  model

``extract_lang_features.py`` が実際に読むのは各イベントの ``type`` /
``text``(空白区切りの語) / ``t_end_sec`` の3つだけ(``result`` の語境界は
トレース用のメタで、特徴抽出には未使用)。本スクリプトはこの3要素を
忠実に組み立てる。生成した JSONL は vapx 側の venv で

    python extract_lang_features.py --asr-jsonl <これ> --output <out>.npz \
        --model sbintuitions/sarashina2.2-0.5b --frame-hz 50

のように後段へ渡す(LM の選択・.npz 化は vapx 側に委ねる)。

JSONL 契約(run_vosk_asr.py / run_sherpa_asr.py と同一)
------------------------------------------------------
    {"t_end_sec": 0.48, "type": "partial", "text": "今日 は"}
    {"t_end_sec": 1.60, "type": "final",   "text": "今日 は 会議 です",
     "result": [{"word": "今日", "start": 0.30, "end": 0.55, "conf": 0.9}, ...]}

Oracle としての時刻づけ
-----------------------
強制アライメントは語の正確な start/end を持つ。「その語が観測される時刻」
(``t_end_sec``)は既定で語の **end**(= 語を聞き終えた瞬間)に合わせる。
vapx 側の音声×言語クロスアテンションは ``t_end_sec`` より未来の言語トークン
には注意できない因果マスクを張るため、end に合わせると「語を発話し終えて
初めて観測される」という最も厳密(かつ現実のストリーミング消費者に近い)
意味づけになる。``--observe start`` でやや寛容(発話開始で観測)にもできる。

発話(utterance)区切り
----------------------
統一スキーマはチャンネル全体のフラットな語列。連続する語の無音ギャップ
(``prev.end`` と ``next.start`` の差)が ``--gap-sec`` を超えたら発話境界と
みなし、直前までを ``final``(= EOS)で閉じて新しい発話を開始する。
発話内では語が1つ確定するごとに ``partial`` を1件出す(累積テキスト)。

チャンネルと出力名
-----------------
vapx の命名は話者ごとに ``-L`` / ``-R``。ASR_align は ch0=L / ch1=R なので、
``--out`` を明示しない場合は入力 JSON の ``channel`` から ``<stem>-<L|R>.jsonl``
を組み立てる(``--out-dir`` 配下)。``--out`` を渡せばそのパスを優先。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CH_TO_SIDE = {0: "L", 1: "R"}


def _clean_word(w: str) -> str:
    """JSONL の text は空白区切りで語を並べる契約。語内部の空白は潰す。"""
    return "".join(w.split())


def _build_events(
    words: list[dict],
    *,
    gap_sec: float,
    observe: str,
    emit_partials: bool,
    include_result: bool,
) -> list[dict]:
    """統一スキーマの words(各 {word,start,end,conf?}) を JSONL イベント列へ。

    語は start 昇順で走査し、無音ギャップで発話に区切る。各発話は
    [partial..., final] の並び(VOSK/sherpa と同じ contract)で吐く。
    """
    # start 昇順(タイ時は end)で安定ソート。強制アライメントの軽微な逆転に強くする。
    ordered = sorted(
        (w for w in words if _clean_word(str(w.get("word", "")))),
        key=lambda w: (float(w["start"]), float(w["end"])),
    )

    def obs_time(w: dict) -> float:
        return float(w["end"] if observe == "end" else w["start"])

    events: list[dict] = []

    def flush(utt: list[dict]) -> None:
        if not utt:
            return
        text = " ".join(_clean_word(str(w["word"])) for w in utt)
        ev: dict = {
            "t_end_sec": round(obs_time(utt[-1]), 4),
            "type": "final",
            "text": text,
        }
        if include_result:
            result = []
            for w in utt:
                item = {
                    "word": _clean_word(str(w["word"])),
                    "start": round(float(w["start"]), 4),
                    "end": round(float(w["end"]), 4),
                }
                if w.get("conf") is not None:
                    item["conf"] = round(float(w["conf"]), 4)
                result.append(item)
            ev["result"] = result
        events.append(ev)

    utt: list[dict] = []
    prev_end: float | None = None
    for w in ordered:
        if utt and prev_end is not None and float(w["start"]) - prev_end > gap_sec:
            flush(utt)
            utt = []
        utt.append(w)
        prev_end = float(w["end"])
        if emit_partials:
            text = " ".join(_clean_word(str(x["word"])) for x in utt)
            events.append({
                "t_end_sec": round(obs_time(w), 4),
                "type": "partial",
                "text": text,
            })
    flush(utt)
    return events


def _resolve_out(args, data: dict) -> Path:
    """出力パス。vapx の Dataset は言語特徴を厳密に ``<entry_id>-{L,R}.npz`` で
    引く(data/dataset.py)。lang.sh は .jsonl の stem をそのまま .npz 名に使うため、
    ここでの JSONL stem は **manifest の id + -L/-R** でなければならない
    (手法の区別はディレクトリ側で行うので、ファイル名に method 中置は付けない)。

    命名の優先順位:
      1. --out          : そのまま使用
      2. --id (+ --side / channel) : ``<out-dir>/<id>-<side>.jsonl``
      3. audio からの推定 : audio basename 末尾の ``_ch{N}`` を落として id とする
         (audio が ``<id>_ch{0,1}`` の形のときだけ正しい。違う場合は --id を渡す)。
    """
    if args.out is not None:
        return Path(args.out)

    ch = int(data.get("channel", 0))
    side = args.side or CH_TO_SIDE.get(ch, str(ch))

    if args.id is not None:
        entry_id = args.id
    else:
        audio = data.get("audio") or Path(args.inp).stem
        entry_id = Path(audio).stem
        for suf in (f"_ch{ch}", f"-ch{ch}"):
            if entry_id.endswith(suf):
                entry_id = entry_id[: -len(suf)]
                break

    return Path(args.out_dir) / f"{entry_id}-{side}.jsonl"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="inp", required=True,
                   help="統一スキーマ JSON (vosk_alignment_ch{N}.json / mms_alignment_ch{N}.json)")
    p.add_argument("--out", default=None,
                   help="出力 JSONL パス。'-' で stdout。省略時は --out-dir/<stem>-<L|R>.<method>.jsonl")
    p.add_argument("--out-dir", default=".",
                   help="--out 省略時の出力先ディレクトリ(default: カレント)")
    p.add_argument("--id", default=None,
                   help="出力 stem に使う vapx manifest の entry id(例 101_1_2)。"
                        "省略時は audio basename 末尾の _ch{N} を落として推定。")
    p.add_argument("--side", choices=("L", "R"), default=None,
                   help="チャンネル側(L/R)を明示。省略時は ch0->L, ch1->R。")
    p.add_argument("--gap-sec", type=float, default=0.5,
                   help="発話境界とみなす無音ギャップ秒(default 0.5)")
    p.add_argument("--observe", choices=("end", "start"), default="end",
                   help="語が観測される時刻を語の end / start のどちらに合わせるか(default end)")
    p.add_argument("--no-partials", action="store_true",
                   help="partial イベントを出さず final のみにする(BOS/EOS は維持)")
    p.add_argument("--no-result", action="store_true",
                   help="final に語境界 result 配列を含めない(text と t_end_sec のみ)")
    args = p.parse_args(argv)

    in_path = Path(args.inp)
    if not in_path.is_file():
        print(f"入力が見つかりません: {in_path}", file=sys.stderr)
        return 2
    with in_path.open(encoding="utf-8") as f:
        data = json.load(f)

    words = data.get("words", [])
    if not words:
        print(f"warn: {in_path} の words が空(転写/アライメント失敗?)", file=sys.stderr)

    events = _build_events(
        words,
        gap_sec=args.gap_sec,
        observe=args.observe,
        emit_partials=not args.no_partials,
        include_result=not args.no_result,
    )

    if args.out == "-":
        out_file = sys.stdout
        owns = False
        out_path = Path("-")
    else:
        out_path = _resolve_out(args, data)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path.open("w", encoding="utf-8")
        owns = True
    try:
        for ev in events:
            out_file.write(json.dumps(ev, ensure_ascii=False) + "\n")
    finally:
        if owns:
            out_file.close()

    n_final = sum(1 for e in events if e["type"] == "final")
    print(
        f"[to_vapx] {in_path.name} ({data.get('method','?')} ch{data.get('channel','?')}): "
        f"{len(words)} words -> {len(events)} events ({n_final} finals) -> {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
