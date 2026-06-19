#!/usr/bin/env python3
"""指定時間窓で、各手法の (word, start, end) を「その手法の粒度のまま」並べる。

トークン単位はそろえない。正解(GT)との目視比較用に、
- long CSV : [source, start, end, dur, word] を start 昇順(全手法 + GT を混在)
- wide HTML: 時間順の階段レイアウト(列=手法/GT、行=トークンを start 順)
を出力する。GT は CSV(start,end,word)を --gt で渡すと先頭列に差し込む。

使い方:
  python compare_window.py --out-dir data/out --channel 0 --t0 0 --t1 30 \
      [--gt data/out/groundtruth_ch0.csv]
"""
import argparse
import csv
import html
import json
import os

# 並べる手法(キー = {key}_alignment_ch{N}.json)。左ほど優先表示。
# NeMo系は参考にならないため除外。Voskテキストに対する強制アライメントを追加。
METHODS = [
    "whisperx", "mfa", "mms", "vosk",
    "whisperx_on_vosk", "mfa_on_vosk", "mms_on_vosk",
]


def load_method(out_dir, m, ch):
    p = os.path.join(out_dir, f"{m}_alignment_ch{ch}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))["words"]


def load_gt(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("start"):
                continue
            s = float(r["start"])
            e = float(r["end"]) if r.get("end") not in (None, "") else s
            rows.append({"word": r.get("word", ""), "start": s, "end": e})
    return rows


def in_window(w, t0, t1):
    s, e = float(w["start"]), float(w["end"])
    return s < t1 and e > t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--channel", type=int, required=True)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=30.0)
    ap.add_argument("--gt", default=None, help="正解CSV(start,end,word)")
    args = ap.parse_args()

    ch, t0, t1 = args.channel, args.t0, args.t1

    # 列順: GT(あれば) -> 各手法
    columns = []
    if args.gt and os.path.exists(args.gt):
        columns.append(("GT", load_gt(args.gt)))
    for m in METHODS:
        ws = load_method(args.out_dir, m, ch)
        if ws is not None:
            columns.append((m, ws))

    # 窓内に絞る
    col_tokens = []
    for name, ws in columns:
        toks = [w for w in ws if in_window(w, t0, t1)]
        toks.sort(key=lambda w: float(w["start"]))
        col_tokens.append((name, toks))

    base = f"compare_window_ch{ch}_{int(t0)}-{int(t1)}s"

    # --- long CSV (start昇順, 全列混在) ---
    long_path = os.path.join(args.out_dir, base + ".csv")
    allrows = []
    for name, toks in col_tokens:
        for w in toks:
            s, e = float(w["start"]), float(w["end"])
            allrows.append([name, round(s, 3), round(e, 3),
                            round(e - s, 3), w["word"]])
    allrows.sort(key=lambda r: (r[1], r[0]))
    with open(long_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["source", "start", "end", "dur", "word"])
        wr.writerows(allrows)

    # --- wide HTML (時間順 階段レイアウト) ---
    # 全トークンを (start, 列index) で行に割り当て、各行は1セルのみ充填。
    events = []  # (start, col_index, html_cell)
    for ci, (name, toks) in enumerate(col_tokens):
        for w in toks:
            s, e = float(w["start"]), float(w["end"])
            cell = f'<b>{html.escape(str(w["word"]))}</b><br><span class="t">{s:.2f}–{e:.2f}</span>'
            events.append((s, ci, cell))
    events.sort(key=lambda x: (x[0], x[1]))

    ncol = len(col_tokens)
    th = "".join(f"<th>{html.escape(n)}</th>" for n, _ in col_tokens)
    rows_html = []
    for s, ci, cell in events:
        tds = []
        for j in range(ncol):
            klass = "gt" if (col_tokens[j][0] == "GT") else ""
            tds.append(f'<td class="{klass}">{cell if j == ci else ""}</td>')
        rows_html.append(f'<tr><td class="time">{s:6.2f}</td>{"".join(tds)}</tr>')

    html_doc = f"""<!doctype html><meta charset="utf-8">
<title>{base}</title>
<style>
 body{{font-family:"Noto Sans CJK JP",sans-serif;font-size:13px;}}
 table{{border-collapse:collapse;}}
 th,td{{border:1px solid #ccc;padding:2px 6px;vertical-align:top;text-align:center;min-width:60px;}}
 th{{position:sticky;top:0;background:#eee;}}
 td.time{{color:#888;font-variant-numeric:tabular-nums;text-align:right;background:#fafafa;}}
 td.gt{{background:#fff7e0;}}
 th:nth-child(2){{background:#ffe6a8;}}
 .t{{color:#0a7;font-size:11px;font-variant-numeric:tabular-nums;}}
</style>
<h3>ch{ch} {t0:.0f}–{t1:.0f}s  各手法のテキスト+timestamp(時間順・粒度そのまま)</h3>
<p>行は start 昇順。各行はその start を持つトークンを該当列に表示。GT列(あれば橙)が基準。</p>
<table><thead><tr><th>start[s]</th>{th}</tr></thead>
<tbody>{"".join(rows_html)}</tbody></table>
"""
    html_path = os.path.join(args.out_dir, base + ".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    print(f"[compare_window] ch{ch} {t0:.0f}-{t1:.0f}s  "
          f"列: {[n for n,_ in col_tokens]}")
    print(f"  long CSV -> {long_path}  ({len(allrows)} tokens)")
    print(f"  wide HTML -> {html_path}")


if __name__ == "__main__":
    main()
