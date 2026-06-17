"""音声入出力ユーティリティ。ffmpeg に依存せず soundfile + scipy で完結させる。

- ステレオ対話音声を L/R チャンネルに分離(話者分離)
- 16kHz mono へリサンプル(各 ASR/アライメントの想定入力)
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SR = 16000


def load_wav(path: str):
    """wav を float32 (n,) or (n, ch) で返す。"""
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (n, ch)
    return data, sr


def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """1次元 float 配列を 16kHz にリサンプル。"""
    if sr == TARGET_SR:
        return x.astype(np.float32)
    from math import gcd
    g = gcd(sr, TARGET_SR)
    up, down = TARGET_SR // g, sr // g
    return resample_poly(x, up, down).astype(np.float32)


def split_and_resample(src_wav: str, out_prefix: str, channels=(0, 1)) -> dict:
    """ステレオ wav を各チャンネルに分離し 16k mono wav として保存。

    返り値: {channel_index: 出力wavパス}
    """
    data, sr = load_wav(src_wav)
    n, ch = data.shape
    print(f"[audio] {src_wav}: sr={sr}, ch={ch}, dur={n/sr:.1f}s")
    out = {}
    for c in channels:
        if c >= ch:
            print(f"[audio] チャンネル {c} は存在しません(ch={ch})。スキップ。")
            continue
        mono = resample_to_16k(data[:, c], sr)
        path = f"{out_prefix}_ch{c}.wav"
        sf.write(path, mono, TARGET_SR, subtype="PCM_16")
        print(f"[audio] ch{c} -> {path} ({len(mono)/TARGET_SR:.1f}s @16k)")
        out[c] = path
    return out


def read_16k_mono(path: str) -> np.ndarray:
    """前処理済み(16k mono)wav を float32 (n,) で読む。"""
    data, sr = sf.read(path, dtype="float32")
    assert sr == TARGET_SR, f"{path} は {sr}Hz です。00_preprocess を先に実行してください。"
    if data.ndim > 1:
        data = data[:, 0]
    return data


def duration_sec(path: str) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate
