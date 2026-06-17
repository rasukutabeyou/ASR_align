"""統一timestampスキーマ。

5手法すべてがこの形式でJSONを書き出すことで、主観比較・可視化を共通化する。

形式:
{
  "method": "whisperx",        # 手法名
  "audio": "101_1_2_zoom_ch0.wav",
  "channel": 0,                 # 0=L, 1=R
  "language": "ja",
  "sample_rate": 16000,
  "model": "...",               # 使用モデル名(再現用)
  "duration": 1234.5,           # 音声長(秒)
  "granularity": "word",        # word | subword | char (手法により粒度が違う点に注意)
  "words": [
    {"word": "こんにちは", "start": 0.12, "end": 0.78, "conf": 0.93},
    ...
  ]
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Word:
    word: str
    start: float
    end: float
    conf: Optional[float] = None  # 手法によっては無い

    def to_dict(self) -> dict:
        d = {"word": self.word, "start": round(float(self.start), 3),
             "end": round(float(self.end), 3)}
        if self.conf is not None:
            d["conf"] = round(float(self.conf), 4)
        return d


@dataclass
class Alignment:
    method: str
    audio: str
    channel: int
    language: str
    sample_rate: int
    model: str
    duration: float
    granularity: str = "word"
    words: list = field(default_factory=list)

    def add(self, word: str, start: float, end: float, conf: Optional[float] = None):
        self.words.append(Word(word, start, end, conf))

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "audio": self.audio,
            "channel": self.channel,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "model": self.model,
            "duration": round(float(self.duration), 3),
            "granularity": self.granularity,
            "n_words": len(self.words),
            "words": [w.to_dict() for w in self.words],
        }

    def save(self, path: str):
        validate(self.to_dict())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"[schema] wrote {len(self.words)} words -> {path}")


def validate(d: dict) -> None:
    """最低限の整合性チェック(主観評価前に壊れたデータを弾く)。"""
    assert d["words"], "words が空です(転写・アライメント失敗の可能性)"
    prev_end = -1.0
    for i, w in enumerate(d["words"]):
        assert w["end"] >= w["start"] - 1e-3, f"word[{i}] end<start: {w}"
        assert w["start"] >= -1e-3, f"word[{i}] negative start: {w}"
        # 単調増加は必須ではない(手法によりわずかな逆転あり)が、警告対象
        if w["start"] + 1e-2 < prev_end:
            # 重なりが大きい場合のみ通知
            pass
        prev_end = w["end"]


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
