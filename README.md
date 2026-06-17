# Oracle Timestamp 生成手法 比較実験

VAP 実験用の Oracle 単語レベル timestamp を、5手法で生成して主観比較するための一式。

| # | 手法 | 種別 | timestamp 粒度 | 共通転写を使う |
|---|------|------|----------------|----------------|
| 1 | Whisper → WhisperX | wav2vec2 forced align | word | ✔ |
| 2 | Whisper → MFA | Kaldi forced align | word | ✔ |
| 3 | Whisper → MMS | torchaudio MMS_FA(ローマ字) | word | ✔ |
| 4 | Vosk | オフライン ASR | word | ✗(自前認識) |
| 5 | NeMo(ReazonSpeech v2) | RNN-T ASR | subword | ✗(自前認識) |

WhisperX / MFA / MMS は **共通の Whisper 転写**(ステップ1)を入力に forced alignment する。
Vosk / NeMo は ASR と timestamp を同時に出す。

## データの扱い

- 入力: ステレオ対話音声(L/R が話者ごと)。例 `101_1_2_zoom.wav`(32kHz/stereo)
- **話者ごとに L/R を分離**し、各チャンネルを **16kHz mono** に変換してから全手法に投入
  (重複発話による mis-alignment を避け、Oracle 品質を最大化)
- 出力はチャンネルごと・手法ごとに統一スキーマ JSON(`common/schema.py`)

## ディレクトリ

```
config.sh                共通設定(AUDIO / CHANNELS / モデル等)
common/                  schema.py(統一JSON), audio_utils.py(分離・16k化)
scripts/00_preprocess.py 〜 06 / 99_inspect.py
setup/                   手法ごとの venv セットアップ
data/work/               前処理wav・MFA中間物
data/out/                各手法のJSON + 比較用 TextGrid/CSV
models/                  Voskモデル
```

## セットアップ(本番 24GB 環境 / venv・pip)

依存衝突を避けるため **手法ごとに独立 venv**。RTX 50系は `TORCH_INDEX` を cu128 に。

```bash
export TORCH_INDEX=https://download.pytorch.org/whl/cu128   # GPUに合わせて
bash setup/setup_tools.sh      # 前処理・可視化
bash setup/setup_whisperx.sh   # Whisper転写 + WhisperX
bash setup/setup_mms.sh
bash setup/setup_vosk.sh       # 日本語モデルもDL
bash setup/setup_nemo.sh
bash setup/setup_mfa.sh        # ※MFAのみ conda 必須(Kaldi依存)
```

> **MFA だけは pip 単体不可。** Kaldi バイナリに依存するため miniforge/conda が事実上必須です。
> venv/pip 方針でも MFA だけは conda 例外として扱ってください(`setup_mfa.sh` が案内します)。

## 実行

```bash
bash run_all.sh                          # 全手法・全チャンネル
METHODS="whisperx vosk" bash run_all.sh  # 一部だけ
AUDIO=/path/other.wav bash run_all.sh    # 別音声
```

各ステップは対応する venv の python で実行される(`run_all.sh` が振り分け)。

## 主観評価

`99_inspect.py` が **チャンネルごとに5手法を tier として並べた TextGrid** を生成する。

```
data/out/compare_ch0.TextGrid
data/out/compare_ch1.TextGrid
data/out/compare_ch0.csv
```

Praat で `data/work/<stem>_ch0.wav` と `compare_ch0.TextGrid` を一緒に開くと、
同一時間軸で5手法の単語境界を重ねて目視・聴取できる。確認観点:

- 単語開始/終了時刻の正確さ
- 発話開始付近・終了付近のズレ
- フィラー(えーと/あの 等)の扱い
- 長い単語の境界
- 言い直し箇所での崩れ

これらを照合し、Oracle 実験で採用するアライメント手法を決定する。

## 注意 / 既知の差異

- **粒度の違い**: NeMo(ReazonSpeech)はサブワード単位。日本語は単語境界が曖昧なため、
  公平な比較というより「どの粒度・精度が VAP に有用か」を見る観点で評価する。
- **分かち書き**: MFA/MMS は fugashi(UniDic)で形態素分割した語を単位にする。
  WhisperX は内部トークナイザ、Vosk はモデル語彙に従うため、語の切れ目は手法間で一致しない。
- **VRAM**: 検証環境は 16GB のため large-v3 + 各モデル同時は厳しい。手法ごとに venv を
  分け逐次実行する設計。本番 24GB なら large-v3(float16)で問題なし。
- このリポジトリのスクリプトは**本番環境向け**。検証環境(本機)では未実行。
