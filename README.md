# TextAnalysis : フランス語詩テクストの韻律・リズム・プロゾディー・音素統合分析システム

フランス近代詩（ボードレール、ランボー、ヴェルレーヌ、マラルメ等）および古典詩を対象とした、**詩的リズム・韻律・音声学（IPA）・統語パターンの高度統合分析プラットフォーム**です。

---

## 主な機能と理論的基盤

### 1. 理論派詩学に基づくリズム・韻律自動解析
- **Benoît de Cornulier 式古典・近代的韻律判定**:
  - 音節（音素）数計測（ミュタ・クム・リクィダ、伝統的詩的母音接続、ディエレズ/シネレズの選択解析）
  - 弱強アタック（Proclitique）と音節重複の厳密判定（$S_6, F_6, M_6, C_6$ 等）
  - 句調・アパント・半休止（Césure）および 6+6（アレクサンドラン標準格）、4+4+4（三拍律）、トリメーターの判定
- **Jean-Michel Gouvard 型押韻・音響詩学解析**:
  - 脚韻構成（抱擁韻・交差韻・平韻・三連韻等）の自動認識
  - 押韻の豊かさ（Pauvre / Suffisante / Riche）の音素別精密判定
  - ヒアチュス（Hiatus）、母音後方無音e（Postvocalic e）、句跨ぎ（Rejet / Contre-rejet / Enjambement）の検出
- **Henri Meschonnic ＆ Gérard Dessons 談話律動・音素連鎖論**:
  - 単語境界を超越した音素連鎖（Accords / Échos sonores）の追跡
  - アクセント対比（Contre-accents / Bipolar Clash）の動的スキャン
- **Albert Di Cristo プロゾディー統語・音響韻律**:
  - 韻律句境界（Minor / Major Phrase Boundary）
  - 連接辞（Enclitics / Proclitics）の弱化と音響衝突の検知

### 2. 高精度音声学（IPA）＆音素遷移トラッカー
- **高精度IPA自動転写**:
  - フランス語文法規則に基づく義務的・詩的リエゾン（Liaisons obligatoires / poétiques）の文脈自動補正
  - 無音のh（h muet）と有音のh（h aspiré）の語彙別完全判定
- **任意IPA音素・音素連鎖（n-gram）検索**:
  - 単一音素（例: `[s]`, `[k]`, `[ɑ̃]`）だけでなく、任意の音素連鎖（例: `[ɔʁ]`, `[wa]`, `[st]`, `[pʁ]`）を行別に瞬時追跡
  - 全36音素の手動追加・調音分類フィルター
- **長短行の歪みを排除する数理正規化指標**:
  - **12音節標準換算数**（$C \times 12 / S$）: 長短行の音節数格差をアレクサンドラン基準へ正規化
  - **標準語期待比（$O/E$）**: フランス語標準語彙コーパス対比の音素濃縮度倍率（基準線 $y=1.0$）
  - **行別統計偏差（$z$-score）**: 行内サンプルサイズを数理補正した統計的有意性指標

### 3. スタイロメトリー・統語プロファイル
- 句読点比率、単位語数、定動詞率、接続詞率（等位・従属）、関係詞頻度
- 品詞（POS）構成比率、語彙多様度（MATTR）、Zipf頻度スコア
- フランス語標準書き言葉コーパス（UD GSD）および話し言葉コーパス（ParisStories）との標準偏差比較

---

## 必要環境とインストール

### 動作環境
- Python 3.10 以上（3.11, 3.12, 3.13 動作確認済）

### 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

### 必要な自然言語処理モデル
```bash
python -m spacy download fr_core_news_md
```

---

## 起動方法

```bash
streamlit run app.py
```
ブラウザ上でローカルサーバー（デフォルト: `http://localhost:8501`）が自動起動します。

---

## テスト実行

全34項目の単体テスト（韻律・音節数・古典詩規範・音素転写・正規化数理モデル等）を実行します：
```bash
pytest tests/ -v
```

---

## ディレクトリ構成

```text
TextAnalysis/
├── app.py                  # Streamlit Web UI アプリケーション
├── prosody_rhythm.py       # Cornulier, Gouvard, Meschonnic, Di Cristo 理論分析エンジン
├── phonetics.py            # 音声学・IPA転写・リエゾン・音素統計モジュール
├── analysis.py             # 統語・スタイロメトリー・形態素解析統合層
├── reference.json          # フランス語標準コーパス分布データ
├── requirements.txt        # 依存ライブラリ一覧
├── pytest.ini              # テスト設定
├── tests/                  # 単体・結合テストスイート
└── README.md               # 本ドキュメント
```
