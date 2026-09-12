# 🏢 Tokyo Real Estate AI

東京都の中古マンションを対象に、**AIによる想定成約価格の予測・売出価格との比較・価格評価・分析レポート生成**を行う機械学習システムです。

さらに、実際の販売開始日・成約日を含む履歴データを投入した場合に、**成約までに必要な日数を学習・予測できるパイプライン**も実装しています。

国土交通省「不動産情報ライブラリ API」から取得した実際の不動産取引データと鉄道駅GISデータを利用し、Python / pandas / CatBoost を用いて構築しています。

```text
国交省API
↓
データクレンジング
↓
駅GISデータ結合
↓
特徴量生成
↓
時系列データ分割
↓
モデル比較・学習
↓
未使用期間による最終評価
↓
CSV / Excel予測
↓
価格評価
↓
分析レポート
```

Ver.5では、**最寄駅・路線・駅座標**を価格予測モデルの特徴量として追加し、同一データ条件のベースラインモデルと比較して予測精度の改善を確認しました。

---

## 📌 実装状況

| 機能 | 状態 |
|---|---|
| 国交省APIから不動産データ取得 | ✅ 完成 |
| 鉄道駅GISデータ取得・結合 | ✅ 完成 |
| 東京都中古マンション抽出 | ✅ 完成 |
| データクレンジング・前処理 | ✅ 完成 |
| CatBoost価格予測モデル | ✅ 完成 |
| 時系列Train / Validation / Test分割 | ✅ 完成 |
| 駅特徴量を利用したモデル改善 | ✅ 完成 |
| CSV価格予測 | ✅ 完成 |
| Excel価格予測 | ✅ 完成 |
| 駅名から路線・駅座標を自動補完 | ✅ 完成 |
| 地区情報から代表駅を補完 | ✅ 完成 |
| 売出価格との価格乖離率算出 | ✅ 完成 |
| 割安 / 適正 / 割高判定 | ✅ 完成 |
| 市区町村別・駅別・価格帯別分析 | ✅ 完成 |
| Docker実行 | ✅ 完成 |
| 成約日数学習用Excelテンプレート | ✅ 完成 |
| 成約日数学習データ前処理 | ✅ 完成 |
| 価格AIと成約日数AIの連携 | ✅ 完成 |
| 成約日数モデル学習コード | ✅ 完成 |
| 成約日数予測用Excel | ✅ 完成 |
| 成約日数推論コード | ✅ 完成 |
| 成約日数モデル本学習・実データ精度評価 | ⏳ 教師データ待ち |

---

# 🎯 目的

不動産会社における、

- 査定業務
- 売出価格設定
- 価格改定判断
- 周辺相場との比較
- 販売期間の見通し
- 売出価格と販売期間の関係分析

を支援することを想定したAIシステムです。

価格予測では、物件情報と売出価格を入力すると以下を算出します。

- AI想定㎡単価
- AI想定成約価格
- 売出価格との差額
- 売出価格との価格乖離率
- 価格評価
  - 割安
  - 適正
  - やや割高
  - 割高

```text
物件情報
   ↓
駅情報補完
   ↓
価格予測AI
   ↓
AI想定成約価格
   ↓
売出価格との比較
   ↓
価格乖離率
   ↓
価格評価
```

成約日数AIでは、教師データを用いてモデルを学習した後、

```text
物件情報
↓
売出価格
↓
価格AIによる適正価格
↓
価格乖離率
↓
成約日数AI
↓
予測成約日数
↓
予測成約日
```

という形で利用できる構成にしています。

---

# 🛠 使用技術

## Language

- Python 3

## Machine Learning / Data Analysis

- CatBoost
- pandas
- NumPy
- scikit-learn
- matplotlib

## API / Data

- requests
- python-dotenv
- 国土交通省 不動産情報ライブラリ API

## Excel

- openpyxl

## Development / Environment

- Visual Studio Code
- Git
- GitHub
- Docker

---

# 📚 データ

国土交通省「不動産情報ライブラリ API」を利用しています。

価格モデルでは主に以下の情報を利用します。

- 市区町村
- 地区
- 成約価格
- 専有面積
- 間取り
- 築年数
- 建物構造
- 都市計画
- 取引年
- 取引四半期
- 最寄駅
- 路線
- 駅緯度
- 駅経度

駅情報は、不動産取引ポイントと鉄道駅GISデータを組み合わせて付与しています。

> GISマッチ時に使用する駅ポイント間の距離は、物件から駅までの徒歩距離を表すものではありません。  
> そのため、徒歩分数や物件－駅間距離の特徴量としては使用していません。

---

# 🗾 データ件数

駅特徴量付きデータ取得・結合後：

```text
194,092件
```

東京都データ：

```text
108,020件
```

駅名取得率：

```text
100.00%
```

外れ値除去などの前処理後、価格モデルに使用したデータ：

```text
106,937件
```

---

# 🤖 価格予測モデル Ver.5

価格予測には `CatBoostRegressor` を使用しています。

成約価格そのものを直接予測するのではなく、**成約㎡単価を予測した後、専有面積を掛けて最終価格を算出**します。

```text
物件情報
   ↓
CatBoost
   ↓
予測㎡単価
   ↓
予測㎡単価 × 専有面積
   ↓
AI予測成約価格
```

学習時の目的変数：

```text
log1p(contract_price_per_m2)
```

高価格物件の影響を抑えるため、㎡単価を対数変換して学習しています。

---

# 🕒 時系列データ分割

未来の情報が学習時に混入しないよう、ランダム分割ではなく**取引年による時系列分割**を採用しています。

```text
2021〜2024
Train
74,745件

2025
Validation
25,770件

2026
Final Test
6,422件
```

2025年データを使用して特徴量セットを比較し、モデルを選択します。

その後、

```text
2021〜2025
```

で最終モデルを再学習し、

```text
2026
```

を最終テストとして使用しています。

**2026年データはモデル選択には使用していません。**

---

# 🚉 駅特徴量

Ver.5では以下を追加しました。

```text
station_name
station_line
station_latitude
station_longitude
```

比較した特徴量セット：

```text
baseline_current
station_name
station_geo
station_geo_line
```

## 2025 Validation

| Feature Set | MAPE | R² |
|---|---:|---:|
| baseline_current | 17.19% | 0.8483 |
| station_name | 16.91% | 0.8530 |
| station_geo | 16.68% | 0.8556 |
| **station_geo_line** | **16.64%** | **0.8567** |

Validationで最も性能が良かった、

```text
station_geo_line
```

を最終モデルとして採用しました。

---

# 🏆 最終テスト結果

2026年データ6,422件による最終評価結果です。

| 指標 | 結果 |
|---|---:|
| Test Rows | 6,422 |
| MAE | **12,168,235円** |
| RMSE | **19,017,665円** |
| MAPE | **17.26%** |
| R² | **0.8655** |
| Median Absolute Error | 7,720,813円 |
| Median Percentage Error | 15.43% |

---

# 📈 駅特徴量の効果

駅特徴量そのものの効果を確認するため、**同じデータ・同じテスト期間**でベースラインと比較しています。

| Model | Test MAPE | Test R² |
|---|---:|---:|
| 駅特徴量なし | 17.76% | 0.8546 |
| **駅特徴量あり** | **17.26%** | **0.8655** |

改善幅：

```text
MAPE
17.76% → 17.26%
-0.50ポイント

R²
0.8546 → 0.8655
+0.0108
```

最寄駅・路線・駅座標を加えることで、同一条件のベースラインより予測精度が改善しました。

---

# 📈 モデル改善履歴

| Version | Test MAPE | Test R² |
|---|---:|---:|
| Ver.1 | 22.26% | 0.7453 |
| Ver.2 | 21.60% | 0.7469 |
| Ver.3 | 21.38% | 0.7560 |
| Ver.4 | 17.71% | 0.8528 |
| **Ver.5** | **17.26%** | **0.8655** |

> Ver.4以前とVer.5ではデータ取得・前処理パイプラインの一部が異なるため、Ver.4 → Ver.5は参考値です。  
> 駅特徴量の正式な比較値には、同一データによる `17.76% → 17.26%` を使用しています。

---

# 🔍 特徴量重要度

Ver.5最終モデルの上位特徴量：

| 順位 | 特徴量 | Importance |
|---:|---|---:|
| 1 | 市区町村 | 26.27 |
| 2 | 築年数 | 24.77 |
| 3 | 最寄駅経度 | 7.99 |
| 4 | 取引年 | 7.85 |
| 5 | 地区名 | 5.28 |
| 6 | 路線 | 5.20 |
| 7 | 専有面積 | 5.17 |
| 8 | 最寄駅 | 4.75 |
| 9 | 最寄駅緯度 | 3.97 |
| 10 | 都市計画 | 3.91 |

---

# 🏠 CSV価格予測

入力：

```text
data/input/prediction_input.csv
```

例：

```csv
property_id,city,district_name,station_name,area_m2,floor_plan,building_age,structure,city_planning,transaction_year,transaction_quarter,asking_price,station_line,station_latitude,station_longitude
A001,足立区,千住,北千住,65.2,3LDK,12,RC,商業地域,2026,3,65000000,,,
A002,世田谷区,三軒茶屋,三軒茶屋,55.0,2LDK,8,RC,近隣商業地域,2026,3,110000000,,,
A003,港区,六本木,六本木,70.0,2LDK,5,RC,商業地域,2026,3,270000000,,,
```

`station_line`・`station_latitude`・`station_longitude` は空欄でも実行できます。

入力された `station_name` を基に、学習データ内の駅情報から自動補完します。

実行：

```bash
python main.py predict
```

出力：

```text
output/price_predictions.csv
```

---

# 📗 Excel価格予測

実行：

```bash
python main.py excel
```

入力：

```text
data/input/prediction_input.xlsx
```

出力：

```text
output/price_predictions.xlsx
```

Excel入力で `station_name` がない場合は、

```text
city + district_name
```

が一致する学習データの中から、**最も出現頻度の高い代表的な駅**を補完します。

これは物件位置から厳密な最寄駅を計算する処理ではありません。

実際の最寄駅が分かっている場合は、入力された `station_name` を優先します。

駅名確定後、

```text
station_name
↓
station_line
station_latitude
station_longitude
```

を自動補完します。

---

# 💡 予測例

## 足立区 千住

```text
最寄駅: 北千住
路線: 常磐線
専有面積: 65.2㎡
```

AI想定成約価格：

```text
74,451,592円
```

売出価格：

```text
65,000,000円
```

価格乖離率：

```text
約 -12.69%
```

評価：

```text
割安
```

## 世田谷区 三軒茶屋

AI想定成約価格：

```text
106,135,213円
```

売出価格：

```text
110,000,000円
```

価格乖離率：

```text
約 +3.64%
```

評価：

```text
適正
```

## 港区 六本木

AI想定成約価格：

```text
234,674,334円
```

売出価格：

```text
270,000,000円
```

価格乖離率：

```text
約 +15.05%
```

評価：

```text
やや割高
```

> 上記の売出価格は動作確認用のサンプル値です。

---

# 📉 分析レポート

実行：

```bash
python main.py report
```

生成内容：

- 実際の成約価格 vs AI予測価格
- AI予測誤差率の分布
- 成約価格と誤差率
- 市区町村別MAPE
- 主要駅別MAPE
- 成約価格帯別MAPE
- 特徴量重要度
- 駅特徴量あり / なし比較

出力：

```text
output/analysis/
```

主なファイル：

```text
analysis_summary.csv
city_metrics.csv
station_metrics.csv
price_band_metrics.csv
model_comparison.csv

actual_vs_predicted.png
error_distribution.png
error_by_price.png
city_mape.png
station_mape.png
price_band_mape.png
feature_importance.png
model_comparison_mape.png
```

---

# ⏱ 成約日数予測AI

成約までの日数を予測するための、**入力テンプレート・前処理・モデル学習・価格AI連携・新規物件推論までのパイプライン**を実装しています。

## 学習フロー

```text
contract_history.xlsx
↓
販売開始日・成約日を検証
↓
成約日数を算出
↓
駅情報を補完
↓
Ver.5価格AIで適正価格を算出
↓
売出価格との価格乖離率を生成
↓
days_training.csv
↓
時間順 Train / Validation / Final Test
↓
CatBoost
↓
days_model.cbm
```

主な入力情報：

- 販売開始日
- 初回売出価格
- 成約日
- 成約価格
- 市区町村
- 地区
- 最寄駅
- 路線
- 駅座標
- 専有面積
- 間取り
- 築年数
- 建物構造
- 都市計画

価格AIから、

```text
ai_estimated_price
ai_price_per_m2
price_gap_ratio
```

も特徴量として成約日数AIへ渡す構成です。

売出価格そのものだけでなく、

```text
売出価格
-
AIによる適正価格
```

の関係を特徴量として利用できます。

## 成約日数モデルの学習設計

教師データが100件以上ある場合、

```text
古い70%
Train

次の15%
Validation

新しい15%
Final Test
```

の時間順で分割します。

ValidationでCatBoostのBest Iterationを決定した後、

```text
Train + Validation
= 85%
```

で最終モデルを再学習し、残り15%をFinal Testとして評価する構成です。

目的変数：

```text
log1p(days_to_contract)
```

評価指標：

```text
MAE
RMSE
Median Absolute Error
MAPE
SMAPE
R²
```

## 新規物件の推論フロー

実成約履歴からモデルを学習した後は、

```text
days_prediction_input.xlsx
↓
物件情報・売出価格
↓
代表駅 / 駅情報補完
↓
Ver.5価格AI
↓
AI想定成約価格
↓
価格乖離率
↓
成約日数AI
↓
予測成約日数
↓
予測成約日
↓
days_predictions.xlsx
```

まで実行できます。

実行：

```bash
python main.py days-predict
```

初回実行時に入力ファイルが存在しない場合、

```text
data/input/days_prediction_input.xlsx
```

を生成します。

モデル学習後の予測結果：

```text
output/days_predictions.xlsx
```

主な出力：

- AI想定成約価格
- AI想定㎡単価
- 売出価格との差額
- 価格乖離率
- 予測成約日数
- 予測成約日
- 成約期間区分

## 現在の状態

成約日数を学習するためには、

```text
販売開始日 → 成約日
```

の実履歴が必要です。

現在利用している公開データからは、価格モデルに必要な成約価格データは取得できますが、**販売開始日を含む十分な教師データを確保できていません。**

そのため現在は、

**成約日数AIの学習・推論パイプラインまで実装済みですが、本学習および実データによる精度評価は未実施です。**

教師データが存在しない場合はエラーとして無理に学習するのではなく、

```text
成約日数AI: 教師データ待ち
```

として正常終了します。

また、100件未満の場合も不安定な精度を算出しないため、本学習を停止する設計です。

架空データや疑似教師データから算出した精度を、実運用精度として扱わない方針です。

## 実成約データ取得後

テンプレート作成：

```bash
python main.py days-template
```

実成約履歴を、

```text
data/input/contract_history.xlsx
```

の `成約履歴` シートへ入力します。

前処理：

```bash
python main.py days-preprocess
```

学習：

```bash
python main.py days-train
```

前処理から学習まで一括：

```bash
python main.py days-full
```

新規物件予測：

```bash
python main.py days-predict
```

## データリークについて

成約日数AIの特徴量として価格AIの予測値を利用します。

過去の成約日数を厳密にバックテストする場合、上流の価格AIについても各時点より未来の価格情報を利用しない**時間順OOF（Out-of-Fold）予測**を作成することが望ましいと考えています。

現在の実装ではこの点を明示し、実成約データ取得後の本格的なモデル評価時に追加検証する設計としています。

---

# 🐳 Docker

Dockerを利用してローカルPython環境に依存せず実行できます。

## Build

```bash
docker build -t real-estate-ai .
```

## 起動確認

```bash
docker run --rm real-estate-ai
```

## CSV価格予測

Windows PowerShell：

```powershell
docker run --rm `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/output:/app/output" `
  real-estate-ai predict
```

Docker環境でも、

- CatBoostモデル読込
- 駅参照データ読込
- 路線自動補完
- 駅座標自動補完
- Ver.5価格予測
- ホスト側へのCSV出力

まで動作確認済みです。

## APIを利用する場合

```powershell
docker run --rm `
  --env-file .env `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/output:/app/output" `
  real-estate-ai station
```

`.env` はDockerイメージおよびGitHubへ含めません。

---

# 🗂 主なディレクトリ構成

```text
real_estate_ai/
├─ data/
│  ├─ raw/
│  │  └─ tokyo_contract_prices_station.csv
│  ├─ processed/
│  │  ├─ price_training.csv
│  │  └─ days_training.csv
│  └─ input/
│     ├─ prediction_input.csv
│     ├─ prediction_input.xlsx
│     ├─ contract_history.xlsx
│     └─ days_prediction_input.xlsx
│
├─ models/
│  ├─ price_model.cbm
│  └─ days_model.cbm
│
├─ output/
│  ├─ analysis/
│  ├─ price_model_metrics.json
│  ├─ price_feature_importance.csv
│  ├─ price_final_test_comparison.csv
│  ├─ price_test_predictions.csv
│  ├─ price_predictions.csv
│  ├─ price_predictions.xlsx
│  ├─ days_model_metrics.json
│  ├─ days_feature_importance.csv
│  ├─ days_test_predictions.csv
│  └─ days_predictions.xlsx
│
├─ collect_mlit.py
├─ collect_station_features.py
├─ preprocessing.py
├─ train_price.py
├─ predict.py
├─ predict_excel.py
├─ analysis_report.py
├─ create_days_template.py
├─ preprocessing_days.py
├─ train_days.py
├─ predict_days.py
├─ main.py
├─ Dockerfile
├─ .dockerignore
├─ requirements.txt
├─ .gitignore
└─ README.md
```

> `days_training.csv` は実成約履歴を前処理した後に生成されます。  
> `days_model.cbm`・成約日数モデル評価ファイルは、実成約履歴を用いて本学習を行った後に生成されます。  
> `days_prediction_input.xlsx` は `days-predict` の初回実行時に自動生成できます。

---

# ▶ セットアップ

## Clone

```bash
git clone https://github.com/Rion-rion/real-estate-ai.git
cd real-estate-ai
```

## 仮想環境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## ライブラリ

```bash
pip install -r requirements.txt
```

## APIキー

プロジェクト直下に `.env` を作成します。

```env
MLIT_API_KEY=YOUR_API_KEY
```

---

# ▶ 主なコマンド

価格予測：

```bash
python main.py predict
python main.py excel
python main.py report
```

データ取得・前処理・学習：

```bash
python main.py station
python main.py preprocess
python main.py train
```

価格AI一括構築：

```bash
python main.py price-full
```

最新データを再取得して価格AIを再構築：

```bash
python main.py price-refresh
```

成約日数AI：

```bash
python main.py days-template
python main.py days-preprocess
python main.py days-train
python main.py days-full
python main.py days-predict
```

システム状態確認：

```bash
python main.py status
```

全体構築：

```bash
python main.py all
```

---

# 🔐 セキュリティ

APIキーはソースコードへ直接記述せず、

```text
.env
```

で管理しています。

`.env` はGit管理対象外です。

また、rawデータ・processedデータ・モデル・出力結果など、ローカル生成物や大容量ファイルは必要に応じてGit管理から除外しています。

---

# ⚠️ モデルの制約

本システムは不動産価格の分析・査定・販売判断支援を目的とした機械学習プロジェクトであり、正式な不動産鑑定評価を代替するものではありません。

現在の公開データでは、以下のような価格形成に重要な情報が十分に取得できていません。

- 駅徒歩分数
- 所在階
- 方角
- 眺望
- 総戸数
- 管理状態
- 室内状態
- リフォーム詳細
- マンションブランド

そのため、実運用では周辺相場や個別物件情報と組み合わせて利用することを想定しています。

成約日数AIについても、販売活動・広告状況・仲介会社・内見件数・価格改定履歴など、現時点で取得していない要因が販売期間へ影響する可能性があります。

---

# 🧠 このプロジェクトで重視したこと

単純にモデル精度だけを追うのではなく、

- 公開実データの取得
- GISデータとの結合
- 欠損・外れ値処理
- データリークの防止
- 時系列Train / Validation / Test分割
- Validationによるモデル選択
- 未使用期間による最終評価
- ベースラインとの比較
- CSV / Excelによる推論
- 業務を想定した価格評価
- 価格AIと成約日数AIの連携
- 入力値チェック
- 教師データ不足時の安全な停止
- 分析結果の可視化
- APIキーの安全な管理
- Dockerによる再現性
- 教師データがない課題では無理に精度を作らない

ことを意識して実装しています。

---

# 📌 今後の改善候補

価格予測AIでは、取得可能になれば以下の追加を検証できます。

- 駅徒歩時間
- 所在階
- 方角
- マンション規模
- 周辺地価
- 周辺施設
- 金利
- 市況指標

成約日数AIでは、実際の販売開始日・成約日を含む成約履歴データを確保した段階で、

- 本学習
- Final Testによる実データ評価
- 上流価格AIの時間順OOF予測
- 価格改定履歴
- 内見件数
- 問い合わせ件数
- 広告掲載状況
- 仲介条件

などを追加検証する予定です。

---

## Author

GitHub: **Rion-rion**