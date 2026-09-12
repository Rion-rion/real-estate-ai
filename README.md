🏢 Tokyo Real Estate AI

東京都の中古マンションを対象に、AIによる想定成約価格の予測・売出価格との比較・価格評価・分析レポート生成を行う機械学習システムです。
国土交通省「不動産情報ライブラリ API」の不動産取引データと鉄道駅GISデータを利用し、Python / pandas / CatBoost で構築しています。
販売開始日・成約日を含む実履歴を投入した場合に、成約までの日数を学習・予測するパイプラインも実装しています。成約日数モデルは現在、十分な教師データを確保できていないため、本学習・実データ精度評価は未実施です。

📌 主な結果

項目

結果

価格モデル学習データ

106,937件

Final Test

2026年 6,422件

Test MAPE

17.26%

Test R²

0.8655

駅特徴量なし MAPE

17.76%

駅特徴量あり MAPE

17.26%

MAPE改善

0.50ポイント

駅名取得率

100.00%

🔄 システム構成

国交省API
  ↓
不動産取引データ / 鉄道駅GIS
  ↓
データクレンジング・駅GIS結合
  ↓
特徴量生成
  ↓
時系列 Train / Validation / Final Test
  ↓
特徴量セット比較・モデル学習
  ↓
未使用期間による最終評価
  ↓
CSV / Excel予測
  ↓
売出価格との比較・価格評価
  ↓
分析レポート

Ver.5では、最寄駅・路線・駅緯度・駅経度を価格モデルの特徴量へ追加し、同一データ・同一テスト期間のベースラインと比較しています。

🎯 目的

不動産会社における次の業務支援を想定しています。
査定業務
売出価格設定
価格改定判断
周辺相場との比較
販売期間の見通し
売出価格と販売期間の関係分析
価格予測では以下を算出します。
AI想定㎡単価
AI想定成約価格
売出価格との差額
価格乖離率
価格評価（割安 / 適正 / やや割高 / 割高）

🛠 使用技術

分類

技術

Language

Python 3

Machine Learning

CatBoost, scikit-learn

Data Analysis

pandas, NumPy

Visualization

matplotlib

API

requests, python-dotenv

Excel

openpyxl

Development

VS Code, Git, GitHub

Environment

Docker

Data Source

国土交通省 不動産情報ライブラリ API

📚 データ

主な特徴量:

市区町村 / 地区

専有面積 / 間取り / 築年数

建物構造 / 都市計画

取引年 / 取引四半期

最寄駅 / 路線 / 駅緯度 / 駅経度

データ件数:

段階

件数

駅特徴量付きデータ取得・結合後

194,092

東京都データ

108,020

価格モデル学習データ

106,937

GISマッチ時の駅ポイント間距離は、物件から駅までの徒歩距離を表すものではないため、徒歩分数や物件－駅間距離の特徴量としては使用していません。

🤖 価格予測モデル Ver.5

価格予測には CatBoostRegressor を使用しています。

成約価格を直接予測せず、成約㎡単価を予測した後に専有面積を掛けて最終価格を算出します。

物件情報
  ↓
CatBoost
  ↓
予測㎡単価
  ↓
予測㎡単価 × 専有面積
  ↓
AI予測成約価格

目的変数:

log1p(contract_price_per_m2)

🕒 時系列分割

期間

用途

件数

2021〜2024

Train

74,745

2025

Validation

25,770

2026

Final Test

6,422

2025年で特徴量セットを比較してモデルを選択し、2021〜2025年で最終モデルを再学習します。

2026年データはモデル選択に使用していません。

🚉 駅特徴量

追加した特徴量:

station_name
station_line
station_latitude
station_longitude

2025 Validation:

Feature Set

MAPE

R²

baseline_current

17.19%

0.8483

station_name

16.91%

0.8530

station_geo

16.68%

0.8556

station_geo_line

16.64%

0.8567

Validationで最も性能が良かった station_geo_line を採用しました。

🏆 Final Test

指標

結果

Test Rows

6,422

MAE

12,168,235円

RMSE

19,017,665円

MAPE

17.26%

R²

0.8655

Median Absolute Error

7,720,813円

Median Percentage Error

15.43%

同一条件ベースラインとの比較:

Model

Test MAPE

Test R²

駅特徴量なし

17.76%

0.8546

駅特徴量あり

17.26%

0.8655

MAPE: 17.76% → 17.26%

R²: 0.8546 → 0.8655

Ver.4以前とはデータ取得・前処理パイプラインの一部が異なるため、正式な駅特徴量比較には同一データの 17.76% → 17.26% を使用しています。

🏠 価格予測

CSV:

python main.py predict

data/input/prediction_input.csv
  ↓
output/price_predictions.csv

Excel:

python main.py excel

data/input/prediction_input.xlsx
  ↓
output/price_predictions.xlsx

station_line・station_latitude・station_longitude は空欄でも実行できます。

station_name がない場合は、city + district_name が一致する学習データから最頻の代表駅を補完します。実際の最寄駅が分かる場合は入力値を優先します。

💡 予測例

地域

最寄駅

売出価格

AI想定成約価格

乖離率

評価

足立区 千住

北千住

65,000,000円

74,451,592円

-12.69%

割安

世田谷区 三軒茶屋

三軒茶屋

110,000,000円

106,135,213円

+3.64%

適正

港区 六本木

六本木

270,000,000円

234,674,334円

+15.05%

やや割高

売出価格は動作確認用のサンプル値です。

📉 分析レポート

python main.py report

生成内容:

実成約価格 vs AI予測価格

誤差率分布

市区町村別MAPE

主要駅別MAPE

価格帯別MAPE

特徴量重要度

駅特徴量あり / なし比較

出力先:

output/analysis/

⏱ 成約日数予測AI

入力テンプレート・前処理・モデル学習・価格AI連携・新規物件推論までのパイプラインを実装しています。

contract_history.xlsx
  ↓
販売開始日・成約日を検証
  ↓
成約日数算出
  ↓
駅情報補完
  ↓
Ver.5価格AI
  ↓
AI想定価格・価格乖離率
  ↓
days_training.csv
  ↓
時間順 Train / Validation / Final Test
  ↓
CatBoost
  ↓
days_model.cbm

価格AIから以下も特徴量として渡します。

ai_estimated_price
ai_price_per_m2
price_gap_ratio

学習設計

教師データが100件以上ある場合:

区分

割合

Train

70%

Validation

15%

Final Test

15%

ValidationでBest Iterationを決定し、Train + Validation（85%）で最終モデルを再学習します。

目的変数:

log1p(days_to_contract)

評価指標:

MAE / RMSE / Median Absolute Error / MAPE / SMAPE / R²

現在の状態

成約日数AIには 販売開始日 → 成約日 の実履歴が必要です。
現在の公開データでは十分な教師データを確保できていないため、学習・推論パイプラインは実装済みですが、本学習・実データ精度評価は未実施です。
教師データなし: 正常な「教師データ待ち」として終了
100件未満: 不安定な精度を出さないため学習停止
架空データ・疑似教師データの精度は実運用精度として扱わない

実データ取得後:

python main.py days-template
python main.py days-preprocess
python main.py days-train
python main.py days-full
python main.py days-predict

データリーク

成約日数AIでは価格AIの予測値を特徴量として利用します。

厳密なバックテストでは、上流の価格AIについても各時点より未来の価格情報を利用しない時間順OOF（Out-of-Fold）予測を作成する必要があります。実成約データ取得後の本格評価で追加検証する設計です。

🐳 Docker

docker build -t real-estate-ai .
docker run --rm real-estate-ai

CSV価格予測（PowerShell）:

docker run --rm `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/output:/app/output" `
  real-estate-ai predict

API利用時:

docker run --rm `
  --env-file .env `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/output:/app/output" `
  real-estate-ai station

🗂 主なファイル

collect_mlit.py
collect_station_features.py
preprocessing.py
train_price.py
predict.py
predict_excel.py
analysis_report.py
create_days_template.py
preprocessing_days.py
train_days.py
predict_days.py
main.py
Dockerfile
requirements.txt
README.md

▶ セットアップ

git clone https://github.com/Rion-rion/real-estate-ai.git
cd real-estate-ai
python -m venv .venv
pip install -r requirements.txt

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

.env:

MLIT_API_KEY=YOUR_API_KEY

▶ 主なコマンド

# 価格AI
python main.py station
python main.py preprocess
python main.py train
python main.py predict
python main.py excel
python main.py report
python main.py price-full
python main.py price-refresh

# 成約日数AI
python main.py days-template
python main.py days-preprocess
python main.py days-train
python main.py days-full
python main.py days-predict

# システム
python main.py status
python main.py all

🔐 セキュリティ

APIキーは .env で管理し、Git管理対象外としています。

raw / processedデータ、モデル、出力結果などのローカル生成物・大容量ファイルも必要に応じてGit管理から除外します。

⚠️ 制約

本システムは分析・査定・販売判断支援を目的とする機械学習プロジェクトであり、正式な不動産鑑定評価を代替するものではありません。
公開データでは、駅徒歩分数・所在階・方角・眺望・総戸数・管理状態・室内状態・リフォーム詳細・マンションブランドなどを十分に取得できていません。
成約日数AIでも、販売活動・広告状況・仲介会社・内見件数・価格改定履歴などの未取得要因があります。
🧠 重視したこと
公開実データの取得
GISデータとの結合
欠損・外れ値処理
データリークの防止
時系列分割
Validationによるモデル選択
未使用期間による最終評価
同一条件のベースライン比較
CSV / Excel推論
業務を想定した価格評価
価格AIと成約日数AIの連携
教師データ不足時の安全な停止
APIキー管理
Dockerによる再現性
教師データがない課題で無理に精度を作らない
📌 今後の改善候補
価格AI:
駅徒歩時間
所在階
方角
周辺地価
周辺施設
金利・市況指標
成約日数AI:
実データによる本学習・Final Test
上流価格AIの時間順OOF予測
価格改定履歴
内見件数 / 問い合わせ件数
広告掲載状況 / 仲介条件

Author
GitHub: Rion-rion