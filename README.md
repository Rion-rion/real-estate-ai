# 🏢 Tokyo Real Estate AI

東京都の中古マンションを対象に、物件情報から**想定成約価格を予測する機械学習システム**です。

国土交通省「不動産情報ライブラリ API」から取得した実際の不動産取引データを使用し、  
Python / pandas / CatBoost を用いてデータ取得・前処理・特徴量生成・モデル学習・評価・新規物件予測までを一連で実装しています。

---

## 📌 プロジェクト概要

本プロジェクトでは、東京都の中古マンションについて以下の処理を自動化します。

1. 国土交通省 API から不動産取引データを取得
2. データクレンジング・前処理
3. 特徴量生成
4. CatBoost による㎡単価予測
5. 予測㎡単価 × 専有面積から成約価格を算出
6. モデル精度評価
7. 新規物件の価格予測
8. 売出価格とAI推定価格の比較
9. CSV形式で予測結果を出力

---

## 🎯 目的

不動産物件の条件から、

- 想定成約価格
- 想定㎡単価
- 売出価格との差額
- 売出価格との乖離率

を算出し、不動産価格査定や販売価格検討の参考情報として利用できる予測基盤を構築することを目的としています。

---

## 🛠 使用技術

### Language

- Python 3

### Libraries

- pandas
- NumPy
- requests
- python-dotenv
- CatBoost
- scikit-learn

### Data Source

- 国土交通省 不動産情報ライブラリ API

### Version Control

- Git
- GitHub

---

## 🤖 機械学習モデル

価格予測には `CatBoostRegressor` を使用しています。

成約価格そのものを直接予測するのではなく、

```text
物件情報
    ↓
CatBoost
    ↓
予測㎡単価
    ↓
専有面積を掛ける
    ↓
予測成約価格

という構成にしています。

目的変数には成約㎡単価を対数変換した値を使用しています。

log_contract_price_per_m2

モデル出力後に逆変換し、

予測成約価格
=
予測㎡単価 × 専有面積

として最終価格を算出します。

📊 モデル精度

2021～2025年のデータをモデル構築に利用し、
2026年のデータを最終テストデータとして評価しています。

最終テスト結果
指標	結果
MAE	12,485,825円
RMSE	19,641,775円
MAPE	17.71%
R²	0.8528
㎡単価 MAE	201,196円/㎡
㎡単価 MAPE	17.71%

テストデータに対して R² = 0.8528 を記録しました。

📈 モデル改善

モデルは複数回改善を行いました。

Version	Test MAPE	Test R²
Ver.1	22.26%	0.7453
Ver.2	21.60%	0.7469
Ver.3	21.38%	0.7560
Ver.4	17.71%	0.8528

Ver.4では特徴量セットを複数パターン比較し、検証データで最も性能の良かった特徴量構成を採用しました。

採用された特徴量セット：

name_compact

さらに2025年の検証データをモデル選択に使用した後、
2021～2025年のデータを用いて最終モデルを再学習し、2026年データで最終評価を行っています。

🔍 主要特徴量

モデルの特徴量重要度上位は以下の通りです。

特徴量	Importance
city	35.67
building_age	26.65
district_name	10.19
transaction_year	8.48
area_m2	6.13
city_planning	5.38
floor_plan	3.82
structure	2.87
transaction_quarter	0.80

東京都の中古マンション価格では、

市区町村
築年数
地区
取引時期
専有面積

などが価格予測に大きく影響していることを確認しました。

🗂 ディレクトリ構成
real_estate_ai/
│
├─ data/
│  ├─ raw/
│  │  └─ tokyo_contract_prices.csv
│  │
│  ├─ processed/
│  │  └─ price_training.csv
│  │
│  └─ input/
│     └─ prediction_input.csv
│
├─ models/
│  └─ price_model.cbm
│
├─ output/
│  ├─ price_model_metrics.json
│  ├─ price_predictions.csv
│  ├─ price_test_predictions.csv
│  ├─ price_feature_importance.csv
│  └─ price_feature_set_comparison.csv
│
├─ collect_mlit.py
├─ preprocessing.py
├─ train_price.py
├─ predict.py
├─ main.py
├─ requirements.txt
├─ .gitignore
└─ README.md
📥 データ取得

collect_mlit.py では、国土交通省「不動産情報ライブラリ API」から東京都の中古マンション取引情報を取得します。

取得したデータは、

data/raw/tokyo_contract_prices.csv

へ保存されます。

APIキーは .env から読み込みます。

.env
MLIT_API_KEY=YOUR_API_KEY

.env は .gitignore に追加し、GitHubには公開しません。

🧹 前処理

preprocessing.py では以下を実施します。

カラム名統一
数値型変換
欠損値処理
築年数生成
取引年生成
四半期生成
㎡単価生成
異常値除外
統計的外れ値除外
カテゴリデータ整理
都市計画情報整理
学習用データ生成

出力：

data/processed/price_training.csv
🧠 モデル学習
python train_price.py

複数の特徴量セットを比較し、検証MAPEが最も小さい構成を自動選択します。

モデルは、

models/price_model.cbm

に保存されます。

精度評価結果は、

output/price_model_metrics.json

へ保存されます。

🏠 新規物件を予測する
1. 入力データ

data/input/prediction_input.csv に予測したい物件を入力します。

例：

property_id,city,district_name,area_m2,floor_plan,building_age,structure,renovation,use,city_planning,coverage_ratio,floor_area_ratio,transaction_year,transaction_quarter,asking_price
A001,足立区,千住,65.2,3LDK,12,RC,未改装,住宅,商業地域,80,400,2026,3,45000000
2. 予測
python predict.py

または、

python main.py
3. 出力
output/price_predictions.csv
💡 予測例
足立区 千住
面積: 65.2㎡

AI予測㎡単価:
856,766円/㎡

AI予測成約価格:
55,861,147円

売出価格:
45,000,000円

価格差:
-10,861,147円

価格乖離率:
-19.44%
世田谷区 三軒茶屋
AI予測成約価格:
58,601,201円

売出価格:
60,000,000円

価格乖離率:
2.39%
港区 六本木
AI予測成約価格:
132,054,525円

売出価格:
120,000,000円

価格乖離率:
-9.13%
▶ 実行方法
1. Repository Clone
git clone https://github.com/Rion-rion/real-estate-ai.git
cd real-estate-ai
2. ライブラリインストール
pip install -r requirements.txt
3. APIキー設定

プロジェクト直下に .env を作成します。

MLIT_API_KEY=YOUR_API_KEY
4. データ取得
python main.py collect
5. 前処理
python main.py preprocess
6. モデル学習
python main.py train
7. 価格予測
python main.py predict

または、

python main.py
8. 一括モデル構築
python main.py full

これにより、

データ取得
↓
前処理
↓
モデル学習

を一括実行できます。

🔐 セキュリティ

APIキーなどの秘密情報は .env に保存しています。

.gitignore

.env
.venv/
__pycache__/
*.pyc

data/raw/
data/processed/

models/*.cbm

output/price_test_predictions.csv

APIキーや学習元データはGitHubへ公開しない構成としています。

⚠ 現在の制約

現在のVer.1では以下の情報を利用していません。

最寄駅
駅徒歩分数
緯度・経度
所在階
周辺施設
地価
金利
マクロ経済指標

特に不動産価格では駅距離や位置情報の影響が大きいため、これらを追加することでさらなる精度改善が期待できます。

また、本モデルの予測値は不動産価格を保証するものではなく、参考値として利用することを想定しています。

🚀 今後の改善予定
駅情報の追加
駅徒歩時間の追加
緯度・経度特徴量の追加
地価情報との結合
エリア別精度評価
価格帯別精度評価
Excel入出力対応
可視化機能
Web UI
API化
Docker対応
AWS / GCPへのデプロイ

また、掲載開始日・成約日を含む教師データを確保できた場合、

成約までの日数を予測する機械学習モデル

の追加を予定しています。

📌 開発で意識した点

単純にモデル精度を出すだけでなく、以下を意識して開発しました。

データリーク防止
時系列を考慮したデータ分割
学習・検証・最終テストの分離
特徴量セットの比較
モデル評価の自動化
特徴量重要度の可視化用データ出力
APIキーの環境変数管理
データ取得から予測までの処理分離
Git / GitHubによるバージョン管理
再現可能なPython環境構築
📄 License

This project is created for portfolio and learning purposes.


これ、**今のGitHubにかなり合ってるREADME**になってる。

特に採用側に見てほしいのは、

**Ver.1 → Ver.4で MAPE 22.26% → 17.71%**  
**R² 0.7453 → 0.8528**

のところ。単にAIを動かしただけじゃなくて、**評価→問題発見→特徴量改善→再評価**までやったのが伝わる。

保存したら、

```powershell
git add README.md
git commit -m "Add project README"
git push