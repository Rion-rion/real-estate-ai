# 🏢 不動産価格査定・案件管理システム

東京都の中古マンションを対象に、過去の実取引データから学習した機械学習モデルを利用して、成約価格の推定、売出価格との比較、査定履歴、案件管理までを一体化した業務支援システムです。

国土交通省「不動産情報ライブラリ API」の不動産取引データと鉄道駅GISデータを利用し、Python / pandas / CatBoost / Streamlit / Supabase で構築しています。

`main` ブランチは公開ポートフォリオ版、`commercial` ブランチは認証・権限管理・案件管理を追加した商用プロトタイプです。

## 🌐 Public Demo

公開ポートフォリオ版:

https://real-estate-ai-hugilvbuq8poka4tqnyvx5.streamlit.app/

> 商用プロトタイプの認証情報は公開していません。

## ✨ 主な機能

- 1件査定
- Excel / CSV 一括査定
- 推定成約価格
- 推定㎡単価
- 売出価格との差額
- 価格乖離率
- 価格評価
- 価格比較グラフ
- ㎡単価比較グラフ
- 査定結果Excel出力
- 営業向け参考コメント
- 説明用サマリー
- Supabase Auth によるユーザー認証
- JWT を利用したAPIアクセス
- Row Level Security（RLS）によるユーザー別データ分離
- ユーザー別査定履歴
- モデルVersion保存
- 案件管理 CRUD
- 案件ステータス管理
- 案件ごとの複数査定履歴
- `cases.id` と `appraisal_history.case_id` の外部キー連携
- ライト / ダーク表示
- 円 / 万円表示

## 📌 価格モデル

| 項目 | 結果 |
|---|---:|
| 最終データ件数 | 106,937件 |
| Train | 2021〜2024 / 74,745件 |
| Validation | 2025 / 25,770件 |
| Final Test | 2026 / 6,422件 |
| MAE | 12,168,235円 |
| RMSE | 19,017,665円 |
| MAPE | 17.26% |
| R² | 0.8655 |
| 駅特徴量なし MAPE | 17.76% |
| 駅特徴量あり MAPE | 17.26% |
| 駅名取得率 | 100.00% |

2026年のデータはモデル選択には使用せず、最終評価用の未使用期間として利用しています。

### モデル概要

価格予測には `CatBoostRegressor` を使用しています。

成約価格を直接予測するのではなく、成約㎡単価を予測し、専有面積を掛けて推定成約価格を算出します。

```text
物件情報
   ↓
CatBoost
   ↓
推定㎡単価
   ↓
推定㎡単価 × 専有面積
   ↓
推定成約価格
```

目的変数:

```text
log1p(contract_price_per_m2)
```

主な特徴量:

- 市区町村
- 地区
- 専有面積
- 間取り
- 築年数
- 建物構造
- 都市計画 / 用途地域
- 取引年
- 取引四半期
- 最寄駅
- 路線
- 駅緯度
- 駅経度

## 🚉 駅特徴量

Ver.5 では以下を追加しています。

- `station_name`
- `station_line`
- `station_latitude`
- `station_longitude`

2025 Validation:

| Feature Set | MAPE | R² |
|---|---:|---:|
| baseline_current | 17.19% | 0.8483 |
| station_name | 16.91% | 0.8530 |
| station_geo | 16.68% | 0.8556 |
| station_geo_line | 16.64% | 0.8567 |

最もValidation MAPEが低かった `station_geo_line` を採用しました。

Final Testでは、MAPEは17.76%から17.26%、R²は0.8546から0.8655へ改善しました。

## 🏗 システム構成

```text
国土交通省 不動産情報ライブラリ API
        ↓
不動産取引データ
鉄道駅GISデータ
        ↓
データクレンジング
駅GIS結合
        ↓
特徴量生成
        ↓
2021〜2024 Train
2025 Validation
2026 Final Test
        ↓
CatBoost
        ↓
price_model.cbm
        ↓
Streamlit
        ↓
Supabase Auth
        ↓
JWT
        ↓
PostgreSQL + RLS
        ↓
cases
  └─ appraisal_history
       └─ case_id
```

Webアプリでは約10.7万件の学習データを毎回読み込むのではなく、学習済みモデル `models/price_model.cbm` を利用して推論します。

駅情報は軽量な `data/reference/station_reference.csv` を利用します。

## 🗂 案件管理

商用プロトタイプでは、査定だけでなく案件そのものを管理できます。

主な管理項目:

- 案件名
- 物件ID
- ステータス
- 市区町村
- 地区
- 最寄駅
- 専有面積
- 間取り
- 築年数
- 売出価格
- メモ
- 作成日時
- 更新日時

ステータス例:

- 査定中
- 提案中
- 販売中
- 成約
- 保留
- 失注

## 🔗 案件と査定履歴

`cases.id` と `appraisal_history.case_id` を外部キーで紐づけています。

```text
cases
└─ 北千住マンション
   ├─ 査定 1 / Ver.5
   ├─ 査定 2 / Ver.5
   └─ 査定 3 / Ver.6
```

これにより、1つの案件に対して複数回の査定結果を時系列で保持できます。

単発査定では `case_id = NULL` とし、案件に紐づけない利用も可能です。

## 🔐 認証・権限管理

Supabase Auth を使用しています。

アプリからSupabaseへアクセスする際は、Publishable keyとログインユーザーのJWTを使用します。

```text
ログイン
  ↓
Supabase Auth
  ↓
Access Token / JWT
  ↓
Streamlit
  ↓
Supabase REST API
  ↓
RLS
```

`cases` と `appraisal_history` はRLSを有効化し、ログインユーザー本人のデータだけを参照・作成・更新・削除できる構成にしています。

異なる2ユーザーで、相互の査定履歴・案件が表示されないことを実動作で確認しています。

### Secrets

ローカル環境:

```text
.streamlit/secrets.toml
```

例:

```toml
[supabase]
url = "https://YOUR_PROJECT.supabase.co"
publishable_key = "YOUR_PUBLISHABLE_KEY"
```

`secrets.toml` はGit管理対象外です。

Service Role / Secret key はWebアプリ側では使用しません。

## 📊 Excel一括査定

入力テンプレートから複数物件をまとめて査定できます。

対応形式:

- `.xlsx`
- `.csv`

査定結果はExcelとして出力できます。

## 🗃 査定履歴

査定結果はSupabase PostgreSQLへ保存します。

保存項目例:

- 査定日時
- ユーザーID
- 案件ID
- モデルVersion
- 案件名
- 担当者
- 物件ID
- 市区町村
- 地区
- 最寄駅
- 専有面積
- 間取り
- 築年数
- 売出価格
- 推定成約価格
- 推定㎡単価
- 価格乖離率
- 価格評価

## 🎯 想定用途

不動産会社における以下の業務支援を想定しています。

- 物件査定
- 売出価格設定
- 価格改定判断
- 営業担当者の価格説明
- 複数物件の一括査定
- 査定履歴の管理
- 案件ステータス管理
- 再査定履歴の管理

## 🛠 使用技術

| 分類 | 技術 |
|---|---|
| Language | Python |
| Machine Learning | CatBoost, scikit-learn |
| Data Analysis | pandas, NumPy |
| Web UI | Streamlit |
| Database | Supabase / PostgreSQL |
| Authentication | Supabase Auth |
| Authorization | JWT / Row Level Security |
| Visualization | Streamlit Charts, matplotlib |
| API | requests |
| Excel | openpyxl |
| Development | VS Code, Git, GitHub |
| Container | Docker |
| Cloud | Streamlit Community Cloud |
| Data Source | 国土交通省 不動産情報ライブラリ API |

## 📚 データ

| 段階 | 件数 |
|---|---:|
| 駅特徴量付き取得・結合後 | 194,092 |
| 東京都データ | 108,020 |
| 最終価格モデルデータ | 106,937 |

GISマッチ時の駅ポイント間距離は、物件から駅までの徒歩距離を意味しないため、徒歩時間や物件－駅間距離としては使用していません。

## ⏱ 成約日数予測

販売開始日から成約日までの実履歴を利用する学習・推論パイプラインも実装しています。

```text
販売開始日
   ↓
成約日
   ↓
成約日数
   ↓
価格モデルによる価格特徴量
   ↓
CatBoost
   ↓
成約日数予測
```

ただし、現在は公開データから十分な教師データを確保できていません。

そのため、前処理・学習・推論パイプラインは実装していますが、実成約履歴による本学習・実データ精度評価は未実施です。架空の教師データによる精度を実運用精度として扱わない方針です。

## ▶ ローカル実行

```bash
git clone https://github.com/Rion-rion/real-estate-ai.git
cd real-estate-ai
git switch commercial
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Supabaseを使用する場合は `.streamlit/secrets.toml` を設定してください。

## ▶ CLI

価格モデル:

```bash
python main.py station
python main.py preprocess
python main.py train
python main.py predict
python main.py excel
python main.py report
python main.py price-full
```

成約日数:

```bash
python main.py days-template
python main.py days-preprocess
python main.py days-train
python main.py days-full
python main.py days-predict
```

状態確認:

```bash
python main.py status
```

## 🗂 主なファイル

```text
app.py
main.py
predict.py
predict_excel.py

collect_mlit.py
collect_station_features.py

preprocessing.py
train_price.py
analysis_report.py

create_days_template.py
preprocessing_days.py
train_days.py
predict_days.py

models/
└── price_model.cbm

data/
└── reference/
    └── station_reference.csv

requirements.txt
Dockerfile
README.md
```

## ⚠️ 制約

本システムは査定・販売判断を支援する機械学習システムであり、正式な不動産鑑定評価を代替するものではありません。

現在の公開データでは以下を十分に取得できていません。

- 駅徒歩時間
- 所在階
- 方角
- 眺望
- 総戸数
- 管理状態
- 室内状態
- リフォーム詳細
- マンションブランド
- 金利
- 短期的な市況変化

## 🚧 Commercial Prototype

`commercial` ブランチでは、公開ポートフォリオ版をベースに以下を追加しています。

- Supabase Auth
- JWT認証
- RLS
- ユーザー別査定履歴
- モデルVersion管理
- 案件管理
- 案件ステータス
- `case_id` による案件と査定履歴の正式なリレーション

今後は自動テスト、CI、監査ログ、例外処理、運用監視などを追加して、より実運用に近い構成へ拡張する予定です。
