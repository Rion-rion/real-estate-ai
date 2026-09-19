# 🏢 不動産価格査定システム

東京都の中古マンションを対象に、過去の実取引データから学習した価格モデルを利用して、成約価格の推定・売出価格との比較・価格評価を行う不動産査定支援システムです。

国土交通省「不動産情報ライブラリ API」の不動産取引データと鉄道駅GISデータを利用し、Python / pandas / CatBoost で構築しています。

現在は Streamlit Community Cloud 上でWebアプリとして公開し、営業担当者がブラウザから物件情報を入力して査定できる構成にしています。

---

## 🌐 Webアプリ

**Live Demo**

https://real-estate-ai-hugilvbuq8poka4tqnyvx5.streamlit.app/

主な機能:

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
- ライト / ダーク表示
- 円 / 万円表示
- アクセスパスワード
- Supabase接続時のクラウド査定履歴

---

## 📌 価格モデルの結果

| 項目 | 結果 |
|---|---:|
| 最終データ件数 | 106,937件 |
| Final Test | 2026年 6,422件 |
| MAE | 12,168,235円 |
| RMSE | 19,017,665円 |
| MAPE | 17.26% |
| R² | 0.8655 |
| 駅特徴量なし MAPE | 17.76% |
| 駅特徴量あり MAPE | 17.26% |
| 駅名取得率 | 100.00% |

2026年のデータはモデル選択には使用せず、最終評価用の未使用期間として利用しています。

---

## 🔄 システム構成

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
学習済み価格モデル
price_model.cbm
        ↓
Streamlit Webアプリ
        ↓
営業担当者が物件情報を入力
        ↓
推定成約価格
推定㎡単価
価格乖離率
価格評価
        ↓
グラフ / Excel出力
        ↓
Supabase
査定履歴保存

Webアプリでは約10.7万件の学習データを毎回読み込むのではなく、学習済みモデル price_model.cbm を利用して新しい物件の価格を推定します。

駅情報については、学習データから作成した軽量な data/reference/station_reference.csv を利用しています。

🎯 想定用途

不動産会社における以下の業務支援を想定しています。

物件査定
売出価格設定
価格改定判断
営業担当者の価格説明
複数物件の一括査定
査定結果のExcel共有
査定履歴の管理
🧠 価格モデル Ver.5

価格予測には CatBoostRegressor を使用しています。

成約価格を直接予測するのではなく、成約㎡単価を予測し、専有面積を掛けて推定成約価格を算出します。

物件情報
   ↓
CatBoost
   ↓
推定㎡単価
   ↓
推定㎡単価 × 専有面積
   ↓
推定成約価格

目的変数:

log1p(contract_price_per_m2)
使用特徴量
市区町村
地区
専有面積
間取り
築年数
建物構造
都市計画 / 用途地域
取引年
取引四半期
最寄駅
路線
駅緯度
駅経度
🕒 時系列分割
期間	用途	件数
2021〜2024	Train	74,745
2025	Validation	25,770
2026	Final Test	6,422

2025年のValidationで特徴量セットを比較し、採用モデルを決定しました。

その後、2021〜2025年で最終モデルを再学習し、2026年のFinal Testで評価しています。

🚉 駅特徴量

Ver.5で以下を追加しました。

station_name
station_line
station_latitude
station_longitude

2025 Validation:

Feature Set	MAPE	R²
baseline_current	17.19%	0.8483
station_name	16.91%	0.8530
station_geo	16.68%	0.8556
station_geo_line	16.64%	0.8567

最もValidation MAPEが低かった station_geo_line を採用しています。

Final Testでは、

MAPE  17.76% → 17.26%
R²    0.8546 → 0.8655

となりました。

💻 Web査定

営業向け画面では、機械学習モデルの内部情報を前面に出さず、業務上必要な情報を中心に表示します。

物件情報入力
      ↓
査定
      ↓
推定成約価格
売出価格
価格乖離率
価格評価
      ↓
価格比較グラフ
㎡単価比較グラフ
      ↓
Excel出力

1件査定ではExcelに近い1行入力形式を採用しています。

📊 Excel一括査定

Web画面から入力テンプレートをダウンロードし、複数物件をまとめて査定できます。

対応形式:

.xlsx
.csv
🗃 査定履歴

Supabase接続時は、査定結果をクラウド上のPostgreSQLへ保存します。

保存項目例:

査定日時
案件名
担当者
市区町村
地区
最寄駅
専有面積
間取り
築年数
売出価格
推定成約価格
推定㎡単価
価格乖離率
価格評価

Web画面の「査定履歴」から直近の査定結果を確認し、Excelとして出力できます。

🔐 セキュリティ

秘密情報はGitHubへコミットせず、Streamlit Community Cloud の Secrets で管理します。

例:

APP_PASSWORD = "YOUR_APP_PASSWORD"

[supabase]
url = "https://YOUR_PROJECT.supabase.co"
secret_key = "YOUR_SUPABASE_SECRET_KEY"

ローカル環境では、

.streamlit/secrets.toml

を利用できます。

.env や secrets.toml はGit管理対象外とします。

☁️ デプロイ

Webアプリは Streamlit Community Cloud で動作します。

Repository:
Rion-rion/real-estate-ai

Branch:
main

Main file:
app.py

main ブランチへpushすると、Webアプリにも変更が反映されます。

🛠 使用技術
分類	技術
Language	Python
Machine Learning	CatBoost, scikit-learn
Data Analysis	pandas, NumPy
Web UI	Streamlit
Database	Supabase / PostgreSQL
Visualization	Streamlit Charts, matplotlib
API	requests
Excel	openpyxl
Development	VS Code, Git, GitHub
Container	Docker
Cloud	Streamlit Community Cloud
Data Source	国土交通省 不動産情報ライブラリ API
📚 データ
段階	件数
駅特徴量付き取得・結合後	194,092
東京都データ	108,020
最終価格モデルデータ	106,937

GISマッチ時の駅ポイント間距離は、物件から駅までの徒歩距離を意味しないため、徒歩時間や物件－駅間距離としては使用していません。

⏱ 成約日数予測

販売開始日から成約日までの実履歴を利用する学習・推論パイプラインも実装しています。

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

ただし、現在は公開データから十分な教師データを確保できていません。

そのため、

学習パイプライン
前処理
推論パイプライン

までは実装済みですが、実成約履歴による本学習・実データ精度評価は未実施です。

架空の教師データによる精度を実運用精度として扱わない設計にしています。

▶ ローカル実行
git clone https://github.com/Rion-rion/real-estate-ai.git

cd real-estate-ai

python -m venv .venv

pip install -r requirements.txt

Windows PowerShell:

.\.venv\Scripts\python.exe -m streamlit run app.py

通常:

streamlit run app.py
▶ CLI

価格モデル:

python main.py station
python main.py preprocess
python main.py train
python main.py predict
python main.py excel
python main.py report
python main.py price-full

成約日数:

python main.py days-template
python main.py days-preprocess
python main.py days-train
python main.py days-full
python main.py days-predict

状態確認:

python main.py status
🗂 主なファイル
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
⚠️ 制約

本システムは査定・販売判断を支援する機械学習システムであり、正式な不動産鑑定評価を代替するものではありません。

現在の公開データでは以下を十分に取得できていません。

駅徒歩時間
所在階
方角
眺望
総戸数
管理状態
室内状態
リフォーム詳細
マンションブランド
金利
短期的な市況変化

これらは今後の改善候補です。

📌 今後の改善候補
価格査定
駅徒歩時間
所在階
方角
周辺地価
周辺施設
金利・市況指標
マンション単位の特徴量
成約日数
実成約履歴による本学習
価格改定履歴
内見件数
問い合わせ件数
広告掲載状況
時間順OOF予測
Web / 運用
ユーザー単位ログイン
権限管理
担当者別履歴
案件検索
帳票の高度化

Author

GitHub: Rion-rion