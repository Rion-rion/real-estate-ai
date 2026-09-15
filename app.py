from datetime import date
from io import BytesIO
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

from predict import (
    enrich_station_features,
    evaluate_price_position,
    load_model_info,
    predict_prices,
    prepare_input_data,
)
from predict_excel import auto_fill_station_name


ROOT = Path(__file__).resolve().parent
PRICE_MODEL = ROOT / "models" / "price_model.cbm"
PRICE_METRICS = ROOT / "output" / "price_model_metrics.json"
DAYS_MODEL = ROOT / "models" / "days_model.cbm"
DAYS_METRICS = ROOT / "output" / "days_model_metrics.json"
ANALYSIS_DIR = ROOT / "output" / "analysis"

st.set_page_config(
    page_title="Tokyo Real Estate AI",
    page_icon="🏢",
    layout="wide",
)


def quarter():
    return (date.today().month - 1) // 3 + 1


def money(value):
    return "-" if pd.isna(value) else f"{float(value):,.0f}円"


def load_json(path):
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_price_model():
    return load_model_info()


def template_df():
    return pd.DataFrame(
        [
            {
                "property_id": "A001",
                "city": "足立区",
                "district_name": "千住",
                "station_name": "北千住",
                "station_line": "",
                "station_latitude": np.nan,
                "station_longitude": np.nan,
                "area_m2": 65.2,
                "floor_plan": "3LDK",
                "building_age": 12,
                "structure": "RC",
                "city_planning": "商業地域",
                "transaction_year": date.today().year,
                "transaction_quarter": quarter(),
                "asking_price": 75_000_000,
            },
            {
                "property_id": "A002",
                "city": "世田谷区",
                "district_name": "三軒茶屋",
                "station_name": "三軒茶屋",
                "station_line": "",
                "station_latitude": np.nan,
                "station_longitude": np.nan,
                "area_m2": 55.0,
                "floor_plan": "2LDK",
                "building_age": 8,
                "structure": "RC",
                "city_planning": "近隣商業地域",
                "transaction_year": date.today().year,
                "transaction_quarter": quarter(),
                "asking_price": 110_000_000,
            },
        ]
    )


def to_excel(df, sheet_name="予測結果"):
    buffer = BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name=sheet_name,
        )

    return buffer.getvalue()


def read_upload(file):
    suffix = Path(file.name).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file)

    excel = pd.ExcelFile(file)

    sheet = (
        "予測入力"
        if "予測入力" in excel.sheet_names
        else excel.sheet_names[0]
    )

    return pd.read_excel(
        excel,
        sheet_name=sheet,
    )


def normalize_input(
    df,
    features,
    categories,
):
    df = df.copy()

    required = [
        "city",
        "district_name",
        "area_m2",
        "floor_plan",
        "building_age",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            "不足列: "
            + ", ".join(missing)
        )

    if "property_id" not in df.columns:
        df["property_id"] = [
            f"WEB{i + 1:04d}"
            for i in range(len(df))
        ]

    for column in [
        "station_name",
        "station_line",
        "structure",
        "city_planning",
    ]:
        if column not in df.columns:
            df[column] = pd.NA

    for column in [
        "station_latitude",
        "station_longitude",
        "asking_price",
    ]:
        if column not in df.columns:
            df[column] = np.nan

    if "transaction_year" not in df.columns:
        df["transaction_year"] = (
            date.today().year
        )

    if "transaction_quarter" not in df.columns:
        df["transaction_quarter"] = (
            quarter()
        )

    df["station_name"] = (
        df["station_name"]
        .astype("string")
    )

    df["station_line"] = (
        df["station_line"]
        .astype("string")
    )

    for column in features:
        if column in df.columns:
            continue

        df[column] = (
            "不明"
            if column in categories
            else np.nan
        )

    return df


def predict_price(df):
    (
        model,
        model_info,
        features,
        categories,
    ) = load_price_model()

    df = normalize_input(
        df,
        features,
        categories,
    )

    df = auto_fill_station_name(df)

    df = enrich_station_features(
        df,
        features,
    )

    prepared = prepare_input_data(
        df,
        features,
        categories,
    )

    (
        predicted_unit_price,
        predicted_price,
    ) = predict_prices(
        model,
        prepared,
        features,
    )

    result = df.copy()

    result["AI予測㎡単価"] = (
        np.round(
            predicted_unit_price
        ).astype(int)
    )

    result["AI予測成約価格"] = (
        np.round(
            predicted_price
        ).astype(int)
    )

    result["asking_price"] = pd.to_numeric(
        result["asking_price"],
        errors="coerce",
    )

    result["売出価格との差額"] = (
        result["asking_price"]
        - result["AI予測成約価格"]
    )

    result["価格乖離率(%)"] = (
        result["売出価格との差額"]
        / result["AI予測成約価格"]
        * 100
    )

    result["価格評価"] = (
        result["価格乖離率(%)"]
        .apply(
            evaluate_price_position
        )
    )

    return result


def show_result(result):
    row = result.iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "AI想定成約価格",
        money(
            row["AI予測成約価格"]
        ),
    )

    c2.metric(
        "売出価格",
        money(
            row["asking_price"]
        ),
    )

    gap = row["価格乖離率(%)"]

    c3.metric(
        "価格乖離率",
        (
            f"{gap:+.2f}%"
            if pd.notna(gap)
            else "-"
        ),
    )

    c4.metric(
        "価格評価",
        row["価格評価"] or "-",
    )

    chart = result[
        [
            "property_id",
            "AI予測成約価格",
            "asking_price",
        ]
    ].copy()

    chart = chart.rename(
        columns={
            "asking_price": "売出価格",
        }
    )

    st.bar_chart(
        chart.set_index(
            "property_id"
        )
    )

    columns = [
        "property_id",
        "city",
        "district_name",
        "station_name",
        "station_line",
        "area_m2",
        "asking_price",
        "AI予測㎡単価",
        "AI予測成約価格",
        "売出価格との差額",
        "価格乖離率(%)",
        "価格評価",
    ]

    st.dataframe(
        result[
            [
                column
                for column in columns
                if column in result.columns
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "📥 結果をExcelで保存",
        data=to_excel(result),
        file_name=(
            "price_predictions_web.xlsx"
        ),
        mime=(
            "application/"
            "vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
    )


def dashboard():
    st.title(
        "🏢 Tokyo Real Estate AI"
    )

    st.caption(
        "東京都中古マンション "
        "価格予測・販売判断支援"
    )

    info = load_json(
        PRICE_METRICS
    )

    metrics = info.get(
        "test_metrics",
        {},
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "価格AI",
        (
            "稼働可能"
            if PRICE_MODEL.exists()
            else "未作成"
        ),
    )

    c2.metric(
        "Test MAPE",
        (
            f"{metrics.get('mape', 0):.2f}%"
            if metrics
            else "-"
        ),
    )

    c3.metric(
        "Test R²",
        (
            f"{metrics.get('r2', 0):.4f}"
            if metrics
            else "-"
        ),
    )

    c4.metric(
        "成約日数AI",
        (
            "学習済み"
            if DAYS_MODEL.exists()
            else "教師データ待ち"
        ),
    )

    st.code(
        "物件情報 → 駅補完 → 価格AI "
        "→ AI想定価格 → 価格乖離率 "
        "→ 価格評価 → Excel / 可視化",
        language="text",
    )


def price_page():
    st.header(
        "💰 AI価格予測"
    )

    manual, batch = st.tabs(
        [
            "1件入力",
            "Excel / CSV 一括",
        ]
    )

    with manual:
        with st.form(
            "manual_form"
        ):
            a, b, c = st.columns(3)

            property_id = a.text_input(
                "物件ID",
                "WEB001",
            )

            city = b.text_input(
                "市区町村",
                "足立区",
            )

            district = c.text_input(
                "地区",
                "千住",
            )

            a, b, c = st.columns(3)

            station = a.text_input(
                "最寄駅",
                "北千住",
            )

            area = b.number_input(
                "専有面積（㎡）",
                min_value=1.0,
                value=65.2,
                step=0.1,
            )

            floor_plan = c.text_input(
                "間取り",
                "3LDK",
            )

            a, b, c = st.columns(3)

            age = a.number_input(
                "築年数",
                min_value=0,
                value=12,
            )

            structure = b.selectbox(
                "構造",
                [
                    "RC",
                    "SRC",
                    "S",
                    "その他",
                    "不明",
                ],
            )

            planning = c.text_input(
                "都市計画",
                "商業地域",
            )

            asking = st.number_input(
                "売出価格（円）",
                min_value=1_000_000,
                value=75_000_000,
                step=1_000_000,
            )

            run = (
                st.form_submit_button(
                    "🚀 AI予測",
                    use_container_width=True,
                )
            )

        if run:
            df = pd.DataFrame(
                [
                    {
                        "property_id": property_id,
                        "city": city,
                        "district_name": district,
                        "station_name": (
                            station
                            or pd.NA
                        ),
                        "area_m2": area,
                        "floor_plan": floor_plan,
                        "building_age": age,
                        "structure": structure,
                        "city_planning": planning,
                        "transaction_year": (
                            date.today().year
                        ),
                        "transaction_quarter": (
                            quarter()
                        ),
                        "asking_price": asking,
                    }
                ]
            )

            try:
                with st.spinner(
                    "予測中..."
                ):
                    st.session_state[
                        "result"
                    ] = predict_price(df)

            except Exception as error:
                st.error(
                    f"予測エラー: {error}"
                )

    with batch:
        st.download_button(
            "📄 入力テンプレート",
            data=to_excel(
                template_df(),
                "予測入力",
            ),
            file_name=(
                "price_prediction_template.xlsx"
            ),
            mime=(
                "application/"
                "vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )

        file = st.file_uploader(
            "Excel / CSVをアップロード",
            type=[
                "xlsx",
                "xls",
                "csv",
            ],
        )

        if file:
            try:
                df = read_upload(file)

                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True,
                )

                if st.button(
                    "🚀 一括予測",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner(
                        "予測中..."
                    ):
                        st.session_state[
                            "result"
                        ] = predict_price(df)

            except Exception as error:
                st.error(
                    f"ファイル処理エラー: {error}"
                )

    if "result" in st.session_state:
        st.divider()

        show_result(
            st.session_state[
                "result"
            ]
        )


def analysis_page():
    st.header(
        "📊 モデル分析"
    )

    info = load_json(
        PRICE_METRICS
    )

    metrics = info.get(
        "test_metrics",
        {},
    )

    if not metrics:
        st.warning(
            "価格モデル評価情報がありません。"
        )
        return

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "MAPE",
        f"{metrics.get('mape', 0):.2f}%",
    )

    c2.metric(
        "R²",
        f"{metrics.get('r2', 0):.4f}",
    )

    c3.metric(
        "MAE",
        money(
            metrics.get("mae")
        ),
    )

    c4.metric(
        "RMSE",
        money(
            metrics.get("rmse")
        ),
    )

    graphs = [
        (
            "実価格 vs AI予測",
            "actual_vs_predicted.png",
        ),
        (
            "誤差分布",
            "error_distribution.png",
        ),
        (
            "市区町村別MAPE",
            "city_mape.png",
        ),
        (
            "駅別MAPE",
            "station_mape.png",
        ),
        (
            "特徴量重要度",
            "feature_importance.png",
        ),
        (
            "モデル比較",
            "model_comparison_mape.png",
        ),
    ]

    for i in range(
        0,
        len(graphs),
        2,
    ):
        cols = st.columns(2)

        for col, item in zip(
            cols,
            graphs[i:i + 2],
        ):
            title, filename = item

            path = (
                ANALYSIS_DIR
                / filename
            )

            with col:
                st.subheader(title)

                if path.exists():
                    st.image(
                        str(path),
                        use_container_width=True,
                    )

                else:
                    st.info(
                        "未生成"
                    )


def days_page():
    st.header(
        "⏱ 成約日数AI"
    )

    if not DAYS_MODEL.exists():
        st.info(
            "教師データ待ちです。"
            "実成約履歴を100件以上投入すると"
            "既存パイプラインで学習できます。"
        )
        return

    metrics = (
        load_json(
            DAYS_METRICS
        )
        .get(
            "test_metrics",
            {},
        )
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        f"{metrics.get('mae_days', 0):.2f}日",
    )

    c2.metric(
        "MAPE",
        f"{metrics.get('mape_percent', 0):.2f}%",
    )

    c3.metric(
        "R²",
        f"{metrics.get('r2', 0):.4f}",
    )


def main():
    st.sidebar.title(
        "🏢 Real Estate AI"
    )

    page = st.sidebar.radio(
        "メニュー",
        [
            "ダッシュボード",
            "AI価格予測",
            "モデル分析",
            "成約日数AI",
        ],
    )

    pages = {
        "ダッシュボード": dashboard,
        "AI価格予測": price_page,
        "モデル分析": analysis_page,
        "成約日数AI": days_page,
    }

    pages[page]()


if __name__ == "__main__":
    main()