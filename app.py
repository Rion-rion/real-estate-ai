from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from predict import (
    evaluate_price_position,
    load_model_info,
    predict_prices,
    prepare_input_data,
)


ROOT = Path(__file__).resolve().parent

MODEL_FILE = ROOT / "models" / "price_model.cbm"
TRAINING_FILE = ROOT / "data" / "processed" / "price_training.csv"
REFERENCE_FILE = ROOT / "data" / "reference" / "station_reference.csv"

COLUMN_ALIASES = {
    "物件ID": "property_id",
    "市区町村": "city",
    "地区": "district_name",
    "最寄駅": "station_name",
    "路線": "station_line",
    "専有面積": "area_m2",
    "専有面積㎡": "area_m2",
    "間取り": "floor_plan",
    "築年数": "building_age",
    "構造": "structure",
    "用途地域": "city_planning",
    "売出価格": "asking_price",
}

REQUIRED_COLUMNS = [
    "city",
    "district_name",
    "area_m2",
    "floor_plan",
    "building_age",
    "asking_price",
]

TEXT_OPTIONAL_COLUMNS = [
    "station_name",
    "station_line",
    "structure",
    "city_planning",
]

NUMERIC_OPTIONAL_COLUMNS = [
    "station_latitude",
    "station_longitude",
]

DISPLAY_COLUMNS = [
    "案件名",
    "担当者",
    "property_id",
    "city",
    "district_name",
    "station_name",
    "station_line",
    "area_m2",
    "floor_plan",
    "building_age",
    "structure",
    "city_planning",
    "asking_price",
    "推定㎡単価",
    "売出㎡単価",
    "推定成約価格",
    "売出価格との差額",
    "価格乖離率(%)",
    "価格評価",
]

DISPLAY_NAMES = {
    "property_id": "物件ID",
    "city": "市区町村",
    "district_name": "地区",
    "station_name": "最寄駅",
    "station_line": "路線",
    "area_m2": "専有面積㎡",
    "floor_plan": "間取り",
    "building_age": "築年数",
    "structure": "構造",
    "city_planning": "用途地域",
    "asking_price": "売出価格",
}

st.set_page_config(
    page_title="不動産価格査定システム",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)


def current_quarter():
    return (date.today().month - 1) // 3 + 1


def clean_text(series):
    return (
        series.astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "<NA>": pd.NA,
            }
        )
    )


def clean_station(series):
    return clean_text(series).str.replace(
        r"駅$",
        "",
        regex=True,
    )


def format_money(value, unit):
    if pd.isna(value):
        return "-"

    value = float(value)

    if unit == "万円":
        return f"{value / 10_000:,.0f}万円"

    return f"{value:,.0f}円"


def apply_theme(theme):
    dark = theme == "ダーク"

    colors = {
        "background": "#0F172A" if dark else "#F7F8FA",
        "panel": "#172033" if dark else "#FFFFFF",
        "sidebar": "#111827" if dark else "#F1F4F8",
        "input": "#1E293B" if dark else "#F4F6F9",
        "text": "#F8FAFC" if dark else "#18202F",
        "muted": "#A7B0C0" if dark else "#667085",
        "border": "#334155" if dark else "#D8DEE8",
    }

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {colors["background"]};
            color: {colors["text"]};
        }}

        [data-testid="stSidebar"] {{
            background: {colors["sidebar"]};
            border-right: 1px solid {colors["border"]};
        }}

        [data-testid="stSidebar"] * {{
            color: {colors["text"]};
        }}

        h1, h2, h3, p, label {{
            color: {colors["text"]};
        }}

        [data-testid="stForm"],
        [data-testid="stMetric"] {{
            background: {colors["panel"]};
            border: 1px solid {colors["border"]};
            border-radius: 14px;
            padding: 1rem;
        }}

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {{
            background: {colors["input"]};
            border-color: {colors["border"]};
        }}

        div[data-baseweb="input"] input,
        div[data-baseweb="select"] span {{
            color: {colors["text"]} !important;
        }}

        .app-subtitle {{
            color: {colors["muted"]};
            margin-top: -0.7rem;
            margin-bottom: 1.4rem;
        }}

        .workflow-card {{
            background: {colors["panel"]};
            border: 1px solid {colors["border"]};
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }}

        .workflow-step {{
            color: {colors["muted"]};
            font-size: 0.8rem;
        }}

        .workflow-title {{
            color: {colors["text"]};
            font-size: 1.05rem;
            font-weight: 700;
        }}

        [data-testid="stToolbar"] {{
            display: none;
        }}

        #MainMenu,
        footer {{
            visibility: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    return load_model_info()


def create_station_reference():
    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            "駅参照データがありません。"
            "data/reference/station_reference.csv を配置してください。"
        )

    columns = [
        "city",
        "district_name",
        "station_name",
        "station_line",
        "station_latitude",
        "station_longitude",
    ]

    df = pd.read_csv(
        TRAINING_FILE,
        usecols=columns,
        low_memory=False,
    )

    df["city"] = clean_text(df["city"])
    df["district_name"] = clean_text(df["district_name"])
    df["station_name"] = clean_station(df["station_name"])
    df["station_line"] = clean_text(df["station_line"]).fillna("不明")

    for column in ["station_latitude", "station_longitude"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "city",
            "district_name",
            "station_name",
            "station_latitude",
            "station_longitude",
        ]
    )

    reference = (
        df.groupby(
            [
                "city",
                "district_name",
                "station_name",
                "station_line",
            ],
            dropna=False,
        )
        .agg(
            station_latitude=("station_latitude", "median"),
            station_longitude=("station_longitude", "median"),
            count=("station_name", "size"),
        )
        .reset_index()
    )

    REFERENCE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference.to_csv(
        REFERENCE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return reference


@st.cache_data
def load_station_reference():
    if REFERENCE_FILE.exists():
        df = pd.read_csv(
            REFERENCE_FILE,
            low_memory=False,
        )
    else:
        df = create_station_reference()

    df["city"] = clean_text(df["city"])
    df["district_name"] = clean_text(df["district_name"])
    df["station_name"] = clean_station(df["station_name"])
    df["station_line"] = clean_text(df["station_line"]).fillna("不明")

    return df


@st.cache_data
def load_station_maps():
    reference = load_station_reference().sort_values(
        "count",
        ascending=False,
    )

    district_best = reference.drop_duplicates(
        ["city", "district_name"]
    )

    detail_best = reference.drop_duplicates(
        [
            "city",
            "district_name",
            "station_name",
        ]
    )

    station_best = reference.drop_duplicates(
        "station_name"
    )

    district_map = {
        (str(row.city), str(row.district_name)): str(row.station_name)
        for row in district_best.itertuples()
    }

    detail_map = {
        (
            str(row.city),
            str(row.district_name),
            str(row.station_name),
        ): {
            "station_line": str(row.station_line),
            "station_latitude": float(row.station_latitude),
            "station_longitude": float(row.station_longitude),
        }
        for row in detail_best.itertuples()
    }

    station_map = {
        str(row.station_name): {
            "station_line": str(row.station_line),
            "station_latitude": float(row.station_latitude),
            "station_longitude": float(row.station_longitude),
        }
        for row in station_best.itertuples()
    }

    return district_map, detail_map, station_map


def fill_station_features(df):
    df = df.copy()

    district_map, detail_map, station_map = load_station_maps()

    df["city"] = clean_text(df["city"])
    df["district_name"] = clean_text(df["district_name"])

    if "station_name" not in df.columns:
        df["station_name"] = pd.NA

    if "station_line" not in df.columns:
        df["station_line"] = pd.NA

    df["station_name"] = clean_station(df["station_name"])
    df["station_line"] = clean_text(df["station_line"])

    for column in NUMERIC_OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    errors = []

    for index, row in df.iterrows():
        city = str(row["city"])
        district = str(row["district_name"])
        station = row["station_name"]

        if pd.isna(station):
            station = district_map.get(
                (city, district)
            )

            if station is None:
                errors.append(
                    f"{index + 1}行目: 最寄駅を補完できません。"
                )
                continue

            df.at[index, "station_name"] = station

        station = str(station)

        info = detail_map.get(
            (city, district, station)
        )

        if info is None:
            info = station_map.get(station)

        if info is None:
            errors.append(
                f"{index + 1}行目: {station}駅の情報がありません。"
            )
            continue

        for column in [
            "station_line",
            "station_latitude",
            "station_longitude",
        ]:
            if pd.isna(row[column]):
                df.at[index, column] = info[column]

    if errors:
        raise ValueError(
            "\n".join(errors)
        )

    return df


def normalize_input(
    df,
    features,
    categories,
):
    df = df.copy().rename(
        columns=COLUMN_ALIASES
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            "必要項目が不足しています: "
            + ", ".join(missing)
        )

    if "property_id" not in df.columns:
        df["property_id"] = [
            f"WEB{i + 1:04d}"
            for i in range(len(df))
        ]

    for column in TEXT_OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    for column in NUMERIC_OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df["transaction_year"] = date.today().year
    df["transaction_quarter"] = current_quarter()

    df["asking_price"] = pd.to_numeric(
        df["asking_price"],
        errors="coerce",
    )

    if df["asking_price"].isna().any():
        raise ValueError(
            "売出価格に未入力または不正な値があります。"
        )

    if (df["asking_price"] <= 0).any():
        raise ValueError(
            "売出価格は0円より大きくしてください。"
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


def assess(df):
    model, _, features, categories = load_model()

    df = normalize_input(
        df,
        features,
        categories,
    )

    df = fill_station_features(df)

    prepared = prepare_input_data(
        df,
        features,
        categories,
    )

    unit_price, price = predict_prices(
        model,
        prepared,
        features,
    )

    result = df.copy()

    result["推定㎡単価"] = np.round(
        unit_price
    ).astype(int)

    result["推定成約価格"] = np.round(
        price
    ).astype(int)

    result["売出㎡単価"] = (
        result["asking_price"]
        / result["area_m2"]
    )

    result["売出価格との差額"] = (
        result["asking_price"]
        - result["推定成約価格"]
    )

    result["価格乖離率(%)"] = (
        result["売出価格との差額"]
        / result["推定成約価格"]
        * 100
    )

    result["価格評価"] = (
        result["価格乖離率(%)"]
        .apply(
            evaluate_price_position
        )
    )

    return result


def sales_view(result):
    columns = [
        column
        for column in DISPLAY_COLUMNS
        if column in result.columns
    ]

    return result[
        columns
    ].rename(
        columns=DISPLAY_NAMES
    )


def excel_bytes(
    df,
    sheet_name,
):
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


def template_excel():
    template = pd.DataFrame(
        [
            {
                "案件名": "北千住マンション",
                "担当者": "山田",
                "物件ID": "A001",
                "市区町村": "足立区",
                "地区": "千住",
                "最寄駅": "北千住",
                "専有面積㎡": 65.2,
                "間取り": "3LDK",
                "築年数": 12,
                "構造": "RC",
                "用途地域": "商業地域",
                "売出価格": 75_000_000,
            }
        ]
    )

    return excel_bytes(
        template,
        "査定入力",
    )


def result_label(result):
    if "案件名" in result.columns:
        values = (
            result["案件名"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        if values.ne("").all():
            return "案件名"

    return "property_id"


def show_charts(
    result,
    unit,
):
    label = result_label(result)

    divisor = (
        10_000
        if unit == "万円"
        else 1
    )

    price = result[
        [
            label,
            "推定成約価格",
            "asking_price",
        ]
    ].copy()

    price["推定成約価格"] /= divisor
    price["asking_price"] /= divisor

    price = price.rename(
        columns={
            "asking_price": "売出価格",
        }
    )

    st.subheader(
        f"価格比較（{unit}）"
    )

    st.bar_chart(
        price.set_index(label)
    )

    unit_price = result[
        [
            label,
            "推定㎡単価",
            "売出㎡単価",
        ]
    ].copy()

    st.subheader(
        "㎡単価比較（円/㎡）"
    )

    st.bar_chart(
        unit_price.set_index(label)
    )

    if len(result) > 1:
        gap = result[
            [
                label,
                "価格乖離率(%)",
            ]
        ]

        st.subheader(
            "価格乖離率（%）"
        )

        st.bar_chart(
            gap.set_index(label)
        )


def show_result(
    result,
    unit,
    show_graph,
):
    first = result.iloc[0]

    st.subheader(
        "査定結果"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "推定成約価格",
        format_money(
            first["推定成約価格"],
            unit,
        ),
    )

    c2.metric(
        "売出価格",
        format_money(
            first["asking_price"],
            unit,
        ),
    )

    gap = first["価格乖離率(%)"]

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
        first["価格評価"] or "-",
    )

    if show_graph:
        show_charts(
            result,
            unit,
        )

    st.subheader(
        "査定結果一覧"
    )

    output = sales_view(
        result
    )

    st.dataframe(
        output,
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "査定結果をExcelで出力",
        data=excel_bytes(
            output,
            "査定結果",
        ),
        file_name="査定結果.xlsx",
        mime=(
            "application/"
            "vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        type="primary",
        use_container_width=True,
    )


def workflow():
    items = [
        ("STEP 1", "物件情報を入力"),
        ("STEP 2", "査定する"),
        ("STEP 3", "Excelで出力"),
    ]

    for column, item in zip(
        st.columns(3),
        items,
    ):
        step, title = item

        with column:
            st.markdown(
                f"""
                <div class="workflow-card">
                    <div class="workflow-step">{step}</div>
                    <div class="workflow-title">{title}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def manual_input():
    with st.form(
        "assessment"
    ):
        st.subheader(
            "物件情報"
        )

        a, b = st.columns(2)

        case = a.text_input(
            "案件名",
            placeholder="例：北千住マンション",
        )

        staff = b.text_input(
            "担当者",
            placeholder="例：山田",
        )

        a, b, c = st.columns(3)

        city = a.text_input(
            "市区町村",
            "足立区",
        )

        district = b.text_input(
            "地区",
            "千住",
        )

        station = c.text_input(
            "最寄駅",
            "北千住",
            help=(
                "空欄の場合は地区情報から"
                "代表駅を補完します。"
            ),
        )

        a, b, c = st.columns(3)

        area = a.number_input(
            "専有面積（㎡）",
            min_value=1.0,
            value=65.2,
            step=0.1,
        )

        plan = b.text_input(
            "間取り",
            "3LDK",
        )

        age = c.number_input(
            "築年数",
            min_value=0,
            value=12,
            step=1,
        )

        a, b = st.columns(2)

        structure = a.selectbox(
            "構造",
            [
                "RC",
                "SRC",
                "S",
                "その他",
                "不明",
            ],
        )

        planning = b.text_input(
            "用途地域",
            "商業地域",
        )

        asking = st.number_input(
            "売出価格（円）",
            min_value=1_000_000,
            value=75_000_000,
            step=1_000_000,
        )

        submitted = (
            st.form_submit_button(
                "査定する",
                type="primary",
                use_container_width=True,
            )
        )

    if not submitted:
        return

    df = pd.DataFrame(
        [
            {
                "案件名": case,
                "担当者": staff,
                "city": city,
                "district_name": district,
                "station_name": (
                    station
                    if station.strip()
                    else pd.NA
                ),
                "area_m2": area,
                "floor_plan": plan,
                "building_age": age,
                "structure": structure,
                "city_planning": planning,
                "asking_price": asking,
            }
        ]
    )

    with st.spinner(
        "査定しています..."
    ):
        st.session_state["result"] = (
            assess(df)
        )


def batch_input():
    st.subheader(
        "Excel一括査定"
    )

    st.caption(
        "テンプレートへ複数物件を入力して"
        "一括査定できます。"
    )

    st.download_button(
        "入力テンプレートをダウンロード",
        data=template_excel(),
        file_name="査定入力テンプレート.xlsx",
        mime=(
            "application/"
            "vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )

    file = st.file_uploader(
        "Excel / CSVを選択",
        type=[
            "xlsx",
            "csv",
        ],
    )

    if file is None:
        return

    suffix = Path(
        file.name
    ).suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file)

    else:
        excel = pd.ExcelFile(file)

        sheet = (
            "査定入力"
            if "査定入力"
            in excel.sheet_names
            else excel.sheet_names[0]
        )

        df = pd.read_excel(
            excel,
            sheet_name=sheet,
        )

    st.caption(
        f"{len(df):,}件を読み込みました。"
    )

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
    )

    if st.button(
        "一括査定する",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner(
            "査定しています..."
        ):
            st.session_state["result"] = (
                assess(df)
            )


def sidebar_settings():
    with st.sidebar:
        st.header(
            "設定"
        )

        theme = st.radio(
            "表示テーマ",
            [
                "ライト",
                "ダーク",
            ],
            horizontal=True,
        )

        unit = st.radio(
            "金額表示",
            [
                "円",
                "万円",
            ],
            horizontal=True,
        )

        show_graph = st.toggle(
            "グラフを表示",
            value=True,
        )

        st.divider()

        st.subheader(
            "システム状態"
        )

        if MODEL_FILE.exists():
            st.success(
                "査定機能：利用可能"
            )
        else:
            st.error(
                "査定機能：利用不可"
            )

        if REFERENCE_FILE.exists():
            st.success(
                "駅情報：利用可能"
            )
        else:
            st.info(
                "駅情報：準備中"
            )

    return (
        theme,
        unit,
        show_graph,
    )


def main():
    theme, unit, show_graph = (
        sidebar_settings()
    )

    apply_theme(theme)

    st.title(
        "不動産価格査定システム"
    )

    st.markdown(
        """
        <div class="app-subtitle">
            東京都中古マンション
            査定・売出価格検討支援
        </div>
        """,
        unsafe_allow_html=True,
    )

    workflow()

    st.write("")

    manual, batch = st.tabs(
        [
            "1件査定",
            "Excel一括査定",
        ]
    )

    try:
        with manual:
            manual_input()

        with batch:
            batch_input()

        if "result" in st.session_state:
            st.divider()

            show_result(
                st.session_state["result"],
                unit,
                show_graph,
            )

    except Exception as error:
        st.error(
            "処理できませんでした。"
        )

        with st.expander(
            "エラー詳細"
        ):
            st.code(
                str(error)
            )

    st.divider()

    st.caption(
        "※ 査定結果は価格検討を支援する参考値です。"
        "正式な不動産鑑定評価を代替するものではありません。"
    )


if __name__ == "__main__":
    main()