from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo
import hmac

import numpy as np
import pandas as pd
import requests
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


DISPLAY_COLUMNS = [
    "査定日時",
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


st.set_page_config(
    page_title="不動産価格査定システム",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 共通
# =========================================================

def current_quarter():
    return (date.today().month - 1) // 3 + 1


def now_jst():
    return datetime.now(
        ZoneInfo("Asia/Tokyo")
    )


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
    return clean_text(
        series
    ).str.replace(
        r"駅$",
        "",
        regex=True,
    )


def format_money(
    value,
    unit="万円",
):
    if pd.isna(value):
        return "-"

    value = float(value)

    if unit == "万円":
        return (
            f"{value / 10_000:,.0f}万円"
        )

    return f"{value:,.0f}円"


def format_percent(value):
    if pd.isna(value):
        return "-"

    return f"{float(value):+.2f}%"


def scalar(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    return value


# =========================================================
# Secrets
# =========================================================

def app_password():
    try:
        return st.secrets.get(
            "APP_PASSWORD",
            "",
        )
    except FileNotFoundError:
        return ""


def supabase_settings():
    try:
        settings = st.secrets.get(
            "supabase",
            {},
        )

        return (
            settings.get(
                "url",
                "",
            ),
            settings.get(
                "secret_key",
                "",
            ),
        )

    except (
        FileNotFoundError,
        KeyError,
    ):
        return "", ""


# =========================================================
# ログイン
# =========================================================

def access_gate():
    expected = app_password()

    if not expected:
        return True

    if st.session_state.get(
        "authenticated"
    ):
        return True

    st.title(
        "🏢 不動産価格査定システム"
    )

    st.caption(
        "営業用画面へアクセスするには"
        "パスワードを入力してください。"
    )

    with st.form(
        "login_form"
    ):
        password = st.text_input(
            "アクセスパスワード",
            type="password",
        )

        submitted = (
            st.form_submit_button(
                "ログイン",
                type="primary",
                use_container_width=True,
            )
        )

    if submitted:
        if hmac.compare_digest(
            str(password),
            str(expected),
        ):
            st.session_state[
                "authenticated"
            ] = True

            st.rerun()

        st.error(
            "パスワードが違います。"
        )

    return False


# =========================================================
# テーマ
# =========================================================

def apply_theme(theme):
    dark = theme == "ダーク"

    colors = {
        "background":
            "#0F172A"
            if dark
            else "#F7F8FA",

        "panel":
            "#172033"
            if dark
            else "#FFFFFF",

        "sidebar":
            "#111827"
            if dark
            else "#F1F4F8",

        "input":
            "#1E293B"
            if dark
            else "#F4F6F9",

        "text":
            "#F8FAFC"
            if dark
            else "#18202F",

        "muted":
            "#A7B0C0"
            if dark
            else "#667085",

        "border":
            "#334155"
            if dark
            else "#D8DEE8",
    }

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-color:
                {colors["background"]};
            color:
                {colors["text"]};
        }}

        [data-testid="stSidebar"] {{
            background-color:
                {colors["sidebar"]};
            border-right:
                1px solid
                {colors["border"]};
        }}

        [data-testid="stSidebar"] * {{
            color:
                {colors["text"]};
        }}

        h1, h2, h3, h4, p, label {{
            color:
                {colors["text"]};
        }}

        [data-testid="stMetric"] {{
            background-color:
                {colors["panel"]};
            border:
                1px solid
                {colors["border"]};
            border-radius: 14px;
            padding: 16px;
        }}

        [data-testid="stDataFrame"] {{
            border:
                1px solid
                {colors["border"]};
            border-radius: 12px;
        }}

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {{
            background-color:
                {colors["input"]};
            border-color:
                {colors["border"]};
        }}

        div[data-baseweb="input"] input,
        div[data-baseweb="select"] span {{
            color:
                {colors["text"]}
                !important;
        }}

        .workflow-card {{
            background:
                {colors["panel"]};
            border:
                1px solid
                {colors["border"]};
            border-radius: 12px;
            padding: 14px;
            text-align: center;
        }}

        .workflow-step {{
            font-size: 12px;
            color:
                {colors["muted"]};
        }}

        .workflow-title {{
            font-size: 16px;
            font-weight: 700;
            color:
                {colors["text"]};
        }}

        .app-subtitle {{
            color:
                {colors["muted"]};
            margin-top: -10px;
            margin-bottom: 20px;
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


# =========================================================
# モデル
# =========================================================

@st.cache_resource
def load_model():
    return load_model_info()


# =========================================================
# 駅情報
# =========================================================

def create_station_reference():
    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            "駅参照データがありません。"
            "data/reference/"
            "station_reference.csv "
            "を配置してください。"
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

    df["city"] = clean_text(
        df["city"]
    )

    df["district_name"] = (
        clean_text(
            df["district_name"]
        )
    )

    df["station_name"] = (
        clean_station(
            df["station_name"]
        )
    )

    df["station_line"] = (
        clean_text(
            df["station_line"]
        )
        .fillna("不明")
    )

    for column in [
        "station_latitude",
        "station_longitude",
    ]:
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
            station_latitude=(
                "station_latitude",
                "median",
            ),

            station_longitude=(
                "station_longitude",
                "median",
            ),

            count=(
                "station_name",
                "size",
            ),
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
        df = (
            create_station_reference()
        )

    df["city"] = clean_text(
        df["city"]
    )

    df["district_name"] = (
        clean_text(
            df["district_name"]
        )
    )

    df["station_name"] = (
        clean_station(
            df["station_name"]
        )
    )

    df["station_line"] = (
        clean_text(
            df["station_line"]
        )
        .fillna("不明")
    )

    return df


@st.cache_data
def load_station_maps():
    reference = (
        load_station_reference()
        .sort_values(
            "count",
            ascending=False,
        )
    )

    district_best = (
        reference.drop_duplicates(
            [
                "city",
                "district_name",
            ]
        )
    )

    detail_best = (
        reference.drop_duplicates(
            [
                "city",
                "district_name",
                "station_name",
            ]
        )
    )

    station_best = (
        reference.drop_duplicates(
            "station_name"
        )
    )

    district_map = {
        (
            str(row.city),
            str(row.district_name),
        ): str(
            row.station_name
        )
        for row
        in district_best.itertuples()
    }

    detail_map = {
        (
            str(row.city),
            str(row.district_name),
            str(row.station_name),
        ): {
            "station_line":
                str(
                    row.station_line
                ),

            "station_latitude":
                float(
                    row.station_latitude
                ),

            "station_longitude":
                float(
                    row.station_longitude
                ),
        }
        for row
        in detail_best.itertuples()
    }

    station_map = {
        str(row.station_name): {
            "station_line":
                str(
                    row.station_line
                ),

            "station_latitude":
                float(
                    row.station_latitude
                ),

            "station_longitude":
                float(
                    row.station_longitude
                ),
        }
        for row
        in station_best.itertuples()
    }

    return (
        district_map,
        detail_map,
        station_map,
    )


def fill_station_features(df):
    df = df.copy()

    (
        district_map,
        detail_map,
        station_map,
    ) = load_station_maps()

    df["city"] = clean_text(
        df["city"]
    )

    df["district_name"] = (
        clean_text(
            df["district_name"]
        )
    )

    if (
        "station_name"
        not in df.columns
    ):
        df["station_name"] = pd.NA

    if (
        "station_line"
        not in df.columns
    ):
        df["station_line"] = pd.NA

    df["station_name"] = (
        clean_station(
            df["station_name"]
        )
    )

    df["station_line"] = (
        clean_text(
            df["station_line"]
        )
    )

    for column in [
        "station_latitude",
        "station_longitude",
    ]:
        if column not in df.columns:
            df[column] = np.nan

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    errors = []

    for index, row in df.iterrows():
        city = str(
            row["city"]
        )

        district = str(
            row["district_name"]
        )

        station = row[
            "station_name"
        ]

        if pd.isna(station):
            station = (
                district_map.get(
                    (
                        city,
                        district,
                    )
                )
            )

            if station is None:
                errors.append(
                    f"{index + 1}行目: "
                    "最寄駅を補完できません。"
                )
                continue

            df.at[
                index,
                "station_name",
            ] = station

        station = str(station)

        info = detail_map.get(
            (
                city,
                district,
                station,
            )
        )

        if info is None:
            info = station_map.get(
                station
            )

        if info is None:
            errors.append(
                f"{index + 1}行目: "
                f"{station}駅の情報がありません。"
            )
            continue

        for column in [
            "station_line",
            "station_latitude",
            "station_longitude",
        ]:
            if pd.isna(
                row[column]
            ):
                df.at[
                    index,
                    column,
                ] = info[column]

    if errors:
        raise ValueError(
            "\n".join(errors)
        )

    return df


# =========================================================
# 入力
# =========================================================

def normalize_input(
    df,
    features,
    categories,
):
    df = (
        df.copy()
        .rename(
            columns=COLUMN_ALIASES
        )
    )

    missing = [
        column
        for column
        in REQUIRED_COLUMNS
        if column
        not in df.columns
    ]

    if missing:
        raise KeyError(
            "必要項目が不足しています: "
            + ", ".join(missing)
        )

    if (
        "property_id"
        not in df.columns
    ):
        df["property_id"] = [
            f"WEB{i + 1:04d}"
            for i in range(
                len(df)
            )
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
    ]:
        if column not in df.columns:
            df[column] = np.nan

    df["transaction_year"] = (
        date.today().year
    )

    df["transaction_quarter"] = (
        current_quarter()
    )

    for column in [
        "area_m2",
        "building_age",
        "asking_price",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if (
        df["asking_price"]
        .isna()
        .any()
    ):
        raise ValueError(
            "売出価格に未入力または"
            "不正な値があります。"
        )

    if (
        df["asking_price"]
        <= 0
    ).any():
        raise ValueError(
            "売出価格は0円より"
            "大きくしてください。"
        )

    if (
        df["area_m2"]
        .isna()
        .any()
        or (
            df["area_m2"]
            <= 0
        ).any()
    ):
        raise ValueError(
            "専有面積を正しく"
            "入力してください。"
        )

    for column in features:
        if column not in df.columns:
            df[column] = (
                "不明"
                if column
                in categories
                else np.nan
            )

    return df


# =========================================================
# 査定
# =========================================================

def assess(df):
    (
        model,
        _,
        features,
        categories,
    ) = load_model()

    df = normalize_input(
        df,
        features,
        categories,
    )

    df = fill_station_features(
        df
    )

    prepared = prepare_input_data(
        df,
        features,
        categories,
    )

    (
        unit_price,
        price,
    ) = predict_prices(
        model,
        prepared,
        features,
    )

    result = df.copy()

    result["査定日時"] = (
        now_jst()
        .strftime(
            "%Y/%m/%d %H:%M"
        )
    )

    result["推定㎡単価"] = (
        np.round(
            unit_price
        )
        .astype(int)
    )

    result["推定成約価格"] = (
        np.round(
            price
        )
        .astype(int)
    )

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


# =========================================================
# Supabase
# =========================================================

def database_configured():
    url, key = (
        supabase_settings()
    )

    return bool(
        url
        and key
    )


def database_headers():
    _, key = (
        supabase_settings()
    )

    return {
        "apikey": key,
        "Content-Type":
            "application/json",
        "Prefer":
            "return=minimal",
    }


def save_history_to_database(
    result,
):
    if not database_configured():
        return False

    base_url, _ = (
        supabase_settings()
    )

    url = (
        base_url.rstrip("/")
        + "/rest/v1/"
        + "appraisal_history"
    )

    rows = []

    for _, row in result.iterrows():
        rows.append(
            {
                "appraised_at":
                    now_jst()
                    .isoformat(),

                "case_name":
                    scalar(
                        row.get(
                            "案件名"
                        )
                    ),

                "staff":
                    scalar(
                        row.get(
                            "担当者"
                        )
                    ),

                "property_id":
                    scalar(
                        row.get(
                            "property_id"
                        )
                    ),

                "city":
                    scalar(
                        row.get(
                            "city"
                        )
                    ),

                "district_name":
                    scalar(
                        row.get(
                            "district_name"
                        )
                    ),

                "station_name":
                    scalar(
                        row.get(
                            "station_name"
                        )
                    ),

                "station_line":
                    scalar(
                        row.get(
                            "station_line"
                        )
                    ),

                "area_m2":
                    scalar(
                        row.get(
                            "area_m2"
                        )
                    ),

                "floor_plan":
                    scalar(
                        row.get(
                            "floor_plan"
                        )
                    ),

                "building_age":
                    scalar(
                        row.get(
                            "building_age"
                        )
                    ),

                "structure":
                    scalar(
                        row.get(
                            "structure"
                        )
                    ),

                "city_planning":
                    scalar(
                        row.get(
                            "city_planning"
                        )
                    ),

                "asking_price":
                    scalar(
                        row.get(
                            "asking_price"
                        )
                    ),

                "estimated_unit_price":
                    scalar(
                        row.get(
                            "推定㎡単価"
                        )
                    ),

                "selling_unit_price":
                    scalar(
                        row.get(
                            "売出㎡単価"
                        )
                    ),

                "estimated_price":
                    scalar(
                        row.get(
                            "推定成約価格"
                        )
                    ),

                "price_difference":
                    scalar(
                        row.get(
                            "売出価格との差額"
                        )
                    ),

                "gap_percent":
                    scalar(
                        row.get(
                            "価格乖離率(%)"
                        )
                    ),

                "price_evaluation":
                    scalar(
                        row.get(
                            "価格評価"
                        )
                    ),
            }
        )

    response = requests.post(
        url,
        headers=database_headers(),
        json=rows,
        timeout=15,
    )

    response.raise_for_status()

    load_database_history.clear()

    return True


@st.cache_data(
    ttl=30
)
def load_database_history():
    if not database_configured():
        return pd.DataFrame()

    base_url, _ = (
        supabase_settings()
    )

    url = (
        base_url.rstrip("/")
        + "/rest/v1/"
        + "appraisal_history"
    )

    response = requests.get(
        url,
        headers=database_headers(),
        params={
            "select": "*",
            "order":
                "appraised_at.desc",
            "limit":
                "50",
        },
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(
        data
    )

    rename = {
        "appraised_at":
            "査定日時",

        "case_name":
            "案件名",

        "staff":
            "担当者",

        "property_id":
            "物件ID",

        "city":
            "市区町村",

        "district_name":
            "地区",

        "station_name":
            "最寄駅",

        "station_line":
            "路線",

        "area_m2":
            "専有面積㎡",

        "floor_plan":
            "間取り",

        "building_age":
            "築年数",

        "structure":
            "構造",

        "city_planning":
            "用途地域",

        "asking_price":
            "売出価格",

        "estimated_unit_price":
            "推定㎡単価",

        "selling_unit_price":
            "売出㎡単価",

        "estimated_price":
            "推定成約価格",

        "price_difference":
            "売出価格との差額",

        "gap_percent":
            "価格乖離率(%)",

        "price_evaluation":
            "価格評価",
    }

    df = df.rename(
        columns=rename
    )

    if (
        "査定日時"
        in df.columns
    ):
        values = pd.to_datetime(
            df["査定日時"],
            errors="coerce",
            utc=True,
        )

        df["査定日時"] = (
            values
            .dt.tz_convert(
                "Asia/Tokyo"
            )
            .dt.strftime(
                "%Y/%m/%d %H:%M"
            )
        )

    for column in [
        "id",
        "created_at",
    ]:
        if column in df.columns:
            df = df.drop(
                columns=[
                    column
                ]
            )

    return df


# =========================================================
# 表示
# =========================================================

def sales_view(result):
    columns = [
        column
        for column
        in DISPLAY_COLUMNS
        if column
        in result.columns
    ]

    return (
        result[columns]
        .rename(
            columns=DISPLAY_NAMES
        )
    )


def sales_comment(row):
    gap = row[
        "価格乖離率(%)"
    ]

    if pd.isna(gap):
        return (
            "売出価格との比較情報を"
            "算出できませんでした。"
        )

    if gap > 20:
        return (
            "売出価格は推定成約価格を"
            "20%以上上回っています。"
            "価格設定や販売状況を"
            "再確認する際の参考にしてください。"
        )

    if gap > 10:
        return (
            "売出価格は推定成約価格を"
            "10%以上上回っています。"
            "反響状況とあわせて価格設定を"
            "確認する余地があります。"
        )

    if gap >= -10:
        return (
            "売出価格は推定成約価格の"
            "±10%以内に収まっています。"
            "現在の価格帯を検討する際の"
            "参考にできます。"
        )

    return (
        "売出価格は推定成約価格を"
        "10%以上下回っています。"
        "価格設定に引き上げ余地がないか"
        "確認する際の参考にしてください。"
    )


def result_summary(
    row,
    unit,
):
    case_name = (
        row.get(
            "案件名",
            "",
        )
        or "物件"
    )

    return (
        f"{case_name}\n"
        f"推定成約価格："
        f"{format_money(row['推定成約価格'], unit)}\n"
        f"売出価格："
        f"{format_money(row['asking_price'], unit)}\n"
        f"価格差："
        f"{format_money(row['売出価格との差額'], unit)}\n"
        f"価格乖離率："
        f"{format_percent(row['価格乖離率(%)'])}\n"
        f"価格評価："
        f"{row['価格評価']}"
    )


# =========================================================
# Excel
# =========================================================

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

        worksheet = writer[
            sheet_name
        ]

        worksheet.freeze_panes = (
            "A2"
        )

        for cells in (
            worksheet.columns
        ):
            letter = (
                cells[0]
                .column_letter
            )

            max_length = max(
                len(
                    ""
                    if cell.value
                    is None
                    else str(
                        cell.value
                    )
                )
                for cell in cells
            )

            worksheet.column_dimensions[
                letter
            ].width = min(
                max_length + 3,
                30,
            )

    return buffer.getvalue()


def template_excel():
    df = pd.DataFrame(
        [
            {
                "案件名":
                    "北千住マンション",

                "担当者":
                    "山田",

                "物件ID":
                    "A001",

                "市区町村":
                    "足立区",

                "地区":
                    "千住",

                "最寄駅":
                    "北千住",

                "専有面積㎡":
                    65.2,

                "間取り":
                    "3LDK",

                "築年数":
                    12,

                "構造":
                    "RC",

                "用途地域":
                    "商業地域",

                "売出価格":
                    75_000_000,
            }
        ]
    )

    return excel_bytes(
        df,
        "査定入力",
    )


# =========================================================
# グラフ
# =========================================================

def result_label(result):
    if (
        "案件名"
        in result.columns
    ):
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
    label = result_label(
        result
    )

    divisor = (
        10_000
        if unit == "万円"
        else 1
    )

    price = (
        result[
            [
                label,
                "推定成約価格",
                "asking_price",
            ]
        ]
        .copy()
    )

    price[
        "推定成約価格"
    ] /= divisor

    price[
        "asking_price"
    ] /= divisor

    price = (
        price.rename(
            columns={
                "asking_price":
                    "売出価格"
            }
        )
    )

    st.subheader(
        f"価格比較（{unit}）"
    )

    st.bar_chart(
        price.set_index(
            label
        )
    )

    unit_price = (
        result[
            [
                label,
                "推定㎡単価",
                "売出㎡単価",
            ]
        ]
        .copy()
    )

    st.subheader(
        "㎡単価比較（円 / ㎡）"
    )

    st.bar_chart(
        unit_price.set_index(
            label
        )
    )

    if len(result) > 1:
        gap = (
            result[
                [
                    label,
                    "価格乖離率(%)",
                ]
            ]
            .copy()
        )

        st.subheader(
            "価格乖離率（%）"
        )

        st.bar_chart(
            gap.set_index(
                label
            )
        )


# =========================================================
# 1件査定
# =========================================================

def single_input():
    st.subheader(
        "物件情報"
    )

    st.caption(
        "1行に物件情報を入力し、"
        "「査定する」を押してください。"
    )

    input_df = pd.DataFrame(
        [
            {
                "案件名":
                    "",

                "担当者":
                    "",

                "city":
                    "足立区",

                "district_name":
                    "千住",

                "station_name":
                    "北千住",

                "area_m2":
                    65.2,

                "floor_plan":
                    "3LDK",

                "building_age":
                    12,

                "structure":
                    "RC",

                "city_planning":
                    "商業地域",

                "asking_price":
                    75_000_000,
            }
        ]
    )

    edited_df = st.data_editor(
        input_df,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        row_height=42,
        key="single_property_editor",

        column_config={
            "案件名":
                st.column_config.TextColumn(
                    "案件名",
                    width="medium",
                ),

            "担当者":
                st.column_config.TextColumn(
                    "担当者",
                    width="small",
                ),

            "city":
                st.column_config.TextColumn(
                    "市区町村",
                    width="small",
                    required=True,
                ),

            "district_name":
                st.column_config.TextColumn(
                    "地区",
                    width="small",
                    required=True,
                ),

            "station_name":
                st.column_config.TextColumn(
                    "最寄駅",
                    width="small",
                ),

            "area_m2":
                st.column_config.NumberColumn(
                    "専有面積㎡",
                    min_value=1.0,
                    step=0.1,
                    format="%.1f",
                    width="small",
                    required=True,
                ),

            "floor_plan":
                st.column_config.TextColumn(
                    "間取り",
                    width="small",
                    required=True,
                ),

            "building_age":
                st.column_config.NumberColumn(
                    "築年数",
                    min_value=0,
                    step=1,
                    format="%d",
                    width="small",
                    required=True,
                ),

            "structure":
                st.column_config.SelectboxColumn(
                    "構造",
                    options=[
                        "RC",
                        "SRC",
                        "S",
                        "その他",
                        "不明",
                    ],
                    width="small",
                ),

            "city_planning":
                st.column_config.TextColumn(
                    "用途地域",
                    width="medium",
                ),

            "asking_price":
                st.column_config.NumberColumn(
                    "売出価格（円）",
                    min_value=1_000_000,
                    step=1_000_000,
                    format="%d",
                    width="medium",
                    required=True,
                ),
        },
    )

    left, right = (
        st.columns(
            [4, 1]
        )
    )

    with left:
        run = st.button(
            "査定する",
            type="primary",
            use_container_width=True,
        )

    with right:
        clear = st.button(
            "結果をクリア",
            use_container_width=True,
        )

    if clear:
        st.session_state.pop(
            "result",
            None,
        )

    if run:
        with st.spinner(
            "査定しています..."
        ):
            result = assess(
                edited_df
            )

            st.session_state[
                "result"
            ] = result

            try:
                if (
                    save_history_to_database(
                        result
                    )
                ):
                    st.toast(
                        "査定履歴を保存しました。"
                    )

            except requests.RequestException:
                st.warning(
                    "査定は完了しましたが、"
                    "履歴のクラウド保存に"
                    "失敗しました。"
                )


# =========================================================
# 一括査定
# =========================================================

def batch_input():
    st.subheader(
        "Excel一括査定"
    )

    st.caption(
        "複数物件をまとめて"
        "査定できます。"
    )

    st.download_button(
        "入力テンプレートをダウンロード",
        data=template_excel(),
        file_name=(
            "査定入力テンプレート.xlsx"
        ),
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

    suffix = (
        Path(
            file.name
        )
        .suffix
        .lower()
    )

    if suffix == ".csv":
        df = pd.read_csv(
            file
        )

    else:
        excel = pd.ExcelFile(
            file
        )

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

    st.success(
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
            result = assess(
                df
            )

            st.session_state[
                "result"
            ] = result

            try:
                save_history_to_database(
                    result
                )

            except requests.RequestException:
                st.warning(
                    "査定は完了しましたが、"
                    "履歴保存に失敗しました。"
                )


# =========================================================
# 結果
# =========================================================

def show_result(
    result,
    unit,
    show_graph,
):
    first = result.iloc[0]

    st.divider()

    st.header(
        "査定結果"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "推定成約価格",
        format_money(
            first[
                "推定成約価格"
            ],
            unit,
        ),
    )

    c2.metric(
        "売出価格",
        format_money(
            first[
                "asking_price"
            ],
            unit,
        ),
    )

    c3.metric(
        "価格乖離率",
        format_percent(
            first[
                "価格乖離率(%)"
            ]
        ),
    )

    c4.metric(
        "価格評価",
        first[
            "価格評価"
        ]
        or "-",
    )

    st.subheader(
        "参考コメント"
    )

    st.info(
        sales_comment(
            first
        )
    )

    difference = first[
        "売出価格との差額"
    ]

    if pd.notna(
        difference
    ):
        if difference > 0:
            st.caption(
                "売出価格は推定成約価格より "
                f"{format_money(abs(difference), unit)} "
                "高く設定されています。"
            )

        elif difference < 0:
            st.caption(
                "売出価格は推定成約価格より "
                f"{format_money(abs(difference), unit)} "
                "低く設定されています。"
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

    left, right = (
        st.columns(2)
    )

    with left:
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

    with right:
        if len(result) == 1:
            with st.popover(
                "説明用サマリー"
            ):
                st.text_area(
                    "コピー用",
                    value=result_summary(
                        first,
                        unit,
                    ),
                    height=160,
                )


# =========================================================
# 履歴
# =========================================================

def history_page():
    st.subheader(
        "査定履歴"
    )

    if not database_configured():
        st.info(
            "クラウド履歴が"
            "設定されていません。"
        )
        return

    try:
        history = (
            load_database_history()
        )

    except requests.RequestException as error:
        st.error(
            "査定履歴を"
            "読み込めませんでした。"
        )

        with st.expander(
            "エラー詳細"
        ):
            st.code(
                str(error)
            )

        return

    if history.empty:
        st.info(
            "まだ査定履歴はありません。"
        )
        return

    st.caption(
        "クラウドに保存された"
        "直近50件を表示しています。"
    )

    st.dataframe(
        history,
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "履歴をExcelで出力",
        data=excel_bytes(
            history,
            "査定履歴",
        ),
        file_name="査定履歴.xlsx",
        mime=(
            "application/"
            "vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
    )


# =========================================================
# UI
# =========================================================

def workflow():
    items = [
        (
            "STEP 1",
            "物件情報を入力",
        ),
        (
            "STEP 2",
            "査定する",
        ),
        (
            "STEP 3",
            "結果を確認・出力",
        ),
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
                    <div class="workflow-step">
                        {step}
                    </div>
                    <div class="workflow-title">
                        {title}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def sidebar_settings():
    with st.sidebar:

        st.header(
            "表示設定"
        )

        theme = st.radio(
            "テーマ",
            [
                "ライト",
                "ダーク",
            ],
            horizontal=True,
        )

        unit = st.radio(
            "金額表示",
            [
                "万円",
                "円",
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
            st.warning(
                "駅情報：未配置"
            )

        if database_configured():
            st.success(
                "査定履歴：クラウド保存"
            )
        else:
            st.info(
                "査定履歴：未接続"
            )

        if (
            st.session_state.get(
                "authenticated"
            )
            and app_password()
        ):
            if st.button(
                "ログアウト",
                use_container_width=True,
            ):
                st.session_state.pop(
                    "authenticated",
                    None,
                )

                st.rerun()

    return (
        theme,
        unit,
        show_graph,
    )


# =========================================================
# Main
# =========================================================

def main():

    if not access_gate():
        st.stop()

    (
        theme,
        unit,
        show_graph,
    ) = sidebar_settings()

    apply_theme(
        theme
    )

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

    (
        single_tab,
        batch_tab,
        history_tab,
    ) = st.tabs(
        [
            "1件査定",
            "Excel一括査定",
            "査定履歴",
        ]
    )

    try:

        with single_tab:
            single_input()

        with batch_tab:
            batch_input()

        with history_tab:
            history_page()

        if (
            "result"
            in st.session_state
        ):
            show_result(
                st.session_state[
                    "result"
                ],
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