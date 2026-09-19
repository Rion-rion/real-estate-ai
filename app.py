from __future__ import annotations

from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo
import hmac

import numpy as np
import pandas as pd
import requests
import streamlit as st
from openpyxl.utils import get_column_letter

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

SUPABASE_TABLE = "appraisal_history"
REQUEST_TIMEOUT = 15
HISTORY_LIMIT = 100

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
    "売出価格（円）": "asking_price",
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

HISTORY_RENAME = {
    "appraised_at": "査定日時",
    "case_name": "案件名",
    "staff": "担当者",
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
    "estimated_unit_price": "推定㎡単価",
    "selling_unit_price": "売出㎡単価",
    "estimated_price": "推定成約価格",
    "price_difference": "売出価格との差額",
    "gap_percent": "価格乖離率(%)",
    "price_evaluation": "価格評価",
}

st.set_page_config(
    page_title="不動産価格査定システム",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)


def current_quarter() -> int:
    return (date.today().month - 1) // 3 + 1


def now_jst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Tokyo"))


def safe_text(value, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else default


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )


def clean_station(series: pd.Series) -> pd.Series:
    return clean_text(series).str.replace(r"駅$", "", regex=True)


def scalar(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def format_money(value, unit: str = "万円") -> str:
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    return f"{value / 10_000:,.0f}万円" if unit == "万円" else f"{value:,.0f}円"


def format_percent(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):+.2f}%"


def generated_property_ids(count: int) -> list[str]:
    stamp = now_jst().strftime("%Y%m%d%H%M%S%f")
    return [f"WEB-{stamp}-{i + 1:03d}" for i in range(count)]


def app_password() -> str:
    try:
        return safe_text(st.secrets.get("APP_PASSWORD", ""))
    except Exception:
        return ""


def supabase_settings() -> tuple[str, str]:
    try:
        settings = st.secrets.get("supabase", {})
        return safe_text(settings.get("url", "")), safe_text(settings.get("secret_key", ""))
    except Exception:
        return "", ""


def access_gate() -> bool:
    expected = app_password()
    if not expected:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("🏢 不動産価格査定システム")
    st.caption("営業用画面へアクセスするにはパスワードを入力してください。")

    with st.form("login_form"):
        password = st.text_input("アクセスパスワード", type="password")
        submitted = st.form_submit_button("ログイン", type="primary", use_container_width=True)

    if submitted:
        if hmac.compare_digest(str(password), str(expected)):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("パスワードが違います。")

    return False


def apply_theme(theme: str) -> None:
    dark = theme == "ダーク"
    colors = {
        "background": "#0F172A" if dark else "#F7F8FA",
        "panel": "#172033" if dark else "#FFFFFF",
        "sidebar": "#111827" if dark else "#F1F4F8",
        "input": "#1E293B" if dark else "#FFFFFF",
        "text": "#F8FAFC" if dark else "#18202F",
        "muted": "#A7B0C0" if dark else "#667085",
        "border": "#334155" if dark else "#D8DEE8",
    }

    st.markdown(
        f"""
        <style>
        .stApp {{ background: {colors['background']}; color: {colors['text']}; }}
        [data-testid="stSidebar"] {{
            background: {colors['sidebar']};
            border-right: 1px solid {colors['border']};
        }}
        [data-testid="stSidebar"] * {{ color: {colors['text']}; }}
        h1, h2, h3, h4, p, label {{ color: {colors['text']}; }}
        [data-testid="stMetric"] {{
            background: {colors['panel']};
            border: 1px solid {colors['border']};
            border-radius: 14px;
            padding: 16px;
        }}
        [data-testid="stDataFrame"] {{
            border: 1px solid {colors['border']};
            border-radius: 12px;
        }}
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {{
            background: {colors['input']};
            border-color: {colors['border']};
        }}
        div[data-baseweb="input"] input,
        div[data-baseweb="select"] span {{ color: {colors['text']} !important; }}
        .workflow-card {{
            background: {colors['panel']};
            border: 1px solid {colors['border']};
            border-radius: 12px;
            padding: 14px;
            text-align: center;
        }}
        .workflow-step {{ font-size: 12px; color: {colors['muted']}; }}
        .workflow-title {{ font-size: 16px; font-weight: 700; color: {colors['text']}; }}
        .app-subtitle {{ color: {colors['muted']}; margin-top: -10px; margin-bottom: 20px; }}
        [data-testid="stToolbar"] {{ display: none; }}
        #MainMenu, footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model():
    return load_model_info()


def create_station_reference() -> pd.DataFrame:
    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            "駅参照データがありません。data/reference/station_reference.csv を配置してください。"
        )

    columns = [
        "city",
        "district_name",
        "station_name",
        "station_line",
        "station_latitude",
        "station_longitude",
    ]
    df = pd.read_csv(TRAINING_FILE, usecols=columns, low_memory=False)
    df["city"] = clean_text(df["city"])
    df["district_name"] = clean_text(df["district_name"])
    df["station_name"] = clean_station(df["station_name"])
    df["station_line"] = clean_text(df["station_line"]).fillna("不明")
    df["station_latitude"] = pd.to_numeric(df["station_latitude"], errors="coerce")
    df["station_longitude"] = pd.to_numeric(df["station_longitude"], errors="coerce")
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
            ["city", "district_name", "station_name", "station_line"],
            dropna=False,
        )
        .agg(
            station_latitude=("station_latitude", "median"),
            station_longitude=("station_longitude", "median"),
            count=("station_name", "size"),
        )
        .reset_index()
    )

    REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(REFERENCE_FILE, index=False, encoding="utf-8-sig")
    return reference


@st.cache_data(show_spinner=False)
def load_station_reference() -> pd.DataFrame:
    df = pd.read_csv(REFERENCE_FILE, low_memory=False) if REFERENCE_FILE.exists() else create_station_reference()

    required = {
        "city",
        "district_name",
        "station_name",
        "station_line",
        "station_latitude",
        "station_longitude",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError("station_reference.csv に必要な列がありません: " + ", ".join(missing))

    df = df.copy()
    df["city"] = clean_text(df["city"])
    df["district_name"] = clean_text(df["district_name"])
    df["station_name"] = clean_station(df["station_name"])
    df["station_line"] = clean_text(df["station_line"]).fillna("不明")
    df["station_latitude"] = pd.to_numeric(df["station_latitude"], errors="coerce")
    df["station_longitude"] = pd.to_numeric(df["station_longitude"], errors="coerce")
    if "count" not in df.columns:
        df["count"] = 1
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1)

    return df.dropna(
        subset=["city", "district_name", "station_name", "station_latitude", "station_longitude"]
    )


@st.cache_data(show_spinner=False)
def load_station_maps():
    reference = load_station_reference().sort_values("count", ascending=False)

    district_best = reference.drop_duplicates(["city", "district_name"])
    detail_best = reference.drop_duplicates(["city", "district_name", "station_name"])
    city_station_best = reference.drop_duplicates(["city", "station_name"])
    station_best = reference.drop_duplicates("station_name")

    def info(row) -> dict:
        return {
            "station_line": safe_text(row.station_line, "不明"),
            "station_latitude": float(row.station_latitude),
            "station_longitude": float(row.station_longitude),
        }

    district_map = {
        (str(row.city), str(row.district_name)): str(row.station_name)
        for row in district_best.itertuples()
    }
    detail_map = {
        (str(row.city), str(row.district_name), str(row.station_name)): info(row)
        for row in detail_best.itertuples()
    }
    city_station_map = {
        (str(row.city), str(row.station_name)): info(row)
        for row in city_station_best.itertuples()
    }
    station_map = {str(row.station_name): info(row) for row in station_best.itertuples()}

    return district_map, detail_map, city_station_map, station_map


def fill_station_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    district_map, detail_map, city_station_map, station_map = load_station_maps()

    df["city"] = clean_text(df["city"])
    df["district_name"] = clean_text(df["district_name"])

    if "station_name" not in df.columns:
        df["station_name"] = pd.Series(pd.NA, index=df.index, dtype="string")
    else:
        df["station_name"] = clean_station(df["station_name"])

    if "station_line" not in df.columns:
        df["station_line"] = pd.Series(pd.NA, index=df.index, dtype="string")
    else:
        df["station_line"] = clean_text(df["station_line"])

    for column in ["station_latitude", "station_longitude"]:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)

    errors = []

    for index, row in df.iterrows():
        city = safe_text(row.get("city"))
        district = safe_text(row.get("district_name"))
        station = safe_text(row.get("station_name"))

        if not station:
            station = district_map.get((city, district), "")
            if not station:
                errors.append(f"{index + 1}行目: {city} {district} の代表駅を補完できません。")
                continue
            df.at[index, "station_name"] = station

        reference = (
            detail_map.get((city, district, station))
            or city_station_map.get((city, station))
            or station_map.get(station)
        )
        if reference is None:
            errors.append(f"{index + 1}行目: {station}駅の駅情報が見つかりません。")
            continue

        if not safe_text(row.get("station_line")):
            df.at[index, "station_line"] = reference["station_line"]
        if pd.isna(row.get("station_latitude")):
            df.at[index, "station_latitude"] = reference["station_latitude"]
        if pd.isna(row.get("station_longitude")):
            df.at[index, "station_longitude"] = reference["station_longitude"]

    if errors:
        raise ValueError("\n".join(errors))

    return df


def normalize_input(df: pd.DataFrame, features: list[str], categories: list[str]) -> pd.DataFrame:
    df = df.copy().rename(columns=COLUMN_ALIASES)
    df.columns = [str(column).strip() for column in df.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        labels = [DISPLAY_NAMES.get(column, column) for column in missing]
        raise KeyError("必要項目が不足しています: " + ", ".join(labels))

    if "property_id" not in df.columns:
        df["property_id"] = generated_property_ids(len(df))
    else:
        ids = clean_text(df["property_id"])
        generated = generated_property_ids(len(df))
        df["property_id"] = [
            safe_text(value, generated[i]) for i, value in enumerate(ids)
        ]

    for column in ["station_name", "station_line", "structure", "city_planning"]:
        if column not in df.columns:
            df[column] = pd.NA

    for column in ["station_latitude", "station_longitude"]:
        if column not in df.columns:
            df[column] = np.nan

    for column in ["city", "district_name", "floor_plan"]:
        df[column] = clean_text(df[column])
        if df[column].isna().any():
            raise ValueError(f"{DISPLAY_NAMES.get(column, column)}に未入力があります。")

    df["station_name"] = clean_station(df["station_name"])
    df["station_line"] = clean_text(df["station_line"])
    df["structure"] = clean_text(df["structure"]).fillna("不明")
    df["city_planning"] = clean_text(df["city_planning"]).fillna("不明")

    df["area_m2"] = pd.to_numeric(df["area_m2"], errors="coerce")
    df["building_age"] = pd.to_numeric(df["building_age"], errors="coerce")
    df["asking_price"] = pd.to_numeric(df["asking_price"], errors="coerce")

    if df["area_m2"].isna().any() or (df["area_m2"] <= 0).any():
        raise ValueError("専有面積は0より大きい数値で入力してください。")
    if df["building_age"].isna().any() or (df["building_age"] < 0).any():
        raise ValueError("築年数は0以上の数値で入力してください。")
    if df["asking_price"].isna().any() or (df["asking_price"] <= 0).any():
        raise ValueError("売出価格は0円より大きい数値で入力してください。")

    df["transaction_year"] = date.today().year
    df["transaction_quarter"] = current_quarter()

    for column in features:
        if column not in df.columns:
            df[column] = "不明" if column in categories else np.nan

    return df


def assess(df: pd.DataFrame) -> pd.DataFrame:
    model, _, features, categories = load_model()
    normalized = normalize_input(df, features, categories)
    enriched = fill_station_features(normalized)
    prepared = prepare_input_data(enriched, features, categories)
    unit_price, price = predict_prices(model, prepared, features)

    result = enriched.copy()
    result["査定日時"] = now_jst().strftime("%Y/%m/%d %H:%M")
    result["推定㎡単価"] = np.round(unit_price).astype(int)
    result["推定成約価格"] = np.round(price).astype(int)
    result["売出㎡単価"] = result["asking_price"] / result["area_m2"]
    result["売出価格との差額"] = result["asking_price"] - result["推定成約価格"]
    result["価格乖離率(%)"] = (
        result["売出価格との差額"] / result["推定成約価格"] * 100
    )
    result["価格評価"] = result["価格乖離率(%)"].apply(evaluate_price_position)
    return result


def database_configured() -> bool:
    url, key = supabase_settings()
    return bool(url and key)


def database_url() -> str:
    base_url, _ = supabase_settings()
    return f"{base_url.rstrip('/')}/rest/v1/{SUPABASE_TABLE}"


def database_headers(*, minimal_return: bool = False) -> dict[str, str]:
    _, key = supabase_settings()
    headers = {
        "apikey": key,
        "Content-Type": "application/json",
    }
    if minimal_return:
        headers["Prefer"] = "return=minimal"
    return headers


def history_payload(result: pd.DataFrame) -> list[dict]:
    appraised_at = now_jst().isoformat()
    rows = []
    for _, row in result.iterrows():
        rows.append(
            {
                "appraised_at": appraised_at,
                "case_name": scalar(row.get("案件名")),
                "staff": scalar(row.get("担当者")),
                "property_id": scalar(row.get("property_id")),
                "city": scalar(row.get("city")),
                "district_name": scalar(row.get("district_name")),
                "station_name": scalar(row.get("station_name")),
                "station_line": scalar(row.get("station_line")),
                "area_m2": scalar(row.get("area_m2")),
                "floor_plan": scalar(row.get("floor_plan")),
                "building_age": scalar(row.get("building_age")),
                "structure": scalar(row.get("structure")),
                "city_planning": scalar(row.get("city_planning")),
                "asking_price": scalar(row.get("asking_price")),
                "estimated_unit_price": scalar(row.get("推定㎡単価")),
                "selling_unit_price": scalar(row.get("売出㎡単価")),
                "estimated_price": scalar(row.get("推定成約価格")),
                "price_difference": scalar(row.get("売出価格との差額")),
                "gap_percent": scalar(row.get("価格乖離率(%)")),
                "price_evaluation": scalar(row.get("価格評価")),
            }
        )
    return rows


def save_history_to_database(result: pd.DataFrame) -> bool:
    if not database_configured():
        return False

    response = requests.post(
        database_url(),
        headers=database_headers(minimal_return=True),
        json=history_payload(result),
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    load_database_history.clear()
    return True


@st.cache_data(ttl=30, show_spinner=False)
def load_database_history() -> pd.DataFrame:
    if not database_configured():
        return pd.DataFrame()

    response = requests.get(
        database_url(),
        headers=database_headers(),
        params={
            "select": "*",
            "order": "appraised_at.desc",
            "limit": str(HISTORY_LIMIT),
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data).rename(columns=HISTORY_RENAME)
    if "査定日時" in df.columns:
        dt = pd.to_datetime(df["査定日時"], errors="coerce", utc=True)
        df["査定日時"] = dt.dt.tz_convert("Asia/Tokyo").dt.strftime("%Y/%m/%d %H:%M")

    return df.drop(columns=[column for column in ["id", "created_at"] if column in df.columns])


def excel_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    """DataFrameをExcel化。pandas 2.xでは writer.sheets からWorksheetを取得する。"""
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for column_index, cells in enumerate(worksheet.columns, start=1):
            max_length = max(
                (len("" if cell.value is None else str(cell.value)) for cell in cells),
                default=0,
            )
            worksheet.column_dimensions[get_column_letter(column_index)].width = min(max_length + 3, 32)

    buffer.seek(0)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def template_excel() -> bytes:
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
    return excel_bytes(template, "査定入力")


def read_uploaded_file(file) -> pd.DataFrame:
    raw = file.getvalue()
    suffix = Path(file.name).suffix.lower()

    if suffix == ".xlsx":
        excel = pd.ExcelFile(BytesIO(raw))
        sheet = "査定入力" if "査定入力" in excel.sheet_names else excel.sheet_names[0]
        return pd.read_excel(excel, sheet_name=sheet)

    if suffix == ".csv":
        last_error = None
        for encoding in ["utf-8-sig", "cp932", "utf-8"]:
            try:
                return pd.read_csv(BytesIO(raw), encoding=encoding)
            except UnicodeDecodeError as error:
                last_error = error
        raise ValueError("CSVの文字コードを読み取れませんでした。UTF-8またはCP932で保存してください。") from last_error

    raise ValueError("対応しているファイル形式は .xlsx / .csv です。")


def sales_view(result: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in DISPLAY_COLUMNS if column in result.columns]
    return result[columns].rename(columns=DISPLAY_NAMES)


def sales_comment(row: pd.Series) -> str:
    gap = row.get("価格乖離率(%)")
    if gap is None or pd.isna(gap):
        return "売出価格との比較情報を算出できませんでした。"
    if gap > 20:
        return "売出価格は推定成約価格を20%以上上回っています。価格設定や販売状況を再確認する際の参考にしてください。"
    if gap > 10:
        return "売出価格は推定成約価格を10%以上上回っています。反響状況とあわせて価格設定を確認する余地があります。"
    if gap >= -10:
        return "売出価格は推定成約価格の±10%以内です。現在の価格帯を検討する際の参考にできます。"
    return "売出価格は推定成約価格を10%以上下回っています。価格設定に引き上げ余地がないか確認する際の参考にしてください。"


def result_summary(row: pd.Series, unit: str) -> str:
    case_name = safe_text(row.get("案件名"), "物件")
    return "\n".join(
        [
            case_name,
            f"推定成約価格：{format_money(row.get('推定成約価格'), unit)}",
            f"売出価格：{format_money(row.get('asking_price'), unit)}",
            f"価格差：{format_money(row.get('売出価格との差額'), unit)}",
            f"価格乖離率：{format_percent(row.get('価格乖離率(%)'))}",
            f"価格評価：{safe_text(row.get('価格評価'), '-')}",
        ]
    )


def result_label(result: pd.DataFrame) -> str:
    if "案件名" in result.columns:
        values = result["案件名"].fillna("").astype(str).str.strip()
        if values.ne("").all() and values.is_unique:
            return "案件名"
    return "property_id"


def show_charts(result: pd.DataFrame, unit: str) -> None:
    divisor = 10_000 if unit == "万円" else 1

    if len(result) == 1:
        row = result.iloc[0]

        price_compare = pd.DataFrame(
            {
                "金額": [
                    float(row["asking_price"]) / divisor,
                    float(row["推定成約価格"]) / divisor,
                ]
            },
            index=["売出価格", "推定成約価格"],
        )
        st.subheader(f"価格比較（{unit}）")
        st.bar_chart(price_compare, height=320)

        unit_price_compare = pd.DataFrame(
            {
                "円 / ㎡": [
                    float(row["売出㎡単価"]),
                    float(row["推定㎡単価"]),
                ]
            },
            index=["売出㎡単価", "推定㎡単価"],
        )
        st.subheader("㎡単価比較（円 / ㎡）")
        st.bar_chart(unit_price_compare, height=320)
        return

    label = result_label(result)

    price = result[[label, "asking_price", "推定成約価格"]].copy()
    price["売出価格"] = price.pop("asking_price") / divisor
    price["推定成約価格"] = price["推定成約価格"] / divisor
    st.subheader(f"価格比較（{unit}）")
    st.bar_chart(
        price,
        x=label,
        y=["売出価格", "推定成約価格"],
        stack=False,
        height=400,
    )

    unit_price = result[[label, "売出㎡単価", "推定㎡単価"]].copy()
    st.subheader("㎡単価比較（円 / ㎡）")
    st.bar_chart(
        unit_price,
        x=label,
        y=["売出㎡単価", "推定㎡単価"],
        stack=False,
        height=400,
    )

    gap = result[[label, "価格乖離率(%)"]].copy()
    st.subheader("価格乖離率（%）")
    st.bar_chart(
        gap,
        x=label,
        y="価格乖離率(%)",
        height=340,
    )

def single_input() -> None:
    st.subheader("物件情報")
    st.caption("上から順番に物件情報を入力してください。")

    with st.form("single_assessment_form", clear_on_submit=False):
        st.markdown("#### 基本情報")
        case_name = st.text_input("案件名", placeholder="例：北千住マンション")
        staff = st.text_input("担当者", placeholder="例：山田")
        property_id = st.text_input("物件ID（任意）", placeholder="例：A001")

        st.divider()
        st.markdown("#### 所在地")
        city = st.text_input("市区町村", value="足立区", placeholder="例：足立区")
        district = st.text_input("地区", value="千住", placeholder="例：千住")
        station = st.text_input(
            "最寄駅",
            value="北千住",
            placeholder="例：北千住",
            help="「駅」は付けても付けなくても大丈夫です。空欄の場合は地区から代表駅を補完します。",
        )

        st.divider()
        st.markdown("#### 物件情報")
        area = st.number_input("専有面積（㎡）", min_value=1.0, value=65.2, step=0.1, format="%.1f")
        floor_plan = st.text_input("間取り", value="3LDK", placeholder="例：3LDK")
        building_age = st.number_input("築年数", min_value=0, value=12, step=1)
        structure = st.selectbox("構造", ["RC", "SRC", "S", "その他", "不明"])
        city_planning = st.text_input("用途地域", value="商業地域", placeholder="例：商業地域")

        st.divider()
        st.markdown("#### 価格")
        asking_price = st.number_input(
            "売出価格（円）",
            min_value=1_000_000,
            value=75_000_000,
            step=1_000_000,
            format="%d",
        )

        submitted = st.form_submit_button("査定する", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "案件名": safe_text(case_name),
                    "担当者": safe_text(staff),
                    "property_id": safe_text(property_id) or pd.NA,
                    "city": city,
                    "district_name": district,
                    "station_name": safe_text(station) or pd.NA,
                    "area_m2": area,
                    "floor_plan": floor_plan,
                    "building_age": building_age,
                    "structure": structure,
                    "city_planning": city_planning,
                    "asking_price": asking_price,
                }
            ]
        )

        with st.spinner("査定しています..."):
            result = assess(input_df)
            st.session_state["result"] = result

            try:
                if save_history_to_database(result):
                    st.toast("査定履歴をクラウドに保存しました。")
            except requests.RequestException as error:
                st.warning("査定は完了しましたが、査定履歴のクラウド保存に失敗しました。")
                with st.expander("履歴保存エラー"):
                    st.code(str(error))

    if "result" in st.session_state:
        if st.button("査定結果をクリア", key="clear_single_result", use_container_width=True):
            st.session_state.pop("result", None)
            st.rerun()


def batch_input() -> None:
    st.subheader("Excel一括査定")
    st.caption("複数物件をまとめて査定できます。")

    st.download_button(
        "入力テンプレートをダウンロード",
        data=template_excel(),
        file_name="査定入力テンプレート.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    file = st.file_uploader("Excel / CSVを選択", type=["xlsx", "csv"])
    if file is None:
        return

    df = read_uploaded_file(file)
    if df.empty:
        st.warning("ファイルに査定対象がありません。")
        return

    st.success(f"{len(df):,}件を読み込みました。")
    st.dataframe(df, hide_index=True, use_container_width=True)

    if st.button("一括査定する", type="primary", use_container_width=True):
        with st.spinner("査定しています..."):
            result = assess(df)
            st.session_state["result"] = result

            try:
                if save_history_to_database(result):
                    st.toast(f"{len(result):,}件の査定履歴をクラウドに保存しました。")
            except requests.RequestException as error:
                st.warning("査定は完了しましたが、査定履歴のクラウド保存に失敗しました。")
                with st.expander("履歴保存エラー"):
                    st.code(str(error))


def show_single_result(first: pd.Series, unit: str) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("推定成約価格", format_money(first.get("推定成約価格"), unit))
    c2.metric("売出価格", format_money(first.get("asking_price"), unit))
    c3.metric("価格乖離率", format_percent(first.get("価格乖離率(%)")))
    c4.metric("価格評価", safe_text(first.get("価格評価"), "-"))

    st.subheader("参考コメント")
    st.info(sales_comment(first))

    difference = first.get("売出価格との差額")
    if difference is not None and pd.notna(difference):
        if difference > 0:
            st.caption(f"売出価格は推定成約価格より {format_money(abs(difference), unit)} 高く設定されています。")
        elif difference < 0:
            st.caption(f"売出価格は推定成約価格より {format_money(abs(difference), unit)} 低く設定されています。")


def show_batch_summary(result: pd.DataFrame, unit: str) -> None:
    average_gap = result["価格乖離率(%)"].mean()
    appropriate_count = int((result["価格評価"] == "適正").sum())
    average_price = result["推定成約価格"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("査定件数", f"{len(result):,}件")
    c2.metric("平均推定成約価格", format_money(average_price, unit))
    c3.metric("平均価格乖離率", format_percent(average_gap))
    c4.metric("適正価格帯", f"{appropriate_count:,}件")


def show_result(result: pd.DataFrame, unit: str, show_graph: bool) -> None:
    if result is None or result.empty:
        return

    st.divider()
    st.header("査定結果")

    if len(result) == 1:
        show_single_result(result.iloc[0], unit)
    else:
        show_batch_summary(result, unit)

    if show_graph:
        show_charts(result, unit)

    st.subheader("査定結果一覧")
    output = sales_view(result)
    st.dataframe(output, hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.download_button(
            "査定結果をExcelで出力",
            data=excel_bytes(output, "査定結果"),
            file_name="査定結果.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )

    with right:
        if len(result) == 1:
            with st.popover("説明用サマリー"):
                st.text_area(
                    "コピー用",
                    value=result_summary(result.iloc[0], unit),
                    height=180,
                )


def history_page() -> None:
    st.subheader("査定履歴")

    if not database_configured():
        st.info("クラウド履歴が設定されていません。Streamlit Secrets の Supabase 設定を確認してください。")
        return

    try:
        history = load_database_history()
    except requests.RequestException as error:
        st.error("査定履歴を読み込めませんでした。")
        with st.expander("エラー詳細"):
            st.code(str(error))
        return

    if history.empty:
        st.info("まだ査定履歴はありません。")
        return

    st.caption(f"クラウドに保存された直近{min(HISTORY_LIMIT, len(history))}件を表示しています。")

    search = st.text_input(
        "履歴を検索",
        placeholder="案件名・担当者・市区町村・駅名など",
        key="history_search",
    )
    filtered = history
    if search.strip():
        needle = search.strip().lower()
        mask = history.astype(str).apply(
            lambda column: column.str.lower().str.contains(needle, regex=False, na=False)
        ).any(axis=1)
        filtered = history[mask].copy()

    st.dataframe(filtered, hide_index=True, use_container_width=True)
    st.download_button(
        "表示中の履歴をExcelで出力",
        data=excel_bytes(filtered, "査定履歴"),
        file_name="査定履歴.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def workflow() -> None:
    items = [
        ("STEP 1", "物件情報を入力"),
        ("STEP 2", "査定する"),
        ("STEP 3", "結果を確認・出力"),
    ]
    for column, (step, title) in zip(st.columns(3), items):
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


def sidebar_settings() -> tuple[str, str, bool]:
    with st.sidebar:
        st.header("表示設定")
        theme = st.radio("テーマ", ["ライト", "ダーク"], horizontal=True)
        unit = st.radio("金額表示", ["万円", "円"], horizontal=True)
        show_graph = st.toggle("グラフを表示", value=True)

        st.divider()
        st.subheader("システム状態")
        st.success("査定機能：利用可能") if MODEL_FILE.exists() else st.error("査定機能：モデルなし")
        st.success("駅情報：利用可能") if REFERENCE_FILE.exists() else st.warning("駅情報：参照CSVなし")
        st.success("査定履歴：クラウド設定済み") if database_configured() else st.info("査定履歴：未設定")
        st.success("アクセス制御：有効") if app_password() else st.warning("アクセス制御：未設定")

        if st.session_state.get("authenticated") and app_password():
            if st.button("ログアウト", use_container_width=True):
                st.session_state.pop("authenticated", None)
                st.rerun()

    return theme, unit, show_graph


def main() -> None:
    if not access_gate():
        st.stop()

    theme, unit, show_graph = sidebar_settings()
    apply_theme(theme)

    st.title("不動産価格査定システム")
    st.markdown(
        '<div class="app-subtitle">東京都中古マンション 査定・売出価格検討支援</div>',
        unsafe_allow_html=True,
    )
    workflow()
    st.write("")

    single_tab, batch_tab, history_tab = st.tabs(["1件査定", "Excel一括査定", "査定履歴"])

    try:
        with single_tab:
            single_input()
        with batch_tab:
            batch_input()
        with history_tab:
            history_page()

        result = st.session_state.get("result")
        if isinstance(result, pd.DataFrame) and not result.empty:
            show_result(result, unit, show_graph)

    except Exception as error:
        st.error("処理できませんでした。入力内容またはシステム設定を確認してください。")
        with st.expander("エラー詳細"):
            st.code(str(error))

    st.divider()
    st.caption(
        "※ 査定結果は価格検討を支援する参考値です。正式な不動産鑑定評価を代替するものではありません。"
    )


if __name__ == "__main__":
    main()
