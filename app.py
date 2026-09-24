from __future__ import annotations

import time
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from openpyxl.utils import get_column_letter

from predict import evaluate_price_position, load_model_info, predict_prices, prepare_input_data

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "price_model.cbm"
REFERENCE_FILE = ROOT / "data" / "reference" / "station_reference.csv"
SUPABASE_TABLE = "appraisal_history"
REQUEST_TIMEOUT = 15
HISTORY_LIMIT = 100

COLUMN_ALIASES = {
    "物件ID": "property_id", "市区町村": "city", "地区": "district_name",
    "最寄駅": "station_name", "路線": "station_line", "専有面積": "area_m2",
    "専有面積㎡": "area_m2", "間取り": "floor_plan", "築年数": "building_age",
    "構造": "structure", "用途地域": "city_planning", "売出価格": "asking_price",
    "売出価格（円）": "asking_price",
}
REQUIRED_COLUMNS = {
    "city": "市区町村", "district_name": "地区", "area_m2": "専有面積㎡",
    "floor_plan": "間取り", "building_age": "築年数", "asking_price": "売出価格",
}
RESULT_COLUMNS = {
    "査定日時": "査定日時", "モデルVersion": "モデルVersion",
    "案件名": "案件名", "担当者": "担当者",
    "property_id": "物件ID", "city": "市区町村", "district_name": "地区",
    "station_name": "最寄駅", "station_line": "路線", "area_m2": "専有面積㎡",
    "floor_plan": "間取り", "building_age": "築年数", "structure": "構造",
    "city_planning": "用途地域", "asking_price": "売出価格", "推定㎡単価": "推定㎡単価",
    "売出㎡単価": "売出㎡単価", "推定成約価格": "推定成約価格",
    "売出価格との差額": "売出価格との差額", "価格乖離率(%)": "価格乖離率(%)",
    "価格評価": "価格評価",
}
DB_FIELDS = {
    "case_name": "案件名", "staff": "担当者", "property_id": "property_id",
    "city": "city", "district_name": "district_name", "station_name": "station_name",
    "station_line": "station_line", "area_m2": "area_m2", "floor_plan": "floor_plan",
    "building_age": "building_age", "structure": "structure", "city_planning": "city_planning",
    "asking_price": "asking_price", "estimated_unit_price": "推定㎡単価",
    "selling_unit_price": "売出㎡単価", "estimated_price": "推定成約価格",
    "price_difference": "売出価格との差額", "gap_percent": "価格乖離率(%)",
    "price_evaluation": "価格評価",
}
HISTORY_RENAME = {
    "appraised_at": "査定日時",
    "model_version": "モデルVersion",
    **{db: RESULT_COLUMNS.get(source, source) for db, source in DB_FIELDS.items()},
}

st.set_page_config(
    page_title="不動産価格査定システム", page_icon="🏢", layout="wide",
    initial_sidebar_state="expanded",
)


def safe_text(value, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return str(value).strip() or default


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA}
    )


def clean_station(series: pd.Series) -> pd.Series:
    return clean_text(series).str.replace(r"駅$", "", regex=True)


def json_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if isinstance(value, np.generic) else value


def now_jst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Tokyo"))


def generated_property_ids(count: int) -> list[str]:
    stamp = now_jst().strftime("%Y%m%d%H%M%S%f")
    return [f"WEB-{stamp}-{i + 1:03d}" for i in range(count)]


def format_money(value, unit: str = "万円") -> str:
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    return f"{value / 10_000:,.0f}万円" if unit == "万円" else f"{value:,.0f}円"


def format_percent(value) -> str:
    return "-" if value is None or pd.isna(value) else f"{float(value):+.2f}%"


def supabase_config() -> dict[str, str]:
    try:
        settings = st.secrets.get("supabase", {})
        return {
            "url": safe_text(settings.get("url")),
            "publishable_key": safe_text(settings.get("publishable_key")),
        }
    except Exception:
        return {"url": "", "publishable_key": ""}


def auth_configured() -> bool:
    cfg = supabase_config()
    return bool(cfg["url"] and cfg["publishable_key"])


def database_configured() -> bool:
    session = st.session_state.get("auth_session")
    return bool(
        auth_configured()
        and isinstance(session, dict)
        and session.get("access_token")
        and session.get("user_id")
    )


def auth_headers() -> dict[str, str]:
    return {"apikey": supabase_config()["publishable_key"], "Content-Type": "application/json"}


def save_auth_session(data: dict) -> None:
    previous = st.session_state.get("auth_session", {})
    user = data.get("user") or {}
    st.session_state["auth_session"] = {
        "access_token": data.get("access_token", ""),
        "refresh_token": data.get("refresh_token", previous.get("refresh_token", "")),
        "expires_at": time.time() + int(data.get("expires_in", 3600)),
        "user_id": user.get("id", previous.get("user_id", "")),
        "email": user.get("email", previous.get("email", "")),
    }


def sign_in(email: str, password: str) -> tuple[bool, str]:
    cfg = supabase_config()
    response = requests.post(
        f"{cfg['url'].rstrip('/')}/auth/v1/token?grant_type=password",
        headers=auth_headers(), json={"email": email.strip(), "password": password},
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        return False, "メールアドレスまたはパスワードを確認してください。"
    data = response.json()
    if not data.get("access_token"):
        return False, "ログイン情報を取得できませんでした。"
    save_auth_session(data)
    return True, ""


def refresh_auth_session() -> bool:
    session = st.session_state.get("auth_session")
    if not isinstance(session, dict) or not session.get("refresh_token"):
        return False
    cfg = supabase_config()
    try:
        response = requests.post(
            f"{cfg['url'].rstrip('/')}/auth/v1/token?grant_type=refresh_token",
            headers=auth_headers(), json={"refresh_token": session["refresh_token"]},
            timeout=REQUEST_TIMEOUT,
        )
        if not response.ok:
            return False
        data = response.json()
        if not data.get("access_token"):
            return False
        save_auth_session(data)
        return True
    except requests.RequestException:
        return False


def current_user() -> dict[str, str] | None:
    session = st.session_state.get("auth_session")
    if not isinstance(session, dict) or not session.get("access_token"):
        return None
    if float(session.get("expires_at", 0)) <= time.time() + 60:
        if not refresh_auth_session():
            st.session_state.pop("auth_session", None)
            return None
        session = st.session_state["auth_session"]
    return {"id": safe_text(session.get("user_id")), "email": safe_text(session.get("email"))}


def sign_out() -> None:
    session = st.session_state.get("auth_session")
    if isinstance(session, dict) and session.get("access_token") and auth_configured():
        try:
            headers = auth_headers() | {"Authorization": f"Bearer {session['access_token']}"}
            cfg = supabase_config()
            requests.post(
                f"{cfg['url'].rstrip('/')}/auth/v1/logout", headers=headers, timeout=REQUEST_TIMEOUT
            )
        except requests.RequestException:
            pass
    st.session_state.pop("auth_session", None)
    st.session_state.pop("result", None)


def current_access_token() -> str:
    if not current_user():
        return ""
    session = st.session_state.get("auth_session", {})
    return safe_text(session.get("access_token"))


def current_model_version() -> str:
    _, model_info, _, _ = load_model()
    return safe_text(model_info.get("version"), "Ver.5")


def access_gate() -> bool:
    if current_user():
        return True
    st.title("🏢 不動産価格査定システム")
    st.caption("担当者アカウントでログインしてください。")

    if not auth_configured():
        st.error("Supabase Authの設定が不足しています。")
        st.code(
            '[supabase]\nurl = "https://YOUR_PROJECT.supabase.co"\n'
            'publishable_key = "sb_publishable_..."'
        )
        return False

    with st.form("login_form"):
        email = st.text_input("メールアドレス", placeholder="example@company.jp")
        password = st.text_input("パスワード", type="password")
        submitted = st.form_submit_button("ログイン", type="primary", use_container_width=True)

    if not submitted:
        return False
    if not email.strip() or not password:
        st.error("メールアドレスとパスワードを入力してください。")
        return False
    try:
        success, message = sign_in(email, password)
    except requests.RequestException:
        st.error("認証サーバーへ接続できませんでした。")
        return False
    if success:
        st.rerun()
    st.error(message)
    return False


def apply_theme(theme: str) -> None:
    dark = theme == "ダーク"
    bg, panel, sidebar = (
        ("#0F172A", "#172033", "#111827") if dark else ("#F7F8FA", "#FFFFFF", "#F1F4F8")
    )
    text, muted, border = (
        ("#F8FAFC", "#A7B0C0", "#334155") if dark else ("#18202F", "#667085", "#D8DEE8")
    )
    st.markdown(
        f"""
        <style>
        .stApp {{background:{bg};color:{text}}}
        [data-testid="stSidebar"] {{background:{sidebar};border-right:1px solid {border}}}
        [data-testid="stSidebar"] *,h1,h2,h3,h4,p,label {{color:{text}}}
        [data-testid="stMetric"] {{background:{panel};border:1px solid {border};border-radius:14px;padding:16px}}
        [data-testid="stDataFrame"] {{border:1px solid {border};border-radius:12px}}
        .workflow-card {{background:{panel};border:1px solid {border};border-radius:12px;padding:14px;text-align:center}}
        .workflow-step {{font-size:12px;color:{muted}}}
        .workflow-title {{font-size:16px;font-weight:700;color:{text}}}
        .app-subtitle {{color:{muted};margin-top:-10px;margin-bottom:20px}}
        [data-testid="stToolbar"] {{display:none}} #MainMenu,footer {{visibility:hidden}}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model():
    return load_model_info()


@st.cache_data(show_spinner=False)
def load_station_reference() -> pd.DataFrame:
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError("data/reference/station_reference.csv が見つかりません。")
    df = pd.read_csv(REFERENCE_FILE, low_memory=False)
    required = {
        "city", "district_name", "station_name", "station_line",
        "station_latitude", "station_longitude",
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
    count = df["count"] if "count" in df.columns else pd.Series(1, index=df.index)
    df["count"] = pd.to_numeric(count, errors="coerce").fillna(1)
    return df.dropna(
        subset=["city", "district_name", "station_name", "station_latitude", "station_longitude"]
    )


@st.cache_data(show_spinner=False)
def load_station_maps():
    reference = load_station_reference().sort_values("count", ascending=False)

    def info(row) -> dict:
        return {
            "station_line": safe_text(row.station_line, "不明"),
            "station_latitude": float(row.station_latitude),
            "station_longitude": float(row.station_longitude),
        }

    district = reference.drop_duplicates(["city", "district_name"])
    detail = reference.drop_duplicates(["city", "district_name", "station_name"])
    city_station = reference.drop_duplicates(["city", "station_name"])
    station = reference.drop_duplicates("station_name")
    return (
        {(str(r.city), str(r.district_name)): str(r.station_name) for r in district.itertuples()},
        {(str(r.city), str(r.district_name), str(r.station_name)): info(r) for r in detail.itertuples()},
        {(str(r.city), str(r.station_name)): info(r) for r in city_station.itertuples()},
        {str(r.station_name): info(r) for r in station.itertuples()},
    )


def fill_station_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    district_map, detail_map, city_station_map, station_map = load_station_maps()
    df["city"] = clean_text(df["city"])
    df["district_name"] = clean_text(df["district_name"])
    df["station_name"] = (
        clean_station(df["station_name"])
        if "station_name" in df else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    df["station_line"] = (
        clean_text(df["station_line"])
        if "station_line" in df else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    for column in ("station_latitude", "station_longitude"):
        if column not in df:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)

    errors = []
    for index, row in df.iterrows():
        city, district = safe_text(row.get("city")), safe_text(row.get("district_name"))
        station = safe_text(row.get("station_name"))
        if not station:
            station = district_map.get((city, district), "")
            if not station:
                errors.append(f"{index + 1}行目: {city} {district} の代表駅を補完できません。")
                continue
            df.at[index, "station_name"] = station

        ref = (
            detail_map.get((city, district, station))
            or city_station_map.get((city, station))
            or station_map.get(station)
        )
        if ref is None:
            errors.append(f"{index + 1}行目: {station}駅の駅情報が見つかりません。")
            continue
        if not safe_text(row.get("station_line")):
            df.at[index, "station_line"] = ref["station_line"]
        for column in ("station_latitude", "station_longitude"):
            if pd.isna(row.get(column)):
                df.at[index, column] = ref[column]

    if errors:
        raise ValueError("\n".join(errors))
    return df


def normalize_input(df: pd.DataFrame, features: list[str], categories: list[str]) -> pd.DataFrame:
    df = df.copy().rename(columns=COLUMN_ALIASES)
    df.columns = [str(column).strip() for column in df.columns]
    missing = [label for column, label in REQUIRED_COLUMNS.items() if column not in df]
    if missing:
        raise KeyError("必要項目が不足しています: " + ", ".join(missing))

    default_staff = (current_user() or {}).get("email", "")
    if "担当者" not in df:
        df["担当者"] = default_staff
    else:
        df["担当者"] = df["担当者"].apply(lambda value: safe_text(value, default_staff))

    generated = generated_property_ids(len(df))
    if "property_id" not in df:
        df["property_id"] = generated
    else:
        df["property_id"] = [
            safe_text(value, generated[i]) for i, value in enumerate(clean_text(df["property_id"]))
        ]

    for column in ("station_name", "station_line", "structure", "city_planning"):
        if column not in df:
            df[column] = pd.NA
    for column in ("station_latitude", "station_longitude"):
        if column not in df:
            df[column] = np.nan

    for column in ("city", "district_name", "floor_plan"):
        df[column] = clean_text(df[column])
        if df[column].isna().any():
            raise ValueError(f"{REQUIRED_COLUMNS[column]}に未入力があります。")

    df["station_name"] = clean_station(df["station_name"])
    df["station_line"] = clean_text(df["station_line"])
    df["structure"] = clean_text(df["structure"]).fillna("不明")
    df["city_planning"] = clean_text(df["city_planning"]).fillna("不明")

    rules = {
        "area_m2": (0, "専有面積は0より大きい数値で入力してください。"),
        "building_age": (-1, "築年数は0以上の数値で入力してください。"),
        "asking_price": (0, "売出価格は0円より大きい数値で入力してください。"),
    }
    for column, (minimum, message) in rules.items():
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any() or (df[column] <= minimum).any():
            raise ValueError(message)

    df["transaction_year"] = date.today().year
    df["transaction_quarter"] = (date.today().month - 1) // 3 + 1
    for column in features:
        if column not in df:
            df[column] = "不明" if column in categories else np.nan
    return df


def assess(df: pd.DataFrame) -> pd.DataFrame:
    model, _, features, categories = load_model()
    enriched = fill_station_features(normalize_input(df, features, categories))
    prepared = prepare_input_data(enriched, features, categories)
    unit_price, price = predict_prices(model, prepared, features)

    result = enriched.copy()
    result["査定日時"] = now_jst().strftime("%Y/%m/%d %H:%M")
    result["モデルVersion"] = current_model_version()
    result["推定㎡単価"] = np.round(unit_price).astype(int)
    result["推定成約価格"] = np.round(price).astype(int)
    result["売出㎡単価"] = result["asking_price"] / result["area_m2"]
    result["売出価格との差額"] = result["asking_price"] - result["推定成約価格"]
    result["価格乖離率(%)"] = result["売出価格との差額"] / result["推定成約価格"] * 100
    result["価格評価"] = result["価格乖離率(%)"].apply(evaluate_price_position)
    return result


def database_url() -> str:
    return f"{supabase_config()['url'].rstrip('/')}/rest/v1/{SUPABASE_TABLE}"


def database_headers(minimal: bool = False) -> dict[str, str]:
    token = current_access_token()
    if not token:
        raise RuntimeError("ログインセッションを確認できません。再ログインしてください。")

    headers = {
        "apikey": supabase_config()["publishable_key"],
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if minimal:
        headers["Prefer"] = "return=minimal"
    return headers


def save_history(result: pd.DataFrame) -> bool:
    if not database_configured():
        return False

    user = current_user()
    if not user or not user.get("id"):
        raise RuntimeError("ログインユーザーを確認できません。再ログインしてください。")

    appraised_at = now_jst().isoformat()
    model_version = current_model_version()
    payload = [
        {
            "appraised_at": appraised_at,
            "user_id": user["id"],
            "model_version": model_version,
            **{db: json_value(row.get(source)) for db, source in DB_FIELDS.items()},
        }
        for _, row in result.iterrows()
    ]

    response = requests.post(
        database_url(),
        headers=database_headers(True),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    load_history.clear()
    return True


@st.cache_data(ttl=30, show_spinner=False)
def load_history() -> pd.DataFrame:
    if not database_configured():
        return pd.DataFrame()

    user = current_user()
    if not user or not user.get("id"):
        return pd.DataFrame()

    response = requests.get(
        database_url(),
        headers=database_headers(),
        params={
            "select": "*",
            "user_id": f"eq.{user['id']}",
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
    if "査定日時" in df:
        dt = pd.to_datetime(df["査定日時"], errors="coerce", utc=True)
        df["査定日時"] = dt.dt.tz_convert("Asia/Tokyo").dt.strftime("%Y/%m/%d %H:%M")

    hidden = [column for column in ("id", "created_at", "user_id") if column in df]
    return df.drop(columns=hidden)


def excel_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        sheet = writer.sheets[sheet_name]
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for index, cells in enumerate(sheet.columns, 1):
            width = max((len("" if cell.value is None else str(cell.value)) for cell in cells), default=0)
            sheet.column_dimensions[get_column_letter(index)].width = min(width + 3, 32)
    buffer.seek(0)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def template_excel() -> bytes:
    return excel_bytes(
        pd.DataFrame([{
            "案件名": "北千住マンション", "担当者": "山田", "物件ID": "A001",
            "市区町村": "足立区", "地区": "千住", "最寄駅": "北千住",
            "専有面積㎡": 65.2, "間取り": "3LDK", "築年数": 12,
            "構造": "RC", "用途地域": "商業地域", "売出価格": 75_000_000,
        }]),
        "査定入力",
    )


def read_uploaded_file(file) -> pd.DataFrame:
    raw, suffix = file.getvalue(), Path(file.name).suffix.lower()
    if suffix == ".xlsx":
        excel = pd.ExcelFile(BytesIO(raw))
        sheet = "査定入力" if "査定入力" in excel.sheet_names else excel.sheet_names[0]
        return pd.read_excel(excel, sheet_name=sheet)
    if suffix == ".csv":
        for encoding in ("utf-8-sig", "cp932", "utf-8"):
            try:
                return pd.read_csv(BytesIO(raw), encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("CSVはUTF-8またはCP932で保存してください。")
    raise ValueError("対応形式は .xlsx / .csv です。")


def sales_view(result: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in RESULT_COLUMNS if column in result]
    return result[columns].rename(columns=RESULT_COLUMNS)


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
    return "\n".join([
        safe_text(row.get("案件名"), "物件"),
        f"推定成約価格：{format_money(row.get('推定成約価格'), unit)}",
        f"売出価格：{format_money(row.get('asking_price'), unit)}",
        f"価格差：{format_money(row.get('売出価格との差額'), unit)}",
        f"価格乖離率：{format_percent(row.get('価格乖離率(%)'))}",
        f"価格評価：{safe_text(row.get('価格評価'), '-')}",
    ])


def result_label(result: pd.DataFrame) -> str:
    if "案件名" in result:
        values = result["案件名"].fillna("").astype(str).str.strip()
        if values.ne("").all() and values.is_unique:
            return "案件名"
    return "property_id"


def show_charts(result: pd.DataFrame, unit: str) -> None:
    divisor = 10_000 if unit == "万円" else 1
    if len(result) == 1:
        row = result.iloc[0]
        st.subheader(f"価格比較（{unit}）")
        st.bar_chart(
            pd.DataFrame({
                "項目": ["売出価格", "推定成約価格"],
                "金額": [float(row["asking_price"]) / divisor, float(row["推定成約価格"]) / divisor],
            }),
            x="項目", y="金額", x_label="", y_label=f"金額（{unit}）",
            horizontal=True, sort=False, height=230,
        )
        st.subheader("㎡単価比較（円 / ㎡）")
        st.bar_chart(
            pd.DataFrame({
                "項目": ["売出㎡単価", "推定㎡単価"],
                "㎡単価": [float(row["売出㎡単価"]), float(row["推定㎡単価"])],
            }),
            x="項目", y="㎡単価", x_label="", y_label="円 / ㎡",
            horizontal=True, sort=False, height=230,
        )
        return

    label = result_label(result)
    price = result[[label, "asking_price", "推定成約価格"]].copy()
    price["売出価格"] = price.pop("asking_price") / divisor
    price["推定成約価格"] /= divisor
    st.subheader(f"価格比較（{unit}）")
    st.bar_chart(
        price, x=label, y=["売出価格", "推定成約価格"], x_label="",
        y_label=f"金額（{unit}）", stack=False, sort=False, height=400,
    )

    st.subheader("㎡単価比較（円 / ㎡）")
    st.bar_chart(
        result[[label, "売出㎡単価", "推定㎡単価"]], x=label,
        y=["売出㎡単価", "推定㎡単価"], x_label="", y_label="円 / ㎡",
        stack=False, sort=False, height=400,
    )
    st.subheader("価格乖離率（%）")
    st.bar_chart(
        result[[label, "価格乖離率(%)"]], x=label, y="価格乖離率(%)",
        x_label="", y_label="乖離率（%）", sort=False, height=340,
    )


def save_result(result: pd.DataFrame) -> None:
    st.session_state["result"] = result
    try:
        if save_history(result):
            st.toast("査定履歴をクラウドに保存しました。")
    except requests.RequestException as error:
        st.warning("査定は完了しましたが、査定履歴の保存に失敗しました。")
        with st.expander("履歴保存エラー"):
            st.code(str(error))


def single_input() -> None:
    email = (current_user() or {}).get("email", "")
    st.subheader("物件情報")
    st.caption("上から順番に物件情報を入力してください。")

    with st.form("single_assessment_form", clear_on_submit=False):
        st.markdown("#### 基本情報")
        case_name = st.text_input("案件名", placeholder="例：北千住マンション")
        staff = st.text_input("担当者", value=email, placeholder="例：山田")
        property_id = st.text_input("物件ID（任意）", placeholder="例：A001")
        st.divider()
        st.markdown("#### 所在地")
        city = st.text_input("市区町村", value="足立区")
        district = st.text_input("地区", value="千住")
        station = st.text_input(
            "最寄駅", value="北千住",
            help="「駅」は付けても付けなくても大丈夫です。空欄の場合は地区から代表駅を補完します。",
        )
        st.divider()
        st.markdown("#### 物件情報")
        area = st.number_input("専有面積（㎡）", min_value=1.0, value=65.2, step=0.1, format="%.1f")
        floor_plan = st.text_input("間取り", value="3LDK")
        building_age = st.number_input("築年数", min_value=0, value=12, step=1)
        structure = st.selectbox("構造", ["RC", "SRC", "S", "その他", "不明"])
        city_planning = st.text_input("用途地域", value="商業地域")
        st.divider()
        st.markdown("#### 価格")
        asking_price = st.number_input(
            "売出価格（円）", min_value=1_000_000, value=75_000_000,
            step=1_000_000, format="%d",
        )
        submitted = st.form_submit_button("査定する", type="primary", use_container_width=True)

    if submitted:
        data = pd.DataFrame([{
            "案件名": safe_text(case_name), "担当者": safe_text(staff, email),
            "property_id": safe_text(property_id) or pd.NA, "city": city,
            "district_name": district, "station_name": safe_text(station) or pd.NA,
            "area_m2": area, "floor_plan": floor_plan, "building_age": building_age,
            "structure": structure, "city_planning": city_planning, "asking_price": asking_price,
        }])
        with st.spinner("査定しています..."):
            save_result(assess(data))

    if "result" in st.session_state and st.button(
        "査定結果をクリア", key="clear_single_result", use_container_width=True
    ):
        st.session_state.pop("result", None)
        st.rerun()


def batch_input() -> None:
    st.subheader("Excel一括査定")
    st.caption("複数物件をまとめて査定できます。")
    st.download_button(
        "入力テンプレートをダウンロード", data=template_excel(),
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
            save_result(assess(df))


def show_result(result: pd.DataFrame, unit: str, show_graph: bool) -> None:
    st.divider()
    st.header("査定結果")
    if len(result) == 1:
        row = result.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("推定成約価格", format_money(row.get("推定成約価格"), unit))
        c2.metric("売出価格", format_money(row.get("asking_price"), unit))
        c3.metric("価格乖離率", format_percent(row.get("価格乖離率(%)")))
        c4.metric("価格評価", safe_text(row.get("価格評価"), "-"))
        st.subheader("参考コメント")
        st.info(sales_comment(row))
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("査定件数", f"{len(result):,}件")
        c2.metric("平均推定成約価格", format_money(result["推定成約価格"].mean(), unit))
        c3.metric("平均価格乖離率", format_percent(result["価格乖離率(%)"].mean()))
        c4.metric("適正価格帯", f"{int((result['価格評価'] == '適正').sum()):,}件")

    if show_graph:
        show_charts(result, unit)

    st.subheader("査定結果一覧")
    output = sales_view(result)
    st.dataframe(output, hide_index=True, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.download_button(
            "査定結果をExcelで出力", data=excel_bytes(output, "査定結果"),
            file_name="査定結果.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary", use_container_width=True,
        )
    with right:
        if len(result) == 1:
            with st.popover("説明用サマリー"):
                st.text_area("コピー用", value=result_summary(result.iloc[0], unit), height=180)


def history_page() -> None:
    st.subheader("査定履歴")
    if not database_configured():
        st.info("クラウド履歴が設定されていません。")
        return
    try:
        history = load_history()
    except requests.RequestException as error:
        st.error("査定履歴を読み込めませんでした。")
        with st.expander("エラー詳細"):
            st.code(str(error))
        return
    if history.empty:
        st.info("まだ査定履歴はありません。")
        return

    st.caption(f"ログイン中の担当者に紐づく直近{min(HISTORY_LIMIT, len(history))}件を表示しています。")
    search = st.text_input("履歴を検索", placeholder="案件名・担当者・市区町村・駅名など")
    filtered = history
    if search.strip():
        needle = search.strip().lower()
        mask = history.astype(str).apply(
            lambda column: column.str.lower().str.contains(needle, regex=False, na=False)
        ).any(axis=1)
        filtered = history[mask].copy()
    st.dataframe(filtered, hide_index=True, use_container_width=True)
    st.download_button(
        "表示中の履歴をExcelで出力", data=excel_bytes(filtered, "査定履歴"),
        file_name="査定履歴.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def sidebar_settings() -> tuple[str, str, bool]:
    user = current_user()
    with st.sidebar:
        st.header("表示設定")
        theme = st.radio("テーマ", ["ライト", "ダーク"], horizontal=True)
        unit = st.radio("金額表示", ["万円", "円"], horizontal=True)
        show_graph = st.toggle("グラフを表示", value=True)
        st.divider()
        st.subheader("ログイン情報")
        st.write(f"👤 {(user or {}).get('email', '-')}")
        st.divider()
        st.subheader("システム状態")

        if MODEL_FILE.exists():
            st.success("査定機能：利用可能")
        else:
            st.error("査定機能：モデルなし")

        if REFERENCE_FILE.exists():
            st.success("駅情報：利用可能")
        else:
            st.warning("駅情報：参照CSVなし")

        if database_configured():
            st.success("査定履歴：ユーザー別保存")
        else:
            st.info("査定履歴：未設定")

        if auth_configured():
            st.success("ユーザー認証：有効")
        else:
            st.error("ユーザー認証：未設定")
        st.divider()
        if st.button("ログアウト", use_container_width=True):
            sign_out()
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

    for column, (step, title) in zip(
        st.columns(3),
        [("STEP 1", "物件情報を入力"), ("STEP 2", "査定する"), ("STEP 3", "結果を確認・出力")],
    ):
        with column:
            st.markdown(
                f'<div class="workflow-card"><div class="workflow-step">{step}</div>'
                f'<div class="workflow-title">{title}</div></div>', unsafe_allow_html=True,
            )
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
    st.caption("※ 査定結果は価格検討を支援する参考値です。正式な不動産鑑定評価を代替するものではありません。")


if __name__ == "__main__":
    main()
