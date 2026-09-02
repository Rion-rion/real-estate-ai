"""
collect_mlit.py

国土交通省「不動産情報ライブラリ API」から
東京都の成約価格情報を取得し、
中古マンションデータだけをCSV保存する。

出力:
data/raw/tokyo_contract_prices.csv
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


BASE_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001"

TOKYO_AREA_CODE = "13"
PRICE_CLASSIFICATION = "02"

START_YEAR = 2021
END_YEAR = 2026

REQUEST_INTERVAL_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 60

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "tokyo_contract_prices.csv"


def get_api_key() -> str:
    load_dotenv()

    api_key = os.getenv("MLIT_API_KEY")

    if not api_key:
        raise RuntimeError(
            "MLIT_API_KEY が見つかりません。\n"
            ".env に MLIT_API_KEY=APIキー と設定してください。"
        )

    return api_key


def fetch_quarter(
    session: requests.Session,
    api_key: str,
    year: int,
    quarter: int,
) -> list[dict]:

    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Accept": "application/json",
    }

    params = {
        "year": year,
        "quarter": quarter,
        "area": TOKYO_AREA_CODE,
        "priceClassification": PRICE_CLASSIFICATION,
        "language": "ja",
    }

    print(f"{year}年 第{quarter}四半期を取得中...")

    try:
        response = session.get(
            BASE_URL,
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

    except requests.exceptions.HTTPError as error:
        print(f"HTTPエラー: {error}")

        if error.response is not None:
            print(f"ステータスコード: {error.response.status_code}")
            print(error.response.text[:500])

        return []

    except requests.exceptions.RequestException as error:
        print(f"通信エラー: {error}")
        return []

    try:
        result = response.json()

    except requests.exceptions.JSONDecodeError:
        print("JSONの読み込みに失敗しました。")
        return []

    if result.get("status") != "OK":
        print("APIから正常なレスポンスが返されませんでした。")
        print(result)
        return []

    records = result.get("data", [])

    print(f"取得件数: {len(records):,}件")

    return records


def filter_used_condominiums(
    records: list[dict],
) -> list[dict]:

    filtered = []

    for record in records:
        property_type = str(record.get("Type", ""))

        if "中古マンション" in property_type:
            filtered.append(record)

    return filtered


def collect_data() -> pd.DataFrame:

    api_key = get_api_key()

    all_records: list[dict] = []

    with requests.Session() as session:

        for year in range(START_YEAR, END_YEAR + 1):

            for quarter in range(1, 5):

                records = fetch_quarter(
                    session=session,
                    api_key=api_key,
                    year=year,
                    quarter=quarter,
                )

                condominiums = filter_used_condominiums(records)

                print(
                    f"中古マンション: "
                    f"{len(condominiums):,}件"
                )

                all_records.extend(condominiums)

                time.sleep(REQUEST_INTERVAL_SECONDS)

    if not all_records:
        raise RuntimeError(
            "中古マンションデータを取得できませんでした。"
        )

    df = pd.DataFrame(all_records)

    return df


def save_data(df: pd.DataFrame) -> None:

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("データ取得完了")
    print(f"総件数: {len(df):,}件")
    print(f"保存先: {OUTPUT_FILE}")

    print()
    print("取得したカラム:")

    for column in df.columns:
        print(f"- {column}")


def main() -> None:

    print("東京都 中古マンション成約価格データを取得します。")
    print()

    df = collect_data()

    save_data(df)


if __name__ == "__main__":
    main()