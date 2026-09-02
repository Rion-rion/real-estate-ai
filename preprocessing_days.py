from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "contract_history.xlsx"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "days_training.csv"
)


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"成約履歴Excelが見つかりません: {INPUT_FILE}"
        )

    df = pd.read_excel(
        INPUT_FILE,
        sheet_name="成約履歴",
    )

    print(
        f"成約履歴読み込み: {len(df):,}件"
    )

    return df


def validate_columns(
    df: pd.DataFrame,
) -> None:

    required_columns = [
        "listing_date",
        "contract_date",
        "asking_price",
        "contract_price",
        "area_m2",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "必要なカラムがありません: "
            + ", ".join(missing_columns)
        )


def clean_dates(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["listing_date"] = pd.to_datetime(
        df["listing_date"],
        errors="coerce",
    )

    df["contract_date"] = pd.to_datetime(
        df["contract_date"],
        errors="coerce",
    )

    return df


def clean_numbers(
    df: pd.DataFrame,
) -> pd.DataFrame:

    numeric_columns = [
        "asking_price",
        "contract_price",
        "area_m2",
        "building_age",
        "coverage_ratio",
        "floor_area_ratio",
    ]

    for column in numeric_columns:

        if column not in df.columns:
            continue

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def create_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["days_to_contract"] = (
        df["contract_date"]
        - df["listing_date"]
    ).dt.days

    df["listing_year"] = (
        df["listing_date"].dt.year
    )

    df["listing_month"] = (
        df["listing_date"].dt.month
    )

    df["listing_quarter"] = (
        df["listing_date"].dt.quarter
    )

    df["asking_price_per_m2"] = (
        df["asking_price"]
        / df["area_m2"]
    )

    df["contract_price_per_m2"] = (
        df["contract_price"]
        / df["area_m2"]
    )

    df["price_gap_amount"] = (
        df["asking_price"]
        - df["contract_price"]
    )

    df["price_gap_ratio"] = (
        df["price_gap_amount"]
        / df["contract_price"]
    )

    return df


def remove_invalid_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:

    before = len(df)

    df = df[
        df["listing_date"].notna()
        & df["contract_date"].notna()
        & df["asking_price"].notna()
        & df["contract_price"].notna()
        & df["area_m2"].notna()
        & df["days_to_contract"].notna()
    ].copy()

    df = df[
        df["days_to_contract"] >= 0
    ].copy()

    df = df[
        df["days_to_contract"] <= 1000
    ].copy()

    df = df[
        df["asking_price"] > 0
    ].copy()

    df = df[
        df["contract_price"] > 0
    ].copy()

    df = df[
        df["area_m2"] > 0
    ].copy()

    removed = before - len(df)

    print(
        f"異常・欠損データ削除: "
        f"{removed:,}件"
    )

    return df


def select_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        "property_id",

        "city",
        "district_name",

        "area_m2",
        "floor_plan",
        "building_age",
        "structure",
        "renovation",
        "use",
        "city_planning",

        "coverage_ratio",
        "floor_area_ratio",

        "listing_date",
        "listing_year",
        "listing_month",
        "listing_quarter",

        "asking_price",
        "asking_price_per_m2",

        "contract_date",
        "contract_price",
        "contract_price_per_m2",

        "price_gap_amount",
        "price_gap_ratio",

        "days_to_contract",
    ]

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    return df[
        available_columns
    ].copy()


def show_summary(
    df: pd.DataFrame,
) -> None:

    print()
    print("成約日数データ 前処理完了")

    print(
        f"最終件数: {len(df):,}件"
    )

    print()
    print("成約日数:")

    print(
        df["days_to_contract"]
        .describe()
        .round(1)
    )

    print()
    print("欠損数:")

    print(
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )


def save_data(
    df: pd.DataFrame,
) -> None:

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "days_training.csv を保存しました。"
    )

    print(
        f"保存先: {OUTPUT_FILE}"
    )


def main() -> None:

    print(
        "中古マンション"
        "成約日数学習データ前処理"
    )

    print()

    df = load_data()

    validate_columns(df)

    df = clean_dates(df)

    df = clean_numbers(df)

    df = create_features(df)

    df = remove_invalid_rows(df)

    df = select_columns(df)

    show_summary(df)

    save_data(df)


if __name__ == "__main__":
    main()