from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "contract_history.xlsx"
)

PRICE_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "price_model.cbm"
)

PRICE_METRICS_FILE = (
    PROJECT_ROOT
    / "output"
    / "price_model_metrics.json"
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


def load_contract_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"成約履歴が見つかりません: {INPUT_FILE}"
        )

    df = pd.read_excel(
        INPUT_FILE,
        sheet_name="成約履歴",
    )

    print(
        f"成約履歴読み込み: {len(df):,}件"
    )

    return df


def load_price_model():
    if not PRICE_MODEL_FILE.exists():
        raise FileNotFoundError(
            "価格予測モデルが見つかりません。"
        )

    if not PRICE_METRICS_FILE.exists():
        raise FileNotFoundError(
            "価格モデル情報が見つかりません。"
        )

    with open(
        PRICE_METRICS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        model_info = json.load(file)

    feature_columns = model_info.get(
        "features",
        []
    )

    categorical_columns = model_info.get(
        "categorical_features",
        []
    )

    model = CatBoostRegressor()

    model.load_model(
        str(PRICE_MODEL_FILE)
    )

    return (
        model,
        feature_columns,
        categorical_columns,
    )


def validate_columns(
    df: pd.DataFrame,
) -> None:

    required_columns = [
        "listing_date",
        "contract_date",
        "asking_price",
        "area_m2",
        "city",
        "district_name",
        "floor_plan",
        "building_age",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            "不足カラム: "
            + ", ".join(missing)
        )


def clean_data(
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

    numeric_columns = [
        "asking_price",
        "contract_price",
        "area_m2",
        "building_age",
        "coverage_ratio",
        "floor_area_ratio",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


def create_date_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["days_to_contract"] = (
        df["contract_date"]
        - df["listing_date"]
    ).dt.days

    df["listing_year"] = (
        df["listing_date"]
        .dt.year
    )

    df["listing_month"] = (
        df["listing_date"]
        .dt.month
    )

    df["listing_quarter"] = (
        df["listing_date"]
        .dt.quarter
    )

    df["asking_price_per_m2"] = (
        df["asking_price"]
        / df["area_m2"]
    )

    return df


def prepare_price_model_input(
    df: pd.DataFrame,
    feature_columns,
    categorical_columns,
) -> pd.DataFrame:

    price_df = df.copy()

    # 価格モデルでは transaction_year / quarter
    # という名前で学習しているため、
    # 売出時点の日付を使用する。
    price_df["transaction_year"] = (
        price_df["listing_year"]
    )

    price_df["transaction_quarter"] = (
        price_df["listing_quarter"]
    )

    for column in feature_columns:

        if column not in price_df.columns:
            price_df[column] = np.nan

        if column in categorical_columns:

            price_df[column] = (
                price_df[column]
                .fillna("不明")
                .astype(str)
            )

        else:

            price_df[column] = pd.to_numeric(
                price_df[column],
                errors="coerce",
            )

    return price_df


def create_price_ai_features(
    df: pd.DataFrame,
    model,
    feature_columns,
    categorical_columns,
) -> pd.DataFrame:

    df = df.copy()

    price_input = prepare_price_model_input(
        df,
        feature_columns,
        categorical_columns,
    )

    predicted_log_unit_price = (
        model.predict(
            price_input[
                feature_columns
            ]
        )
    )

    predicted_unit_price = (
        np.expm1(
            predicted_log_unit_price
        )
    )

    predicted_unit_price = np.maximum(
        predicted_unit_price,
        0,
    )

    df["ai_price_per_m2"] = (
        predicted_unit_price
    )

    df["ai_estimated_price"] = (
        df["ai_price_per_m2"]
        * df["area_m2"]
    )

    df["price_gap_amount"] = (
        df["asking_price"]
        - df["ai_estimated_price"]
    )

    df["price_gap_ratio"] = (
        df["price_gap_amount"]
        / df["ai_estimated_price"]
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
        & df["area_m2"].notna()
        & df["days_to_contract"].notna()
        & df["ai_estimated_price"].notna()
    ].copy()

    df = df[
        df["days_to_contract"].between(
            1,
            1000,
        )
    ].copy()

    df = df[
        df["asking_price"] > 0
    ].copy()

    df = df[
        df["area_m2"] > 0
    ].copy()

    df = df[
        df["ai_estimated_price"] > 0
    ].copy()

    print(
        f"除外件数: "
        f"{before - len(df):,}件"
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

        "ai_price_per_m2",
        "ai_estimated_price",

        "price_gap_amount",
        "price_gap_ratio",

        # 評価・確認用。学習特徴量には使わない
        "contract_date",
        "contract_price",

        "days_to_contract",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    return df[
        available
    ].copy()


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
        "成約日数学習データ完成"
    )

    print(
        f"件数: {len(df):,}件"
    )

    print(
        f"保存先: {OUTPUT_FILE}"
    )


def main():

    print(
        "成約日数学習データ作成 Ver.2"
    )

    df = load_contract_data()

    validate_columns(df)

    df = clean_data(df)

    df = create_date_features(df)

    (
        price_model,
        price_features,
        price_categories,
    ) = load_price_model()

    df = create_price_ai_features(
        df,
        price_model,
        price_features,
        price_categories,
    )

    df = remove_invalid_rows(df)

    df = select_columns(df)

    save_data(df)


if __name__ == "__main__":
    main()