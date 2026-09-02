from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "tokyo_contract_prices.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "price_training.csv"
)


def load_data() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"CSVが見つかりません: {RAW_FILE}"
        )

    df = pd.read_csv(
        RAW_FILE,
        low_memory=False,
    )

    print(f"読み込み件数: {len(df):,}件")

    return df


def rename_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rename_map = {
        "TradePrice": "contract_price",
        "Area": "area_m2",
        "FloorPlan": "floor_plan",
        "BuildingYear": "build_year",
        "Structure": "structure",
        "Period": "period",
        "Municipality": "city",
        "MunicipalityCode": "city_code",
        "DistrictName": "district_name",
        "DistrictCode": "district_code",
        "Renovation": "renovation",
        "Use": "use",
        "CityPlanning": "city_planning",
        "CoverageRatio": "coverage_ratio",
        "FloorAreaRatio": "floor_area_ratio",
        "TotalFloorArea": "total_floor_area",
        "Region": "region",
    }

    available_columns = {
        old: new
        for old, new in rename_map.items()
        if old in df.columns
    }

    return df.rename(
        columns=available_columns
    )


def extract_number(value):
    if pd.isna(value):
        return np.nan

    text = str(value)

    text = (
        text
        .replace(",", "")
        .replace("円", "")
        .replace("㎡", "")
        .replace("%", "")
        .strip()
    )

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if not match:
        return np.nan

    return float(
        match.group()
    )


def parse_build_year(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    western_year = re.search(
        r"(18\d{2}|19\d{2}|20\d{2})",
        text,
    )

    if western_year:
        return int(
            western_year.group(1)
        )

    era_years = {
        "明治": 1867,
        "大正": 1911,
        "昭和": 1925,
        "平成": 1988,
        "令和": 2018,
    }

    for era, base_year in era_years.items():

        if era not in text:
            continue

        if "元年" in text:
            return base_year + 1

        match = re.search(
            r"(\d+)",
            text,
        )

        if match:
            return (
                base_year
                + int(match.group(1))
            )

    return np.nan


def extract_transaction_year(value):
    if pd.isna(value):
        return np.nan

    match = re.search(
        r"(20\d{2})",
        str(value),
    )

    if not match:
        return np.nan

    return int(
        match.group(1)
    )


def extract_transaction_quarter(value):
    if pd.isna(value):
        return np.nan

    match = re.search(
        r"第?([1-4])四半期",
        str(value),
    )

    if not match:
        return np.nan

    return int(
        match.group(1)
    )


def clean_categories(
    df: pd.DataFrame,
) -> pd.DataFrame:

    categorical_columns = [
        "city",
        "city_code",
        "district_name",
        "district_code",
        "floor_plan",
        "structure",
        "renovation",
        "use",
        "city_planning",
        "region",
    ]

    for column in categorical_columns:

        if column not in df.columns:
            continue

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace(
                {
                    "": pd.NA,
                    "nan": pd.NA,
                    "None": pd.NA,
                }
            )
        )

    return df


def create_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    numeric_columns = [
        "contract_price",
        "area_m2",
        "coverage_ratio",
        "floor_area_ratio",
        "total_floor_area",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .apply(extract_number)
            )

    if "build_year" in df.columns:

        df["build_year"] = (
            df["build_year"]
            .apply(parse_build_year)
        )

    if "period" in df.columns:

        df["transaction_year"] = (
            df["period"]
            .apply(
                extract_transaction_year
            )
        )

        df["transaction_quarter"] = (
            df["period"]
            .apply(
                extract_transaction_quarter
            )
        )

    if (
        "build_year" in df.columns
        and
        "transaction_year" in df.columns
    ):

        df["building_age"] = (
            df["transaction_year"]
            - df["build_year"]
        )

        df.loc[
            (
                df["building_age"] < 0
            )
            |
            (
                df["building_age"] > 150
            ),
            "building_age",
        ] = np.nan

    if (
        "contract_price" in df.columns
        and
        "area_m2" in df.columns
    ):

        df["contract_price_per_m2"] = (
            df["contract_price"]
            / df["area_m2"]
        )

    if "transaction_year" in df.columns:

        minimum_year = (
            df["transaction_year"]
            .min()
        )

        df["market_year_index"] = (
            df["transaction_year"]
            - minimum_year
        )

    if (
        "total_floor_area" in df.columns
        and
        "area_m2" in df.columns
    ):

        df["unit_area_ratio"] = (
            df["area_m2"]
            / df["total_floor_area"]
        )

        df.loc[
            (
                df["unit_area_ratio"] <= 0
            )
            |
            (
                df["unit_area_ratio"] > 1
            ),
            "unit_area_ratio",
        ] = np.nan

    return df


def remove_invalid_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:

    before = len(df)

    df = df[
        df["contract_price"].notna()
    ].copy()

    df = df[
        df["area_m2"].notna()
    ].copy()

    df = df[
        df["contract_price"].between(
            1_000_000,
            1_000_000_000,
        )
    ].copy()

    df = df[
        df["area_m2"].between(
            10,
            300,
        )
    ].copy()

    if "coverage_ratio" in df.columns:

        df.loc[
            ~df["coverage_ratio"].between(
                0,
                100,
            ),
            "coverage_ratio",
        ] = np.nan

    if "floor_area_ratio" in df.columns:

        df.loc[
            ~df["floor_area_ratio"].between(
                0,
                2000,
            ),
            "floor_area_ratio",
        ] = np.nan

    if "total_floor_area" in df.columns:

        df.loc[
            df["total_floor_area"] <= 0,
            "total_floor_area",
        ] = np.nan

    removed = (
        before
        - len(df)
    )

    print(
        f"明らかな異常値削除: "
        f"{removed:,}件"
    )

    return df


def remove_price_outliers(
    df: pd.DataFrame,
) -> pd.DataFrame:

    before = len(df)

    lower = (
        df["contract_price_per_m2"]
        .quantile(0.005)
    )

    upper = (
        df["contract_price_per_m2"]
        .quantile(0.995)
    )

    df = df[
        df["contract_price_per_m2"]
        .between(
            lower,
            upper,
        )
    ].copy()

    removed = (
        before
        - len(df)
    )

    print(
        f"㎡単価外れ値削除: "
        f"{removed:,}件"
    )

    print(
        f"㎡単価範囲: "
        f"{lower:,.0f} ～ "
        f"{upper:,.0f}円/㎡"
    )

    return df


def remove_constant_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    protected = {
        "contract_price",
        "contract_price_per_m2",
        "area_m2",
    }

    removable = []

    for column in df.columns:

        if column in protected:
            continue

        unique_count = (
            df[column]
            .nunique(
                dropna=True
            )
        )

        if unique_count <= 1:
            removable.append(column)

    if removable:

        print()
        print(
            "情報量のない列を削除:"
        )

        for column in removable:
            print(f"- {column}")

        df = df.drop(
            columns=removable
        )

    return df


def select_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        "contract_price",
        "contract_price_per_m2",

        "city",
        "city_code",

        "district_name",
        "district_code",

        "region",

        "area_m2",
        "floor_plan",

        "build_year",
        "building_age",

        "structure",
        "renovation",

        "use",
        "city_planning",

        "coverage_ratio",
        "floor_area_ratio",

        "total_floor_area",
        "unit_area_ratio",

        "period",
        "transaction_year",
        "transaction_quarter",
        "market_year_index",
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
    print("前処理 Ver.3 完了")

    print(
        f"最終件数: "
        f"{len(df):,}件"
    )

    print(
        f"カラム数: "
        f"{len(df.columns)}"
    )

    print()
    print("学習候補カラム:")

    for column in df.columns:
        print(f"- {column}")

    print()
    print("欠損率:")

    missing_ratio = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    for column, ratio in (
        missing_ratio.items()
    ):
        print(
            f"{column}: "
            f"{ratio:.1f}%"
        )

    print()
    print("成約価格:")

    print(
        df["contract_price"]
        .describe()
        .round(0)
    )

    print()
    print("㎡単価:")

    print(
        df["contract_price_per_m2"]
        .describe()
        .round(0)
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
        "price_training.csv を更新しました。"
    )

    print(
        f"保存先: {OUTPUT_FILE}"
    )


def main() -> None:

    print(
        "東京都中古マンション"
        "前処理 Ver.3"
    )

    print()

    df = load_data()

    df = rename_columns(
        df
    )

    df = clean_categories(
        df
    )

    df = create_features(
        df
    )

    df = remove_invalid_rows(
        df
    )

    df = remove_price_outliers(
        df
    )

    df = select_columns(
        df
    )

    df = remove_constant_columns(
        df
    )

    show_summary(
        df
    )

    save_data(
        df
    )


if __name__ == "__main__":
    main()