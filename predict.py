from pathlib import Path
import json

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "price_model.cbm"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "output"
    / "price_model_metrics.json"
)

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "input"
)

INPUT_FILE = (
    INPUT_DIR
    / "prediction_input.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "price_predictions.csv"
)


def load_model_info():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"モデルが見つかりません: {MODEL_FILE}"
        )

    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"モデル情報が見つかりません: {METRICS_FILE}"
        )

    with open(
        METRICS_FILE,
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

    selected_feature_set = model_info.get(
        "selected_feature_set",
        "unknown",
    )

    if not feature_columns:
        raise RuntimeError(
            "price_model_metrics.json に"
            "features が保存されていません。"
        )

    model = CatBoostRegressor()

    model.load_model(
        str(MODEL_FILE)
    )

    print(
        f"モデル読み込み完了"
    )

    print(
        f"特徴量セット: "
        f"{selected_feature_set}"
    )

    return (
        model,
        feature_columns,
        categorical_columns,
    )


def create_input_template(
    feature_columns,
):
    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    template = {
        "property_id": [
            "A001",
            "A002",
            "A003",
        ],
        "city": [
            "足立区",
            "世田谷区",
            "港区",
        ],
        "district_name": [
            "千住",
            "三軒茶屋",
            "六本木",
        ],
        "area_m2": [
            65.2,
            55.0,
            70.0,
        ],
        "floor_plan": [
            "3LDK",
            "2LDK",
            "2LDK",
        ],
        "building_age": [
            12,
            8,
            5,
        ],
        "structure": [
            "RC",
            "RC",
            "RC",
        ],
        "renovation": [
            "未改装",
            "改装済",
            "未改装",
        ],
        "use": [
            "住宅",
            "住宅",
            "住宅",
        ],
        "city_planning": [
            "商業地域",
            "近隣商業地域",
            "商業地域",
        ],
        "coverage_ratio": [
            80,
            80,
            80,
        ],
        "floor_area_ratio": [
            400,
            300,
            600,
        ],
        "transaction_year": [
            2026,
            2026,
            2026,
        ],
        "transaction_quarter": [
            3,
            3,
            3,
        ],
        "asking_price": [
            45_000_000,
            60_000_000,
            120_000_000,
        ],
    }

    df = pd.DataFrame(
        template
    )

    required_input_features = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    for column in required_input_features:
        df[column] = np.nan

    df.to_csv(
        INPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "入力テンプレートを作成しました。"
    )

    print(
        f"保存先: {INPUT_FILE}"
    )

    print()
    print(
        "prediction_input.csv を編集して"
        "もう一度実行してください。"
    )


def load_input_data(
    feature_columns,
):
    if not INPUT_FILE.exists():
        return None

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    if df.empty:
        raise RuntimeError(
            "prediction_input.csv が空です。"
        )

    print()
    print(
        f"予測対象: {len(df):,}件"
    )

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        print()
        print(
            "不足しているカラム:"
        )

        for column in missing_columns:
            print(
                f"- {column}"
            )

        raise KeyError(
            "予測に必要なカラムが不足しています。"
        )

    return df


def prepare_input_data(
    df,
    feature_columns,
    categorical_columns,
):
    df = df.copy()

    if "area_m2" not in df.columns:
        raise KeyError(
            "area_m2 が必要です。"
        )

    df["area_m2"] = pd.to_numeric(
        df["area_m2"],
        errors="coerce",
    )

    if df["area_m2"].isna().any():
        raise ValueError(
            "area_m2 に数値ではない値があります。"
        )

    if (
        df["area_m2"] <= 0
    ).any():
        raise ValueError(
            "area_m2 は0より大きい値にしてください。"
        )

    for column in feature_columns:

        if column in categorical_columns:

            df[column] = (
                df[column]
                .fillna("不明")
                .astype(str)
            )

        else:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


def predict_prices(
    model,
    df,
    feature_columns,
):
    X = df[
        feature_columns
    ]

    predicted_log_unit_price = (
        model.predict(X)
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

    predicted_contract_price = (
        predicted_unit_price
        * df["area_m2"].to_numpy()
    )

    return (
        predicted_unit_price,
        predicted_contract_price,
    )


def create_prediction_result(
    df,
    predicted_unit_price,
    predicted_contract_price,
):
    result = df.copy()

    result[
        "predicted_price_per_m2"
    ] = np.round(
        predicted_unit_price
    ).astype(int)

    result[
        "predicted_contract_price"
    ] = np.round(
        predicted_contract_price
    ).astype(int)

    if "asking_price" in result.columns:

        result["asking_price"] = (
            pd.to_numeric(
                result["asking_price"],
                errors="coerce",
            )
        )

        result[
            "price_gap_amount"
        ] = (
            result["asking_price"]
            - result[
                "predicted_contract_price"
            ]
        )

        result[
            "price_gap_ratio"
        ] = (
            result[
                "price_gap_amount"
            ]
            / result[
                "predicted_contract_price"
            ]
        )

        result[
            "price_gap_ratio_percent"
        ] = (
            result[
                "price_gap_ratio"
            ]
            * 100
        ).round(2)

    return result


def save_predictions(
    result,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "予測完了"
    )

    print(
        f"保存先: {OUTPUT_FILE}"
    )


def show_predictions(
    result,
):
    print()
    print(
        "予測結果"
    )

    for index, row in result.iterrows():

        print()
        print(
            f"物件 {index + 1}"
        )

        if "property_id" in result.columns:
            print(
                f"ID: "
                f"{row['property_id']}"
            )

        if "city" in result.columns:
            print(
                f"地域: "
                f"{row['city']}"
            )

        if "district_name" in result.columns:
            print(
                f"地区: "
                f"{row['district_name']}"
            )

        print(
            f"面積: "
            f"{row['area_m2']}㎡"
        )

        print(
            "AI予測㎡単価: "
            f"{row['predicted_price_per_m2']:,.0f}"
            "円/㎡"
        )

        print(
            "AI予測成約価格: "
            f"{row['predicted_contract_price']:,.0f}"
            "円"
        )

        if (
            "asking_price"
            in result.columns
            and
            pd.notna(
                row["asking_price"]
            )
        ):

            print(
                "売出価格: "
                f"{row['asking_price']:,.0f}"
                "円"
            )

            print(
                "売出価格との差: "
                f"{row['price_gap_amount']:,.0f}"
                "円"
            )

            print(
                "価格乖離率: "
                f"{row['price_gap_ratio_percent']:.2f}"
                "%"
            )


def main():
    print(
        "東京都中古マンション"
        "成約価格予測"
    )

    (
        model,
        feature_columns,
        categorical_columns,
    ) = load_model_info()

    if not INPUT_FILE.exists():

        create_input_template(
            feature_columns
        )

        return

    df = load_input_data(
        feature_columns
    )

    df = prepare_input_data(
        df,
        feature_columns,
        categorical_columns,
    )

    (
        predicted_unit_price,
        predicted_contract_price,
    ) = predict_prices(
        model,
        df,
        feature_columns,
    )

    result = create_prediction_result(
        df,
        predicted_unit_price,
        predicted_contract_price,
    )

    save_predictions(
        result
    )

    show_predictions(
        result
    )


if __name__ == "__main__":
    main()