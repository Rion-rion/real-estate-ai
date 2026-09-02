from pathlib import Path
import json

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent

TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "price_training.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

MODEL_FILE = MODELS_DIR / "price_model.cbm"

METRICS_FILE = (
    OUTPUT_DIR
    / "price_model_metrics.json"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "price_test_predictions.csv"
)

FEATURE_IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "price_feature_importance.csv"
)

TARGET_COLUMN = "contract_price_per_m2"

FEATURE_COLUMNS = [
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
    "transaction_year",
    "transaction_quarter",
    "market_year_index",
]

CATEGORICAL_COLUMNS = [
    "city",
    "city_code",
    "district_name",
    "district_code",
    "region",
    "floor_plan",
    "structure",
    "renovation",
    "use",
    "city_planning",
]


def load_data() -> pd.DataFrame:
    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"学習データが見つかりません: {TRAINING_FILE}"
        )

    df = pd.read_csv(
        TRAINING_FILE,
        low_memory=False,
    )

    print(
        f"学習データ: {len(df):,}件"
    )

    return df


def prepare_data(df: pd.DataFrame):
    df = df.copy()

    required_columns = [
        "contract_price",
        "contract_price_per_m2",
        "area_m2",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise KeyError(
                f"{column} がありません。"
            )

    feature_columns = [
        column
        for column in FEATURE_COLUMNS
        if column in df.columns
    ]

    categorical_columns = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in feature_columns
    ]

    for column in categorical_columns:
        df[column] = (
            df[column]
            .fillna("不明")
            .astype(str)
        )

    numeric_columns = [
        column
        for column in feature_columns
        if column not in categorical_columns
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["contract_price"] = pd.to_numeric(
        df["contract_price"],
        errors="coerce",
    )

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    df["area_m2"] = pd.to_numeric(
        df["area_m2"],
        errors="coerce",
    )

    df = df[
        df["contract_price"].notna()
        & df[TARGET_COLUMN].notna()
        & df["area_m2"].notna()
    ].copy()

    df = df[
        (df["contract_price"] > 0)
        & (df[TARGET_COLUMN] > 0)
        & (df["area_m2"] > 0)
    ].copy()

    print()
    print("使用特徴量:")

    for column in feature_columns:
        print(f"- {column}")

    return (
        df,
        feature_columns,
        categorical_columns,
    )


def temporal_split(df: pd.DataFrame):
    if "transaction_year" in df.columns:

        train_df = df[
            df["transaction_year"] <= 2024
        ].copy()

        validation_df = df[
            df["transaction_year"] == 2025
        ].copy()

        test_df = df[
            df["transaction_year"] >= 2026
        ].copy()

        if (
            len(train_df) > 0
            and len(validation_df) > 0
            and len(test_df) > 0
        ):
            print()
            print("時間順分割")

            print(
                f"学習: {len(train_df):,}件"
            )

            print(
                f"検証: {len(validation_df):,}件"
            )

            print(
                f"テスト: {len(test_df):,}件"
            )

            return (
                train_df,
                validation_df,
                test_df,
            )

    raise RuntimeError(
        "2024年以前 / 2025年 / 2026年の"
        "時間順分割ができません。"
    )


def calculate_mape(
    actual,
    predicted,
) -> float:

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    valid = actual != 0

    return float(
        np.mean(
            np.abs(
                (
                    actual[valid]
                    - predicted[valid]
                )
                / actual[valid]
            )
        )
        * 100
    )


def train_model(
    train_df,
    validation_df,
    feature_columns,
    categorical_columns,
):

    X_train = train_df[
        feature_columns
    ]

    X_validation = validation_df[
        feature_columns
    ]

    y_train = np.log1p(
        train_df[TARGET_COLUMN]
    )

    y_validation = np.log1p(
        validation_df[TARGET_COLUMN]
    )

    model = CatBoostRegressor(
        iterations=3500,
        learning_rate=0.025,
        depth=10,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        l2_leaf_reg=8,
        random_strength=0.3,
        bagging_temperature=0.5,
        verbose=100,
        allow_writing_files=False,
    )

    print()
    print(
        "CatBoost Ver.3 学習開始"
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_columns,
        eval_set=(
            X_validation,
            y_validation,
        ),
        use_best_model=True,
        early_stopping_rounds=200,
    )

    return model


def predict_prices(
    model,
    df,
    feature_columns,
):

    predicted_log_unit_price = (
        model.predict(
            df[feature_columns]
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

    predicted_price = (
        predicted_unit_price
        * df["area_m2"].to_numpy()
    )

    return (
        predicted_unit_price,
        predicted_price,
    )


def evaluate_model(
    model,
    df,
    feature_columns,
    name,
):

    (
        predicted_unit_price,
        predicted_price,
    ) = predict_prices(
        model,
        df,
        feature_columns,
    )

    actual_price = (
        df["contract_price"]
        .to_numpy()
    )

    actual_unit_price = (
        df[TARGET_COLUMN]
        .to_numpy()
    )

    mae = mean_absolute_error(
        actual_price,
        predicted_price,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_price,
            predicted_price,
        )
    )

    mape = calculate_mape(
        actual_price,
        predicted_price,
    )

    r2 = r2_score(
        actual_price,
        predicted_price,
    )

    unit_mae = mean_absolute_error(
        actual_unit_price,
        predicted_unit_price,
    )

    unit_mape = calculate_mape(
        actual_unit_price,
        predicted_unit_price,
    )

    print()
    print(f"{name} 精度")

    print(
        f"MAE : {mae:,.0f}円"
    )

    print(
        f"RMSE: {rmse:,.0f}円"
    )

    print(
        f"MAPE: {mape:.2f}%"
    )

    print(
        f"R²  : {r2:.4f}"
    )

    print()
    print(
        f"㎡単価MAE : "
        f"{unit_mae:,.0f}円/㎡"
    )

    print(
        f"㎡単価MAPE: "
        f"{unit_mape:.2f}%"
    )

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
        "price_per_m2_mae": (
            float(unit_mae)
        ),
        "price_per_m2_mape": (
            float(unit_mape)
        ),
    }

    return (
        metrics,
        predicted_unit_price,
        predicted_price,
    )


def save_predictions(
    test_df,
    predicted_unit_price,
    predicted_price,
):

    result = test_df.copy()

    result[
        "predicted_price_per_m2"
    ] = predicted_unit_price

    result[
        "predicted_contract_price"
    ] = predicted_price

    result[
        "absolute_error"
    ] = (
        result[
            "predicted_contract_price"
        ]
        - result[
            "contract_price"
        ]
    ).abs()

    result[
        "percentage_error"
    ] = (
        result["absolute_error"]
        / result["contract_price"]
        * 100
    )

    result.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def save_feature_importance(
    model,
    feature_columns,
):

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": (
                model.get_feature_importance()
            ),
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("重要特徴量 TOP10")

    print(
        importance
        .head(10)
        .to_string(index=False)
    )


def save_results(
    model,
    metrics,
    feature_columns,
    categorical_columns,
):

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(
        MODEL_FILE
    )

    result = {
        "version": "3.0",
        "model": "CatBoostRegressor",
        "target": (
            "log_contract_price_per_m2"
        ),
        "features": feature_columns,
        "categorical_features": (
            categorical_columns
        ),
        "metrics": metrics,
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main():

    print(
        "東京都中古マンション"
        "価格予測モデル Ver.3"
    )

    df = load_data()

    (
        df,
        feature_columns,
        categorical_columns,
    ) = prepare_data(df)

    (
        train_df,
        validation_df,
        test_df,
    ) = temporal_split(df)

    model = train_model(
        train_df,
        validation_df,
        feature_columns,
        categorical_columns,
    )

    validation_metrics, _, _ = (
        evaluate_model(
            model,
            validation_df,
            feature_columns,
            "検証データ",
        )
    )

    (
        test_metrics,
        predicted_unit_price,
        predicted_price,
    ) = evaluate_model(
        model,
        test_df,
        feature_columns,
        "テストデータ",
    )

    metrics = {
        "validation": (
            validation_metrics
        ),
        "test": (
            test_metrics
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_predictions(
        test_df,
        predicted_unit_price,
        predicted_price,
    )

    save_feature_importance(
        model,
        feature_columns,
    )

    save_results(
        model,
        metrics,
        feature_columns,
        categorical_columns,
    )

    print()
    print(
        "Ver.3 完了"
    )

    print(
        f"モデル: {MODEL_FILE}"
    )

    print(
        f"精度: {METRICS_FILE}"
    )

    print(
        f"予測結果: {PREDICTIONS_FILE}"
    )

    print(
        f"特徴量重要度: "
        f"{FEATURE_IMPORTANCE_FILE}"
    )


if __name__ == "__main__":
    main()