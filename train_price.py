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

FEATURE_COMPARISON_FILE = (
    OUTPUT_DIR
    / "price_feature_set_comparison.csv"
)

TARGET_COLUMN = "contract_price_per_m2"


FEATURE_SETS = {
    "code_compact": [
        "city_code",
        "district_code",
        "area_m2",
        "floor_plan",
        "building_age",
        "structure",
        "renovation",
        "use",
        "city_planning",
        "coverage_ratio",
        "floor_area_ratio",
        "transaction_year",
        "transaction_quarter",
    ],

    "name_compact": [
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
        "transaction_year",
        "transaction_quarter",
    ],

    "mixed_compact": [
        "city_code",
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
        "transaction_year",
        "transaction_quarter",
    ],

    "mixed_extended": [
        "city_code",
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
        "total_floor_area",
        "unit_area_ratio",
        "transaction_year",
        "transaction_quarter",
    ],
}


ALL_CATEGORICAL_COLUMNS = {
    "city",
    "city_code",
    "district_name",
    "district_code",
    "floor_plan",
    "structure",
    "renovation",
    "use",
    "city_planning",
}


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
        f"学習データ読み込み: "
        f"{len(df):,}件"
    )

    return df


def prepare_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    required_columns = [
        "contract_price",
        "contract_price_per_m2",
        "area_m2",
        "transaction_year",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise KeyError(
                f"{column} が見つかりません。"
            )

    df["contract_price"] = pd.to_numeric(
        df["contract_price"],
        errors="coerce",
    )

    df["contract_price_per_m2"] = (
        pd.to_numeric(
            df["contract_price_per_m2"],
            errors="coerce",
        )
    )

    df["area_m2"] = pd.to_numeric(
        df["area_m2"],
        errors="coerce",
    )

    df["transaction_year"] = pd.to_numeric(
        df["transaction_year"],
        errors="coerce",
    )

    for column in df.columns:

        if column in ALL_CATEGORICAL_COLUMNS:

            df[column] = (
                df[column]
                .fillna("不明")
                .astype(str)
            )

    df = df[
        df["contract_price"].notna()
        & df[TARGET_COLUMN].notna()
        & df["area_m2"].notna()
        & df["transaction_year"].notna()
    ].copy()

    df = df[
        (df["contract_price"] > 0)
        & (df[TARGET_COLUMN] > 0)
        & (df["area_m2"] > 0)
    ].copy()

    return df


def split_data(
    df: pd.DataFrame,
):

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
        len(train_df) == 0
        or len(validation_df) == 0
        or len(test_df) == 0
    ):
        raise RuntimeError(
            "2024年以前 / 2025年 / 2026年"
            "の分割ができません。"
        )

    print()
    print("時間順データ分割")
    print(
        f"学習: {len(train_df):,}件"
    )
    print(
        f"検証: {len(validation_df):,}件"
    )
    print(
        f"最終テスト: {len(test_df):,}件"
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


def get_available_features(
    df: pd.DataFrame,
    feature_set: list[str],
):

    features = [
        column
        for column in feature_set
        if column in df.columns
    ]

    categorical = [
        column
        for column in features
        if column in ALL_CATEGORICAL_COLUMNS
    ]

    return (
        features,
        categorical,
    )


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
):

    result = df.copy()

    for column in feature_columns:

        if column in categorical_columns:

            result[column] = (
                result[column]
                .fillna("不明")
                .astype(str)
            )

        else:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def calculate_mape(
    actual,
    predicted,
):

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


def create_time_weights(
    df: pd.DataFrame,
):

    year = df[
        "transaction_year"
    ].astype(float)

    minimum_year = year.min()

    weights = (
        1.0
        + (
            year
            - minimum_year
        )
        * 0.15
    )

    return weights.to_numpy()


def train_candidate(
    train_df,
    validation_df,
    feature_columns,
    categorical_columns,
):

    train_df = prepare_features(
        train_df,
        feature_columns,
        categorical_columns,
    )

    validation_df = prepare_features(
        validation_df,
        feature_columns,
        categorical_columns,
    )

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

    sample_weights = (
        create_time_weights(
            train_df
        )
    )

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=9,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        l2_leaf_reg=8,
        random_strength=0.4,
        bagging_temperature=0.5,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_columns,
        sample_weight=sample_weights,
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
    categorical_columns,
):

    df = prepare_features(
        df,
        feature_columns,
        categorical_columns,
    )

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


def evaluate(
    model,
    df,
    feature_columns,
    categorical_columns,
):

    (
        predicted_unit_price,
        predicted_price,
    ) = predict_prices(
        model,
        df,
        feature_columns,
        categorical_columns,
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

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
        "price_per_m2_mae": float(
            unit_mae
        ),
        "price_per_m2_mape": float(
            unit_mape
        ),
    }

    return (
        metrics,
        predicted_unit_price,
        predicted_price,
    )


def compare_feature_sets(
    train_df,
    validation_df,
):

    results = []

    models = {}

    print()
    print(
        "特徴量セット比較開始"
    )

    for (
        name,
        feature_set,
    ) in FEATURE_SETS.items():

        (
            feature_columns,
            categorical_columns,
        ) = get_available_features(
            train_df,
            feature_set,
        )

        print()
        print(
            f"[{name}]"
        )

        print(
            f"特徴量数: "
            f"{len(feature_columns)}"
        )

        model = train_candidate(
            train_df,
            validation_df,
            feature_columns,
            categorical_columns,
        )

        (
            metrics,
            _,
            _,
        ) = evaluate(
            model,
            validation_df,
            feature_columns,
            categorical_columns,
        )

        best_iteration = (
            model.get_best_iteration()
        )

        if best_iteration < 0:
            best_iteration = (
                model.tree_count_
                - 1
            )

        print(
            f"MAPE: "
            f"{metrics['mape']:.2f}%"
        )

        print(
            f"R²: "
            f"{metrics['r2']:.4f}"
        )

        print(
            f"Best iteration: "
            f"{best_iteration + 1}"
        )

        results.append(
            {
                "feature_set": name,
                "feature_count": (
                    len(feature_columns)
                ),
                "validation_mae": (
                    metrics["mae"]
                ),
                "validation_rmse": (
                    metrics["rmse"]
                ),
                "validation_mape": (
                    metrics["mape"]
                ),
                "validation_r2": (
                    metrics["r2"]
                ),
                "best_iteration": (
                    best_iteration + 1
                ),
            }
        )

        models[name] = {
            "model": model,
            "features": feature_columns,
            "categorical": (
                categorical_columns
            ),
            "best_iteration": (
                best_iteration + 1
            ),
        }

    comparison = pd.DataFrame(
        results
    )

    comparison = (
        comparison
        .sort_values(
            "validation_mape"
        )
        .reset_index(
            drop=True
        )
    )

    best_name = comparison.iloc[
        0
    ][
        "feature_set"
    ]

    print()
    print(
        "特徴量比較結果"
    )

    print(
        comparison[
            [
                "feature_set",
                "validation_mape",
                "validation_r2",
                "best_iteration",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"採用特徴量セット: "
        f"{best_name}"
    )

    return (
        best_name,
        models[best_name],
        comparison,
    )


def train_final_model(
    train_validation_df,
    feature_columns,
    categorical_columns,
    iterations,
):

    train_validation_df = (
        prepare_features(
            train_validation_df,
            feature_columns,
            categorical_columns,
        )
    )

    X = train_validation_df[
        feature_columns
    ]

    y = np.log1p(
        train_validation_df[
            TARGET_COLUMN
        ]
    )

    sample_weights = (
        create_time_weights(
            train_validation_df
        )
    )

    final_iterations = max(
        int(iterations),
        300,
    )

    model = CatBoostRegressor(
        iterations=final_iterations,
        learning_rate=0.03,
        depth=9,
        loss_function="RMSE",
        random_seed=42,
        l2_leaf_reg=8,
        random_strength=0.4,
        bagging_temperature=0.5,
        verbose=100,
        allow_writing_files=False,
    )

    print()
    print(
        "2021～2025年で"
        "最終モデルを再学習します。"
    )

    model.fit(
        X,
        y,
        cat_features=categorical_columns,
        sample_weight=sample_weights,
    )

    return model


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
        result[
            "absolute_error"
        ]
        / result[
            "contract_price"
        ]
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
            "feature": (
                feature_columns
            ),
            "importance": (
                model
                .get_feature_importance()
            ),
        }
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    importance.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "重要特徴量 TOP10"
    )

    print(
        importance
        .head(10)
        .to_string(
            index=False
        )
    )


def save_results(
    model,
    selected_name,
    feature_columns,
    categorical_columns,
    validation_comparison,
    test_metrics,
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

    validation_comparison.to_csv(
        FEATURE_COMPARISON_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    result = {
        "version": "4.0",
        "model": "CatBoostRegressor",
        "selected_feature_set": (
            selected_name
        ),
        "target": (
            "log_contract_price_per_m2"
        ),
        "final_training_period": (
            "2021-2025"
        ),
        "final_test_period": (
            "2026"
        ),
        "features": (
            feature_columns
        ),
        "categorical_features": (
            categorical_columns
        ),
        "test_metrics": (
            test_metrics
        ),
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
        "価格予測モデル Ver.4"
    )

    df = load_data()

    df = prepare_data(
        df
    )

    (
        train_df,
        validation_df,
        test_df,
    ) = split_data(
        df
    )

    (
        selected_name,
        selected_data,
        comparison,
    ) = compare_feature_sets(
        train_df,
        validation_df,
    )

    feature_columns = (
        selected_data[
            "features"
        ]
    )

    categorical_columns = (
        selected_data[
            "categorical"
        ]
    )

    best_iteration = (
        selected_data[
            "best_iteration"
        ]
    )

    train_validation_df = (
        pd.concat(
            [
                train_df,
                validation_df,
            ],
            ignore_index=True,
        )
    )

    final_model = train_final_model(
        train_validation_df,
        feature_columns,
        categorical_columns,
        best_iteration,
    )

    (
        test_metrics,
        predicted_unit_price,
        predicted_price,
    ) = evaluate(
        final_model,
        test_df,
        feature_columns,
        categorical_columns,
    )

    print()
    print(
        "最終テストデータ 精度"
    )

    print(
        f"MAE : "
        f"{test_metrics['mae']:,.0f}円"
    )

    print(
        f"RMSE: "
        f"{test_metrics['rmse']:,.0f}円"
    )

    print(
        f"MAPE: "
        f"{test_metrics['mape']:.2f}%"
    )

    print(
        f"R²  : "
        f"{test_metrics['r2']:.4f}"
    )

    print()
    print(
        f"㎡単価MAE : "
        f"{test_metrics['price_per_m2_mae']:,.0f}"
        f"円/㎡"
    )

    print(
        f"㎡単価MAPE: "
        f"{test_metrics['price_per_m2_mape']:.2f}%"
    )

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
        final_model,
        feature_columns,
    )

    save_results(
        final_model,
        selected_name,
        feature_columns,
        categorical_columns,
        comparison,
        test_metrics,
    )

    print()
    print(
        "Ver.4 完了"
    )

    print(
        f"採用セット: "
        f"{selected_name}"
    )

    print(
        f"モデル: {MODEL_FILE}"
    )

    print(
        f"精度: {METRICS_FILE}"
    )

    print(
        f"特徴量比較: "
        f"{FEATURE_COMPARISON_FILE}"
    )

    print(
        f"特徴量重要度: "
        f"{FEATURE_IMPORTANCE_FILE}"
    )

    print(
        f"予測結果: "
        f"{PREDICTIONS_FILE}"
    )


if __name__ == "__main__":
    main()