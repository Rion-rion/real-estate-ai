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

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
)

MODEL_FILE = (
    MODELS_DIR
    / "price_model.cbm"
)

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

FINAL_TEST_COMPARISON_FILE = (
    OUTPUT_DIR
    / "price_final_test_comparison.csv"
)

TARGET_COLUMN = (
    "contract_price_per_m2"
)

VER4_REFERENCE_MAPE = 17.71
VER4_REFERENCE_R2 = 0.8528


BASELINE_FEATURES = [
    "city",
    "district_name",
    "area_m2",
    "floor_plan",
    "building_age",
    "structure",
    "city_planning",
    "transaction_year",
    "transaction_quarter",
]


FEATURE_SETS = {
    "baseline_current": [
        *BASELINE_FEATURES,
    ],

    "station_name": [
        *BASELINE_FEATURES,
        "station_name",
    ],

    "station_geo": [
        *BASELINE_FEATURES,
        "station_name",
        "station_latitude",
        "station_longitude",
    ],

    "station_geo_line": [
        *BASELINE_FEATURES,
        "station_name",
        "station_line",
        "station_latitude",
        "station_longitude",
    ],
}


ALL_CATEGORICAL_COLUMNS = {
    "city",
    "city_code",
    "district_name",
    "district_code",
    "floor_plan",
    "structure",
    "city_planning",
    "station_name",
    "station_code",
    "station_group_code",
    "station_company",
    "station_line",
}


def load_data() -> pd.DataFrame:
    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"学習データが見つかりません: "
            f"{TRAINING_FILE}"
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
        TARGET_COLUMN,
        "area_m2",
        "transaction_year",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise KeyError(
                f"{column} が見つかりません。"
            )

    numeric_columns = [
        "contract_price",
        TARGET_COLUMN,
        "area_m2",
        "building_age",
        "station_latitude",
        "station_longitude",
        "transaction_year",
        "transaction_quarter",
    ]

    for column in numeric_columns:
        if column not in df.columns:
            continue

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    for column in df.columns:
        if column not in ALL_CATEGORICAL_COLUMNS:
            continue

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

    print(
        f"学習利用可能件数: "
        f"{len(df):,}件"
    )

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
            "2024年以前 / 2025年 / "
            "2026年の時系列分割ができません。"
        )

    print()
    print(
        "時間順データ分割"
    )

    print(
        f"学習 2021-2024: "
        f"{len(train_df):,}件"
    )

    print(
        f"検証 2025: "
        f"{len(validation_df):,}件"
    )

    print(
        f"最終テスト 2026: "
        f"{len(test_df):,}件"
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


def validate_feature_sets(
    df: pd.DataFrame,
) -> None:

    print()
    print(
        "Ver.5 特徴量セット"
    )

    for (
        name,
        feature_set,
    ) in FEATURE_SETS.items():

        available = [
            column
            for column in feature_set
            if column in df.columns
        ]

        missing = [
            column
            for column in feature_set
            if column not in df.columns
        ]

        print()
        print(
            f"[{name}] "
            f"{len(available)}特徴量"
        )

        if missing:
            print(
                "使用不可: "
                + ", ".join(missing)
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

    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
        & (actual != 0)
    )

    if not valid.any():
        return np.nan

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

    year = (
        df[
            "transaction_year"
        ]
        .astype(float)
    )

    minimum_year = (
        year.min()
    )

    weights = (
        1.0
        + (
            year
            - minimum_year
        )
        * 0.15
    )

    return (
        weights.to_numpy()
    )


def create_model(
    iterations: int,
    verbose=False,
):

    return CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.03,
        depth=9,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        l2_leaf_reg=8,
        random_strength=0.4,
        bagging_temperature=0.5,
        verbose=verbose,
        allow_writing_files=False,
        thread_count=-1,
    )


def train_candidate(
    train_df,
    validation_df,
    feature_columns,
    categorical_columns,
):

    train_work = prepare_features(
        train_df,
        feature_columns,
        categorical_columns,
    )

    validation_work = prepare_features(
        validation_df,
        feature_columns,
        categorical_columns,
    )

    X_train = train_work[
        feature_columns
    ]

    X_validation = validation_work[
        feature_columns
    ]

    y_train = np.log1p(
        train_work[
            TARGET_COLUMN
        ]
    )

    y_validation = np.log1p(
        validation_work[
            TARGET_COLUMN
        ]
    )

    sample_weights = (
        create_time_weights(
            train_work
        )
    )

    model = create_model(
        iterations=3000,
        verbose=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=(
            categorical_columns
        ),
        sample_weight=(
            sample_weights
        ),
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

    work = prepare_features(
        df,
        feature_columns,
        categorical_columns,
    )

    predicted_log_unit_price = (
        model.predict(
            work[
                feature_columns
            ]
        )
    )

    predicted_unit_price = (
        np.expm1(
            predicted_log_unit_price
        )
    )

    predicted_unit_price = (
        np.maximum(
            predicted_unit_price,
            0,
        )
    )

    predicted_price = (
        predicted_unit_price
        * work[
            "area_m2"
        ].to_numpy()
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
        df[
            "contract_price"
        ]
        .to_numpy()
    )

    actual_unit_price = (
        df[
            TARGET_COLUMN
        ]
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

    unit_mae = (
        mean_absolute_error(
            actual_unit_price,
            predicted_unit_price,
        )
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
        "Ver.5 特徴量セット比較開始"
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

        if not feature_columns:
            continue

        print()
        print(
            f"[{name}]"
        )

        print(
            f"特徴量数: "
            f"{len(feature_columns)}"
        )

        print(
            "特徴量: "
            + ", ".join(
                feature_columns
            )
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

        best_iteration += 1

        print(
            f"Validation MAPE: "
            f"{metrics['mape']:.2f}%"
        )

        print(
            f"Validation R²: "
            f"{metrics['r2']:.4f}"
        )

        print(
            f"Best iteration: "
            f"{best_iteration}"
        )

        results.append(
            {
                "feature_set": name,
                "feature_count": (
                    len(feature_columns)
                ),
                "uses_station": (
                    "station_name"
                    in feature_columns
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
                    best_iteration
                ),
            }
        )

        models[name] = {
            "model": model,
            "features": (
                feature_columns
            ),
            "categorical": (
                categorical_columns
            ),
            "best_iteration": (
                best_iteration
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

    best_name = (
        comparison.iloc[0][
            "feature_set"
        ]
    )

    print()
    print(
        "2025年 検証結果"
    )

    print(
        comparison[
            [
                "feature_set",
                "feature_count",
                "validation_mape",
                "validation_r2",
                "best_iteration",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "2025年の検証結果のみで"
        "モデルを選択します。"
    )

    print(
        f"採用候補: "
        f"{best_name}"
    )

    return (
        best_name,
        models,
        comparison,
    )


def train_final_model(
    train_validation_df,
    feature_columns,
    categorical_columns,
    iterations,
    model_name,
):

    work = prepare_features(
        train_validation_df,
        feature_columns,
        categorical_columns,
    )

    X = work[
        feature_columns
    ]

    y = np.log1p(
        work[
            TARGET_COLUMN
        ]
    )

    sample_weights = (
        create_time_weights(
            work
        )
    )

    final_iterations = max(
        int(iterations),
        300,
    )

    model = create_model(
        iterations=(
            final_iterations
        ),
        verbose=100,
    )

    print()
    print(
        f"[{model_name}]"
    )

    print(
        "2021～2025年で"
        "最終モデルを再学習します。"
    )

    print(
        f"Iterations: "
        f"{final_iterations}"
    )

    model.fit(
        X,
        y,
        cat_features=(
            categorical_columns
        ),
        sample_weight=(
            sample_weights
        ),
    )

    return model


def run_final_test(
    train_validation_df,
    test_df,
    name,
    candidate_data,
):

    model = train_final_model(
        train_validation_df,
        candidate_data[
            "features"
        ],
        candidate_data[
            "categorical"
        ],
        candidate_data[
            "best_iteration"
        ],
        name,
    )

    (
        metrics,
        predicted_unit_price,
        predicted_price,
    ) = evaluate(
        model,
        test_df,
        candidate_data[
            "features"
        ],
        candidate_data[
            "categorical"
        ],
    )

    return (
        model,
        metrics,
        predicted_unit_price,
        predicted_price,
    )


def compare_final_test(
    train_validation_df,
    test_df,
    selected_name,
    models,
):

    comparison_names = [
        "baseline_current",
    ]

    if (
        selected_name
        not in comparison_names
    ):
        comparison_names.append(
            selected_name
        )

    results = {}
    rows = []

    print()
    print(
        "2026年 最終テスト"
    )

    print(
        "※ここではモデル選択をしません。"
    )

    print(
        "2025年検証で選んだモデルを"
        "未使用の2026年データで評価します。"
    )

    for name in comparison_names:

        candidate_data = (
            models[name]
        )

        (
            model,
            metrics,
            predicted_unit_price,
            predicted_price,
        ) = run_final_test(
            train_validation_df,
            test_df,
            name,
            candidate_data,
        )

        rows.append(
            {
                "feature_set": name,
                "is_selected": (
                    name
                    == selected_name
                ),
                "test_mae": (
                    metrics["mae"]
                ),
                "test_rmse": (
                    metrics["rmse"]
                ),
                "test_mape": (
                    metrics["mape"]
                ),
                "test_r2": (
                    metrics["r2"]
                ),
                "price_per_m2_mae": (
                    metrics[
                        "price_per_m2_mae"
                    ]
                ),
                "price_per_m2_mape": (
                    metrics[
                        "price_per_m2_mape"
                    ]
                ),
            }
        )

        results[name] = {
            "model": model,
            "metrics": metrics,
            "predicted_unit_price": (
                predicted_unit_price
            ),
            "predicted_price": (
                predicted_price
            ),
        }

    comparison = pd.DataFrame(
        rows
    )

    print()
    print(
        comparison[
            [
                "feature_set",
                "test_mape",
                "test_r2",
                "test_mae",
            ]
        ]
        .to_string(
            index=False
        )
    )

    return (
        results,
        comparison,
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
    final_model,
    selected_name,
    selected_data,
    validation_comparison,
    final_test_comparison,
    test_metrics,
    baseline_metrics,
):

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_model.save_model(
        MODEL_FILE
    )

    validation_comparison.to_csv(
        FEATURE_COMPARISON_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    final_test_comparison.to_csv(
        FINAL_TEST_COMPARISON_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    station_features = [
        feature
        for feature in selected_data[
            "features"
        ]
        if feature.startswith(
            "station_"
        )
    ]

    result = {
        "version": "5.0",
        "model": (
            "CatBoostRegressor"
        ),
        "data_source": (
            "MLIT XPT001 + XKT015"
        ),
        "selected_feature_set": (
            selected_name
        ),
        "selection_rule": (
            "Lowest 2025 validation MAPE"
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
            selected_data[
                "features"
            ]
        ),
        "categorical_features": (
            selected_data[
                "categorical"
            ]
        ),
        "station_features": (
            station_features
        ),
        "test_metrics": (
            test_metrics
        ),
        "baseline_same_dataset_metrics": (
            baseline_metrics
        ),
        "ver4_reference": {
            "mape": (
                VER4_REFERENCE_MAPE
            ),
            "r2": (
                VER4_REFERENCE_R2
            ),
            "note": (
                "Reference only. "
                "Ver.4 used a different "
                "data extraction pipeline."
            ),
        },
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


def show_final_result(
    selected_name,
    selected_metrics,
    baseline_metrics,
):

    print()
    print(
        "Ver.5 最終結果"
    )

    print()
    print(
        f"採用モデル: "
        f"{selected_name}"
    )

    print(
        f"MAE : "
        f"{selected_metrics['mae']:,.0f}円"
    )

    print(
        f"RMSE: "
        f"{selected_metrics['rmse']:,.0f}円"
    )

    print(
        f"MAPE: "
        f"{selected_metrics['mape']:.2f}%"
    )

    print(
        f"R²  : "
        f"{selected_metrics['r2']:.4f}"
    )

    print()
    print(
        "同一データの駅なしベースライン"
    )

    print(
        f"MAPE: "
        f"{baseline_metrics['mape']:.2f}%"
    )

    print(
        f"R²  : "
        f"{baseline_metrics['r2']:.4f}"
    )

    mape_difference = (
        baseline_metrics["mape"]
        - selected_metrics["mape"]
    )

    r2_difference = (
        selected_metrics["r2"]
        - baseline_metrics["r2"]
    )

    print()
    print(
        "駅特徴量の効果"
    )

    print(
        f"MAPE改善: "
        f"{mape_difference:+.2f}"
        "ポイント"
    )

    print(
        f"R²改善: "
        f"{r2_difference:+.4f}"
    )

    print()
    print(
        "旧Ver.4参考値"
    )

    print(
        f"MAPE: "
        f"{VER4_REFERENCE_MAPE:.2f}%"
    )

    print(
        f"R²  : "
        f"{VER4_REFERENCE_R2:.4f}"
    )

    print(
        "※Ver.4とは取得データ構成が違うため、"
        "参考比較です。"
    )


def main():

    print(
        "東京都中古マンション "
        "価格予測モデル Ver.5"
    )

    print(
        "駅特徴量追加版"
    )

    df = load_data()

    df = prepare_data(
        df
    )

    validate_feature_sets(
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
        models,
        validation_comparison,
    ) = compare_feature_sets(
        train_df,
        validation_df,
    )

    selected_data = (
        models[
            selected_name
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

    (
        test_results,
        final_test_comparison,
    ) = compare_final_test(
        train_validation_df,
        test_df,
        selected_name,
        models,
    )

    selected_result = (
        test_results[
            selected_name
        ]
    )

    baseline_result = (
        test_results[
            "baseline_current"
        ]
    )

    final_model = (
        selected_result[
            "model"
        ]
    )

    test_metrics = (
        selected_result[
            "metrics"
        ]
    )

    baseline_metrics = (
        baseline_result[
            "metrics"
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_predictions(
        test_df,
        selected_result[
            "predicted_unit_price"
        ],
        selected_result[
            "predicted_price"
        ],
    )

    save_feature_importance(
        final_model,
        selected_data[
            "features"
        ],
    )

    save_results(
        final_model,
        selected_name,
        selected_data,
        validation_comparison,
        final_test_comparison,
        test_metrics,
        baseline_metrics,
    )

    show_final_result(
        selected_name,
        test_metrics,
        baseline_metrics,
    )

    print()
    print(
        "Ver.5 完了"
    )

    print(
        f"モデル: "
        f"{MODEL_FILE}"
    )

    print(
        f"精度: "
        f"{METRICS_FILE}"
    )

    print(
        f"検証比較: "
        f"{FEATURE_COMPARISON_FILE}"
    )

    print(
        f"最終テスト比較: "
        f"{FINAL_TEST_COMPARISON_FILE}"
    )

    print(
        f"特徴量重要度: "
        f"{FEATURE_IMPORTANCE_FILE}"
    )

    print(
        f"テスト予測: "
        f"{PREDICTIONS_FILE}"
    )


if __name__ == "__main__":
    main()