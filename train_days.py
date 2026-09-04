from pathlib import Path
import json

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent

TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "days_training.csv"
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
    / "days_model.cbm"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "days_model_metrics.json"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "days_test_predictions.csv"
)

FEATURE_IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "days_feature_importance.csv"
)

TARGET_COLUMN = "days_to_contract"

MINIMUM_ROWS = 100


FEATURE_COLUMNS = [
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

    "listing_year",
    "listing_month",
    "listing_quarter",

    "asking_price",
    "asking_price_per_m2",

    "ai_estimated_price",
    "ai_price_per_m2",

    "price_gap_ratio",
]


CATEGORICAL_COLUMNS = [
    "city",
    "district_name",
    "floor_plan",
    "structure",
    "renovation",
    "use",
    "city_planning",
]


def load_data():

    if not TRAINING_FILE.exists():

        raise FileNotFoundError(
            f"days_training.csv がありません: "
            f"{TRAINING_FILE}"
        )

    df = pd.read_csv(
        TRAINING_FILE,
        low_memory=False,
    )

    print(
        f"成約日数学習データ: "
        f"{len(df):,}件"
    )

    if len(df) < MINIMUM_ROWS:

        raise RuntimeError(
            f"\n現在 {len(df):,}件しかありません。\n"
            f"最低でも {MINIMUM_ROWS}件以上の"
            "実成約データを用意してください。\n"
            "サンプルデータだけでモデル精度を"
            "算出しないよう停止しました。"
        )

    return df


def prepare_data(df):

    df = df.copy()

    features = [
        column
        for column in FEATURE_COLUMNS
        if column in df.columns
    ]

    categorical = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in features
    ]

    for column in categorical:

        df[column] = (
            df[column]
            .fillna("不明")
            .astype(str)
        )

    numeric_features = [
        column
        for column in features
        if column not in categorical
    ]

    for column in numeric_features:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    df = df[
        df[TARGET_COLUMN].notna()
    ].copy()

    df = df[
        df[TARGET_COLUMN].between(
            1,
            1000,
        )
    ].copy()

    return (
        df,
        features,
        categorical,
    )


def split_data(df):

    if "listing_date" in df.columns:

        df["listing_date"] = pd.to_datetime(
            df["listing_date"],
            errors="coerce",
        )

        df = df.sort_values(
            "listing_date"
        ).reset_index(
            drop=True
        )

    train_end = int(
        len(df) * 0.70
    )

    validation_end = int(
        len(df) * 0.85
    )

    train = df.iloc[
        :train_end
    ].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[
        validation_end:
    ].copy()

    print()
    print("時間順分割")
    print(
        f"学習: {len(train):,}件"
    )
    print(
        f"検証: {len(validation):,}件"
    )
    print(
        f"テスト: {len(test):,}件"
    )

    return (
        train,
        validation,
        test,
    )


def train_model(
    train,
    validation,
    features,
    categorical,
):

    X_train = train[
        features
    ]

    X_validation = validation[
        features
    ]

    # 成約日数は右裾が長くなりやすいためlog変換
    y_train = np.log1p(
        train[TARGET_COLUMN]
    )

    y_validation = np.log1p(
        validation[TARGET_COLUMN]
    )

    model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        l2_leaf_reg=8,
        verbose=100,
        allow_writing_files=False,
    )

    print()
    print(
        "成約日数モデル学習開始"
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical,
        eval_set=(
            X_validation,
            y_validation,
        ),
        use_best_model=True,
        early_stopping_rounds=150,
    )

    return model


def predict_days(
    model,
    df,
    features,
):

    predicted_log = model.predict(
        df[features]
    )

    predicted_days = np.expm1(
        predicted_log
    )

    predicted_days = np.maximum(
        predicted_days,
        1,
    )

    return predicted_days


def evaluate(
    model,
    df,
    features,
    name,
):

    predicted = predict_days(
        model,
        df,
        features,
    )

    actual = (
        df[TARGET_COLUMN]
        .to_numpy()
    )

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    median_ae = median_absolute_error(
        actual,
        predicted,
    )

    r2 = r2_score(
        actual,
        predicted,
    )

    print()
    print(
        f"{name} 精度"
    )

    print(
        f"MAE      : {mae:.2f}日"
    )

    print(
        f"RMSE     : {rmse:.2f}日"
    )

    print(
        f"Median AE: {median_ae:.2f}日"
    )

    print(
        f"R²       : {r2:.4f}"
    )

    return (
        {
            "mae_days": float(mae),
            "rmse_days": float(rmse),
            "median_absolute_error_days": (
                float(median_ae)
            ),
            "r2": float(r2),
        },
        predicted,
    )


def save_results(
    model,
    features,
    categorical,
    validation_metrics,
    test_metrics,
    test_df,
    predicted,
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

    metrics = {
        "version": "1.0",
        "model": "CatBoostRegressor",
        "target": "log_days_to_contract",
        "features": features,
        "categorical_features": categorical,
        "validation_metrics": (
            validation_metrics
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
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    result = test_df.copy()

    result[
        "predicted_days_to_contract"
    ] = np.round(
        predicted
    ).astype(int)

    result[
        "absolute_error_days"
    ] = (
        result[
            "predicted_days_to_contract"
        ]
        - result[
            TARGET_COLUMN
        ]
    ).abs()

    result.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    importance = pd.DataFrame(
        {
            "feature": features,
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

    print()
    print(
        f"モデル: {MODEL_FILE}"
    )

    print(
        f"精度: {METRICS_FILE}"
    )


def main():

    print(
        "東京都中古マンション"
        "成約日数予測モデル"
    )

    df = load_data()

    (
        df,
        features,
        categorical,
    ) = prepare_data(
        df
    )

    (
        train,
        validation,
        test,
    ) = split_data(
        df
    )

    model = train_model(
        train,
        validation,
        features,
        categorical,
    )

    (
        validation_metrics,
        _,
    ) = evaluate(
        model,
        validation,
        features,
        "検証データ",
    )

    (
        test_metrics,
        predicted,
    ) = evaluate(
        model,
        test,
        features,
        "テストデータ",
    )

    save_results(
        model,
        features,
        categorical,
        validation_metrics,
        test_metrics,
        test,
        predicted,
    )


if __name__ == "__main__":
    main()