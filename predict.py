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

TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "price_training.csv"
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


def normalize_text(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    if (
        not text
        or text.lower() == "nan"
        or text == "<NA>"
    ):
        return None

    return text


def load_model_info():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"モデルが見つかりません: "
            f"{MODEL_FILE}"
        )

    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"モデル情報が見つかりません: "
            f"{METRICS_FILE}"
        )

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        model_info = json.load(
            file
        )

    feature_columns = (
        model_info.get(
            "features",
            [],
        )
    )

    categorical_columns = (
        model_info.get(
            "categorical_features",
            [],
        )
    )

    selected_feature_set = (
        model_info.get(
            "selected_feature_set",
            "unknown",
        )
    )

    version = (
        model_info.get(
            "version",
            "unknown",
        )
    )

    test_metrics = (
        model_info.get(
            "test_metrics",
            {},
        )
    )

    if not feature_columns:
        raise RuntimeError(
            "price_model_metrics.json に "
            "features が保存されていません。"
        )

    model = CatBoostRegressor()

    model.load_model(
        str(MODEL_FILE)
    )

    print(
        "モデル読み込み完了"
    )

    print(
        f"Version: "
        f"{version}"
    )

    print(
        f"特徴量セット: "
        f"{selected_feature_set}"
    )

    if "mape" in test_metrics:
        print(
            f"Test MAPE: "
            f"{test_metrics['mape']:.2f}%"
        )

    if "r2" in test_metrics:
        print(
            f"Test R²: "
            f"{test_metrics['r2']:.4f}"
        )

    return (
        model,
        model_info,
        feature_columns,
        categorical_columns,
    )


def create_station_reference():
    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            "駅情報を自動補完するための "
            "price_training.csv が"
            "見つかりません。"
        )

    header = pd.read_csv(
        TRAINING_FILE,
        nrows=0,
    )

    required_columns = [
        "station_name",
        "station_line",
        "station_latitude",
        "station_longitude",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in header.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "駅参照データに必要な列が"
            "ありません: "
            + ", ".join(
                missing_columns
            )
        )

    df = pd.read_csv(
        TRAINING_FILE,
        usecols=required_columns,
        low_memory=False,
    )

    df["station_name"] = (
        df["station_name"]
        .astype("string")
        .str.strip()
    )

    df["station_line"] = (
        df["station_line"]
        .astype("string")
        .str.strip()
    )

    df["station_latitude"] = (
        pd.to_numeric(
            df["station_latitude"],
            errors="coerce",
        )
    )

    df["station_longitude"] = (
        pd.to_numeric(
            df["station_longitude"],
            errors="coerce",
        )
    )

    df = df[
        df["station_name"].notna()
        & df["station_latitude"].notna()
        & df["station_longitude"].notna()
    ].copy()

    df["station_line"] = (
        df["station_line"]
        .fillna("不明")
        .astype("string")
    )

    detail = (
        df.groupby(
            [
                "station_name",
                "station_line",
            ],
            dropna=False,
        )
        .agg(
            station_latitude=(
                "station_latitude",
                "median",
            ),
            station_longitude=(
                "station_longitude",
                "median",
            ),
            count=(
                "station_name",
                "size",
            ),
        )
        .reset_index()
    )

    detail = (
        detail
        .sort_values(
            [
                "station_name",
                "count",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    default_station = (
        detail
        .drop_duplicates(
            subset=[
                "station_name"
            ],
            keep="first",
        )
        .copy()
    )

    exact_reference = {}

    for _, row in detail.iterrows():
        station_name = (
            normalize_text(
                row["station_name"]
            )
        )

        station_line = (
            normalize_text(
                row["station_line"]
            )
        )

        if not station_name:
            continue

        if not station_line:
            station_line = "不明"

        key = (
            station_name,
            station_line,
        )

        exact_reference[
            key
        ] = {
            "station_latitude": float(
                row[
                    "station_latitude"
                ]
            ),
            "station_longitude": float(
                row[
                    "station_longitude"
                ]
            ),
        }

    default_reference = {}

    for _, row in (
        default_station.iterrows()
    ):
        station_name = (
            normalize_text(
                row["station_name"]
            )
        )

        station_line = (
            normalize_text(
                row["station_line"]
            )
        )

        if not station_name:
            continue

        if not station_line:
            station_line = "不明"

        default_reference[
            station_name
        ] = {
            "station_line": (
                station_line
            ),
            "station_latitude": float(
                row[
                    "station_latitude"
                ]
            ),
            "station_longitude": float(
                row[
                    "station_longitude"
                ]
            ),
        }

    print()
    print(
        f"駅参照データ: "
        f"{len(default_reference):,}駅"
    )

    return (
        exact_reference,
        default_reference,
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

        "station_name": [
            "北千住",
            "三軒茶屋",
            "六本木",
        ],

        "station_line": [
            pd.NA,
            pd.NA,
            pd.NA,
        ],

        "station_latitude": [
            np.nan,
            np.nan,
            np.nan,
        ],

        "station_longitude": [
            np.nan,
            np.nan,
            np.nan,
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

        "city_planning": [
            "商業地域",
            "近隣商業地域",
            "商業地域",
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

    df["station_name"] = (
        df["station_name"]
        .astype("string")
    )

    df["station_line"] = (
        df["station_line"]
        .astype("string")
    )

    for column in feature_columns:
        if column not in df.columns:
            df[column] = np.nan

    df.to_csv(
        INPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "Ver.5用入力テンプレートを"
        "作成しました。"
    )

    print(
        f"保存先: "
        f"{INPUT_FILE}"
    )

    print()
    print(
        "station_name を入力すれば、"
        "路線・緯度・経度は"
        "自動補完されます。"
    )


def load_input_data():
    if not INPUT_FILE.exists():
        return None

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    if df.empty:
        raise RuntimeError(
            "prediction_input.csv "
            "が空です。"
        )

    print()
    print(
        f"予測対象: "
        f"{len(df):,}件"
    )

    return df


def upgrade_input_schema(
    df,
    feature_columns,
):
    df = df.copy()

    changed = False
    columns_to_add = []

    for column in feature_columns:
        if column not in df.columns:
            df[column] = np.nan

            columns_to_add.append(
                column
            )

            changed = True

    optional_columns = [
        "station_name",
        "station_line",
        "station_latitude",
        "station_longitude",
    ]

    for column in optional_columns:
        if column not in df.columns:
            if column in {
                "station_name",
                "station_line",
            }:
                df[column] = pd.Series(
                    pd.NA,
                    index=df.index,
                    dtype="string",
                )

            else:
                df[column] = np.nan

            columns_to_add.append(
                column
            )

            changed = True

    if "station_name" in df.columns:
        df["station_name"] = (
            df["station_name"]
            .astype("string")
        )

    if "station_line" in df.columns:
        df["station_line"] = (
            df["station_line"]
            .astype("string")
        )

    if "station_latitude" in df.columns:
        df["station_latitude"] = (
            pd.to_numeric(
                df["station_latitude"],
                errors="coerce",
            )
        )

    if "station_longitude" in df.columns:
        df["station_longitude"] = (
            pd.to_numeric(
                df["station_longitude"],
                errors="coerce",
            )
        )

    if changed:
        df.to_csv(
            INPUT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        print()
        print(
            "旧prediction_input.csvを"
            "Ver.5形式へ更新しました。"
        )

        print(
            "追加された列:"
        )

        for column in columns_to_add:
            print(
                f"- {column}"
            )

        print()
        print(
            "必要項目を入力してから"
            "もう一度実行してください。"
        )

    return (
        df,
        changed,
    )


def enrich_station_features(
    df,
    feature_columns,
):
    station_features_used = any(
        column.startswith(
            "station_"
        )
        for column in feature_columns
    )

    if not station_features_used:
        return df

    if "station_name" not in df.columns:
        raise KeyError(
            "Ver.5モデルでは "
            "station_name が必要です。"
        )

    (
        exact_reference,
        default_reference,
    ) = create_station_reference()

    df = df.copy()

    # 文字列列を明示的にstring型へ
    # pandasのLossySetitemError対策
    if "station_name" not in df.columns:
        df["station_name"] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="string",
        )
    else:
        df["station_name"] = (
            df["station_name"]
            .astype("string")
        )

    if "station_line" not in df.columns:
        df["station_line"] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="string",
        )
    else:
        df["station_line"] = (
            df["station_line"]
            .astype("string")
        )

    # 数値列は明示的にfloatへ
    if "station_latitude" not in df.columns:
        df["station_latitude"] = np.nan
    else:
        df["station_latitude"] = (
            pd.to_numeric(
                df["station_latitude"],
                errors="coerce",
            )
            .astype(float)
        )

    if "station_longitude" not in df.columns:
        df["station_longitude"] = np.nan
    else:
        df["station_longitude"] = (
            pd.to_numeric(
                df["station_longitude"],
                errors="coerce",
            )
            .astype(float)
        )

    lookup_status = []

    auto_line_count = 0
    auto_coordinate_count = 0

    errors = []

    for index, row in (
        df.iterrows()
    ):
        station_name = (
            normalize_text(
                row[
                    "station_name"
                ]
            )
        )

        if not station_name:
            errors.append(
                f"{index + 1}行目: "
                "station_name が空です。"
            )

            lookup_status.append(
                "エラー"
            )

            continue

        station_line = (
            normalize_text(
                row.get(
                    "station_line"
                )
            )
        )

        latitude = pd.to_numeric(
            row.get(
                "station_latitude"
            ),
            errors="coerce",
        )

        longitude = pd.to_numeric(
            row.get(
                "station_longitude"
            ),
            errors="coerce",
        )

        default = (
            default_reference.get(
                station_name
            )
        )

        if station_line is None:
            if default is None:
                errors.append(
                    f"{index + 1}行目: "
                    f"{station_name}駅が"
                    "学習データ内にありません。"
                )

                lookup_status.append(
                    "駅不明"
                )

                continue

            station_line = (
                default[
                    "station_line"
                ]
            )

            df.at[
                index,
                "station_line",
            ] = str(
                station_line
            )

            auto_line_count += 1

        exact = (
            exact_reference.get(
                (
                    station_name,
                    station_line,
                )
            )
        )

        if (
            pd.isna(latitude)
            or pd.isna(longitude)
        ):
            reference = (
                exact
                if exact is not None
                else default
            )

            if reference is None:
                errors.append(
                    f"{index + 1}行目: "
                    f"{station_name}駅の座標を"
                    "自動取得できません。"
                )

                lookup_status.append(
                    "座標不明"
                )

                continue

            latitude = (
                reference[
                    "station_latitude"
                ]
            )

            longitude = (
                reference[
                    "station_longitude"
                ]
            )

            df.at[
                index,
                "station_latitude",
            ] = float(
                latitude
            )

            df.at[
                index,
                "station_longitude",
            ] = float(
                longitude
            )

            auto_coordinate_count += 1

            lookup_status.append(
                "自動補完"
            )

        else:
            lookup_status.append(
                "入力値使用"
            )

    if errors:
        print()
        print(
            "駅情報エラー:"
        )

        for error in errors:
            print(
                f"- {error}"
            )

        raise ValueError(
            "駅情報を確認してください。"
        )

    df[
        "station_lookup_status"
    ] = lookup_status

    print()
    print(
        "駅情報補完"
    )

    print(
        f"路線自動補完: "
        f"{auto_line_count:,}件"
    )

    print(
        f"座標自動補完: "
        f"{auto_coordinate_count:,}件"
    )

    return df


def validate_input_columns(
    df,
    feature_columns,
):
    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "予測に必要な列が不足しています: "
            + ", ".join(
                missing_columns
            )
        )


def prepare_input_data(
    df,
    feature_columns,
    categorical_columns,
):
    df = df.copy()

    validate_input_columns(
        df,
        feature_columns,
    )

    if "area_m2" not in df.columns:
        raise KeyError(
            "area_m2 が必要です。"
        )

    df["area_m2"] = (
        pd.to_numeric(
            df["area_m2"],
            errors="coerce",
        )
    )

    if (
        df["area_m2"]
        .isna()
        .any()
    ):
        raise ValueError(
            "area_m2 に"
            "数値ではない値があります。"
        )

    if (
        df["area_m2"]
        <= 0
    ).any():
        raise ValueError(
            "area_m2 は"
            "0より大きくしてください。"
        )

    for column in feature_columns:
        if column in categorical_columns:
            df[column] = (
                df[column]
                .fillna("不明")
                .astype(str)
            )

        else:
            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )
            )

    required_numeric_columns = [
        column
        for column in feature_columns
        if column
        not in categorical_columns
    ]

    for column in required_numeric_columns:
        if df[column].isna().any():
            raise ValueError(
                f"{column} に"
                "未入力または不正な値があります。"
            )

    if (
        "station_latitude"
        in feature_columns
    ):
        invalid_latitude = (
            ~df[
                "station_latitude"
            ].between(
                20,
                50,
            )
        )

        if invalid_latitude.any():
            raise ValueError(
                "station_latitude が"
                "不正です。"
            )

    if (
        "station_longitude"
        in feature_columns
    ):
        invalid_longitude = (
            ~df[
                "station_longitude"
            ].between(
                120,
                155,
            )
        )

        if invalid_longitude.any():
            raise ValueError(
                "station_longitude が"
                "不正です。"
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
        model.predict(
            X
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

    predicted_contract_price = (
        predicted_unit_price
        * df[
            "area_m2"
        ].to_numpy()
    )

    return (
        predicted_unit_price,
        predicted_contract_price,
    )


def evaluate_price_position(
    gap_percent,
):
    if pd.isna(
        gap_percent
    ):
        return ""

    if gap_percent <= -10:
        return "割安"

    if gap_percent <= 10:
        return "適正"

    if gap_percent <= 20:
        return "やや割高"

    return "割高"


def create_prediction_result(
    df,
    predicted_unit_price,
    predicted_contract_price,
):
    result = df.copy()

    result[
        "predicted_price_per_m2"
    ] = (
        np.round(
            predicted_unit_price
        )
        .astype(int)
    )

    result[
        "predicted_contract_price"
    ] = (
        np.round(
            predicted_contract_price
        )
        .astype(int)
    )

    if "asking_price" in result.columns:
        result[
            "asking_price"
        ] = pd.to_numeric(
            result[
                "asking_price"
            ],
            errors="coerce",
        )

        result[
            "price_gap_amount"
        ] = (
            result[
                "asking_price"
            ]
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

        result[
            "price_evaluation"
        ] = (
            result[
                "price_gap_ratio_percent"
            ]
            .apply(
                evaluate_price_position
            )
        )

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
        f"保存先: "
        f"{OUTPUT_FILE}"
    )


def show_predictions(
    result,
):
    print()
    print(
        "Ver.5 AI価格予測結果"
    )

    for index, row in (
        result.iterrows()
    ):
        print()
        print(
            "-" * 50
        )

        print(
            f"物件 {index + 1}"
        )

        if (
            "property_id"
            in result.columns
        ):
            print(
                f"ID: "
                f"{row['property_id']}"
            )

        if (
            "city"
            in result.columns
        ):
            print(
                f"地域: "
                f"{row['city']}"
            )

        if (
            "district_name"
            in result.columns
        ):
            print(
                f"地区: "
                f"{row['district_name']}"
            )

        if (
            "station_name"
            in result.columns
        ):
            print(
                f"最寄駅: "
                f"{row['station_name']}"
            )

        if (
            "station_line"
            in result.columns
        ):
            print(
                f"路線: "
                f"{row['station_line']}"
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
                row[
                    "asking_price"
                ]
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

            print(
                "価格評価: "
                f"{row['price_evaluation']}"
            )


def main():
    print(
        "東京都中古マンション "
        "成約価格予測 Ver.5"
    )

    (
        model,
        model_info,
        feature_columns,
        categorical_columns,
    ) = load_model_info()

    if not INPUT_FILE.exists():
        create_input_template(
            feature_columns
        )

        return

    df = load_input_data()

    (
        df,
        schema_changed,
    ) = upgrade_input_schema(
        df,
        feature_columns,
    )

    if schema_changed:
        return

    df = enrich_station_features(
        df,
        feature_columns,
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

    result = (
        create_prediction_result(
            df,
            predicted_unit_price,
            predicted_contract_price,
        )
    )

    save_predictions(
        result
    )

    show_predictions(
        result
    )


if __name__ == "__main__":
    main()