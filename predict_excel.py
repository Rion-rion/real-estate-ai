from pathlib import Path

import numpy as np
import pandas as pd

from openpyxl import load_workbook
from openpyxl.styles import (
    Alignment,
    Border,
    Font,
    PatternFill,
    Side,
)

from predict import (
    TRAINING_FILE,
    enrich_station_features,
    evaluate_price_position,
    load_model_info,
    predict_prices,
    prepare_input_data,
)


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "input"
)

INPUT_FILE = (
    INPUT_DIR
    / "prediction_input.xlsx"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "price_predictions.xlsx"
)

INPUT_SHEET = "予測入力"
RESULT_SHEET = "予測結果"
MODEL_SHEET = "モデル情報"


def create_excel_template(
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

    for column in feature_columns:
        if column not in df.columns:
            df[column] = np.nan

    df.to_excel(
        INPUT_FILE,
        index=False,
        sheet_name=INPUT_SHEET,
    )

    style_excel(
        INPUT_FILE,
        INPUT_SHEET,
    )

    print()
    print(
        "Ver.5 Excel入力テンプレートを"
        "作成しました。"
    )

    print(
        f"保存先: "
        f"{INPUT_FILE}"
    )

    print()
    print(
        "テンプレートにはサンプル物件が"
        "入力されています。"
    )

    print(
        "もう一度実行すると"
        "そのまま予測できます。"
    )


def load_excel_input():
    if not INPUT_FILE.exists():
        return None

    try:
        df = pd.read_excel(
            INPUT_FILE,
            sheet_name=INPUT_SHEET,
        )

    except ValueError:
        df = pd.read_excel(
            INPUT_FILE,
            sheet_name=0,
        )

    if df.empty:
        raise RuntimeError(
            "Excelに予測対象がありません。"
        )

    print()
    print(
        f"Excel予測対象: "
        f"{len(df):,}件"
    )

    return df


def upgrade_excel_schema(
    df,
    feature_columns,
):
    df = df.copy()

    added_columns = []

    required_columns = list(
        feature_columns
    )

    extra_columns = [
        "property_id",
        "station_name",
        "station_line",
        "station_latitude",
        "station_longitude",
        "asking_price",
    ]

    for column in extra_columns:
        if column not in required_columns:
            required_columns.append(
                column
            )

    for column in required_columns:
        if column in df.columns:
            continue

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

        added_columns.append(
            column
        )

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

    if added_columns:
        print()
        print(
            "旧Excel入力ファイルを"
            "Ver.5形式へ更新しました。"
        )

        print(
            "追加された列:"
        )

        for column in added_columns:
            print(
                f"- {column}"
            )

        print()
        print(
            "駅情報は可能な範囲で"
            "自動補完して処理を続行します。"
        )

    return df


def normalize_station_text(
    series,
):
    return (
        series
        .astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "<NA>": pd.NA,
            }
        )
    )


def auto_fill_station_name(
    df,
):
    df = df.copy()

    if "station_name" not in df.columns:
        df["station_name"] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="string",
        )

    df["station_name"] = (
        normalize_station_text(
            df["station_name"]
        )
    )

    missing_mask = (
        df["station_name"]
        .isna()
    )

    missing_count = int(
        missing_mask.sum()
    )

    if missing_count == 0:
        print()
        print(
            "最寄駅名: "
            "入力済み"
        )

        return df

    if "city" not in df.columns:
        raise KeyError(
            "station_name 自動推定には "
            "city が必要です。"
        )

    if "district_name" not in df.columns:
        raise KeyError(
            "station_name 自動推定には "
            "district_name が必要です。"
        )

    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            "駅名自動推定に必要な "
            "price_training.csv が"
            "見つかりません。"
        )

    reference = pd.read_csv(
        TRAINING_FILE,
        usecols=[
            "city",
            "district_name",
            "station_name",
        ],
        low_memory=False,
    )

    reference["city"] = (
        normalize_station_text(
            reference["city"]
        )
    )

    reference[
        "district_name"
    ] = normalize_station_text(
        reference[
            "district_name"
        ]
    )

    reference[
        "station_name"
    ] = normalize_station_text(
        reference[
            "station_name"
        ]
    )

    reference = reference[
        reference["city"].notna()
        & reference[
            "district_name"
        ].notna()
        & reference[
            "station_name"
        ].notna()
    ].copy()

    station_counts = (
        reference
        .groupby(
            [
                "city",
                "district_name",
                "station_name",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    station_counts = (
        station_counts
        .sort_values(
            [
                "city",
                "district_name",
                "count",
                "station_name",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    best_station = (
        station_counts
        .drop_duplicates(
            subset=[
                "city",
                "district_name",
            ],
            keep="first",
        )
    )

    station_map = {}

    for _, row in (
        best_station.iterrows()
    ):
        key = (
            str(
                row["city"]
            ).strip(),
            str(
                row[
                    "district_name"
                ]
            ).strip(),
        )

        station_map[
            key
        ] = str(
            row[
                "station_name"
            ]
        ).strip()

    df["city"] = (
        normalize_station_text(
            df["city"]
        )
    )

    df[
        "district_name"
    ] = normalize_station_text(
        df[
            "district_name"
        ]
    )

    filled_count = 0
    errors = []

    for index in (
        df.index[
            missing_mask
        ]
    ):
        city = df.at[
            index,
            "city",
        ]

        district_name = df.at[
            index,
            "district_name",
        ]

        if (
            pd.isna(city)
            or pd.isna(
                district_name
            )
        ):
            errors.append(
                f"{index + 1}行目: "
                "city または "
                "district_name が空です。"
            )

            continue

        key = (
            str(city).strip(),
            str(
                district_name
            ).strip(),
        )

        station_name = (
            station_map.get(
                key
            )
        )

        if station_name is None:
            errors.append(
                f"{index + 1}行目: "
                f"{city} / "
                f"{district_name} の"
                "最寄駅を推定できません。"
            )

            continue

        df.at[
            index,
            "station_name",
        ] = station_name

        filled_count += 1

    print()
    print(
        "最寄駅名自動推定"
    )

    print(
        f"station_name 自動補完: "
        f"{filled_count:,}件"
    )

    if errors:
        print()

        print(
            "最寄駅推定エラー:"
        )

        for error in errors:
            print(
                f"- {error}"
            )

        raise ValueError(
            "最寄駅名を推定できない"
            "物件があります。"
        )

    return df


def save_updated_input(
    df,
):
    save_df = df.copy()

    removable_columns = [
        "station_lookup_status",
    ]

    for column in removable_columns:
        if column in save_df.columns:
            save_df = save_df.drop(
                columns=[
                    column
                ]
            )

    save_df.to_excel(
        INPUT_FILE,
        index=False,
        sheet_name=INPUT_SHEET,
    )

    style_excel(
        INPUT_FILE,
        INPUT_SHEET,
    )

    print()
    print(
        "Excel入力ファイルの"
        "駅情報も更新しました。"
    )


def create_result(
    df,
    predicted_unit_price,
    predicted_contract_price,
):
    result = df.copy()

    result[
        "AI予測㎡単価"
    ] = (
        np.round(
            predicted_unit_price
        )
        .astype(int)
    )

    result[
        "AI予測成約価格"
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
            "売出価格との差額"
        ] = (
            result[
                "asking_price"
            ]
            - result[
                "AI予測成約価格"
            ]
        )

        result[
            "価格乖離率"
        ] = (
            result[
                "売出価格との差額"
            ]
            / result[
                "AI予測成約価格"
            ]
        )

        result[
            "価格評価"
        ] = (
            result[
                "価格乖離率"
            ]
            .mul(100)
            .apply(
                evaluate_price_position
            )
        )

    return result


def create_model_info(
    model_info,
):
    metrics = (
        model_info.get(
            "test_metrics",
            {},
        )
    )

    baseline = (
        model_info.get(
            "baseline_same_dataset_metrics",
            {},
        )
    )

    station_features = (
        model_info.get(
            "station_features",
            [],
        )
    )

    rows = [
        [
            "モデル",
            model_info.get(
                "model",
                "CatBoostRegressor",
            ),
        ],

        [
            "バージョン",
            model_info.get(
                "version",
                "",
            ),
        ],

        [
            "特徴量セット",
            model_info.get(
                "selected_feature_set",
                "",
            ),
        ],

        [
            "テストMAPE",
            metrics.get(
                "mape",
                "",
            ),
        ],

        [
            "テストR²",
            metrics.get(
                "r2",
                "",
            ),
        ],

        [
            "テストMAE",
            metrics.get(
                "mae",
                "",
            ),
        ],

        [
            "テストRMSE",
            metrics.get(
                "rmse",
                "",
            ),
        ],

        [
            "駅なしMAPE",
            baseline.get(
                "mape",
                "",
            ),
        ],

        [
            "駅なしR²",
            baseline.get(
                "r2",
                "",
            ),
        ],

        [
            "駅特徴量",
            ", ".join(
                station_features
            ),
        ],

        [
            "最終テスト",
            "2026年",
        ],

        [
            "注意",
            (
                "最寄駅自動推定は"
                "city + district_name 内で"
                "学習データ上最頻の駅を使用"
            ),
        ],
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "項目",
            "値",
        ],
    )


def save_excel(
    result,
    model_info,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_df = (
        create_model_info(
            model_info
        )
    )

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl",
    ) as writer:

        result.to_excel(
            writer,
            index=False,
            sheet_name=RESULT_SHEET,
        )

        model_df.to_excel(
            writer,
            index=False,
            sheet_name=MODEL_SHEET,
        )

    style_excel(
        OUTPUT_FILE,
        RESULT_SHEET,
    )

    style_excel(
        OUTPUT_FILE,
        MODEL_SHEET,
    )

    print()
    print(
        "Ver.5 Excel予測完了"
    )

    print(
        f"保存先: "
        f"{OUTPUT_FILE}"
    )


def style_excel(
    file_path,
    sheet_name,
):
    workbook = load_workbook(
        file_path
    )

    worksheet = workbook[
        sheet_name
    ]

    header_fill = PatternFill(
        fill_type="solid",
        fgColor="1F4E78",
    )

    header_font = Font(
        color="FFFFFF",
        bold=True,
    )

    thin_border = Border(
        bottom=Side(
            style="thin",
            color="D9E2F3",
        )
    )

    for cell in worksheet[1]:
        cell.fill = (
            header_fill
        )

        cell.font = (
            header_font
        )

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )

    worksheet.freeze_panes = "A2"

    for column_cells in (
        worksheet.columns
    ):
        max_length = 0

        column_letter = (
            column_cells[0]
            .column_letter
        )

        for cell in column_cells:
            if cell.value is None:
                continue

            text_length = len(
                str(
                    cell.value
                )
            )

            max_length = max(
                max_length,
                text_length,
            )

            cell.border = (
                thin_border
            )

        worksheet.column_dimensions[
            column_letter
        ].width = min(
            max(
                max_length + 2,
                10,
            ),
            30,
        )

    header_map = {
        cell.value: cell.column
        for cell in worksheet[1]
    }

    money_columns = [
        "asking_price",
        "AI予測㎡単価",
        "AI予測成約価格",
        "売出価格との差額",
    ]

    for column_name in (
        money_columns
    ):
        column_number = (
            header_map.get(
                column_name
            )
        )

        if not column_number:
            continue

        for row in range(
            2,
            worksheet.max_row + 1,
        ):
            worksheet.cell(
                row=row,
                column=column_number,
            ).number_format = (
                "#,##0"
            )

    ratio_column = (
        header_map.get(
            "価格乖離率"
        )
    )

    if ratio_column:
        for row in range(
            2,
            worksheet.max_row + 1,
        ):
            worksheet.cell(
                row=row,
                column=ratio_column,
            ).number_format = (
                "0.00%"
            )

    latitude_column = (
        header_map.get(
            "station_latitude"
        )
    )

    longitude_column = (
        header_map.get(
            "station_longitude"
        )
    )

    for column_number in [
        latitude_column,
        longitude_column,
    ]:
        if not column_number:
            continue

        for row in range(
            2,
            worksheet.max_row + 1,
        ):
            worksheet.cell(
                row=row,
                column=column_number,
            ).number_format = (
                "0.000000"
            )

    workbook.save(
        file_path
    )


def show_result(
    result,
):
    print()
    print(
        "Ver.5 Excel AI価格予測結果"
    )

    for index, row in (
        result.iterrows()
    ):
        print()
        print(
            "-" * 50
        )

        print(
            f"物件 "
            f"{index + 1}"
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

        if "station_name" in result.columns:
            print(
                f"最寄駅: "
                f"{row['station_name']}"
            )

        if "station_line" in result.columns:
            print(
                f"路線: "
                f"{row['station_line']}"
            )

        if "area_m2" in result.columns:
            print(
                f"面積: "
                f"{row['area_m2']}㎡"
            )

        print(
            "AI予測㎡単価: "
            f"{row['AI予測㎡単価']:,.0f}"
            "円/㎡"
        )

        print(
            "AI予測成約価格: "
            f"{row['AI予測成約価格']:,.0f}"
            "円"
        )

        if (
            "asking_price"
            in result.columns
            and pd.notna(
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
                f"{row['売出価格との差額']:,.0f}"
                "円"
            )

            print(
                "価格乖離率: "
                f"{row['価格乖離率'] * 100:.2f}"
                "%"
            )

            print(
                "価格評価: "
                f"{row['価格評価']}"
            )


def main():
    print(
        "東京都中古マンション "
        "Excel成約価格予測 Ver.5"
    )

    (
        model,
        model_info,
        feature_columns,
        categorical_columns,
    ) = load_model_info()

    if not INPUT_FILE.exists():
        create_excel_template(
            feature_columns
        )

        return

    df = load_excel_input()

    df = upgrade_excel_schema(
        df,
        feature_columns,
    )

    df = auto_fill_station_name(
        df
    )

    df = enrich_station_features(
        df,
        feature_columns,
    )

    save_updated_input(
        df
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

    result = create_result(
        df,
        predicted_unit_price,
        predicted_contract_price,
    )

    save_excel(
        result,
        model_info,
    )

    show_result(
        result
    )


if __name__ == "__main__":
    main()