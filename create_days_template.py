from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_FILE = INPUT_DIR / "contract_history.xlsx"

COLUMNS = [
    "property_id", "city", "district_name", "station_name", "station_line",
    "station_latitude", "station_longitude", "area_m2", "floor_plan",
    "building_age", "structure", "renovation", "use", "city_planning",
    "coverage_ratio", "floor_area_ratio", "listing_date", "asking_price",
    "contract_date", "contract_price",
]

SAMPLE_ROW = {
    "property_id": "A001",
    "city": "足立区",
    "district_name": "千住",
    "station_name": "北千住",
    "station_line": "",
    "station_latitude": "",
    "station_longitude": "",
    "area_m2": 65.2,
    "floor_plan": "3LDK",
    "building_age": 12,
    "structure": "RC",
    "renovation": "未改装",
    "use": "住宅",
    "city_planning": "商業地域",
    "coverage_ratio": 80,
    "floor_area_ratio": 400,
    "listing_date": "2026-04-01",
    "asking_price": 75_000_000,
    "contract_date": "2026-05-21",
    "contract_price": 72_000_000,
}

FIELD_DESCRIPTIONS = [
    ("property_id", "必須", "物件を識別するID", "A001"),
    ("city", "必須", "東京都の市区町村", "足立区"),
    ("district_name", "必須", "町・地区名", "千住"),
    ("station_name", "推奨", "最寄駅名。入力されている場合はこの駅を優先", "北千住"),
    ("station_line", "任意", "路線名。空欄の場合は駅情報から補完可能", "常磐線"),
    ("station_latitude", "任意", "駅緯度。空欄の場合は駅情報から補完可能", "35.749"),
    ("station_longitude", "任意", "駅経度。空欄の場合は駅情報から補完可能", "139.805"),
    ("area_m2", "必須", "専有面積（㎡）", "65.2"),
    ("floor_plan", "必須", "間取り", "3LDK"),
    ("building_age", "必須", "売出時点の築年数", "12"),
    ("structure", "推奨", "建物構造", "RC"),
    ("renovation", "任意", "リフォーム・改装状況", "未改装"),
    ("use", "任意", "物件用途", "住宅"),
    ("city_planning", "推奨", "都市計画・用途地域", "商業地域"),
    ("coverage_ratio", "任意", "建ぺい率（%）", "80"),
    ("floor_area_ratio", "任意", "容積率（%）", "400"),
    ("listing_date", "必須", "販売を開始した日", "2026-04-01"),
    ("asking_price", "必須", "販売開始時点の売出価格（円）", "75000000"),
    ("contract_date", "必須", "成約した日", "2026-05-21"),
    ("contract_price", "推奨", "実際の成約価格（円）。評価・確認用", "72000000"),
]

REQUIRED_COLUMNS = [
    "property_id", "city", "district_name", "area_m2", "floor_plan",
    "building_age", "listing_date", "asking_price", "contract_date",
]
RECOMMENDED_COLUMNS = [
    "station_name", "structure", "city_planning", "contract_price",
]
NUMBER_FORMATS = {
    "yyyy-mm-dd": ["listing_date", "contract_date"],
    '#,##0"円"': ["asking_price", "contract_price"],
    "0.0": ["area_m2"],
}


def create_template() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            "\ncontract_history.xlsx はすでに存在します。\n"
            "既存の成約履歴を保護するため、上書きを停止しました。\n\n"
            f"対象ファイル: {OUTPUT_FILE}\n\n"
            "新しく作り直す場合は、既存ファイルを別名で保存するか"
            "削除してから再実行してください。"
        )

    sheets = {
        "成約履歴": pd.DataFrame(columns=COLUMNS),
        "入力例": pd.DataFrame([SAMPLE_ROW], columns=COLUMNS),
        "項目説明": pd.DataFrame(
            FIELD_DESCRIPTIONS,
            columns=["項目名", "入力区分", "説明", "入力例"],
        ),
    }

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for sheet_name, dataframe in sheets.items():
            dataframe.to_excel(
                writer,
                index=False,
                sheet_name=sheet_name,
            )

    style_excel()

    print()
    print("=" * 60)
    print("成約日数AI 学習用テンプレート作成完了")
    print("=" * 60)
    print()
    print(f"保存先: {OUTPUT_FILE}")
    print()
    print("シート:")
    print("  成約履歴 : 実データ入力用")
    print("  入力例   : 入力方法のサンプル")
    print("  項目説明 : 各項目の説明")
    print()
    print("成約履歴シートへ実際の販売・成約履歴を入力してください。")
    print()
    print(
        "station_line / station_latitude / "
        "station_longitude は空欄でも構いません。"
    )
    print()
    print(
        "注意: 現時点では教師データがないため、"
        "モデル精度の評価は行っていません。"
    )


def _style_header(worksheet, fill: PatternFill, font: Font) -> None:
    for cell in worksheet[1]:
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )


def _adjust_column_widths(worksheet) -> None:
    for column_cells in worksheet.columns:
        max_length = max(
            (
                len(str(cell.value))
                for cell in column_cells
                if cell.value is not None
            ),
            default=0,
        )
        column_letter = get_column_letter(column_cells[0].column)
        worksheet.column_dimensions[column_letter].width = min(
            max(max_length + 2, 12),
            45,
        )


def _fill_headers(
    worksheet,
    header_map: dict,
    columns: list[str],
    fill: PatternFill,
) -> None:
    for column in columns:
        if (column_index := header_map.get(column)) is not None:
            worksheet.cell(row=1, column=column_index).fill = fill


def _apply_number_formats(
    worksheet,
    header_map: dict,
) -> None:
    for number_format, columns in NUMBER_FORMATS.items():
        for column in columns:
            column_index = header_map.get(column)

            if column_index is None:
                continue

            for row in range(2, 1002):
                worksheet.cell(
                    row=row,
                    column=column_index,
                ).number_format = number_format


def style_excel() -> None:
    workbook = load_workbook(OUTPUT_FILE)

    header_fill = PatternFill(fill_type="solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    required_fill = PatternFill(fill_type="solid", fgColor="FFF2CC")
    recommended_fill = PatternFill(fill_type="solid", fgColor="E2F0D9")

    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        _style_header(worksheet, header_fill, header_font)
        _adjust_column_widths(worksheet)

    history_sheet = workbook["成約履歴"]
    header_map = {
        cell.value: cell.column
        for cell in history_sheet[1]
    }

    _fill_headers(
        history_sheet,
        header_map,
        REQUIRED_COLUMNS,
        required_fill,
    )
    _fill_headers(
        history_sheet,
        header_map,
        RECOMMENDED_COLUMNS,
        recommended_fill,
    )
    _apply_number_formats(history_sheet, header_map)

    description_sheet = workbook["項目説明"]
    fill_by_type = {
        "必須": required_fill,
        "推奨": recommended_fill,
    }

    for row in range(2, description_sheet.max_row + 1):
        input_type_cell = description_sheet.cell(row=row, column=2)

        if fill := fill_by_type.get(input_type_cell.value):
            input_type_cell.fill = fill

    workbook.save(OUTPUT_FILE)


def main() -> None:
    create_template()


if __name__ == "__main__":
    main()
