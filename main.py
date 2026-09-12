import argparse
import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_DIR = PROJECT_ROOT / "data" / "input"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

STATION_RAW_FILE = RAW_DIR / "tokyo_contract_prices_station.csv"
PRICE_TRAINING_FILE = PROCESSED_DIR / "price_training.csv"
PRICE_MODEL_FILE = MODELS_DIR / "price_model.cbm"
PRICE_METRICS_FILE = OUTPUT_DIR / "price_model_metrics.json"

DAYS_INPUT_FILE = INPUT_DIR / "contract_history.xlsx"
DAYS_TRAINING_FILE = PROCESSED_DIR / "days_training.csv"
DAYS_MODEL_FILE = MODELS_DIR / "days_model.cbm"
DAYS_PREDICTION_INPUT_FILE = INPUT_DIR / "days_prediction_input.xlsx"
DAYS_PREDICTION_OUTPUT_FILE = OUTPUT_DIR / "days_predictions.xlsx"

MINIMUM_DAYS_ROWS = 100

STATUS_FILES = [
    ("駅特徴量付き生データ", STATION_RAW_FILE),
    ("価格学習データ", PRICE_TRAINING_FILE),
    ("価格AIモデル", PRICE_MODEL_FILE),
    ("価格AI評価情報", PRICE_METRICS_FILE),
    ("成約履歴Excel", DAYS_INPUT_FILE),
    ("成約日数学習データ", DAYS_TRAINING_FILE),
    ("成約日数AIモデル", DAYS_MODEL_FILE),
    ("成約日数予測入力", DAYS_PREDICTION_INPUT_FILE),
    ("成約日数予測結果", DAYS_PREDICTION_OUTPUT_FILE),
]

MENU_LINES = [
    "価格予測AI",
    "  collect         旧価格データ取得",
    "  station         Ver.5駅特徴量付きデータ取得",
    "  preprocess      価格データ前処理",
    "  train           Ver.5価格AI学習",
    "  predict         CSV価格予測",
    "  excel           Excel価格予測",
    "  report          Ver.5分析レポート",
    "  price-full      既存データで価格AI一括構築",
    "  price-refresh   データ再取得＋価格AI再構築",
    "",
    "成約日数AI",
    "  days-template   成約履歴Excel作成",
    "  days-preprocess 成約日数学習データ作成",
    "  days-train      成約日数AI学習",
    "  days-full       成約日数AI一括構築",
    "  days-predict    新規物件の成約日数予測",
    "",
    "システム",
    "  status          ファイル・モデル状態確認",
    "  all             全システム構築",
]


def print_header(text: str) -> None:
    print()
    print("=" * 60)
    print(text)
    print("=" * 60)
    print()


def run_script(script_name: str) -> None:
    script_path = PROJECT_ROOT / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"{script_name} が見つかりません。")

    print_header(f"{script_name} を実行します。")

    result = subprocess.run(
        [sys.executable, "-u", str(script_path)],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} の実行に失敗しました。")


def count_csv_rows(file_path: Path) -> int:
    if not file_path.exists():
        return 0

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        row_count = sum(1 for _ in csv.reader(file))

    return max(row_count - 1, 0)


def collect_price_data() -> None:
    run_script("collect_mlit.py")


def collect_station_data() -> None:
    run_script("collect_station_features.py")


def preprocess_price_data() -> None:
    run_script("preprocessing.py")


def train_price_model() -> None:
    run_script("train_price.py")


def predict_price_csv() -> None:
    run_script("predict.py")


def predict_price_excel() -> None:
    run_script("predict_excel.py")


def create_analysis_report() -> None:
    run_script("analysis_report.py")


def preprocess_days_data() -> None:
    run_script("preprocessing_days.py")


def predict_days_model() -> None:
    run_script("predict_days.py")


def create_days_template() -> None:
    if not DAYS_INPUT_FILE.exists():
        run_script("create_days_template.py")
        return

    print_header("成約履歴Excelは作成済みです")
    print(f"ファイル: {DAYS_INPUT_FILE}")
    print()
    print("既存の成約履歴を保護するため、上書きしません。")
    print()
    print(
        "新しく作成し直す場合は、"
        "既存ファイルを別名で保存するか"
        "削除してください。"
    )


def show_days_waiting_message() -> None:
    print_header("成約日数AI: 教師データ待ち")
    print("販売開始日・成約日を含む実成約履歴がまだありません。")
    print()
    print("data/input/contract_history.xlsx")
    print("の「成約履歴」シートへ実データを入力してください。")
    print()
    print("教師データ投入後:")
    print("python main.py days-full")


def train_days_model() -> bool:
    if not DAYS_TRAINING_FILE.exists():
        show_days_waiting_message()
        return False

    row_count = count_csv_rows(DAYS_TRAINING_FILE)

    if row_count < MINIMUM_DAYS_ROWS:
        print_header("成約日数AI: 教師データ不足")
        print(f"現在の学習データ: {row_count:,}件")
        print(f"必要件数: {MINIMUM_DAYS_ROWS:,}件以上")
        print()
        print(
            "少数データから不安定な精度を算出しないため、"
            "モデル学習を停止します。"
        )
        print()
        print("実成約履歴を追加した後、")
        print("python main.py days-full")
        print("を実行してください。")
        return False

    run_script("train_days.py")
    return True


def prepare_price_source() -> None:
    if STATION_RAW_FILE.exists():
        print()
        print("Ver.5駅特徴量付きデータを再利用します。")
        print(f"データ: {STATION_RAW_FILE}")
        print()
        print("最新データを再取得したい場合は")
        print("python main.py station")
        print("を先に実行してください。")
        return

    print()
    print("Ver.5駅特徴量付きデータがまだありません。")
    print("国交省APIから駅特徴量付きデータを取得します。")
    collect_station_data()


def _run_price_pipeline(refresh: bool) -> None:
    if refresh:
        collect_station_data()
    else:
        prepare_price_source()

    preprocess_price_data()
    train_price_model()
    create_analysis_report()


def build_price_model() -> None:
    print_header("価格予測AI Ver.5 一括構築開始")
    _run_price_pipeline(refresh=False)
    print_header("価格予測AI Ver.5 構築完了")
    print("モデル:")
    print(PRICE_MODEL_FILE)
    print()
    print("モデル評価:")
    print(PRICE_METRICS_FILE)


def refresh_price_model() -> None:
    print_header("価格予測AI Ver.5 データ再取得＋再構築開始")
    _run_price_pipeline(refresh=True)
    print_header("価格予測AI Ver.5 データ再取得＋再構築完了")


def build_days_model() -> bool:
    print_header("成約日数AI 一括構築開始")
    preprocess_days_data()

    if not DAYS_TRAINING_FILE.exists():
        show_days_waiting_message()
        return False

    if not train_days_model():
        return False

    print_header("成約日数AI 構築完了")
    print(f"モデル: {DAYS_MODEL_FILE}")
    print()
    print("新規物件を予測する場合:")
    print("python main.py days-predict")
    return True


def build_all() -> None:
    print_header("東京都中古マンション AIシステム全体構築開始")
    build_price_model()

    print()
    print("価格AI構築完了")
    print()
    print("成約日数AIを確認します。")

    try:
        build_days_model()
    except (
        RuntimeError,
        FileNotFoundError,
        KeyError,
        ValueError,
    ) as error:
        print_header("成約日数AIを構築できませんでした")
        print(f"理由: {error}")
        print()
        print(
            "入力データまたは"
            "成約日数AI関連ファイルを"
            "確認してください。"
        )

    print_header("全体処理終了")


def show_status() -> None:
    print_header("東京都中古マンション AIシステム ステータス")

    for name, path in STATUS_FILES:
        status = "OK" if path.exists() else "未作成"
        print(f"{status:4} | {name}")
        print(f"       {path}")

    if not DAYS_TRAINING_FILE.exists():
        print()
        print("成約日数AI: 教師データ待ち")
        print()
        return

    row_count = count_csv_rows(DAYS_TRAINING_FILE)

    print()
    print(f"成約日数学習件数: {row_count:,}件")

    if row_count < MINIMUM_DAYS_ROWS:
        print(
            f"状態: 教師データ不足 "
            f"({MINIMUM_DAYS_ROWS:,}件以上必要)"
        )
    elif DAYS_MODEL_FILE.exists():
        print("状態: 成約日数AI学習済み")
    else:
        print("状態: 成約日数AI学習可能")

    print()


def show_menu() -> None:
    print_header("東京都中古マンション AI予測システム Ver.5")

    for line in MENU_LINES:
        print(line)

    print()


def main() -> None:
    commands = {
        "collect": collect_price_data,
        "station": collect_station_data,
        "preprocess": preprocess_price_data,
        "train": train_price_model,
        "predict": predict_price_csv,
        "excel": predict_price_excel,
        "report": create_analysis_report,
        "price-full": build_price_model,
        "price-refresh": refresh_price_model,
        "days-template": create_days_template,
        "days-preprocess": preprocess_days_data,
        "days-train": train_days_model,
        "days-full": build_days_model,
        "days-predict": predict_days_model,
        "status": show_status,
        "all": build_all,
    }

    parser = argparse.ArgumentParser(
        description=(
            "東京都中古マンション "
            "成約価格・成約日数 "
            "AI予測システム Ver.5"
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        choices=list(commands),
    )

    command = parser.parse_args().command

    if command is None:
        show_menu()
        return

    commands[command]()


if __name__ == "__main__":
    main()
