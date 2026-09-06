import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

STATION_RAW_FILE = (
    RAW_DIR
    / "tokyo_contract_prices_station.csv"
)

PRICE_TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "price_training.csv"
)

PRICE_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "price_model.cbm"
)

PRICE_METRICS_FILE = (
    PROJECT_ROOT
    / "output"
    / "price_model_metrics.json"
)

DAYS_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "contract_history.xlsx"
)

DAYS_TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "days_training.csv"
)

DAYS_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "days_model.cbm"
)


def print_header(
    text: str,
) -> None:
    print()
    print(
        "=" * 60
    )
    print(
        text
    )
    print(
        "=" * 60
    )
    print()


def run_script(
    script_name: str,
) -> None:
    script_path = (
        PROJECT_ROOT
        / script_name
    )

    if not script_path.exists():
        raise FileNotFoundError(
            f"{script_name} が"
            "見つかりません。"
        )

    print_header(
        f"{script_name} を実行します。"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(
                script_path
            ),
        ],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} の"
            "実行に失敗しました。"
        )


def collect_price_data() -> None:
    run_script(
        "collect_mlit.py"
    )


def collect_station_data() -> None:
    run_script(
        "collect_station_features.py"
    )


def preprocess_price_data() -> None:
    run_script(
        "preprocessing.py"
    )


def train_price_model() -> None:
    run_script(
        "train_price.py"
    )


def predict_price_csv() -> None:
    run_script(
        "predict.py"
    )


def predict_price_excel() -> None:
    run_script(
        "predict_excel.py"
    )


def create_analysis_report() -> None:
    run_script(
        "analysis_report.py"
    )


def create_days_template() -> None:
    run_script(
        "create_days_template.py"
    )


def preprocess_days_data() -> None:
    run_script(
        "preprocessing_days.py"
    )


def train_days_model() -> None:
    run_script(
        "train_days.py"
    )


def prepare_price_source() -> None:
    if STATION_RAW_FILE.exists():
        print()
        print(
            "Ver.5駅特徴量付きデータを"
            "再利用します。"
        )

        print(
            f"データ: "
            f"{STATION_RAW_FILE}"
        )

        print()
        print(
            "最新データを再取得したい場合は"
        )

        print(
            "python main.py station"
        )

        print(
            "を先に実行してください。"
        )

        return

    print()
    print(
        "Ver.5駅特徴量付きデータが"
        "まだありません。"
    )

    print(
        "国交省APIから"
        "駅特徴量付きデータを取得します。"
    )

    collect_station_data()


def build_price_model() -> None:
    print_header(
        "価格予測AI Ver.5 "
        "一括構築開始"
    )

    prepare_price_source()

    preprocess_price_data()

    train_price_model()

    create_analysis_report()

    print_header(
        "価格予測AI Ver.5 "
        "構築完了"
    )

    print(
        "モデル:"
    )

    print(
        PRICE_MODEL_FILE
    )

    print()
    print(
        "モデル評価:"
    )

    print(
        PRICE_METRICS_FILE
    )


def refresh_price_model() -> None:
    print_header(
        "価格予測AI Ver.5 "
        "データ再取得＋再構築開始"
    )

    collect_station_data()

    preprocess_price_data()

    train_price_model()

    create_analysis_report()

    print_header(
        "価格予測AI Ver.5 "
        "データ再取得＋再構築完了"
    )


def build_days_model() -> None:
    print_header(
        "成約日数AI "
        "一括構築開始"
    )

    preprocess_days_data()

    train_days_model()

    print_header(
        "成約日数AI "
        "構築完了"
    )


def build_all() -> None:
    print_header(
        "東京都中古マンション "
        "AIシステム全体構築開始"
    )

    build_price_model()

    print()
    print(
        "価格AI構築完了"
    )

    print()
    print(
        "成約日数AIを確認します。"
    )

    try:
        build_days_model()

    except (
        RuntimeError,
        FileNotFoundError,
        ValueError,
    ) as error:
        print()
        print(
            "成約日数AIは"
            "まだ構築されませんでした。"
        )

        print()
        print(
            f"理由: "
            f"{error}"
        )

        print()
        print(
            "実成約履歴データが"
            "100件以上集まり次第、"
        )

        print(
            "python main.py days-full"
        )

        print(
            "を実行してください。"
        )

    print_header(
        "全体処理終了"
    )


def show_status() -> None:
    print_header(
        "東京都中古マンション "
        "AIシステム ステータス"
    )

    files = [
        (
            "駅特徴量付き生データ",
            STATION_RAW_FILE,
        ),
        (
            "価格学習データ",
            PRICE_TRAINING_FILE,
        ),
        (
            "価格AIモデル",
            PRICE_MODEL_FILE,
        ),
        (
            "価格AI評価情報",
            PRICE_METRICS_FILE,
        ),
        (
            "成約履歴Excel",
            DAYS_INPUT_FILE,
        ),
        (
            "成約日数学習データ",
            DAYS_TRAINING_FILE,
        ),
        (
            "成約日数AIモデル",
            DAYS_MODEL_FILE,
        ),
    ]

    for name, path in files:
        status = (
            "OK"
            if path.exists()
            else "未作成"
        )

        print(
            f"{status:4} | "
            f"{name}"
        )

        print(
            f"       {path}"
        )

    print()


def show_menu() -> None:
    print_header(
        "東京都中古マンション "
        "AI予測システム Ver.5"
    )

    print(
        "価格予測AI"
    )

    print(
        "  collect"
        "         旧価格データ取得"
    )

    print(
        "  station"
        "         Ver.5駅特徴量付きデータ取得"
    )

    print(
        "  preprocess"
        "      価格データ前処理"
    )

    print(
        "  train"
        "           Ver.5価格AI学習"
    )

    print(
        "  predict"
        "         CSV価格予測"
    )

    print(
        "  excel"
        "           Excel価格予測"
    )

    print(
        "  report"
        "          Ver.5分析レポート"
    )

    print(
        "  price-full"
        "      既存データで価格AI一括構築"
    )

    print(
        "  price-refresh"
        "   データ再取得＋価格AI再構築"
    )

    print()
    print(
        "成約日数AI"
    )

    print(
        "  days-template"
        "   成約履歴Excel作成"
    )

    print(
        "  days-preprocess"
        " 成約日数学習データ作成"
    )

    print(
        "  days-train"
        "      成約日数AI学習"
    )

    print(
        "  days-full"
        "       成約日数AI一括構築"
    )

    print()
    print(
        "システム"
    )

    print(
        "  status"
        "          ファイル・モデル状態確認"
    )

    print(
        "  all"
        "             全システム構築"
    )

    print()


def main() -> None:
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
        choices=[
            "collect",
            "station",
            "preprocess",
            "train",
            "predict",
            "excel",
            "report",
            "price-full",
            "price-refresh",
            "days-template",
            "days-preprocess",
            "days-train",
            "days-full",
            "status",
            "all",
        ],
    )

    args = (
        parser.parse_args()
    )

    command = (
        args.command
    )

    if command is None:
        show_menu()
        return

    if command == "collect":
        collect_price_data()

    elif command == "station":
        collect_station_data()

    elif command == "preprocess":
        preprocess_price_data()

    elif command == "train":
        train_price_model()

    elif command == "predict":
        predict_price_csv()

    elif command == "excel":
        predict_price_excel()

    elif command == "report":
        create_analysis_report()

    elif command == "price-full":
        build_price_model()

    elif command == "price-refresh":
        refresh_price_model()

    elif command == "days-template":
        create_days_template()

    elif command == "days-preprocess":
        preprocess_days_data()

    elif command == "days-train":
        train_days_model()

    elif command == "days-full":
        build_days_model()

    elif command == "status":
        show_status()

    elif command == "all":
        build_all()


if __name__ == "__main__":
    main()