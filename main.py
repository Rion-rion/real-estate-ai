import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_script(script_name: str) -> None:
    script_path = PROJECT_ROOT / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"{script_name} が見つかりません。"
        )

    print()
    print(f"{script_name} を実行します。")
    print("-" * 60)

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} の実行に失敗しました。"
        )


def collect_price_data() -> None:
    run_script(
        "collect_mlit.py"
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


def build_price_model() -> None:
    print()
    print(
        "価格予測モデルの構築を開始します。"
    )

    collect_price_data()
    preprocess_price_data()
    train_price_model()

    print()
    print(
        "価格予測モデルの構築が完了しました。"
    )


def build_days_model() -> None:
    print()
    print(
        "成約日数モデルの構築を開始します。"
    )

    preprocess_days_data()
    train_days_model()

    print()
    print(
        "成約日数モデルの構築が完了しました。"
    )


def show_menu() -> None:
    print()
    print(
        "東京都中古マンション"
        " AI予測システム"
    )

    print()
    print("利用可能なコマンド")
    print()
    print(
        "collect"
        "          国交省APIから価格データ取得"
    )
    print(
        "preprocess"
        "       価格学習データ前処理"
    )
    print(
        "train"
        "            価格モデル学習"
    )
    print(
        "predict"
        "          CSV価格予測"
    )
    print(
        "excel"
        "            Excel価格予測"
    )
    print(
        "price-full"
        "       価格モデル一括構築"
    )
    print(
        "days-template"
        "    成約履歴Excel作成"
    )
    print(
        "days-preprocess"
        "  成約日数学習データ作成"
    )
    print(
        "days-train"
        "       成約日数モデル学習"
    )
    print(
        "days-full"
        "        成約日数モデル一括構築"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "東京都中古マンション"
            "価格・成約日数予測システム"
        )
    )

    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        choices=[
            "collect",
            "preprocess",
            "train",
            "predict",
            "excel",
            "price-full",
            "days-template",
            "days-preprocess",
            "days-train",
            "days-full",
        ],
    )

    args = parser.parse_args()

    if args.command is None:
        show_menu()
        return

    if args.command == "collect":
        collect_price_data()

    elif args.command == "preprocess":
        preprocess_price_data()

    elif args.command == "train":
        train_price_model()

    elif args.command == "predict":
        predict_price_csv()

    elif args.command == "excel":
        predict_price_excel()

    elif args.command == "price-full":
        build_price_model()

    elif args.command == "days-template":
        create_days_template()

    elif args.command == "days-preprocess":
        preprocess_days_data()

    elif args.command == "days-train":
        train_days_model()

    elif args.command == "days-full":
        build_days_model()


if __name__ == "__main__":
    main()