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
    print("-" * 50)

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


def collect() -> None:
    run_script(
        "collect_mlit.py"
    )


def preprocess() -> None:
    run_script(
        "preprocessing.py"
    )


def train_price() -> None:
    run_script(
        "train_price.py"
    )


def predict() -> None:
    run_script(
        "predict.py"
    )


def full_training() -> None:
    print()
    print(
        "東京都中古マンション"
        "価格予測モデルを構築します。"
    )

    collect()
    preprocess()
    train_price()

    print()
    print(
        "価格予測モデルの構築が完了しました。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "東京都中古マンション"
            "価格予測システム"
        )
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="predict",
        choices=[
            "collect",
            "preprocess",
            "train",
            "predict",
            "full",
        ],
        help=(
            "実行処理を指定します。"
        ),
    )

    args = parser.parse_args()

    if args.command == "collect":
        collect()

    elif args.command == "preprocess":
        preprocess()

    elif args.command == "train":
        train_price()

    elif args.command == "predict":
        predict()

    elif args.command == "full":
        full_training()


if __name__ == "__main__":
    main()