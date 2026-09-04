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
    print("=" * 60)
    print(f"{script_name} を実行します。")
    print("=" * 60)
    print()

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
    run_script("collect_mlit.py")


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


def create_days_template() -> None:
    run_script("create_days_template.py")


def preprocess_days_data() -> None:
    run_script("preprocessing_days.py")


def train_days_model() -> None:
    run_script("train_days.py")


def build_price_model() -> None:
    print()
    print("価格予測モデルの一括構築を開始します。")

    collect_price_data()
    preprocess_price_data()
    train_price_model()
    create_analysis_report()

    print()
    print("価格予測モデルの構築が完了しました。")


def build_days_model() -> None:
    print()
    print("成約日数モデルの一括構築を開始します。")

    preprocess_days_data()
    train_days_model()

    print()
    print("成約日数モデルの構築が完了しました。")


def build_all() -> None:
    print()
    print("不動産AIシステム全体の構築を開始します。")

    build_price_model()

    print()
    print("価格AI構築完了")
    print()

    try:
        build_days_model()

    except RuntimeError as error:
        print()
        print("成約日数モデルは構築されませんでした。")
        print(error)
        print()
        print(
            "実成約データが100件以上集まり次第、"
            "days-full を実行してください。"
        )

    print()
    print("処理を終了しました。")


def show_menu() -> None:
    print()
    print("=" * 60)
    print("東京都中古マンション AI予測システム")
    print("=" * 60)

    print()
    print("価格予測")
    print("  collect           国交省APIからデータ取得")
    print("  preprocess        価格データ前処理")
    print("  train             価格モデル学習")
    print("  predict           CSV価格予測")
    print("  excel             Excel価格予測")
    print("  report            価格モデル分析レポート")
    print("  price-full        価格AIを一括構築")

    print()
    print("成約日数予測")
    print("  days-template     成約履歴Excel作成")
    print("  days-preprocess   成約日数学習データ作成")
    print("  days-train        成約日数モデル学習")
    print("  days-full         成約日数AIを一括構築")

    print()
    print("全体")
    print("  all               全システム構築")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "東京都中古マンション "
            "成約価格・成約日数予測システム"
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
            "report",
            "price-full",
            "days-template",
            "days-preprocess",
            "days-train",
            "days-full",
            "all",
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

    elif args.command == "report":
        create_analysis_report()

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

    elif args.command == "all":
        build_all()


if __name__ == "__main__":
    main()