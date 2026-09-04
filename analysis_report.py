from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parent

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "output"
    / "price_test_predictions.csv"
)

FEATURE_IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "output"
    / "price_feature_importance.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "output"
    / "analysis"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "analysis_summary.csv"
)

CITY_METRICS_FILE = (
    REPORT_DIR
    / "city_metrics.csv"
)

PRICE_BAND_METRICS_FILE = (
    REPORT_DIR
    / "price_band_metrics.csv"
)


FEATURE_NAME_JP = {
    "city": "市区町村",
    "city_code": "市区町村コード",
    "district_name": "地区名",
    "district_code": "地区コード",
    "area_m2": "専有面積",
    "floor_plan": "間取り",
    "build_year": "建築年",
    "building_age": "築年数",
    "structure": "建物構造",
    "renovation": "改装状況",
    "use": "用途",
    "city_planning": "都市計画",
    "coverage_ratio": "建ぺい率",
    "floor_area_ratio": "容積率",
    "transaction_year": "取引年",
    "transaction_quarter": "取引四半期",
    "market_year_index": "市場年指数",
    "total_floor_area": "延床面積",
    "unit_area_ratio": "専有面積比率",
}


def setup_japanese_font() -> None:
    preferred_fonts = [
        "Yu Gothic",
        "Yu Gothic UI",
        "Meiryo",
        "MS Gothic",
        "Noto Sans CJK JP",
    ]

    installed_fonts = {
        font.name
        for font in font_manager.fontManager.ttflist
    }

    for font_name in preferred_fonts:

        if font_name in installed_fonts:

            plt.rcParams[
                "font.family"
            ] = font_name

            print(
                f"日本語フォント: "
                f"{font_name}"
            )

            break

    plt.rcParams[
        "axes.unicode_minus"
    ] = False


def yen_to_man_yen(
    value,
    position=None,
):
    return (
        f"{value / 10_000:,.0f}"
    )


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            "テスト予測結果が見つかりません。\n"
            f"{PREDICTIONS_FILE}"
        )

    df = pd.read_csv(
        PREDICTIONS_FILE,
        low_memory=False,
    )

    required_columns = [
        "contract_price",
        "predicted_contract_price",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise KeyError(
            "必要なカラムがありません: "
            + ", ".join(
                missing_columns
            )
        )

    df[
        "contract_price"
    ] = pd.to_numeric(
        df[
            "contract_price"
        ],
        errors="coerce",
    )

    df[
        "predicted_contract_price"
    ] = pd.to_numeric(
        df[
            "predicted_contract_price"
        ],
        errors="coerce",
    )

    df = df[
        df["contract_price"].notna()
        & df[
            "predicted_contract_price"
        ].notna()
    ].copy()

    df = df[
        (df["contract_price"] > 0)
        & (
            df[
                "predicted_contract_price"
            ]
            > 0
        )
    ].copy()

    df[
        "prediction_error"
    ] = (
        df[
            "predicted_contract_price"
        ]
        - df[
            "contract_price"
        ]
    )

    df[
        "absolute_error"
    ] = (
        df[
            "prediction_error"
        ]
        .abs()
    )

    df[
        "percentage_error"
    ] = (
        df[
            "absolute_error"
        ]
        / df[
            "contract_price"
        ]
        * 100
    )

    print(
        f"テストデータ: "
        f"{len(df):,}件"
    )

    return df


def calculate_basic_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    actual = (
        df[
            "contract_price"
        ]
        .to_numpy()
    )

    predicted = (
        df[
            "predicted_contract_price"
        ]
        .to_numpy()
    )

    mae = np.mean(
        np.abs(
            actual
            - predicted
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                actual
                - predicted
            )
            ** 2
        )
    )

    mape = (
        np.mean(
            np.abs(
                (
                    actual
                    - predicted
                )
                / actual
            )
        )
        * 100
    )

    denominator = np.sum(
        (
            actual
            - np.mean(actual)
        )
        ** 2
    )

    r2 = (
        1
        - np.sum(
            (
                actual
                - predicted
            )
            ** 2
        )
        / denominator
    )

    median_absolute_error = (
        np.median(
            df[
                "absolute_error"
            ]
        )
    )

    median_percentage_error = (
        np.median(
            df[
                "percentage_error"
            ]
        )
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "MAE",
                "RMSE",
                "MAPE",
                "R2",
                "Median Absolute Error",
                "Median Percentage Error",
                "Test Rows",
            ],
            "value": [
                mae,
                rmse,
                mape,
                r2,
                median_absolute_error,
                median_percentage_error,
                len(df),
            ],
        }
    )

    return summary


def plot_actual_vs_predicted(
    df: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.scatter(
        df[
            "contract_price"
        ],
        df[
            "predicted_contract_price"
        ],
        alpha=0.35,
        s=15,
    )

    minimum = min(
        df[
            "contract_price"
        ].min(),
        df[
            "predicted_contract_price"
        ].min(),
    )

    maximum = max(
        df[
            "contract_price"
        ].max(),
        df[
            "predicted_contract_price"
        ].max(),
    )

    ax.plot(
        [
            minimum,
            maximum,
        ],
        [
            minimum,
            maximum,
        ],
        linestyle="--",
        label="完全一致ライン",
    )

    formatter = FuncFormatter(
        yen_to_man_yen
    )

    ax.xaxis.set_major_formatter(
        formatter
    )

    ax.yaxis.set_major_formatter(
        formatter
    )

    ax.set_title(
        "実際の成約価格 vs AI予測価格"
    )

    ax.set_xlabel(
        "実際の成約価格（万円）"
    )

    ax.set_ylabel(
        "AI予測成約価格（万円）"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR
        / "actual_vs_predicted.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_error_distribution(
    df: pd.DataFrame,
) -> None:

    upper_limit = (
        df[
            "percentage_error"
        ]
        .quantile(
            0.99
        )
    )

    plot_data = (
        df[
            "percentage_error"
        ]
        .clip(
            upper=upper_limit
        )
    )

    median_value = (
        df[
            "percentage_error"
        ]
        .median()
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.hist(
        plot_data,
        bins=40,
    )

    ax.axvline(
        median_value,
        linestyle="--",
        label=(
            "中央値 "
            f"{median_value:.1f}%"
        ),
    )

    ax.set_title(
        "AI予測誤差率の分布"
    )

    ax.set_xlabel(
        "絶対誤差率（%）"
    )

    ax.set_ylabel(
        "物件数"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR
        / "error_distribution.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_city_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if "city" not in df.columns:
        return pd.DataFrame()

    city_metrics = (
        df.groupby(
            "city",
            dropna=False,
        )
        .agg(
            count=(
                "contract_price",
                "size",
            ),
            actual_average=(
                "contract_price",
                "mean",
            ),
            predicted_average=(
                "predicted_contract_price",
                "mean",
            ),
            mae=(
                "absolute_error",
                "mean",
            ),
            mape=(
                "percentage_error",
                "mean",
            ),
        )
        .reset_index()
    )

    return (
        city_metrics
        .sort_values(
            "mape"
        )
        .reset_index(
            drop=True
        )
    )


def plot_city_mape(
    city_metrics: pd.DataFrame,
) -> None:

    if city_metrics.empty:
        return

    reliable = (
        city_metrics[
            city_metrics[
                "count"
            ]
            >= 20
        ]
        .copy()
    )

    if reliable.empty:

        reliable = (
            city_metrics
            .sort_values(
                "count",
                ascending=False,
            )
            .head(20)
        )

    reliable = (
        reliable
        .sort_values(
            "mape",
            ascending=False,
        )
        .head(20)
        .sort_values(
            "mape"
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.barh(
        reliable[
            "city"
        ],
        reliable[
            "mape"
        ],
    )

    ax.set_title(
        "市区町村別 AI予測誤差率"
    )

    ax.set_xlabel(
        "MAPE（%）"
    )

    ax.set_ylabel(
        "市区町村"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR
        / "city_mape.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_price_band_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    bins = [
        0,
        30_000_000,
        50_000_000,
        70_000_000,
        100_000_000,
        150_000_000,
        np.inf,
    ]

    labels = [
        "3,000万円未満",
        "3,000〜5,000万円",
        "5,000〜7,000万円",
        "7,000万円〜1億円",
        "1億〜1億5,000万円",
        "1億5,000万円以上",
    ]

    work_df = df.copy()

    work_df[
        "price_band"
    ] = pd.cut(
        work_df[
            "contract_price"
        ],
        bins=bins,
        labels=labels,
        right=False,
    )

    metrics = (
        work_df.groupby(
            "price_band",
            observed=True,
        )
        .agg(
            count=(
                "contract_price",
                "size",
            ),
            actual_average=(
                "contract_price",
                "mean",
            ),
            mae=(
                "absolute_error",
                "mean",
            ),
            mape=(
                "percentage_error",
                "mean",
            ),
        )
        .reset_index()
    )

    return metrics


def plot_price_band_mape(
    metrics: pd.DataFrame,
) -> None:

    if metrics.empty:
        return

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        metrics[
            "price_band"
        ].astype(str),
        metrics[
            "mape"
        ],
    )

    ax.set_title(
        "成約価格帯別 AI予測誤差率"
    )

    ax.set_xlabel(
        "成約価格帯"
    )

    ax.set_ylabel(
        "MAPE（%）"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR
        / "price_band_mape.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_feature_importance() -> None:

    if not FEATURE_IMPORTANCE_FILE.exists():

        print(
            "特徴量重要度CSVがないため"
            "グラフをスキップします。"
        )

        return

    df = pd.read_csv(
        FEATURE_IMPORTANCE_FILE
    )

    if (
        "feature"
        not in df.columns
        or
        "importance"
        not in df.columns
    ):
        return

    df[
        "feature_jp"
    ] = (
        df[
            "feature"
        ]
        .map(
            FEATURE_NAME_JP
        )
        .fillna(
            df[
                "feature"
            ]
        )
    )

    df = (
        df
        .sort_values(
            "importance",
            ascending=False,
        )
        .head(10)
        .sort_values(
            "importance"
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        df[
            "feature_jp"
        ],
        df[
            "importance"
        ],
    )

    ax.set_title(
        "価格予測に重要な特徴量 TOP10"
    )

    ax.set_xlabel(
        "重要度"
    )

    ax.set_ylabel(
        "特徴量"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR
        / "feature_importance.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_error_by_actual_price(
    df: pd.DataFrame,
) -> None:

    upper_error = (
        df[
            "percentage_error"
        ]
        .quantile(
            0.99
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.scatter(
        df[
            "contract_price"
        ],
        df[
            "percentage_error"
        ],
        alpha=0.3,
        s=15,
    )

    formatter = FuncFormatter(
        yen_to_man_yen
    )

    ax.xaxis.set_major_formatter(
        formatter
    )

    ax.set_ylim(
        0,
        upper_error,
    )

    ax.set_title(
        "成約価格とAI予測誤差率の関係"
    )

    ax.set_xlabel(
        "実際の成約価格（万円）"
    )

    ax.set_ylabel(
        "絶対誤差率（%）"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR
        / "error_by_price.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def show_summary(
    summary: pd.DataFrame,
) -> None:

    values = dict(
        zip(
            summary[
                "metric"
            ],
            summary[
                "value"
            ],
        )
    )

    print()
    print(
        "価格モデル分析結果"
    )

    print(
        f"MAE : "
        f"{values['MAE']:,.0f}円"
    )

    print(
        f"RMSE: "
        f"{values['RMSE']:,.0f}円"
    )

    print(
        f"MAPE: "
        f"{values['MAPE']:.2f}%"
    )

    print(
        f"R²  : "
        f"{values['R2']:.4f}"
    )

    print(
        "誤差率中央値: "
        f"{values['Median Percentage Error']:.2f}%"
    )

    print(
        "テスト件数: "
        f"{int(values['Test Rows']):,}件"
    )


def main() -> None:

    print(
        "不動産価格AI "
        "分析レポート作成"
    )

    setup_japanese_font()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_predictions()

    summary = (
        calculate_basic_metrics(
            df
        )
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    city_metrics = (
        create_city_metrics(
            df
        )
    )

    if not city_metrics.empty:

        city_metrics.to_csv(
            CITY_METRICS_FILE,
            index=False,
            encoding="utf-8-sig",
        )

    price_band_metrics = (
        create_price_band_metrics(
            df
        )
    )

    price_band_metrics.to_csv(
        PRICE_BAND_METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "グラフを作成しています..."
    )

    plot_actual_vs_predicted(
        df
    )

    plot_error_distribution(
        df
    )

    plot_city_mape(
        city_metrics
    )

    plot_price_band_mape(
        price_band_metrics
    )

    plot_feature_importance()

    plot_error_by_actual_price(
        df
    )

    show_summary(
        summary
    )

    print()
    print(
        "分析レポート作成完了"
    )

    print(
        f"保存先: "
        f"{REPORT_DIR}"
    )

    print()
    print(
        "生成ファイル:"
    )

    print(
        "- actual_vs_predicted.png"
    )

    print(
        "- error_distribution.png"
    )

    print(
        "- city_mape.png"
    )

    print(
        "- price_band_mape.png"
    )

    print(
        "- feature_importance.png"
    )

    print(
        "- error_by_price.png"
    )


if __name__ == "__main__":
    main()