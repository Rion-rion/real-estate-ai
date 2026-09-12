from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
PREDICTIONS_FILE = OUTPUT_DIR / 'price_test_predictions.csv'
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / 'price_feature_importance.csv'
METRICS_FILE = OUTPUT_DIR / 'price_model_metrics.json'
REPORT_DIR = OUTPUT_DIR / 'analysis'
SUMMARY_FILE = REPORT_DIR / 'analysis_summary.csv'
CITY_METRICS_FILE = REPORT_DIR / 'city_metrics.csv'
PRICE_BAND_METRICS_FILE = REPORT_DIR / 'price_band_metrics.csv'
STATION_METRICS_FILE = REPORT_DIR / 'station_metrics.csv'
MODEL_COMPARISON_FILE = REPORT_DIR / 'model_comparison.csv'
FEATURE_NAME_JP = {
    'city': '市区町村',
    'city_code': '市区町村コード',
    'district_name': '地区名',
    'district_code': '地区コード',
    'station_name': '最寄駅',
    'station_code': '駅コード',
    'station_group_code': '駅グループコード',
    'station_company': '鉄道事業者',
    'station_line': '路線',
    'station_latitude': '最寄駅緯度',
    'station_longitude': '最寄駅経度',
    'station_geometry_match_m': '駅GISマッチ距離',
    'area_m2': '専有面積',
    'floor_plan': '間取り',
    'build_year': '建築年',
    'building_age': '築年数',
    'structure': '建物構造',
    'renovation': '改装状況',
    'use': '用途',
    'city_planning': '都市計画',
    'coverage_ratio': '建ぺい率',
    'floor_area_ratio': '容積率',
    'transaction_year': '取引年',
    'transaction_quarter': '取引四半期',
    'market_year_index': '市場年指数',
    'total_floor_area': '延床面積',
    'unit_area_ratio': '専有面積比率',
}


def setup_japanese_font():
    preferred_fonts = ['Yu Gothic', 'Yu Gothic UI', 'Meiryo', 'MS Gothic', 'Noto Sans CJK JP']
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in installed_fonts:
            plt.rcParams['font.family'] = font_name
            print(f'日本語フォント: {font_name}')
            break
    plt.rcParams['axes.unicode_minus'] = False


def yen_to_man_yen(value, position=None):
    return f'{value / 10_000:,.0f}'


def load_model_metrics():
    if not METRICS_FILE.exists():
        print('price_model_metrics.json が見つからないため、モデル比較をスキップします。')
        return {}
    with open(METRICS_FILE, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_predictions():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f'テスト予測結果が見つかりません。\n{PREDICTIONS_FILE}')
    df = pd.read_csv(PREDICTIONS_FILE, low_memory=False)
    required_columns = ['contract_price', 'predicted_contract_price']
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError('必要なカラムがありません: ' + ', '.join(missing_columns))
    df['contract_price'] = pd.to_numeric(df['contract_price'], errors='coerce')
    df['predicted_contract_price'] = pd.to_numeric(
        df['predicted_contract_price'], errors='coerce'
    )
    df = df[
        df['contract_price'].notna() & df['predicted_contract_price'].notna()
    ].copy()
    df = df[(df['contract_price'] > 0) & (df['predicted_contract_price'] > 0)].copy()
    df['prediction_error'] = df['predicted_contract_price'] - df['contract_price']
    df['absolute_error'] = df['prediction_error'].abs()
    df['percentage_error'] = df['absolute_error'] / df['contract_price'] * 100
    df['signed_percentage_error'] = df['prediction_error'] / df['contract_price'] * 100
    print(f'テストデータ: {len(df):,}件')
    return df


def calculate_basic_metrics(df):
    actual = df['contract_price'].to_numpy()
    predicted = df['predicted_contract_price'].to_numpy()
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    if denominator == 0:
        r2 = np.nan
    else:
        r2 = 1 - np.sum((actual - predicted) ** 2) / denominator
    median_absolute_error = np.median(df['absolute_error'])
    median_percentage_error = np.median(df['percentage_error'])
    mean_signed_error = df['prediction_error'].mean()
    mean_signed_percentage_error = df['signed_percentage_error'].mean()
    summary = pd.DataFrame(
        {
            'metric': [
                'MAE',
                'RMSE',
                'MAPE',
                'R2',
                'Median Absolute Error',
                'Median Percentage Error',
                'Mean Signed Error',
                'Mean Signed Percentage Error',
                'Test Rows',
            ],
            'value': [
                mae,
                rmse,
                mape,
                r2,
                median_absolute_error,
                median_percentage_error,
                mean_signed_error,
                mean_signed_percentage_error,
                len(df),
            ],
        }
    )
    return summary


def create_city_metrics(df):
    if 'city' not in df.columns:
        return pd.DataFrame()
    metrics = (
        df.groupby('city', dropna=False)
        .agg(
            count=('contract_price', 'size'),
            actual_average=('contract_price', 'mean'),
            predicted_average=('predicted_contract_price', 'mean'),
            mae=('absolute_error', 'mean'),
            mape=('percentage_error', 'mean'),
            signed_error_percent=('signed_percentage_error', 'mean'),
        )
        .reset_index()
    )
    return metrics.sort_values('mape').reset_index(drop=True)


def create_station_metrics(df):
    if 'station_name' not in df.columns:
        return pd.DataFrame()
    work_df = df.copy()
    work_df['station_name'] = work_df['station_name'].astype('string').fillna('不明')
    group_columns = ['station_name']
    if 'station_line' in work_df.columns:
        work_df['station_line'] = work_df['station_line'].astype('string').fillna('不明')
        group_columns.append('station_line')
    metrics = (
        work_df.groupby(group_columns, dropna=False)
        .agg(
            count=('contract_price', 'size'),
            actual_average=('contract_price', 'mean'),
            predicted_average=('predicted_contract_price', 'mean'),
            mae=('absolute_error', 'mean'),
            mape=('percentage_error', 'mean'),
            signed_error_percent=('signed_percentage_error', 'mean'),
        )
        .reset_index()
    )
    return metrics.sort_values(
        ['count', 'mape'], ascending=[False, True]
    ).reset_index(drop=True)


def create_price_band_metrics(df):
    bins = [0, 30_000_000, 50_000_000, 70_000_000, 100_000_000, 150_000_000, np.inf]
    labels = [
        '3,000万円未満',
        '3,000〜5,000万円',
        '5,000〜7,000万円',
        '7,000万円〜1億円',
        '1億〜1億5,000万円',
        '1億5,000万円以上',
    ]
    work_df = df.copy()
    work_df['price_band'] = pd.cut(work_df['contract_price'], bins=bins, labels=labels, right=False)
    metrics = (
        work_df.groupby('price_band', observed=True)
        .agg(
            count=('contract_price', 'size'),
            actual_average=('contract_price', 'mean'),
            predicted_average=('predicted_contract_price', 'mean'),
            mae=('absolute_error', 'mean'),
            mape=('percentage_error', 'mean'),
            signed_error_percent=('signed_percentage_error', 'mean'),
        )
        .reset_index()
    )
    return metrics


def create_model_comparison(model_info):
    if not model_info:
        return pd.DataFrame()
    test_metrics = model_info.get('test_metrics', {})
    baseline_metrics = model_info.get('baseline_same_dataset_metrics', {})
    selected_name = model_info.get('selected_feature_set', 'Ver.5')
    rows = []
    if baseline_metrics:
        rows.append(
            {
                'model': 'baseline_current',
                'description': '駅特徴量なし',
                'mape': baseline_metrics.get('mape'),
                'r2': baseline_metrics.get('r2'),
                'mae': baseline_metrics.get('mae'),
                'rmse': baseline_metrics.get('rmse'),
            }
        )
    if test_metrics:
        rows.append(
            {
                'model': selected_name,
                'description': '駅特徴量あり',
                'mape': test_metrics.get('mape'),
                'r2': test_metrics.get('r2'),
                'mae': test_metrics.get('mae'),
                'rmse': test_metrics.get('rmse'),
            }
        )
    return pd.DataFrame(rows)


def plot_actual_vs_predicted(df):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(df['contract_price'], df['predicted_contract_price'], alpha=0.35, s=15)
    minimum = min(df['contract_price'].min(), df['predicted_contract_price'].min())
    maximum = max(df['contract_price'].max(), df['predicted_contract_price'].max())
    ax.plot([minimum, maximum], [minimum, maximum], linestyle='--', label='完全一致ライン')
    formatter = FuncFormatter(yen_to_man_yen)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_title('Ver.5 実際の成約価格 vs AI予測価格')
    ax.set_xlabel('実際の成約価格（万円）')
    ax.set_ylabel('AI予測成約価格（万円）')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'actual_vs_predicted.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_error_distribution(df):
    upper_limit = df['percentage_error'].quantile(0.99)
    plot_data = df['percentage_error'].clip(upper=upper_limit)
    median_value = df['percentage_error'].median()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(plot_data, bins=40)
    ax.axvline(median_value, linestyle='--', label=f'中央値 {median_value:.1f}%')
    ax.set_title('Ver.5 AI予測誤差率の分布')
    ax.set_xlabel('絶対誤差率（%）')
    ax.set_ylabel('物件数')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'error_distribution.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_error_by_price(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['contract_price'], df['percentage_error'], alpha=0.3, s=15)
    formatter = FuncFormatter(yen_to_man_yen)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_title('成約価格とAI予測誤差率')
    ax.set_xlabel('実際の成約価格（万円）')
    ax.set_ylabel('絶対誤差率（%）')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'error_by_price.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_city_mape(metrics):
    if metrics.empty:
        return
    reliable = metrics[metrics['count'] >= 20].copy()
    if reliable.empty:
        reliable = metrics.sort_values('count', ascending=False).head(20)
    reliable = reliable.sort_values('mape', ascending=False).head(20).sort_values('mape')
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(reliable['city'], reliable['mape'])
    ax.set_title('市区町村別 AI予測誤差率')
    ax.set_xlabel('MAPE（%）')
    ax.set_ylabel('市区町村')
    ax.grid(axis='x', alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'city_mape.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_station_mape(metrics):
    if metrics.empty:
        return
    reliable = metrics[metrics['count'] >= 20].copy()
    if reliable.empty:
        reliable = metrics.sort_values('count', ascending=False).head(20)
    reliable = reliable.sort_values('count', ascending=False).head(30)
    reliable = reliable.sort_values('mape', ascending=False).head(20).sort_values('mape')
    if 'station_line' in reliable.columns:
        labels = (
            reliable['station_name'].astype(str)
            + ' / '
            + reliable['station_line'].astype(str)
        )
    else:
        labels = reliable['station_name'].astype(str)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(labels, reliable['mape'])
    ax.set_title('主要駅別 AI予測誤差率')
    ax.set_xlabel('MAPE（%）')
    ax.set_ylabel('最寄駅 / 路線')
    ax.grid(axis='x', alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'station_mape.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_price_band_mape(metrics):
    if metrics.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(metrics['price_band'].astype(str), metrics['mape'])
    ax.set_title('成約価格帯別 AI予測誤差率')
    ax.set_xlabel('成約価格帯')
    ax.set_ylabel('MAPE（%）')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'price_band_mape.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_feature_importance():
    if not FEATURE_IMPORTANCE_FILE.exists():
        print('特徴量重要度CSVがないためグラフをスキップします。')
        return
    df = pd.read_csv(FEATURE_IMPORTANCE_FILE)
    required_columns = ['feature', 'importance']
    if not all((column in df.columns for column in required_columns)):
        print('特徴量重要度CSVの形式が想定と異なります。')
        return
    df['importance'] = pd.to_numeric(df['importance'], errors='coerce')
    df = df[df['importance'].notna()].copy()
    df['feature_jp'] = df['feature'].map(FEATURE_NAME_JP).fillna(df['feature'])
    top = df.sort_values('importance', ascending=False).head(15).sort_values('importance')
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top['feature_jp'], top['importance'])
    ax.set_title('Ver.5 CatBoost 特徴量重要度')
    ax.set_xlabel('重要度')
    ax.set_ylabel('特徴量')
    ax.grid(axis='x', alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'feature_importance.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_model_comparison(comparison):
    if comparison.empty:
        return
    if 'mape' not in comparison.columns:
        return
    work_df = comparison[comparison['mape'].notna()].copy()
    if work_df.empty:
        return
    labels = work_df['description'].fillna(work_df['model'])
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, work_df['mape'])
    ax.set_title('駅特徴量追加によるMAPE比較')
    ax.set_ylabel('MAPE（%）')
    ax.set_xlabel('モデル')
    ax.grid(axis='y', alpha=0.2)
    for bar, value in zip(bars, work_df['mape']):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{value:.2f}%',
            ha='center',
            va='bottom',
        )
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'model_comparison_mape.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def save_csv_files(
    summary,
    city_metrics,
    price_band_metrics,
    station_metrics,
    model_comparison,
):
    summary.to_csv(SUMMARY_FILE, index=False, encoding='utf-8-sig')
    if not city_metrics.empty:
        city_metrics.to_csv(CITY_METRICS_FILE, index=False, encoding='utf-8-sig')
    if not price_band_metrics.empty:
        price_band_metrics.to_csv(
            PRICE_BAND_METRICS_FILE,
            index=False,
            encoding='utf-8-sig',
        )
    if not station_metrics.empty:
        station_metrics.to_csv(STATION_METRICS_FILE, index=False, encoding='utf-8-sig')
    if not model_comparison.empty:
        model_comparison.to_csv(MODEL_COMPARISON_FILE, index=False, encoding='utf-8-sig')


def print_summary(summary, model_info, model_comparison):
    metrics = {row['metric']: row['value'] for _, row in summary.iterrows()}
    print()
    print('=' * 60)
    print('Ver.5 AI価格モデル 分析結果')
    print('=' * 60)
    version = model_info.get('version') if model_info else None
    feature_set = model_info.get('selected_feature_set') if model_info else None
    if version is not None:
        print(f'Version: {version}')
    if feature_set:
        print(f'特徴量セット: {feature_set}')
    print()
    print(f"Test Rows: {int(metrics['Test Rows']):,}")
    print(f"MAE: {metrics['MAE']:,.0f}円")
    print(f"RMSE: {metrics['RMSE']:,.0f}円")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"R²: {metrics['R2']:.4f}")
    print(f"Median Absolute Error: {metrics['Median Absolute Error']:,.0f}円")
    print(f"Median Percentage Error: {metrics['Median Percentage Error']:.2f}%")
    if len(model_comparison) >= 2:
        baseline_row = model_comparison.iloc[0]
        station_row = model_comparison.iloc[-1]
        if pd.notna(baseline_row['mape']) and pd.notna(station_row['mape']):
            improvement = baseline_row['mape'] - station_row['mape']
            print()
            print('駅特徴量の効果')
            print(f"駅なし MAPE: {baseline_row['mape']:.2f}%")
            print(f"駅あり MAPE: {station_row['mape']:.2f}%")
            print(f'MAPE改善: {improvement:.2f}ポイント')
        if pd.notna(baseline_row['r2']) and pd.notna(station_row['r2']):
            r2_improvement = station_row['r2'] - baseline_row['r2']
            print(f'R²改善: +{r2_improvement:.4f}')
    print()
    print(f'分析結果保存先: {REPORT_DIR}')


def main():
    print('東京都中古マンション AI価格モデル分析レポート Ver.5')
    setup_japanese_font()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model_info = load_model_metrics()
    predictions = load_predictions()
    summary = calculate_basic_metrics(predictions)
    city_metrics = create_city_metrics(predictions)
    station_metrics = create_station_metrics(predictions)
    price_band_metrics = create_price_band_metrics(predictions)
    model_comparison = create_model_comparison(model_info)
    save_csv_files(
        summary,
        city_metrics,
        price_band_metrics,
        station_metrics,
        model_comparison,
    )
    plot_actual_vs_predicted(predictions)
    plot_error_distribution(predictions)
    plot_error_by_price(predictions)
    plot_city_mape(city_metrics)
    plot_station_mape(station_metrics)
    plot_price_band_mape(price_band_metrics)
    plot_feature_importance()
    plot_model_comparison(model_comparison)
    print_summary(summary, model_info, model_comparison)
    print()
    print('分析レポート生成完了')


if __name__ == '__main__':
    main()
