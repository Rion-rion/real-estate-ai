from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_FILE = PROJECT_ROOT / 'data' / 'processed' / 'days_training.csv'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODEL_FILE = MODELS_DIR / 'days_model.cbm'
METRICS_FILE = OUTPUT_DIR / 'days_model_metrics.json'
PREDICTIONS_FILE = OUTPUT_DIR / 'days_test_predictions.csv'
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / 'days_feature_importance.csv'
TARGET_COLUMN = 'days_to_contract'
MINIMUM_ROWS = 100
FEATURE_COLUMNS = [
    'city',
    'district_name',
    'station_name',
    'station_line',
    'station_latitude',
    'station_longitude',
    'area_m2',
    'floor_plan',
    'building_age',
    'structure',
    'renovation',
    'use',
    'city_planning',
    'coverage_ratio',
    'floor_area_ratio',
    'listing_year',
    'listing_month',
    'listing_quarter',
    'asking_price',
    'asking_price_per_m2',
    'ai_estimated_price',
    'ai_price_per_m2',
    'price_gap_ratio',
]
CATEGORICAL_COLUMNS = [
    'city',
    'district_name',
    'station_name',
    'station_line',
    'floor_plan',
    'structure',
    'renovation',
    'use',
    'city_planning',
]

def print_header(text: str) -> None:
    print()
    print('=' * 60)
    print(text)
    print('=' * 60)
    print()

def show_waiting_message(current_rows: int=0) -> None:
    print_header('成約日数AI: 教師データ不足')
    print(f'現在の学習データ: {current_rows:,}件')
    print(f'最低必要件数: {MINIMUM_ROWS:,}件')
    print()
    print('少数データから不安定な精度を算出しないため、モデル学習は実行しません。')
    print()
    print('実成約履歴を追加した後、')
    print('python main.py days-full')
    print('を実行してください。')

def load_data():
    if not TRAINING_FILE.exists():
        print_header('成約日数AI: 教師データ待ち')
        print('days_training.csv がまだ作成されていません。')
        print()
        print('先に実成約履歴を入力し、')
        print('python main.py days-preprocess')
        print('を実行してください。')
        return None
    df = pd.read_csv(TRAINING_FILE, low_memory=False)
    df = df.dropna(how='all').reset_index(drop=True)
    print(f'成約日数学習データ: {len(df):,}件')
    if len(df) < MINIMUM_ROWS:
        show_waiting_message(len(df))
        return None
    return df

def validate_schema(df: pd.DataFrame) -> None:
    required_columns = ['listing_date', TARGET_COLUMN, *FEATURE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            '\n成約日数学習データに必要な列が不足しています。\n\n'
            '不足列:\n- '
            + '\n- '.join(missing_columns)
            + '\n\npreprocessing_days.py を最新版で再実行してください。'
        )

def prepare_data(df: pd.DataFrame):
    df = df.copy()
    validate_schema(df)
    df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
    if df['listing_date'].isna().any():
        raise ValueError('listing_date に不正な値があります。')
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].fillna('不明').astype(str)
    numeric_features = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]
    for column in numeric_features:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    before = len(df)
    df = df[df[TARGET_COLUMN].notna()].copy()
    df = df[df[TARGET_COLUMN].between(1, 1000)].copy()
    removed = before - len(df)
    if removed:
        print()
        print(f'不正な成約日数を除外: {removed:,}件')
    if len(df) < MINIMUM_ROWS:
        show_waiting_message(len(df))
        return None
    df = df.sort_values('listing_date').reset_index(drop=True)
    return df

def split_data(df: pd.DataFrame):
    train_end = int(len(df) * 0.7)
    validation_end = int(len(df) * 0.85)
    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()
    if len(train) == 0 or len(validation) == 0 or len(test) == 0:
        raise RuntimeError('Train / Validation / Test のいずれかが0件です。')
    print_header('時間順データ分割')
    print(f'Train      : {len(train):,}件 (70%)')
    print(f'Validation : {len(validation):,}件 (15%)')
    print(f'Final Test : {len(test):,}件 (15%)')
    print()
    print('Train期間:')
    print(
        f"{train['listing_date'].min().date()} ～ "
        f"{train['listing_date'].max().date()}"
    )
    print()
    print('Validation期間:')
    print(
        f"{validation['listing_date'].min().date()} ～ "
        f"{validation['listing_date'].max().date()}"
    )
    print()
    print('Final Test期間:')
    print(
        f"{test['listing_date'].min().date()} ～ "
        f"{test['listing_date'].max().date()}"
    )
    return (train, validation, test)

def create_model(iterations: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.03,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        l2_leaf_reg=8,
        verbose=100,
        allow_writing_files=False,
    )

def train_selection_model(train: pd.DataFrame, validation: pd.DataFrame):
    print_header('成約日数AI モデル選択学習')
    model = create_model(iterations=2500)
    X_train = train[FEATURE_COLUMNS]
    X_validation = validation[FEATURE_COLUMNS]
    y_train = np.log1p(train[TARGET_COLUMN])
    y_validation = np.log1p(validation[TARGET_COLUMN])
    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_COLUMNS,
        eval_set=(X_validation, y_validation),
        use_best_model=True,
        early_stopping_rounds=150,
    )
    best_iteration = model.get_best_iteration()
    if best_iteration is None or best_iteration < 0:
        best_iterations = 2500
    else:
        best_iterations = int(best_iteration) + 1
    print()
    print(f'Best iteration: {best_iterations:,}')
    return (model, best_iterations)

def predict_days(model: CatBoostRegressor, df: pd.DataFrame):
    predicted_log = model.predict(df[FEATURE_COLUMNS])
    predicted_days = np.expm1(predicted_log)
    predicted_days = np.maximum(predicted_days, 1)
    return predicted_days

def calculate_mape(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)

def calculate_smape(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual) + np.abs(predicted)
    valid = denominator > 0
    if not valid.any():
        return 0.0
    return float(np.mean(2 * np.abs(predicted[valid] - actual[valid]) / denominator[valid]) * 100)

def evaluate(model: CatBoostRegressor, df: pd.DataFrame, name: str):
    predicted = predict_days(model, df)
    actual = df[TARGET_COLUMN].to_numpy(dtype=float)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    median_ae = median_absolute_error(actual, predicted)
    mape = calculate_mape(actual, predicted)
    smape = calculate_smape(actual, predicted)
    r2 = r2_score(actual, predicted)
    print_header(f'{name} 評価')
    print(f'MAE       : {mae:.2f}日')
    print(f'RMSE      : {rmse:.2f}日')
    print(f'Median AE : {median_ae:.2f}日')
    print(f'MAPE      : {mape:.2f}%')
    print(f'SMAPE     : {smape:.2f}%')
    print(f'R²        : {r2:.4f}')
    metrics = {
        'mae_days': float(mae),
        'rmse_days': float(rmse),
        'median_absolute_error_days': float(median_ae),
        'mape_percent': float(mape),
        'smape_percent': float(smape),
        'r2': float(r2),
    }
    return (metrics, predicted)

def train_final_model(train: pd.DataFrame, validation: pd.DataFrame, best_iterations: int):
    print_header('最終モデル再学習')
    development = pd.concat([train, validation], ignore_index=True)
    development = development.sort_values('listing_date').reset_index(drop=True)
    print(f'Train + Validation: {len(development):,}件')
    print(f'Iterations: {best_iterations:,}')
    model = create_model(iterations=best_iterations)
    X = development[FEATURE_COLUMNS]
    y = np.log1p(development[TARGET_COLUMN])
    model.fit(X, y, cat_features=CATEGORICAL_COLUMNS)
    return (model, development)

def create_date_range(df: pd.DataFrame):
    return {
        'start': str(df['listing_date'].min().date()),
        'end': str(df['listing_date'].max().date()),
    }

def save_results(
    final_model: CatBoostRegressor,
    best_iterations: int,
    all_rows: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    development: pd.DataFrame,
    test: pd.DataFrame,
    validation_metrics,
    test_metrics,
    test_predictions,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(MODEL_FILE))
    metrics = {
        'version': '1.0',
        'status': 'trained',
        'model': 'CatBoostRegressor',
        'target': 'log1p(days_to_contract)',
        'minimum_training_rows': MINIMUM_ROWS,
        'total_rows': int(all_rows),
        'best_iterations': int(best_iterations),
        'features': FEATURE_COLUMNS,
        'categorical_features': CATEGORICAL_COLUMNS,
        'split': {
            'method': 'chronological_70_15_15',
            'train_rows': int(len(train)),
            'validation_rows': int(len(validation)),
            'development_rows': int(len(development)),
            'test_rows': int(len(test)),
            'train_period': create_date_range(train),
            'validation_period': create_date_range(validation),
            'test_period': create_date_range(test),
        },
        'validation_metrics': validation_metrics,
        'test_metrics': test_metrics,
        'price_ai_dependency': {
            'used': True,
            'features': [
                'ai_estimated_price',
                'ai_price_per_m2',
                'price_gap_ratio',
            ],
        },
        'notes': [
            '販売開始日時点で取得可能な特徴量のみを成約日数AIへ使用',
            'contract_date と contract_price は学習特徴量に使用しない',
            'Final Testはモデル選択に使用しない',
        ],
    }
    with open(METRICS_FILE, 'w', encoding='utf-8') as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    result = test.copy().reset_index(drop=True)
    result['predicted_days_to_contract'] = np.round(test_predictions).astype(int)
    result['prediction_error_days'] = result['predicted_days_to_contract'] - result[TARGET_COLUMN]
    result['absolute_error_days'] = result['prediction_error_days'].abs()
    result['absolute_percentage_error'] = (
        result['absolute_error_days']
        / result[TARGET_COLUMN]
        * 100
    )
    result.to_csv(PREDICTIONS_FILE, index=False, encoding='utf-8-sig')
    importance = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'importance': final_model.get_feature_importance(),
    })
    importance = importance.sort_values('importance', ascending=False).reset_index(drop=True)
    importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False, encoding='utf-8-sig')
    print_header('重要特徴量 TOP10')
    print(importance.head(10).to_string(index=False))
    print()
    print(f'モデル: {MODEL_FILE}')
    print(f'評価情報: {METRICS_FILE}')
    print(f'Final Test予測: {PREDICTIONS_FILE}')
    print(f'特徴量重要度: {FEATURE_IMPORTANCE_FILE}')

def main() -> None:
    print_header('東京都中古マンション 成約日数予測AI Ver.1')
    df = load_data()
    if df is None:
        return
    df = prepare_data(df)
    if df is None:
        return
    all_rows = len(df)
    train, validation, test = split_data(df)
    selection_model, best_iterations = train_selection_model(train, validation)
    validation_metrics, _ = evaluate(selection_model, validation, 'Validation')
    final_model, development = train_final_model(train, validation, best_iterations)
    test_metrics, test_predictions = evaluate(final_model, test, 'Final Test')
    save_results(
        final_model=final_model,
        best_iterations=best_iterations,
        all_rows=all_rows,
        train=train,
        validation=validation,
        development=development,
        test=test,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        test_predictions=test_predictions,
    )
    print_header('成約日数予測AI 学習完了')
if __name__ == '__main__':
    main()
