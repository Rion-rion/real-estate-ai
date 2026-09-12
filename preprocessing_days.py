from pathlib import Path
import pandas as pd
from predict import (
    enrich_station_features,
    load_model_info,
    predict_prices,
    prepare_input_data,
)
from predict_excel import auto_fill_station_name
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_ROOT / 'data' / 'input' / 'contract_history.xlsx'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_FILE = PROCESSED_DIR / 'days_training.csv'
INPUT_SHEET = '成約履歴'
REQUIRED_COLUMNS = [
    'property_id',
    'city',
    'district_name',
    'area_m2',
    'floor_plan',
    'building_age',
    'listing_date',
    'asking_price',
    'contract_date',
]
NUMERIC_COLUMNS = [
    'area_m2',
    'building_age',
    'asking_price',
    'contract_price',
    'coverage_ratio',
    'floor_area_ratio',
    'station_latitude',
    'station_longitude',
]
TEXT_COLUMNS = [
    'property_id',
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

def load_contract_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f'\n成約履歴Excelが見つかりません。\n'
            f'{INPUT_FILE}\n\n'
            '先に以下を実行してください。\n'
            'python main.py days-template'
        )
    try:
        df = pd.read_excel(INPUT_FILE, sheet_name=INPUT_SHEET)
    except ValueError as error:
        raise RuntimeError(f"\nExcel内に '{INPUT_SHEET}' シートが見つかりません。") from error
    df = df.dropna(how='all').reset_index(drop=True)
    if df.empty:
        # 過去の学習データを新しい空テンプレートで誤再利用しないよう削除する。
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()
        print_header('成約日数AI: 教師データ待ち')
        print('contract_history.xlsx の「成約履歴」シートに実成約データがありません。')
        print()
        print('販売開始日・成約日を含む実成約履歴を入力すると、学習データを自動生成できます。')
        print()
        print('現時点では成約日数モデルの学習・精度評価は行いません。')
        return None
    print(f'成約履歴読み込み: {len(df):,}件')
    return df

def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError('\n成約日数学習に必要な列が不足しています。\n\n不足列:\n- ' + '\n- '.join(missing))

def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in TEXT_COLUMNS:
        if column not in df.columns:
            continue
        df[column] = (
            df[column]
            .astype('string')
            .str.strip()
            .replace({
                '': pd.NA,
                'nan': pd.NA,
                'None': pd.NA,
                '<NA>': pd.NA,
            })
        )
    return df

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
    df['contract_date'] = pd.to_datetime(df['contract_date'], errors='coerce')
    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def validate_rows(df: pd.DataFrame) -> None:
    errors = []
    required_text = ['property_id', 'city', 'district_name', 'floor_plan']
    for index, row in df.iterrows():
        excel_row = index + 2
        for column in required_text:
            if pd.isna(row.get(column)):
                errors.append(f'Excel {excel_row}行目: {column} が空です。')
        area = row.get('area_m2')
        if pd.isna(area) or area <= 0:
            errors.append(f'Excel {excel_row}行目: area_m2 が不正です。')
        building_age = row.get('building_age')
        if pd.isna(building_age) or building_age < 0:
            errors.append(f'Excel {excel_row}行目: building_age が不正です。')
        asking_price = row.get('asking_price')
        if pd.isna(asking_price) or asking_price <= 0:
            errors.append(f'Excel {excel_row}行目: asking_price が不正です。')
        listing_date = row.get('listing_date')
        contract_date = row.get('contract_date')
        if pd.isna(listing_date):
            errors.append(f'Excel {excel_row}行目: listing_date が不正です。')
        if pd.isna(contract_date):
            errors.append(f'Excel {excel_row}行目: contract_date が不正です。')
        if (
            pd.notna(listing_date)
            and pd.notna(contract_date)
            and contract_date <= listing_date
        ):
            errors.append(f'Excel {excel_row}行目: contract_date はlisting_date より後の日付にしてください。')
    duplicate_mask = df['property_id'].notna() & df['property_id'].duplicated(keep=False)
    if duplicate_mask.any():
        duplicated_ids = df.loc[duplicate_mask, 'property_id'].astype(str).unique().tolist()
        errors.append('property_id が重複しています: ' + ', '.join(duplicated_ids))
    if errors:
        print()
        print('入力データエラー')
        for error in errors[:20]:
            print(f'- {error}')
        if len(errors) > 20:
            print(f'...ほか {len(errors) - 20:,}件')
        raise ValueError('\n成約履歴Excelを修正してください。')

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['days_to_contract'] = (df['contract_date'] - df['listing_date']).dt.days
    invalid_days = ~df['days_to_contract'].between(1, 1000)
    if invalid_days.any():
        rows = (df.index[invalid_days] + 2).tolist()
        raise ValueError(f'\n成約日数が1〜1000日の範囲外のデータがあります。\nExcel行: {rows[:20]}')
    df['listing_year'] = df['listing_date'].dt.year.astype(int)
    df['listing_month'] = df['listing_date'].dt.month.astype(int)
    df['listing_quarter'] = df['listing_date'].dt.quarter.astype(int)
    df['asking_price_per_m2'] = df['asking_price'] / df['area_m2']
    return df

def create_price_ai_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    price_model, model_info, price_features, price_categories = load_model_info()
    print()
    print('成約履歴へVer.5価格AIを適用します。')

    # 地区情報から代表駅を補完し、駅名から路線・座標を補完する。
    df = auto_fill_station_name(df)
    df = enrich_station_features(df, price_features)

    # 売出日を価格AIの時点特徴量として使用する。
    df['transaction_year'] = df['listing_year']
    df['transaction_quarter'] = df['listing_quarter']
    price_input = prepare_input_data(df, price_features, price_categories)
    predicted_unit_price, predicted_contract_price = predict_prices(
        price_model,
        price_input,
        price_features,
    )
    df['ai_price_per_m2'] = predicted_unit_price
    df['ai_estimated_price'] = predicted_contract_price
    df['price_gap_amount'] = df['asking_price'] - df['ai_estimated_price']
    df['price_gap_ratio'] = df['price_gap_amount'] / df['ai_estimated_price']
    df['price_gap_ratio_percent'] = df['price_gap_ratio'] * 100
    version = model_info.get('version', 'unknown')
    df['price_model_version'] = str(version)
    return df

def show_leakage_warning(df: pd.DataFrame) -> None:
    if (df['listing_year'] <= 2025).any():
        print()
        print('注意:')
        print('Ver.5価格AIは2021〜2025年の価格データで最終学習しています。')
        print()
        print(
            '2025年以前の履歴を利用して'
            '成約日数AIの過去精度を厳密に評価する場合は、'
        )
        print(
            '価格AI特徴量についても時間順OOF予測を使用することで、'
            'より厳密にデータリークを防止できます。'
        )

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        'property_id',
        'city',
        'district_name',
        'station_name',
        'station_line',
        'station_latitude',
        'station_longitude',
        'station_lookup_status',
        'area_m2',
        'floor_plan',
        'building_age',
        'structure',
        'renovation',
        'use',
        'city_planning',
        'coverage_ratio',
        'floor_area_ratio',
        'listing_date',
        'listing_year',
        'listing_month',
        'listing_quarter',
        'asking_price',
        'asking_price_per_m2',
        'ai_price_per_m2',
        'ai_estimated_price',
        'price_gap_amount',
        'price_gap_ratio',
        'price_gap_ratio_percent',
        'price_model_version',
        # 教師データ・評価確認用。新規予測時の特徴量には使用しない。
        'contract_date',
        'contract_price',
        'days_to_contract',
    ]
    available = [column for column in columns if column in df.columns]
    return df[available].copy()

def save_data(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print_header('成約日数学習データ作成完了')
    print(f'件数: {len(df):,}件')
    print(f'保存先: {OUTPUT_FILE}')

def show_summary(df: pd.DataFrame) -> None:
    print()
    print('成約日数')
    print(f"平均: {df['days_to_contract'].mean():.1f}日")
    print(f"中央値: {df['days_to_contract'].median():.1f}日")
    print(f"最短: {df['days_to_contract'].min():.0f}日")
    print(f"最長: {df['days_to_contract'].max():.0f}日")
    print()
    print('価格AIとの乖離率')
    print(f"平均: {df['price_gap_ratio_percent'].mean():.2f}%")
    print(f"中央値: {df['price_gap_ratio_percent'].median():.2f}%")

def main() -> None:
    print_header('東京都中古マンション 成約日数学習データ作成 Ver.5')
    df = load_contract_data()

    # 教師データがない場合は正常な「データ待ち」状態として終了する。
    if df is None:
        return
    validate_columns(df)
    df = normalize_text_columns(df)
    df = convert_types(df)
    validate_rows(df)
    df = create_date_features(df)
    df = create_price_ai_features(df)
    show_leakage_warning(df)
    df = select_columns(df)
    save_data(df)
    show_summary(df)
    print()
    print('次のステップ:')
    print('python main.py days-train')
if __name__ == '__main__':
    main()
