import numpy as np
import pandas as pd
import pytest

from predict import (
    create_prediction_result,
    evaluate_price_position,
    normalize_text,
    predict_prices,
    prepare_input_data,
    validate_input_columns,
)


class DummyModel:
    def __init__(self, unit_prices):
        self.unit_prices = np.asarray(unit_prices, dtype=float)

    def predict(self, X):
        return np.log1p(self.unit_prices[: len(X)])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (np.nan, None),
        ("", None),
        ("   ", None),
        ("nan", None),
        ("<NA>", None),
        (" 北千住 ", "北千住"),
        (123, "123"),
    ],
)
def test_normalize_text(value, expected):
    assert normalize_text(value) == expected


@pytest.mark.parametrize(
    ("gap_percent", "expected"),
    [
        (np.nan, ""),
        (-25, "割安"),
        (-10, "割安"),
        (-9.99, "適正"),
        (0, "適正"),
        (10, "適正"),
        (10.01, "やや割高"),
        (20, "やや割高"),
        (20.01, "割高"),
    ],
)
def test_evaluate_price_position(gap_percent, expected):
    assert evaluate_price_position(gap_percent) == expected


def test_validate_input_columns_success():
    df = pd.DataFrame({"area_m2": [50], "building_age": [10]})
    validate_input_columns(df, ["area_m2", "building_age"])


def test_validate_input_columns_missing():
    df = pd.DataFrame({"area_m2": [50]})

    with pytest.raises(KeyError, match="building_age"):
        validate_input_columns(df, ["area_m2", "building_age"])


def test_prepare_input_data_success():
    df = pd.DataFrame(
        {
            "area_m2": [65.2],
            "building_age": [12],
            "city": ["足立区"],
            "station_latitude": [35.75],
            "station_longitude": [139.80],
        }
    )
    feature_columns = [
        "area_m2",
        "building_age",
        "city",
        "station_latitude",
        "station_longitude",
    ]
    categorical_columns = ["city"]

    result = prepare_input_data(df, feature_columns, categorical_columns)

    assert result.loc[0, "area_m2"] == pytest.approx(65.2)
    assert result.loc[0, "building_age"] == pytest.approx(12)
    assert result.loc[0, "city"] == "足立区"


def test_prepare_input_data_fills_missing_category():
    df = pd.DataFrame(
        {
            "area_m2": [50],
            "city": [None],
        }
    )

    result = prepare_input_data(
        df,
        feature_columns=["area_m2", "city"],
        categorical_columns=["city"],
    )

    assert result.loc[0, "city"] == "不明"


def test_prepare_input_data_rejects_zero_area():
    df = pd.DataFrame({"area_m2": [0]})

    with pytest.raises(ValueError, match="0より大きく"):
        prepare_input_data(
            df,
            feature_columns=["area_m2"],
            categorical_columns=[],
        )


def test_prepare_input_data_rejects_invalid_latitude():
    df = pd.DataFrame(
        {
            "area_m2": [50],
            "station_latitude": [80],
        }
    )

    with pytest.raises(ValueError, match="station_latitude"):
        prepare_input_data(
            df,
            feature_columns=["area_m2", "station_latitude"],
            categorical_columns=[],
        )


def test_prepare_input_data_rejects_invalid_longitude():
    df = pd.DataFrame(
        {
            "area_m2": [50],
            "station_longitude": [10],
        }
    )

    with pytest.raises(ValueError, match="station_longitude"):
        prepare_input_data(
            df,
            feature_columns=["area_m2", "station_longitude"],
            categorical_columns=[],
        )


def test_predict_prices():
    df = pd.DataFrame(
        {
            "area_m2": [50.0, 80.0],
            "building_age": [10, 5],
        }
    )
    model = DummyModel([1_000_000, 800_000])

    unit_prices, contract_prices = predict_prices(
        model,
        df,
        feature_columns=["area_m2", "building_age"],
    )

    np.testing.assert_allclose(unit_prices, [1_000_000, 800_000], rtol=1e-10)
    np.testing.assert_allclose(contract_prices, [50_000_000, 64_000_000], rtol=1e-10)


def test_predict_prices_clips_negative_unit_price():
    df = pd.DataFrame({"area_m2": [50.0]})

    class NegativeModel:
        def predict(self, X):
            return np.array([-1.0])

    unit_prices, contract_prices = predict_prices(
        NegativeModel(),
        df,
        feature_columns=["area_m2"],
    )

    assert unit_prices[0] == 0
    assert contract_prices[0] == 0


def test_create_prediction_result():
    df = pd.DataFrame(
        {
            "property_id": ["A001"],
            "area_m2": [50.0],
            "asking_price": [55_000_000],
        }
    )

    result = create_prediction_result(
        df,
        predicted_unit_price=np.array([1_000_000.0]),
        predicted_contract_price=np.array([50_000_000.0]),
    )

    assert result.loc[0, "predicted_price_per_m2"] == 1_000_000
    assert result.loc[0, "predicted_contract_price"] == 50_000_000
    assert result.loc[0, "price_gap_amount"] == 5_000_000
    assert result.loc[0, "price_gap_ratio_percent"] == pytest.approx(10.0)
    assert result.loc[0, "price_evaluation"] == "適正"
