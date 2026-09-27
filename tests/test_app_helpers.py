import pytest

from app import case_payload, format_money, format_percent, safe_text


def test_safe_text():
    assert safe_text(None) == ""
    assert safe_text("  北千住  ") == "北千住"
    assert safe_text("", "未設定") == "未設定"


def test_format_money_man_yen():
    assert format_money(75_000_000, "万円") == "7,500万円"


def test_format_money_yen():
    assert format_money(75_000_000, "円") == "75,000,000円"


def test_format_percent():
    assert format_percent(12.345) == "+12.35%"
    assert format_percent(-5) == "-5.00%"


def test_case_payload_valid_status():
    result = case_payload(
        case_name="北千住マンション",
        property_id="A001",
        status="提案中",
        city="足立区",
        district="千住",
        station="北千住",
        area_m2=65.2,
        floor_plan="3LDK",
        building_age=12,
        asking_price=75_000_000,
        memo="テスト",
    )

    assert result["case_name"] == "北千住マンション"
    assert result["property_id"] == "A001"
    assert result["status"] == "提案中"
    assert result["area_m2"] == pytest.approx(65.2)
    assert result["asking_price"] == 75_000_000


def test_case_payload_invalid_status_falls_back():
    result = case_payload(
        case_name="テスト案件",
        property_id="A001",
        status="存在しない状態",
        city="",
        district="",
        station="",
        area_m2=0,
        floor_plan="",
        building_age=0,
        asking_price=0,
        memo="",
    )

    assert result["status"] == "査定中"
    assert result["city"] is None
    assert result["area_m2"] is None
    assert result["asking_price"] is None
