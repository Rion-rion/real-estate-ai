from __future__ import annotations

import math
import os
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.neighbors import BallTree


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

OUTPUT_FILE = (
    RAW_DIR
    / "tokyo_contract_prices_station.csv"
)


PRICE_POINT_URL = (
    "https://www.reinfolib.mlit.go.jp/"
    "ex-api/external/XPT001"
)

STATION_URL = (
    "https://www.reinfolib.mlit.go.jp/"
    "ex-api/external/XKT015"
)


# 東京都本土部（23区 + 多摩地域）をおおむねカバー
MIN_LONGITUDE = 138.90
MAX_LONGITUDE = 140.05

MIN_LATITUDE = 35.40
MAX_LATITUDE = 35.95


ZOOM_LEVEL = 11

FROM_PERIOD = "20211"
TO_PERIOD = "20262"

PRICE_CLASSIFICATION = "02"

# 中古マンション等
LAND_TYPE_CODE = "07"

REQUEST_TIMEOUT = 90
REQUEST_INTERVAL = 0.3

EARTH_RADIUS_M = 6_371_000

# XPT001のポイントとXKT015駅GISの
# マッチング距離がこれ以上なら不明扱い
MAX_STATION_MATCH_DISTANCE_M = 1500


def get_api_key() -> str:
    load_dotenv()

    api_key = os.getenv(
        "MLIT_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "MLIT_API_KEY が見つかりません。\n"
            ".env にAPIキーを設定してください。"
        )

    return api_key


def normalize_text(
    value,
) -> str | None:

    if value is None:
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    return unicodedata.normalize(
        "NFKC",
        text,
    )


def parse_number(
    value,
) -> float:

    text = normalize_text(
        value
    )

    if not text:
        return np.nan

    text = (
        text
        .replace(",", "")
        .replace("㎡", "")
        .replace("m2", "")
        .replace("%", "")
        .replace("円", "")
        .strip()
    )

    number = ""

    for char in text:

        if (
            char.isdigit()
            or char in ".-"
        ):
            number += char

        elif number:
            break

    if not number:
        return np.nan

    try:
        return float(number)

    except ValueError:
        return np.nan


def parse_japanese_money(
    value,
) -> float:

    text = normalize_text(
        value
    )

    if not text:
        return np.nan

    text = (
        text
        .replace(",", "")
        .replace("円", "")
        .strip()
    )

    total = 0.0

    try:
        if "億" in text:

            oku_part, text = (
                text.split(
                    "億",
                    1,
                )
            )

            total += (
                float(oku_part)
                * 100_000_000
            )

        if "万" in text:

            man_part, text = (
                text.split(
                    "万",
                    1,
                )
            )

            if man_part:
                total += (
                    float(man_part)
                    * 10_000
                )

        if text:
            remaining = ""

            for char in text:

                if (
                    char.isdigit()
                    or char == "."
                ):
                    remaining += char

            if remaining:
                total += float(
                    remaining
                )

        if total > 0:
            return total

    except ValueError:
        pass

    return parse_number(
        value
    )


def lonlat_to_tile(
    longitude: float,
    latitude: float,
    zoom: int,
) -> tuple[int, int]:

    n = 2 ** zoom

    x = int(
        (
            longitude
            + 180.0
        )
        / 360.0
        * n
    )

    latitude_rad = math.radians(
        latitude
    )

    y = int(
        (
            1.0
            - math.asinh(
                math.tan(
                    latitude_rad
                )
            )
            / math.pi
        )
        / 2.0
        * n
    )

    return x, y


def get_target_tiles() -> list[
    tuple[int, int]
]:

    min_x, max_y = (
        lonlat_to_tile(
            MIN_LONGITUDE,
            MIN_LATITUDE,
            ZOOM_LEVEL,
        )
    )

    max_x, min_y = (
        lonlat_to_tile(
            MAX_LONGITUDE,
            MAX_LATITUDE,
            ZOOM_LEVEL,
        )
    )

    tiles = []

    for x in range(
        min_x,
        max_x + 1,
    ):

        for y in range(
            min_y,
            max_y + 1,
        ):

            tiles.append(
                (x, y)
            )

    return tiles


def request_geojson(
    session: requests.Session,
    url: str,
    params: dict,
    api_key: str,
) -> dict:

    headers = {
        "Ocp-Apim-Subscription-Key": (
            api_key
        ),
        "Accept": "application/json",
    }

    response = session.get(
        url,
        headers=headers,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code in {
        204,
        404,
    }:
        return {
            "type": "FeatureCollection",
            "features": [],
        }

    response.raise_for_status()

    if not response.content:
        return {
            "type": "FeatureCollection",
            "features": [],
        }

    return response.json()


def collect_station_geojson(
    session: requests.Session,
    api_key: str,
    tiles: list[
        tuple[int, int]
    ],
) -> list[dict]:

    features = []

    print()
    print(
        "駅GISデータを取得します。"
    )

    for index, (
        x,
        y,
    ) in enumerate(
        tiles,
        start=1,
    ):

        print(
            f"[駅 {index}/{len(tiles)}] "
            f"z={ZOOM_LEVEL} "
            f"x={x} y={y}"
        )

        params = {
            "response_format": (
                "geojson"
            ),
            "z": ZOOM_LEVEL,
            "x": x,
            "y": y,
        }

        try:
            data = request_geojson(
                session,
                STATION_URL,
                params,
                api_key,
            )

            features.extend(
                data.get(
                    "features",
                    [],
                )
            )

        except (
            requests.RequestException,
            ValueError,
        ) as error:

            print(
                f"  駅API取得失敗: "
                f"{error}"
            )

        time.sleep(
            REQUEST_INTERVAL
        )

    print()
    print(
        f"駅GIS Feature数: "
        f"{len(features):,}"
    )

    return features


def collect_price_geojson(
    session: requests.Session,
    api_key: str,
    tiles: list[
        tuple[int, int]
    ],
) -> list[dict]:

    features = []

    print()
    print(
        "中古マンション成約価格"
        "ポイントを取得します。"
    )

    for index, (
        x,
        y,
    ) in enumerate(
        tiles,
        start=1,
    ):

        print(
            f"[価格 {index}/{len(tiles)}] "
            f"z={ZOOM_LEVEL} "
            f"x={x} y={y}"
        )

        params = {
            "response_format": (
                "geojson"
            ),
            "z": ZOOM_LEVEL,
            "x": x,
            "y": y,
            "from": FROM_PERIOD,
            "to": TO_PERIOD,
            "priceClassification": (
                PRICE_CLASSIFICATION
            ),
            "landTypeCode": (
                LAND_TYPE_CODE
            ),
        }

        try:
            data = request_geojson(
                session,
                PRICE_POINT_URL,
                params,
                api_key,
            )

            tile_features = (
                data.get(
                    "features",
                    [],
                )
            )

            features.extend(
                tile_features
            )

            print(
                f"  → "
                f"{len(tile_features):,}件"
            )

        except (
            requests.RequestException,
            ValueError,
        ) as error:

            print(
                f"  価格API取得失敗: "
                f"{error}"
            )

        time.sleep(
            REQUEST_INTERVAL
        )

    print()
    print(
        f"価格Feature総数: "
        f"{len(features):,}"
    )

    return features


def flatten_geometry_coordinates(
    geometry: dict,
) -> list[
    tuple[float, float]
]:

    if not geometry:
        return []

    geometry_type = geometry.get(
        "type"
    )

    coordinates = geometry.get(
        "coordinates"
    )

    if coordinates is None:
        return []

    points = []

    if geometry_type == "Point":

        if len(coordinates) >= 2:

            points.append(
                (
                    float(
                        coordinates[0]
                    ),
                    float(
                        coordinates[1]
                    ),
                )
            )

    elif geometry_type == "LineString":

        for coordinate in coordinates:

            if len(coordinate) >= 2:

                points.append(
                    (
                        float(
                            coordinate[0]
                        ),
                        float(
                            coordinate[1]
                        ),
                    )
                )

    elif geometry_type == "MultiLineString":

        for line in coordinates:

            for coordinate in line:

                if len(coordinate) >= 2:

                    points.append(
                        (
                            float(
                                coordinate[0]
                            ),
                            float(
                                coordinate[1]
                            ),
                        )
                    )

    elif geometry_type == "MultiPoint":

        for coordinate in coordinates:

            if len(coordinate) >= 2:

                points.append(
                    (
                        float(
                            coordinate[0]
                        ),
                        float(
                            coordinate[1]
                        ),
                    )
                )

    return points


def build_station_index(
    station_features: list[
        dict
    ],
):

    coordinate_rows = []
    metadata_rows = []

    seen = set()

    for feature in station_features:

        properties = (
            feature.get(
                "properties",
                {}
            )
        )

        station_name = (
            normalize_text(
                properties.get(
                    "S12_001_ja"
                )
            )
        )

        station_code = (
            normalize_text(
                properties.get(
                    "S12_001c"
                )
            )
        )

        station_group_code = (
            normalize_text(
                properties.get(
                    "S12_001g"
                )
            )
        )

        company = (
            normalize_text(
                properties.get(
                    "S12_002_ja"
                )
            )
        )

        line_name = (
            normalize_text(
                properties.get(
                    "S12_003_ja"
                )
            )
        )

        coordinates = (
            flatten_geometry_coordinates(
                feature.get(
                    "geometry",
                    {}
                )
            )
        )

        for (
            longitude,
            latitude,
        ) in coordinates:

            key = (
                round(
                    longitude,
                    7,
                ),
                round(
                    latitude,
                    7,
                ),
                station_name,
                company,
                line_name,
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            coordinate_rows.append(
                [
                    latitude,
                    longitude,
                ]
            )

            metadata_rows.append(
                {
                    "station_name": (
                        station_name
                    ),
                    "station_code": (
                        station_code
                    ),
                    "station_group_code": (
                        station_group_code
                    ),
                    "station_company": (
                        company
                    ),
                    "station_line": (
                        line_name
                    ),
                }
            )

    if not coordinate_rows:
        raise RuntimeError(
            "駅GIS座標を取得できませんでした。"
        )

    coordinates_array = np.asarray(
        coordinate_rows,
        dtype=float,
    )

    tree = BallTree(
        np.radians(
            coordinates_array
        ),
        metric="haversine",
    )

    print(
        f"駅GIS検索ポイント: "
        f"{len(coordinate_rows):,}"
    )

    return (
        tree,
        coordinates_array,
        metadata_rows,
    )


def property_feature_to_record(
    feature: dict,
) -> dict | None:

    geometry = feature.get(
        "geometry",
        {}
    )

    if (
        geometry.get(
            "type"
        )
        != "Point"
    ):
        return None

    coordinates = geometry.get(
        "coordinates"
    )

    if (
        not coordinates
        or len(coordinates) < 2
    ):
        return None

    longitude = float(
        coordinates[0]
    )

    latitude = float(
        coordinates[1]
    )

    properties = feature.get(
        "properties",
        {}
    )

    record = {
        "PriceCategory": (
            normalize_text(
                properties.get(
                    "price_information_category_name_ja"
                )
            )
        ),

        "Type": (
            normalize_text(
                properties.get(
                    "land_type_name_ja"
                )
            )
        ),

        "Region": (
            normalize_text(
                properties.get(
                    "use_category_name_ja"
                )
            )
        ),

        "MunicipalityCode": (
            normalize_text(
                properties.get(
                    "city_code"
                )
            )
        ),

        "Prefecture": (
            normalize_text(
                properties.get(
                    "prefecture_name_ja"
                )
            )
        ),

        "Municipality": (
            normalize_text(
                properties.get(
                    "city_name_ja"
                )
            )
        ),

        "DistrictName": (
            normalize_text(
                properties.get(
                    "district_name_ja"
                )
            )
        ),

        "DistrictCode": (
            normalize_text(
                properties.get(
                    "district_code"
                )
            )
        ),

        "TradePrice": (
            parse_japanese_money(
                properties.get(
                    "u_transaction_price_total_ja"
                )
            )
        ),

        "FloorPlan": (
            normalize_text(
                properties.get(
                    "floor_plan_name_ja"
                )
            )
        ),

        "Area": (
            parse_number(
                properties.get(
                    "u_area_ja"
                )
            )
        ),

        "TotalFloorArea": (
            parse_number(
                properties.get(
                    "u_building_total_floor_area_ja"
                )
            )
        ),

        "BuildingYear": (
            normalize_text(
                properties.get(
                    "u_construction_year_ja"
                )
            )
        ),

        "Structure": (
            normalize_text(
                properties.get(
                    "building_structure_name_ja"
                )
            )
        ),

        "Use": (
            normalize_text(
                properties.get(
                    "building_use_name_ja"
                )
            )
        ),

        "Purpose": (
            normalize_text(
                properties.get(
                    "future_use_purpose_name_ja"
                )
            )
        ),

        "CityPlanning": (
            normalize_text(
                properties.get(
                    "land_use_name_ja"
                )
            )
        ),

        "CoverageRatio": (
            parse_number(
                properties.get(
                    "u_building_coverage_ratio_ja"
                )
            )
        ),

        "FloorAreaRatio": (
            parse_number(
                properties.get(
                    "u_floor_area_ratio_ja"
                )
            )
        ),

        "Period": (
            normalize_text(
                properties.get(
                    "point_in_time_name_ja"
                )
            )
        ),

        "Renovation": (
            normalize_text(
                properties.get(
                    "remark_renovation_name_ja"
                )
            )
        ),

        # XPT001が返す地点は
        # 「対象不動産の最寄駅ポイント」
        "station_longitude": longitude,
        "station_latitude": latitude,
    }

    return record


def create_property_dataframe(
    property_features: list[
        dict
    ],
) -> pd.DataFrame:

    records = []

    for feature in property_features:

        record = (
            property_feature_to_record(
                feature
            )
        )

        if record is not None:
            records.append(
                record
            )

    if not records:
        raise RuntimeError(
            "中古マンション価格データを"
            "取得できませんでした。"
        )

    df = pd.DataFrame(
        records
    )

    before = len(df)

    df = df.drop_duplicates()

    print(
        f"価格ポイント: "
        f"{before:,}件 → "
        f"重複除去後 {len(df):,}件"
    )

    return df


def match_station_names(
    df: pd.DataFrame,
    station_tree: BallTree,
    station_metadata: list[
        dict
    ],
) -> pd.DataFrame:

    df = df.copy()

    query_coordinates = (
        df[
            [
                "station_latitude",
                "station_longitude",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    distances, indices = (
        station_tree.query(
            np.radians(
                query_coordinates
            ),
            k=1,
        )
    )

    distances_m = (
        distances[
            :,
            0
        ]
        * EARTH_RADIUS_M
    )

    matched_indices = (
        indices[
            :,
            0
        ]
    )

    station_names = []
    station_codes = []
    station_group_codes = []
    station_companies = []
    station_lines = []

    for (
        distance_m,
        station_index,
    ) in zip(
        distances_m,
        matched_indices,
    ):

        if (
            distance_m
            > MAX_STATION_MATCH_DISTANCE_M
        ):

            station_names.append(
                "不明"
            )

            station_codes.append(
                None
            )

            station_group_codes.append(
                None
            )

            station_companies.append(
                None
            )

            station_lines.append(
                None
            )

            continue

        metadata = (
            station_metadata[
                int(
                    station_index
                )
            ]
        )

        station_names.append(
            metadata[
                "station_name"
            ]
            or "不明"
        )

        station_codes.append(
            metadata[
                "station_code"
            ]
        )

        station_group_codes.append(
            metadata[
                "station_group_code"
            ]
        )

        station_companies.append(
            metadata[
                "station_company"
            ]
        )

        station_lines.append(
            metadata[
                "station_line"
            ]
        )

    df[
        "station_name"
    ] = station_names

    df[
        "station_code"
    ] = station_codes

    df[
        "station_group_code"
    ] = station_group_codes

    df[
        "station_company"
    ] = station_companies

    df[
        "station_line"
    ] = station_lines

    df[
        "station_geometry_match_m"
    ] = np.round(
        distances_m,
        1,
    )

    matched_count = (
        df[
            "station_name"
        ]
        .ne(
            "不明"
        )
        .sum()
    )

    print()
    print(
        f"駅名マッチ成功: "
        f"{matched_count:,}"
        f" / {len(df):,}"
    )

    print(
        f"マッチ率: "
        f"{matched_count / len(df) * 100:.2f}%"
    )

    print(
        "※ station_geometry_match_m は"
        "物件から駅までの距離ではありません。"
    )

    return df


def save_data(
    df: pd.DataFrame,
) -> None:

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "駅情報付き成約価格データ"
        "を保存しました。"
    )

    print(
        f"件数: "
        f"{len(df):,}件"
    )

    print(
        f"保存先: "
        f"{OUTPUT_FILE}"
    )

    print()
    print(
        "駅名 TOP20:"
    )

    print(
        df[
            "station_name"
        ]
        .value_counts()
        .head(20)
        .to_string()
    )


def main() -> None:

    print(
        "東京都中古マンション "
        "駅特徴量取得 Ver.1"
    )

    api_key = get_api_key()

    tiles = get_target_tiles()

    print(
        f"対象タイル数: "
        f"{len(tiles)}"
    )

    print(
        "対象: 東京都本土部 "
        "（23区・多摩地域）"
    )

    with requests.Session() as session:

        station_features = (
            collect_station_geojson(
                session,
                api_key,
                tiles,
            )
        )

        property_features = (
            collect_price_geojson(
                session,
                api_key,
                tiles,
            )
        )

    (
        station_tree,
        _,
        station_metadata,
    ) = build_station_index(
        station_features
    )

    df = (
        create_property_dataframe(
            property_features
        )
    )

    df = match_station_names(
        df,
        station_tree,
        station_metadata,
    )

    save_data(
        df
    )


if __name__ == "__main__":
    main()