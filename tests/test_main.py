from pathlib import Path

from main import count_csv_rows


def test_count_csv_rows_missing_file(tmp_path: Path):
    assert count_csv_rows(tmp_path / "missing.csv") == 0


def test_count_csv_rows_header_only(tmp_path: Path):
    file_path = tmp_path / "header_only.csv"
    file_path.write_text("a,b\n", encoding="utf-8-sig")

    assert count_csv_rows(file_path) == 0


def test_count_csv_rows_counts_data_rows(tmp_path: Path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text(
        "a,b\n1,2\n3,4\n5,6\n",
        encoding="utf-8-sig",
    )

    assert count_csv_rows(file_path) == 3
