import pandas as pd
import pytest

import fetch_prices


def test_default_output_path_uses_market_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_prices, "DEFAULT_OUTPUT_DIR", tmp_path)
    path = fetch_prices._default_output_path("SOXL", "2024-01-01", "2024-01-31", "1d")
    assert path.parent == tmp_path
    assert path.name == "SOXL_2024-01-01_2024-01-31_1d.csv"


def test_main_writes_csv(monkeypatch, tmp_path):
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    download_frame = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.5, 10.5],
            "Close": [10.5, 11.5],
            "Adj Close": [10.4, 11.4],
            "Volume": [1000, 1200],
        },
        index=dates,
    )

    def fake_download(*args, **kwargs):
        return download_frame

    monkeypatch.setattr(fetch_prices.yf, "download", fake_download)
    output = tmp_path / "prices.csv"

    exit_code = fetch_prices.main(
        [
            "SOXL",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-03",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    assert output.exists()
    written = pd.read_csv(output)
    assert list(written.columns) == [
        "date",
        "adj_close",
        "close",
        "high",
        "low",
        "open",
        "volume",
    ]
    assert len(written) == 2


def test_fetch_price_frame_errors_when_empty(monkeypatch):
    empty = pd.DataFrame()

    def fake_download(*args, **kwargs):
        return empty

    monkeypatch.setattr(fetch_prices.yf, "download", fake_download)
    with pytest.raises(ValueError):
        fetch_prices.fetch_price_frame("SOXL", "2024-01-01", "2024-01-10", "1d")
