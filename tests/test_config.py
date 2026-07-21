# tests/test_config.py

from src.config import (
    SOURCE_TABLE,
    BRONZE_TABLE,
    SILVER_TABLE,
    GOLD_TABLE,
    PREDICTION_TABLE,
)


def test_table_names_are_defined():
    assert SOURCE_TABLE
    assert BRONZE_TABLE
    assert SILVER_TABLE
    assert GOLD_TABLE
    assert PREDICTION_TABLE


def test_prediction_table_is_weather_related():
    assert "weather" in PREDICTION_TABLE