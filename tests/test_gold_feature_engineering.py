# tests/test_gold_feature_engineering.py

from datetime import date

from src.etl.gold_feature_engineering import GoldFeatureEngineeringPipeline


def test_gold_feature_table_has_expected_columns(spark):
    rows = [
        (
            "S1",
            date(2024, 1, 15),
            20.0,
            80.0,
            100,
            "Station 1",
            10,
            0,
            2024,
            1,
            15,
        ),
    ]

    columns = [
        "station",
        "date",
        "latitude",
        "longitude",
        "elevation",
        "name",
        "precipitation",
        "snowfall",
        "year",
        "month",
        "day_of_year",
    ]

    silver_df = spark.createDataFrame(rows, columns)

    pipeline = GoldFeatureEngineeringPipeline(spark)
    gold_df = pipeline.create_features(silver_df)

    expected_columns = [
        "station",
        "date",
        "latitude",
        "longitude",
        "elevation",
        "elevation_km",
        "absolute_latitude",
        "year",
        "month",
        "day_of_year",
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "precipitation",
    ]

    assert gold_df.columns == expected_columns


def test_gold_feature_engineering_creates_non_null_engineered_features(spark):
    rows = [
        (
            "S1",
            date(2024, 1, 15),
            20.0,
            80.0,
            100,
            "Station 1",
            10,
            0,
            2024,
            1,
            15,
        ),
    ]

    columns = [
        "station",
        "date",
        "latitude",
        "longitude",
        "elevation",
        "name",
        "precipitation",
        "snowfall",
        "year",
        "month",
        "day_of_year",
    ]

    silver_df = spark.createDataFrame(rows, columns)

    pipeline = GoldFeatureEngineeringPipeline(spark)
    gold_df = pipeline.create_features(silver_df)

    result = gold_df.collect()[0]

    assert result["elevation_km"] is not None
    assert result["absolute_latitude"] is not None
    assert result["month_sin"] is not None
    assert result["month_cos"] is not None
    assert result["day_of_year_sin"] is not None
    assert result["day_of_year_cos"] is not None