# tests/test_silver_transformations.py

from datetime import date

from src.etl.silver_transformations import SilverTransformationPipeline


def test_silver_transformation_removes_invalid_rows(spark):
    rows = [
        ("S1", date(2024, 1, 1), 20.0, 80.0, 100, "Station 1", 10, 0),
        ("S2", date(2024, 1, 1), None, 80.0, 100, "Station 2", 20, 0),
        ("S3", date(2024, 1, 1), 25.0, 85.0, 100, "Station 3", None, 0),
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
    ]

    bronze_df = spark.createDataFrame(rows, columns)

    pipeline = SilverTransformationPipeline(spark)
    silver_df = pipeline.transform(bronze_df)

    assert silver_df.count() == 1


def test_silver_transformation_adds_date_features(spark):
    rows = [
        ("S1", date(2024, 1, 15), 20.0, 80.0, 100, "Station 1", 10, 0),
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
    ]

    bronze_df = spark.createDataFrame(rows, columns)

    pipeline = SilverTransformationPipeline(spark)
    silver_df = pipeline.transform(bronze_df)

    assert "year" in silver_df.columns
    assert "month" in silver_df.columns
    assert "day_of_year" in silver_df.columns