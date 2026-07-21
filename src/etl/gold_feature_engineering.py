# src/etl/gold_feature_engineering.py

from pyspark.sql import functions as F

from src.config import SILVER_TABLE, GOLD_TABLE
from src.utils.logging_utils import log_step, log_success


class GoldFeatureEngineeringPipeline:
    def __init__(self, spark):
        self.spark = spark

    def read_silver_table(self):
        log_step(f"Reading Silver table: {SILVER_TABLE}")

        return self.spark.table(SILVER_TABLE)

    def create_features(self, silver_df):
        log_step("Creating ML-ready Gold feature table with engineered features")

        pi_value = 3.141592653589793

        gold_df = (
            silver_df
            .withColumn(
                "month_sin",
                F.sin(2 * F.lit(pi_value) * F.col("month") / F.lit(12))
            )
            .withColumn(
                "month_cos",
                F.cos(2 * F.lit(pi_value) * F.col("month") / F.lit(12))
            )
            .withColumn(
                "day_of_year_sin",
                F.sin(2 * F.lit(pi_value) * F.col("day_of_year") / F.lit(365))
            )
            .withColumn(
                "day_of_year_cos",
                F.cos(2 * F.lit(pi_value) * F.col("day_of_year") / F.lit(365))
            )
            .withColumn(
                "absolute_latitude",
                F.abs(F.col("latitude"))
            )
            .withColumn(
                "elevation_km",
                F.col("elevation") / F.lit(1000)
            )
            .select(
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
            )
        )

        return gold_df

    def write_gold_table(self, gold_df) -> None:
        log_step(f"Writing Gold table: {GOLD_TABLE}")

        (
            gold_df.write
            .format("delta")
            .mode("overwrite")
            .saveAsTable(GOLD_TABLE)
        )

        log_success(f"Gold table created: {GOLD_TABLE}")

    def run(self):
        silver_df = self.read_silver_table()

        gold_df = self.create_features(silver_df)

        self.write_gold_table(gold_df)

        return gold_df