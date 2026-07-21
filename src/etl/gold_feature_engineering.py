# src/etl/gold_feature_engineering.py

from src.config import SILVER_TABLE, GOLD_TABLE
from src.utils.logging_utils import log_step, log_success


class GoldFeatureEngineeringPipeline:
    def __init__(self, spark):
        self.spark = spark

    def read_silver_table(self):
        log_step(f"Reading Silver table: {SILVER_TABLE}")

        return self.spark.table(SILVER_TABLE)

    def create_features(self, silver_df):
        log_step("Creating ML-ready Gold feature table")

        gold_df = (
            silver_df
            .select(
                "station",
                "date",
                "latitude",
                "longitude",
                "elevation",
                "year",
                "month",
                "day_of_year",
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