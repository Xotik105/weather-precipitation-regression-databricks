# src/etl/silver_transformations.py

from pyspark.sql import functions as F

from src.config import BRONZE_TABLE, SILVER_TABLE
from src.utils.logging_utils import log_step, log_success


class SilverTransformationPipeline:
    def __init__(self, spark):
        self.spark = spark

    def read_bronze_table(self):
        log_step(f"Reading Bronze table: {BRONZE_TABLE}")

        return self.spark.table(BRONZE_TABLE)

    def transform(self, bronze_df):
        log_step("Cleaning weather data for Silver table")

        silver_df = (
            bronze_df
            .dropDuplicates(["station", "date"])
            .filter(F.col("station").isNotNull())
            .filter(F.col("date").isNotNull())
            .filter(F.col("latitude").isNotNull())
            .filter(F.col("longitude").isNotNull())
            .filter(F.col("elevation").isNotNull())
            .filter(F.col("precipitation").isNotNull())
            .withColumn("date", F.to_date("date"))
            .withColumn("year", F.year("date"))
            .withColumn("month", F.month("date"))
            .withColumn("day_of_year", F.dayofyear("date"))
            .filter(F.col("year").isNotNull())
            .filter(F.col("month").isNotNull())
            .filter(F.col("day_of_year").isNotNull())
            .filter(F.col("precipitation") >= 0)
            .filter(F.col("elevation") > -500)
            .filter(F.col("elevation") < 9000)
        )

        return silver_df

    def write_silver_table(self, silver_df) -> None:
        log_step(f"Writing Silver table: {SILVER_TABLE}")

        (
            silver_df.write
            .format("delta")
            .mode("overwrite")
            .saveAsTable(SILVER_TABLE)
        )

        log_success(f"Silver table created: {SILVER_TABLE}")

    def run(self):
        bronze_df = self.read_bronze_table()

        silver_df = self.transform(bronze_df)

        self.write_silver_table(silver_df)

        return silver_df