# src/ingestion/marketplace_ingestion.py

from pyspark.sql import functions as F

from src.config import (
    CATALOG_NAME,
    SCHEMA_NAME,
    SOURCE_TABLE,
    BRONZE_TABLE,
)
from src.utils.logging_utils import log_step, log_success


class MarketplaceIngestionPipeline:
    def __init__(self, spark):
        self.spark = spark

    def create_schema(self) -> None:
        log_step(f"Creating schema if not exists: {CATALOG_NAME}.{SCHEMA_NAME}")

        self.spark.sql(
            f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}"
        )

    def read_source_table(self):
        log_step(f"Reading source Marketplace table: {SOURCE_TABLE}")

        source_df = (
            self.spark
            .table(SOURCE_TABLE)
            .select(
                "station",
                "date",
                "latitude",
                "longitude",
                "elevation",
                "name",
                "precipitation",
                "snowfall",
            )
        )

        return source_df

    def create_bronze_dataframe(self, source_df):
        log_step("Creating Bronze DataFrame with ingestion metadata")

        bronze_df = (
            source_df
            .withColumn("_ingestion_timestamp", F.current_timestamp())
            .withColumn("_source_table", F.lit(SOURCE_TABLE))
        )

        return bronze_df

    def write_bronze_table(self, bronze_df) -> None:
        log_step(f"Writing Bronze table: {BRONZE_TABLE}")

        (
            bronze_df.write
            .format("delta")
            .mode("overwrite")
            .saveAsTable(BRONZE_TABLE)
        )

        log_success(f"Bronze table created: {BRONZE_TABLE}")

    def run(self):
        self.create_schema()

        source_df = self.read_source_table()

        bronze_df = self.create_bronze_dataframe(source_df)

        self.write_bronze_table(bronze_df)

        return bronze_df