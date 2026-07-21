# main.py

import sys
from pathlib import Path

from pyspark.sql import SparkSession


PROJECT_ROOT = Path(
    "/Workspace/Users/ai.with.sourabh@gmail.com/weather-precipitation-regression-databricks"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.config import (
    BRONZE_TABLE,
    SILVER_TABLE,
    GOLD_TABLE,
    PREDICTION_TABLE,
)
from src.ingestion.marketplace_ingestion import MarketplaceIngestionPipeline
from src.etl.silver_transformations import SilverTransformationPipeline
from src.etl.gold_feature_engineering import GoldFeatureEngineeringPipeline
from src.ml.train_regression_model import PrecipitationRegressionTrainer
from src.utils.logging_utils import log_section, log_success


class WeatherPrecipitationRegressionApp:
    def __init__(self):
        self.spark = (
            SparkSession.builder
            .appName("WeatherPrecipitationRegression")
            .getOrCreate()
        )

    def run_ingestion(self):
        pipeline = MarketplaceIngestionPipeline(self.spark)
        bronze_df = pipeline.run()
        print(f"Bronze rows: {bronze_df.count()}")
        return bronze_df

    def run_silver_etl(self):
        pipeline = SilverTransformationPipeline(self.spark)
        silver_df = pipeline.run()
        print(f"Silver rows: {silver_df.count()}")
        return silver_df

    def run_gold_etl(self):
        pipeline = GoldFeatureEngineeringPipeline(self.spark)
        gold_df = pipeline.run()
        print(f"Gold rows: {gold_df.count()}")
        return gold_df

    def run_model_training(self):
        trainer = PrecipitationRegressionTrainer(self.spark)
        model, prediction_df, metrics = trainer.run()

        print(f"Prediction rows: {prediction_df.count()}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE : {metrics['mae']:.4f}")
        print(f"R2  : {metrics['r2']:.4f}")

        return model, prediction_df, metrics

    def run(self):
        log_section("Weather Precipitation Regression Pipeline Started")

        self.run_ingestion()
        self.run_silver_etl()
        self.run_gold_etl()
        self.run_model_training()

        log_section("Weather Precipitation Regression Pipeline Completed")

        log_success(f"Bronze table     : {BRONZE_TABLE}")
        log_success(f"Silver table     : {SILVER_TABLE}")
        log_success(f"Gold table       : {GOLD_TABLE}")
        log_success(f"Prediction table : {PREDICTION_TABLE}")


if __name__ == "__main__":
    app = WeatherPrecipitationRegressionApp()
    app.run()