# src/ml/train_regression_model.py

from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

from src.config import (
    GOLD_TABLE,
    PREDICTION_TABLE,
    RANDOM_SEED,
    TRAIN_TEST_SPLIT,
    TARGET_COLUMN,
    PREDICTION_COLUMN,
)
from src.utils.logging_utils import log_step, log_success


class PrecipitationRegressionTrainer:
    def __init__(self, spark):
        self.spark = spark

    def read_gold_table(self):
        log_step(f"Reading Gold table: {GOLD_TABLE}")

        return self.spark.table(GOLD_TABLE)

    def split_data(self, df):
        log_step("Splitting data into train and test")

        train_df, test_df = df.randomSplit(TRAIN_TEST_SPLIT, seed=RANDOM_SEED)

        return train_df, test_df

    def build_pipeline(self):
        log_step("Building Spark ML regression pipeline")

        feature_columns = [
            "latitude",
            "longitude",
            "elevation",
            "year",
            "month",
            "day_of_year",
        ]

        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features",
            handleInvalid="skip",
        )

        linear_regression = LinearRegression(
            featuresCol="features",
            labelCol=TARGET_COLUMN,
            predictionCol=PREDICTION_COLUMN,
        )

        return Pipeline(stages=[assembler, linear_regression])

    def train_model(self, pipeline, train_df):
        log_step("Training Linear Regression model")

        return pipeline.fit(train_df)

    def generate_predictions(self, model, test_df):
        log_step("Generating predictions on test data only")

        return model.transform(test_df)

    def evaluate_model(self, predictions):
        log_step("Evaluating model")

        evaluators = {
            "rmse": RegressionEvaluator(
                labelCol=TARGET_COLUMN,
                predictionCol=PREDICTION_COLUMN,
                metricName="rmse",
            ),
            "mae": RegressionEvaluator(
                labelCol=TARGET_COLUMN,
                predictionCol=PREDICTION_COLUMN,
                metricName="mae",
            ),
            "r2": RegressionEvaluator(
                labelCol=TARGET_COLUMN,
                predictionCol=PREDICTION_COLUMN,
                metricName="r2",
            ),
        }

        metrics = {
            name: evaluator.evaluate(predictions)
            for name, evaluator in evaluators.items()
        }

        return metrics

    def write_predictions(self, predictions):
        log_step(f"Writing prediction table: {PREDICTION_TABLE}")

        prediction_df = (
            predictions
            .select(
                "station",
                "date",
                "latitude",
                "longitude",
                "elevation",
                "year",
                "month",
                "day_of_year",
                F.col(TARGET_COLUMN).alias("actual_precipitation"),
                F.col(PREDICTION_COLUMN).alias("predicted_precipitation"),
            )
            .withColumn("dataset_split", F.lit("test"))
            .withColumn("prediction_timestamp", F.current_timestamp())
        )

        (
            prediction_df.write
            .format("delta")
            .mode("overwrite")
            .saveAsTable(PREDICTION_TABLE)
        )

        log_success(f"Prediction table created: {PREDICTION_TABLE}")

        return prediction_df

    def run(self):
        gold_df = self.read_gold_table()

        train_df, test_df = self.split_data(gold_df)

        pipeline = self.build_pipeline()

        model = self.train_model(pipeline, train_df)

        predictions = self.generate_predictions(model, test_df)

        metrics = self.evaluate_model(predictions)

        prediction_df = self.write_predictions(predictions)

        return model, prediction_df, metrics