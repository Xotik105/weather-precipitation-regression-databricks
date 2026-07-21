# src/config.py

CATALOG_NAME = "workspace"
SCHEMA_NAME = "weather_regression"

SOURCE_TABLE = (
    "rearc_daily_weather_observations_noaa."
    "esg_noaa_ghcn."
    "noaa_ghcn_daily"
)

BRONZE_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.bronze_noaa_daily_weather"
SILVER_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.silver_noaa_daily_weather"
GOLD_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.gold_precipitation_features"
PREDICTION_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.weather_precipitation_predictions"

RANDOM_SEED = 42

TRAIN_TEST_SPLIT = [0.8, 0.2]

TARGET_COLUMN = "precipitation"
PREDICTION_COLUMN = "predicted_precipitation"