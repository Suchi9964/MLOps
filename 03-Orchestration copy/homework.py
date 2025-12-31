#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
from prefect import task, flow

# ---------------- MLflow configuration ----------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path("models")
models_folder.mkdir(exist_ok=True)

# ---------------- Prefect Tasks ----------------

@task
def read_dataframe(year: int, month: int) -> pd.DataFrame:
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(url)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    return df


@task
def create_X(df: pd.DataFrame, dv: DictVectorizer = None):
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


@task
def train_model(X_train, y_train, X_val, y_val, dv) -> str:
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:linear",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # Save and log preprocessor
        preprocessor_path = "models/preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

        # Log model
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id
    
# helper function to calculate months

from datetime import date
from dateutil.relativedelta import relativedelta

def get_train_and_val_months(execution_date: date):
    train_date = execution_date - relativedelta(months=2)
    val_date = execution_date - relativedelta(months=1)

    return (
        train_date.year, train_date.month,
        val_date.year, val_date.month
    )


# ---------------- Prefect Flow (Orchestration) ----------------

@flow(name="nyc-taxi-monthly-training")
def taxi_training_flow(execution_date: date = date.today()):

    train_year, train_month, val_year, val_month = \
        get_train_and_val_months(execution_date)

    # Load data
    df_train = read_dataframe(train_year, train_month)
    df_val = read_dataframe(val_year, val_month)

    # Feature engineering
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    # Train model
    run_id = train_model(X_train, y_train, X_val, y_val, dv)

    print(
        f"Training: {train_year}-{train_month}, "
        f"Validation: {val_year}-{val_month}, "
        f"MLflow run_id: {run_id}"
    )

    return run_id


# # ---------------- CLI Entry Point ----------------

# if __name__ == "__main__":
#     # import argparse

#     # parser = argparse.ArgumentParser(
#     #     description="Train a model to predict taxi trip duration."
#     # )
#     # parser.add_argument("--year", type=int, required=True)
#     # parser.add_argument("--month", type=int, required=True)
#     # args = parser.parse_args()

#     run_id = taxi_training_flow()

#     with open("run_id.txt", "w") as f:
#         f.write(run_id)
