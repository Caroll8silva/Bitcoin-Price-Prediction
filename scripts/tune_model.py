import optuna
import lightgbm as lgb
import pandas as pd
import sys
import os
from sklearn.metrics import mean_absolute_error
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.helpers.feature_engineering_helper import create_features_for_training

HISTORICAL_DATA_PATH = "training_data.csv"
VALIDATION_SET_SIZE = 0.15
N_TRIALS = 300  # aq é o número de combinações diferentes p testar
BEST_PARAMS_FILE = "best_params.json"


def objective(trial, X_train, y_train, X_val, y_val):

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 2000,
        "verbosity": -1,
        "n_jobs": -1,
        "seed": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)

    return mae


def main():
    print("--- Starting Hyperparameter Tuning with Optuna ---")

    print(f"Loading data from: {HISTORICAL_DATA_PATH}")
    df = pd.read_csv(HISTORICAL_DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    print("Engineering features...")
    features_df = create_features_for_training(df)
    features_df.dropna(inplace=True)

    val_size = int(len(features_df) * VALIDATION_SET_SIZE)
    train_df = features_df.iloc[:-val_size]
    val_df = features_df.iloc[-val_size:]

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_val = val_df.drop(columns=["target"])
    y_val = val_df["target"]

    study = optuna.create_study(direction="minimize")

    print(
        f"Running {N_TRIALS} trials to find the best parameters. This will take a long time..."
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
    )

    print("\nTuning complete!")
    print(f"  Best MAE on validation set: ${study.best_value:,.2f}")
    print("  Best parameters found:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\nBest parameters saved to {BEST_PARAMS_FILE}")


if __name__ == "__main__":
    main()
