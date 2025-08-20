import sys
import os
import pandas as pd
import lightgbm as lgb
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.helpers.feature_engineering_helper import create_features_for_training

load_dotenv()
MODEL_PATH = os.getenv("PREDICTION_MODEL_PATH")
HISTORICAL_DATA_PATH = "training_data.csv"
PARAMS_FILE_PATH = "best_params.json"

def train():
    print("--- Starting Final Model Training ---")
    df = pd.read_csv(HISTORICAL_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp')
    features_df = create_features_for_training(df)
    features_df.dropna(inplace=True)

    y_train = features_df['target']
    X_train = features_df.drop(columns=['target', 'price'])
    
    print(" Training final LightGBM model on the entire dataset...")
    try:
        with open(PARAMS_FILE_PATH, 'r') as f:
            best_params = json.load(f)
        print(f"Loaded best parameters from {PARAMS_FILE_PATH}")
    except FileNotFoundError:
        print(f" WARNING: '{PARAMS_FILE_PATH}' not found. Using default parameters.")
        best_params = {}
    best_params.update({'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 2000, 'random_state': 42, 'n_jobs': -1})
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)
    print(f" Saving final trained model to: {MODEL_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    final_model.booster_.save_model(MODEL_PATH)
    print("Final model training complete.")

if __name__ == "__main__":
    train()