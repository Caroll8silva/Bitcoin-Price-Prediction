import lightgbm as lgb
import pandas as pd
import os

from src.helpers.feature_engineering_helper import create_features_for_training
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("PREDICTION_MODEL_PATH")
HISTORICAL_DATA_PATH = os.getenv("HISTORICAL_DATA_PATH")

def train():
    print("Starting Model training")

    print(f"Loading historical training data from: {HISTORICAL_DATA_PATH}")
    try:
        historical_df = pd.read_csv(HISTORICAL_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: training data file not found at '{HISTORICAL_DATA_PATH}'.")
        return
        
    historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
    historical_df = historical_df.set_index('timestamp')

    features_df = create_features_for_training(historical_df)
    
    features_df.dropna(inplace=True)

    X = features_df.drop(columns=['target_price'])
    y = features_df['target_price']

    print("Training LightGBM model...")
    
    lgbm = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    lgbm.fit(X, y)

    print(f"Saving trained model to: {MODEL_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    lgbm.booster_.save_model(MODEL_PATH)
    
    print("Training complete.")
    print(f"Successs. Saved at '{MODEL_PATH}'.")

if __name__ == "__main__":
    train()