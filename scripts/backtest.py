import pandas as pd
import lightgbm as lgb
import os
import sys
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.helpers.feature_engineering_helper import create_features_for_training
from src.utils import constants

MODEL_PATH = "models/btc_predictor.txt"
BACKTEST_DATA_PATH = "testing_data.csv"

def run_backtest():
    print("--- Starting Performance Backtest ---")
    model = lgb.Booster(model_file=MODEL_PATH)
    
    test_df = pd.read_csv(BACKTEST_DATA_PATH)

    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], utc=True)
    
    print("ℹ️ Engineering features for the test dataset...")
    features_df = create_features_for_training(test_df.set_index('timestamp'))
    features_df.dropna(inplace=True)

    current_price = features_df['price']
    y_true_change = features_df['target']
    
    X_test = features_df.drop(columns=['target', 'price'])

    print(f"ℹ️ Generating predictions for {len(X_test)} data points...")
    predicted_change = model.predict(X_test)

    final_prediction = current_price + predicted_change
    final_actual = current_price + y_true_change
    
    mae = mean_absolute_error(final_actual, final_prediction)
    rmse = root_mean_squared_error(final_actual, final_prediction)

    print("\n--- Backtest Results ---")
    print(f"Mean Absolute Error (MAE):    ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")

if __name__ == "__main__":
    run_backtest()