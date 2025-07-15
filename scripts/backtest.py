import lightgbm as lgb
import pandas as pd
import os

from src.helpers.feature_engineering_helper import create_features_for_prediction
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

BACKTEST_DATA_PATH = os.getenv("BACKTEST_DATA_PATH")
MODEL_PATH = "models/btc_predictor.txt"

def run_backtest():
    print("Starting performance")
    print(f"loading model from: {MODEL_PATH}")
    try:
        model = lgb.Booster(model_file=MODEL_PATH)
    except lgb.basic.LightGBMError:
        print(f"ERROR: model file not found at '{MODEL_PATH}'")
        return

    print(f"loading test data from: {BACKTEST_DATA_PATH}")
    try:
        test_df = pd.read_csv(BACKTEST_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: test data file not found at '{BACKTEST_DATA_PATH}'.")
        return

    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    predictions = []
    actuals = []
    
    start_index = 60 
    simulation_size = len(test_df) - start_index
    
    print(f"simulating predictions for {simulation_size}")

    for i in range(start_index, len(test_df)):
        historical_slice = test_df.iloc[:i]
        actual_price = test_df.iloc[i]['price']
        
        features = create_features_for_prediction(historical_slice.set_index('timestamp'))
        predicted_price = model.predict(features)[0]
        
        predictions.append(predicted_price)
        actuals.append(actual_price)

    print("backtest simulation complete.")

    mae = mean_absolute_error(actuals, predictions)
    rmse = root_mean_squared_error(actuals, predictions)

    print(f"mean absolute error (MAE):    ${mae:,.2f}")
    print(f"root mean squared error (RMSE): ${rmse:,.2f}")

if __name__ == "__main__":
    run_backtest()