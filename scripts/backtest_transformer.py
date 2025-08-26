import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sys
import os
import csv
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.utils import constants

MODEL_V2_PATH = 'models/transformer_model.keras'
TEST_DATA_PATH = constants.DataFile.TESTING_DATA
LOG_OUTPUT_PATH = "backtest_log_v2.csv"
HISTORY_SIZE = 180
TARGET_DISTANCE = 60

def create_test_sequences(input_df, history_size, target_distance):
    try:
        eth_df = pd.read_csv(constants.DataFile.ETH_DATA, index_col='timestamp', parse_dates=True)
        spy_df = pd.read_csv(constants.DataFile.SPY_DATA, header=0)
        spy_df = spy_df.iloc[2:].copy()
        spy_df.rename(columns={'Price': 'Datetime'}, inplace=True)
        spy_df['Datetime'] = pd.to_datetime(spy_df['Datetime'], utc=True)
        spy_df.set_index('Datetime', inplace=True)

        if input_df.index.tz is None: input_df.index = input_df.index.tz_localize('UTC')
        if eth_df.index.tz is None: eth_df.index = eth_df.index.tz_localize('UTC')
        

        if 'price' in eth_df.columns:
            eth_df = eth_df[['price']].rename(columns={'price': 'eth_price'})
        elif 'Close' in eth_df.columns:
            eth_df = eth_df[['Close']].rename(columns={'Close': 'eth_price'})
        else:
            raise KeyError("could not find a price column")
        
        spy_df = spy_df[['Close']].rename(columns={'Close': 'spy_price'})

        df_merged = pd.merge_asof(input_df, eth_df, left_index=True, right_index=True, direction='backward')
        df_merged = pd.merge_asof(df_merged, spy_df, left_index=True, right_index=True, direction='backward')

        df_merged['price'] = pd.to_numeric(df_merged['price'], errors='coerce')
        df_merged['eth_price'] = pd.to_numeric(df_merged['eth_price'], errors='coerce')
        df_merged['spy_price'] = pd.to_numeric(df_merged['spy_price'], errors='coerce')
        df_merged.ffill(inplace=True)
        
        df_merged['eth_btc_ratio'] = df_merged['eth_price'] / df_merged['price']
        df_merged['spy_hourly_return'] = df_merged['spy_price'].pct_change()
        df_merged.dropna(inplace=True)
        
    except Exception as e:
        print(f"ERROR: Failed to load or process external data for backtest. Details: {e}")
        return None, None, None, None

    features_to_use = ['price', 'eth_btc_ratio', 'spy_hourly_return']
    data = df_merged[features_to_use].values
    timestamps = df_merged.index
    
    X, y_change, current_prices, final_timestamps = [], [], [], []
    start_index = history_size
    end_index = len(data) - target_distance
    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        X.append(data[indices])
        current_price = data[i - 1][0]
        future_price = data[i + target_distance - 1][0]
        y_change.append(future_price - current_price)
        current_prices.append(current_price)
        final_timestamps.append(timestamps[i + target_distance - 1])
    return np.array(X), np.array(y_change), np.array(current_prices), final_timestamps

def main():
    print("Starting Backtest")
    model = keras.models.load_model(MODEL_V2_PATH)
    test_df_raw = pd.read_csv(TEST_DATA_PATH)
    test_df_raw['timestamp'] = pd.to_datetime(test_df_raw['timestamp'], utc=True)
    
    X_test, y_true_change, current_prices, timestamps = create_test_sequences(
        test_df_raw.set_index('timestamp'), HISTORY_SIZE, TARGET_DISTANCE
    )

    if X_test is None: return

    print(f"Generating predictions for {len(X_test)} data points...")
    predicted_change = model.predict(X_test).flatten()

    final_prediction = current_prices + predicted_change
    final_actual = current_prices + y_true_change

    print("Backtest complete. Calculating performance metrics...")
    mae = mean_absolute_error(final_actual, final_prediction)
    rmse = root_mean_squared_error(final_actual, final_prediction)

    print(f"Writing detailed log to {LOG_OUTPUT_PATH}...")
    with open(LOG_OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "ActualPrice", "PredictedPrice", "AbsoluteError"])
        for i in range(len(final_prediction)):
            error = abs(final_actual[i] - final_prediction[i])
            writer.writerow([timestamps[i], final_actual[i], final_prediction[i], error])
    
    print("\nBacktest Results")
    print(f"Mean Absolute Error (MAE):    ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")

if __name__ == "__main__":
    main()