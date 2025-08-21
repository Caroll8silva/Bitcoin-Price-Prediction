import pandas as pd
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.utils import constants

BTC_DATA_PATH = constants.DataFile.TRAINING_DATA
ETH_DATA_PATH = constants.DataFile.ETH_DATA
SPY_DATA_PATH = constants.DataFile.SPY_DATA

HISTORY_SIZE = 180
TARGET_DISTANCE = 60

def create_sequences(input_df, history_size, target_distance):
    features_to_use = ['price', 'eth_btc_ratio', 'spy_hourly_return']
    data = input_df[features_to_use].values
    
    X, y = [], []
    start_index = history_size
    end_index = len(data) - target_distance

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        X.append(data[indices])
        
        target_change = data[i + target_distance - 1][0] - data[i - 1][0]
        y.append(target_change)

    return np.array(X), np.array(y)

def main():
    print("--- Preparing Data for Transformer Model ---")
    
    print("ℹ️ Loading raw data files (BTC, ETH, SPY)...")
    btc_df = pd.read_csv(BTC_DATA_PATH, index_col='timestamp', parse_dates=True)
    eth_df = pd.read_csv(ETH_DATA_PATH, index_col='timestamp', parse_dates=True)
    
    spy_df = pd.read_csv(SPY_DATA_PATH, header=0)
    spy_df = spy_df.iloc[2:].copy()
    spy_df.rename(columns={'Price': 'Datetime'}, inplace=True)
    spy_df['Datetime'] = pd.to_datetime(spy_df['Datetime'], utc=True)
    spy_df.set_index('Datetime', inplace=True)

    if btc_df.index.tz is None: btc_df.index = btc_df.index.tz_localize('UTC')
    if eth_df.index.tz is None: eth_df.index = eth_df.index.tz_localize('UTC')

    eth_df = eth_df[['price']].rename(columns={'price': 'eth_price'})
    spy_df = spy_df[['Close']].rename(columns={'Close': 'spy_price'})
    spy_df['spy_price'] = pd.to_numeric(spy_df['spy_price'], errors='coerce')

    print("ℹ️ Merging and aligning time-series data...")
    df_merged = pd.merge_asof(btc_df, eth_df, left_index=True, right_index=True, direction='backward')
    df_merged = pd.merge_asof(df_merged, spy_df, left_index=True, right_index=True, direction='backward')
    df_merged.ffill(inplace=True)

    df_merged['eth_btc_ratio'] = df_merged['eth_price'] / df_merged['price']
    df_merged['spy_hourly_return'] = df_merged['spy_price'].pct_change()
    df_merged.dropna(inplace=True)

    print("ℹ️ Creating training sequences...")
    X_train, y_train = create_sequences(df_merged, HISTORY_SIZE, TARGET_DISTANCE)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    
    print("ℹ️ Saving processed data to .npy files...")
    np.save('train_X_tf.npy', X_train)
    np.save('train_y_tf.npy', y_train)
    
    print("✅ Data preparation for Transformer is complete.")

if __name__ == "__main__":
    main()