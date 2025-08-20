import pandas as pd
import numpy as np
from ..utils import constants

def _load_external_data():
    try:
        eth_df = pd.read_csv(constants.DataFile.ETH_DATA, index_col='timestamp', parse_dates=True)
        if eth_df.index.tz is None:
            eth_df.index = eth_df.index.tz_localize('UTC')
        else:
            eth_df.index = eth_df.index.tz_convert('UTC')
        eth_df = eth_df[['price']].rename(columns={'price': 'eth_price'})

        spy_df = pd.read_csv(constants.DataFile.SPY_DATA, header=0)
        spy_df = spy_df.iloc[2:].copy()
        spy_df.rename(columns={'Price': 'Datetime'}, inplace=True)
        spy_df['Datetime'] = pd.to_datetime(spy_df['Datetime'], utc=True)
        spy_df.set_index('Datetime', inplace=True)
        spy_df = spy_df[['Close']].rename(columns={'Close': 'spy_price'})
        spy_df['spy_price'] = pd.to_numeric(spy_df['spy_price'], errors='coerce')
        
        return eth_df, spy_df
    except Exception as e:
        print(f"ERROR: Could not load or parse external data files. Details: {e}")
        return None, None


def _create_base_features(df: pd.DataFrame) -> pd.DataFrame:
   
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
        
    eth_df, spy_df = _load_external_data()
    if eth_df is None or spy_df is None:
        raise RuntimeError("Could not proceed without external data.")

    df_merged = pd.merge_asof(df, eth_df, left_index=True, right_index=True, direction='backward')
    df_merged = pd.merge_asof(df_merged, spy_df, left_index=True, right_index=True, direction='backward')
    
    df_merged['spy_price'] = df_merged['spy_price'].ffill()
    df_merged['eth_price'] = df_merged['eth_price'].ffill()
    
    df_merged['eth_btc_ratio'] = df_merged['eth_price'] / df_merged['price']
    df_merged['spy_hourly_return'] = df_merged['spy_price'].pct_change()

    df_resampled = df_merged.resample('15min').agg({
        'price': 'last', 'eth_price': 'last', 'spy_price': 'last',
        'eth_btc_ratio': 'last', 'spy_hourly_return': 'last',
    })

    df_resampled['hour'] = df_resampled.index.hour
    df_resampled['dayofweek'] = df_resampled.index.dayofweek

    for lag in [1, 2, 4, 8]:
        df_resampled[f'price_lag_{lag}'] = df_resampled['price'].shift(lag)
        df_resampled[f'eth_btc_ratio_lag_{lag}'] = df_resampled['eth_btc_ratio'].shift(lag)

    for window in [4, 8, 12, 24]:
        df_resampled[f'rolling_mean_{window}'] = df_resampled['price'].rolling(window=window).mean()
        df_resampled[f'rolling_std_{window}'] = df_resampled['price'].rolling(window=window).std()

    delta = df_resampled['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df_resampled['rsi'] = 100 - (100 / (1 + (gain/loss)))
    
    return df_resampled

def create_features_for_training(df: pd.DataFrame) -> pd.DataFrame:
    features_df = _create_base_features(df)
    future_price = features_df['price'].shift(-4)
    features_df['target'] = future_price - features_df['price']
    return features_df.drop(columns=['eth_price', 'spy_price'])

def create_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    features_df = create_features_for_training(df).drop(columns=['target'])
    return features_df.iloc[[-1]]