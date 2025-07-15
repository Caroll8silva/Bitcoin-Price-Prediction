import pandas as pd

from ..utils import constants

def _create_base_features(df: pd.DataFrame) -> pd.DataFrame:

    df_resampled = df.resample('1min').last()
    
    for window in constants.FeatureEngineering.MOVING_AVERAGE_WINDOWS:
        df_resampled[f'ma_{window}'] = df_resampled['price'].rolling(window=window).mean()

    delta = df_resampled['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=constants.FeatureEngineering.RSI_WINDOW).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=constants.FeatureEngineering.RSI_WINDOW).mean()
    rs = gain / loss
    df_resampled['rsi'] = 100 - (100 / (1 + rs))
    
    rolling_mean = df_resampled['price'].rolling(window=constants.FeatureEngineering.BOLLINGER_WINDOW).mean()
    rolling_std = df_resampled['price'].rolling(window=constants.FeatureEngineering.BOLLINGER_WINDOW).std()
    df_resampled['bollinger_upper'] = rolling_mean + (rolling_std * constants.FeatureEngineering.BOLLINGER_STD_DEV)
    df_resampled['bollinger_lower'] = rolling_mean - (rolling_std * constants.FeatureEngineering.BOLLINGER_STD_DEV)
    df_resampled['bollinger_width'] = df_resampled['bollinger_upper'] - df_resampled['bollinger_lower']
    
    return df_resampled

def create_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    features_df = _create_base_features(df)
    return features_df.iloc[[-1]].drop(columns=['price'])

def create_features_for_training(df: pd.DataFrame) -> pd.DataFrame:
    features_df = _create_base_features(df)
    features_df['target_price'] = features_df['price'].shift(-1)
    return features_df.drop(columns=['price'])