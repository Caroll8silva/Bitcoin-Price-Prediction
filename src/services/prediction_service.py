from dotenv import load_dotenv
from arch import arch_model
import lightgbm as lgb
import pandas as pd
import numpy as np
import warnings
import os

from ..helpers import feature_engineering_helper
from ..utils import constants

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

class BitcoinPredictionService:
    def __init__(self):
        model_path = os.getenv("PREDICTION_MODEL_PATH")
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = lgb.Booster(model_file=model_path)

    def _calculate_dynamic_interval(self, price_series: pd.Series) -> float:
        resampled_prices = price_series.resample(constants.Garch.RESAMPLE_PERIOD).last()
        returns = 100 * resampled_prices.pct_change().dropna()
        
        if returns.empty or returns.std() == 0:
            return 250.0 

        garch = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
        garch_fit = garch.fit(disp='off', show_warning=False)
        
        forecast = garch_fit.forecast(horizon=1)
        predicted_variance = forecast.variance.iloc[-1, 0]
        
        predicted_volatility = np.sqrt(predicted_variance)
        interval_delta = predicted_volatility * constants.Garch.CONFIDENCE_MULTIPLIER
        
        if not np.isfinite(interval_delta) or interval_delta <= 0:
            return returns.std() * constants.Garch.CONFIDENCE_MULTIPLIER

        return interval_delta

    def create_prediction(self, price_data: pd.DataFrame) -> tuple[float, tuple[float, float]]:
        features = feature_engineering_helper.create_features_for_prediction(price_data)
        
        point_estimate = self.model.predict(features)[0]
        
        interval_delta = self._calculate_dynamic_interval(price_data['price'])
        
        prediction_interval = (point_estimate - interval_delta, point_estimate + interval_delta)
        
        return point_estimate, prediction_interval