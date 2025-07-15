class FeatureEngineering:
    MOVING_AVERAGE_WINDOWS = [5, 15, 60]  # In minutes
    RSI_WINDOW = 14
    BOLLINGER_WINDOW = 20
    BOLLINGER_STD_DEV = 2

class Garch:
    RESAMPLE_PERIOD = "1min"
    CONFIDENCE_MULTIPLIER = 1.96  # For a 95% confidence interval