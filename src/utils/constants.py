
class DataFile:
    ETH_DATA = "eth_data_1m.csv"
    SPY_DATA = "sp500_data_1h.csv"
    TRAINING_DATA = "training_data.csv"
    TESTING_DATA = "testing_data.csv"

class FeatureEngineering:
    MOVING_AVERAGE_WINDOWS = [5, 15, 60]
    RSI_WINDOW = 14
    BOLLINGER_WINDOW = 20
    BOLLINGER_STD_DEV = 2

class Garch:
    RESAMPLE_PERIOD = "1min"
    CONFIDENCE_MULTIPLIER = 1.96