import pandas as pd
import numpy as np

from src.controllers.prediction_controller import run_prediction_flow
from datetime import datetime, timezone

class MockSynapse:
    def __init__(self, timestamp: str):
        self.timestamp = timestamp
        self.prediction = None
        self.interval = None

class MockCMClient:
    def get_CM_ReferenceRate(self, **kwargs) -> pd.DataFrame:
        print("MockCMClient, generating mock historical data")
        end_dt = pd.to_datetime(kwargs.get("end"))
        limit = kwargs.get("limit_per_asset", 604800)
        
        timestamps = pd.to_datetime(pd.date_range(end=end_dt, periods=limit, freq="1s"))
        price = 60000 + np.linspace(0, 500, limit) + np.random.randn(limit) * 25
        
        return pd.DataFrame({'time': timestamps, 'ReferenceRateUSD': price})

def main():
    print("Running prediction flow with >>>> mock <<<<")
    
    mock_cm_client = MockCMClient()
    mock_synapse = MockSynapse(timestamp=datetime.now(timezone.utc).isoformat())

    updated_synapse = run_prediction_flow(mock_synapse, mock_cm_client)
    
    print("\nFlow finished")
    if updated_synapse.prediction is not None:
        print("Successss")
        print(f"Point estimate: ${updated_synapse.prediction:,.2f}")
        print(f"Prediction interval: (${updated_synapse.interval[0]:,.2f}, ${updated_synapse.interval[1]:,.2f})")
    else:
        print("Prediction failed")


if __name__ == "__main__":
    main()