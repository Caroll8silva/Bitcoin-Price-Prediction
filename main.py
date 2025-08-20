import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.controllers.prediction_controller import run_prediction_flow

class MockSynapse:
    def __init__(self, timestamp: str):
        self.timestamp = timestamp
        self.prediction = None
        self.interval = None
    def __str__(self) -> str:
        if self.prediction is not None:
            return (f"Synapse(prediction=${self.prediction:,.2f}, "
                    f"interval=(${self.interval[0]:,.2f}, ${self.interval[1]:,.2f}))")
        return "Synapse(prediction=None, interval=None)"

class MockCMClient:
    def get_CM_ReferenceRate(self, **kwargs) -> pd.DataFrame:
        print("INFO: MockCMClient: Generating mock historical data...")
        end_dt = pd.to_datetime(kwargs.get("end"))
        limit = kwargs.get("limit_per_asset")
        freq = kwargs.get("frequency")
        timestamps = pd.to_datetime(pd.date_range(end=end_dt, periods=limit, freq=freq))
        price = 60000 + np.linspace(0, 500, limit) + np.random.randn(limit) * 25
        return pd.DataFrame({'time': timestamps, 'ReferenceRateUSD': price})

def main():
    print("--- Running Prediction Flow with Mock Data ---")
    mock_cm_client = MockCMClient()
    mock_synapse = MockSynapse(timestamp=datetime.now(timezone.utc).isoformat())
    run_prediction_flow(mock_synapse, mock_cm_client)
    print("\n--- Flow Finished ---")
    if mock_synapse.prediction is not None:
        print("Prediction Successful")
        print(f"Final Synapse State: {mock_synapse}")
    else:
        print("Prediction Failed")

if __name__ == "__main__":
    main()