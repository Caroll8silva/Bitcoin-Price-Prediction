from ..services.prediction_service import BitcoinPredictionService
from ..providers.crypto_data_provider import CryptoDataProvider
from typing import Any

def run_prediction_flow(synapse: Any, cm_client: Any, bt: Any = None):
    try:
        if bt:
            bt.logging.info(f"event=prediction_flow_started, timestamp={synapse.timestamp}")

        data_provider = CryptoDataProvider(cm_client)
        prediction_service = BitcoinPredictionService()

        price_data = data_provider.get_btc_price_data(synapse.timestamp)
        point_estimate, interval = prediction_service.create_prediction(price_data)

        synapse.prediction = point_estimate
        synapse.interval = interval

        if bt:
            bt.logging.success(
                "event=prediction_successful",
                prediction=f"{point_estimate:,.2f}",
                lower_bound=f"{interval[0]:,.2f}",
                upper_bound=f"{interval[1]:,.2f}"
            )

    except Exception as e:
        if bt:
            bt.logging.error(f"event=prediction_flow_failed, error_message='{e}'")
        print(f"An error occurred during prediction: {e}")
        synapse.prediction = None
        synapse.interval = None
        
    return synapse