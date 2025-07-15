import pandas as pd

from typing import Any

class CryptoDataProvider:
    def __init__(self, cm_client: Any):
        if not hasattr(cm_client, 'get_CM_ReferenceRate'):
            raise AttributeError("The provided client must have a 'get_CM_ReferenceRate' method.")
        self.client = cm_client

    def get_btc_price_data(self, end_timestamp_str: str) -> pd.DataFrame:
        end_dt = pd.to_datetime(end_timestamp_str)
        
        price_data = self.client.get_CM_ReferenceRate(
            assets="BTC",
            end=end_dt.isoformat(),
            frequency="1s",
            limit_per_asset=604800,
            paging_from="end",
            use_cache=False,
        )
        
        if price_data.empty:
            raise ValueError("Failed to retrieve price data.")
        
        price_data = price_data.rename(columns={"ReferenceRateUSD": "price"})
        price_data['timestamp'] = pd.to_datetime(price_data['time'])
        price_data = price_data.set_index('timestamp')
        
        return price_data[['price']]