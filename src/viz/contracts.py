from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class PriceFrame:
    df: pd.DataFrame  # index: DatetimeIndex; cols: close, ma50, ma200, rsi14, volatility
