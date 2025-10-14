import pandas as pd
from viz.contracts import PriceFrame

NEEDED = ["close", "ma50", "ma200", "rsi14", "volatility"]

def load_price_frame(ticker: str) -> PriceFrame:
    df = pd.read_parquet(f"data/{ticker}/prices.parquet").sort_index()
    for c in NEEDED:
        if c not in df.columns:
            df[c] = pd.NA
    return PriceFrame(df=df[NEEDED])
