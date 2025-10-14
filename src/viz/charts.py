from typing import Literal
import plotly.graph_objects as go
from viz.contracts import PriceFrame

ChartId = Literal["price_ma"]

def make_price_ma(pf: PriceFrame):
    df = pf.df
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ma50"],  name="MA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ma200"], name="MA200"))
    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), title="Price & Moving Averages")
    return fig

REGISTRY = {"price_ma": make_price_ma}

def make_chart(chart_id: ChartId, pf: PriceFrame):
    return REGISTRY[chart_id](pf)
