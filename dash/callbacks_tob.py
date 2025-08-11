# dash/callbacks_tob.py
import plotly.graph_objects as go
from dash import Input, Output, no_update

from data import load_latest_tob


def register_callbacks(app):
    @app.callback(
        Output("tob-graph", "figure"),
        Output("spread-graph", "figure"),
        Input("refresh", "n_intervals"),
        Input("symbol-dropdown", "value"),
        Input("limit-input", "value"),
    )
    def update_graphs(_, symbol, limit):
        if not symbol:
            return no_update, no_update
        limit = int(limit or 3000)

        df = load_latest_tob(symbol, limit=limit)
        if df.empty:
            # empty figs
            return go.Figure(), go.Figure()

        # Price figure (bid/ask + mid)
        f_price = go.Figure()
        f_price.add_trace(go.Scatter(x=df["ts"], y=df["bid"], name="Bid", mode="lines"))
        f_price.add_trace(go.Scatter(x=df["ts"], y=df["ask"], name="Ask", mode="lines"))
        f_price.add_trace(go.Scatter(x=df["ts"], y=df["mid"], name="Mid", mode="lines"))
        f_price.update_layout(
            margin=dict(l=40, r=10, t=30, b=30),
            xaxis_title="Time (UTC)",
            yaxis_title=f"{symbol} Price",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        # Spread figure (in bps)
        f_spread = go.Figure()
        f_spread.add_trace(go.Scatter(x=df["ts"], y=df["spread_bps"], name="Spread (bps)", mode="lines"))
        f_spread.update_layout(
            margin=dict(l=40, r=10, t=10, b=30),
            xaxis_title="Time (UTC)",
            yaxis_title="Spread (bps)",
            showlegend=False,
        )

        return f_price, f_spread
