from datetime import datetime

import plotly.graph_objects as go
import pytz
from dash import Input, Output
from plotly.subplots import make_subplots

from dashboard.callbacks_tob import fx_reset_window_for_date
from dashboard.data_loader import load_tob_range, load_fills_range, BACKTEST_DB
from dashboard.utils import _to_utc_ts

NY = pytz.timezone("America/New_York")


def register_backtest_callbacks(app):
    @app.callback(
        Output("bt-graph", "figure"),
        Input("bt-symbol", "value"),
        Input("bt-date", "date"),
    )
    def update_backtest(symbol, date_str):
        # 1) time window
        if not date_str:
            date_str = str(datetime.now(NY).date())
        start_utc, end_utc = fx_reset_window_for_date(date_str)
        start_iso, end_iso = start_utc.isoformat(), end_utc.isoformat()
        print(start_iso, end_iso)

        # Load Data SQL
        df_mid = _to_utc_ts(load_tob_range(symbol, start_iso, end_iso))
        df_exec = _to_utc_ts(load_fills_range(symbol, start_iso, end_iso, BACKTEST_DB))
        print("df_exec: ", df_exec)

        pnl_rows = [{
            "ts": "TOTAL",
            "side": "", "usd": 0.0, "qty": 0.0, "price": "", "mid_exec": "",
            "order_type": "", "liq": "",
            "spread_pnl": 0.0, "market_pnl": 0.0, "commission": 0.0, "row_total": 0.0,
        }]
        if df_mid.empty:
            return go.Figure()

        # ---- figure with shared x-axis
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.06, row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol} Price & Executions", "PnL"),
        )

        # ----- Figure Row 1: Price
        fig.add_trace(go.Scatter(x=df_mid["ts"], y=df_mid["bid"], name="Bid", mode="lines"), row=1,
                      col=1)
        fig.add_trace(go.Scatter(x=df_mid["ts"], y=df_mid["ask"], name="Ask", mode="lines"), row=1,
                      col=1)
        fig.add_trace(go.Scatter(x=df_mid["ts"], y=df_mid["mid"], name="Mid", mode="lines"), row=1,
                      col=1)

        # ---- layout / axes
        fig.update_layout(
            margin=dict(l=40, r=10, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        )
        fig.update_yaxes(title_text=f"{symbol} Price", row=1, col=1)
        fig.update_yaxes(title_text="PnL (quote ccy)", row=2, col=1)
        fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)

        x0, x1 = df_mid["ts"].min(), df_mid["ts"].max()
        fig.update_xaxes(range=[x0, x1], row=1, col=1)
        fig.update_xaxes(range=[x0, x1], row=2, col=1)

        if df_exec.empty:
            return fig

        # ----- Figure Row 1: Executions
        df_buy = df_exec[df_exec["side"].str.upper().isin(["BUY", "BOT"])]
        df_sell = df_exec[df_exec["side"].str.upper().isin(["SELL", "SLD"])]

        if not df_buy.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_buy["ts"], y=df_buy["price"], mode="markers",
                    name="Buy fills", marker=dict(symbol="triangle-up", size=10),
                    hovertemplate="BUY<br>%{x|%Y-%m-%d %H:%M:%S}<br>px=%{y:.6f}<extra></extra>",
                ),
                row=1, col=1
            )
        if not df_sell.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_sell["ts"], y=df_sell["price"], mode="markers",
                    name="Sell fills", marker=dict(symbol="triangle-down", size=10),
                    hovertemplate="SELL<br>%{x|%Y-%m-%d %H:%M:%S}<br>px=%{y:.6f}<extra></extra>",
                ),
                row=1, col=1
            )

        return fig
