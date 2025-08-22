import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytz
from dash import Input, Output
from plotly.subplots import make_subplots

from dashboard.data_loader import load_tob_range
from dashboard.utils import _to_utc_ts

NY = pytz.timezone("America/New_York")
ORDERS_BT_DB = Path(__file__).resolve().parents[1] / "data" / "db" / "orders_backtest.db"


def _fx_reset_window_for_date(date_str: str):
    y, m, d = map(int, date_str.split("-"))
    day_ny = NY.localize(datetime(y, m, d))
    start_ny = day_ny.replace(hour=16, minute=55, second=0, microsecond=0)
    end_ny = start_ny + timedelta(days=1)
    return start_ny.astimezone(timezone.utc), end_ny.astimezone(timezone.utc)


def _load_bt_exec_range(symbol: str, start_iso: str, end_iso: str, db_path=ORDERS_BT_DB):
    """
    Load executions from backtest orders DB for [start,end).
    Expect columns: ts TEXT(ISO8601), price, qty, side
    """
    if not Path(db_path).exists():
        return pd.DataFrame(columns=["ts", "price", "qty", "side"])
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(
            """
            SELECT ts, price, qty, side, order_type
            FROM executions
            WHERE symbol = ? AND ts >= ? AND ts < ?
            ORDER BY ts ASC
            """,
            con, params=(symbol, start_iso, end_iso)
        )
    return df


def register_backtest_callbacks(app):
    @app.callback(
        Output("bt-graph", "figure"),
        Input("bt-symbol", "value"),
        Input("bt-date", "date"),
    )
    def update_backtest(symbol, date_str):
        try:
            # 1) time window
            if not date_str:
                date_str = str(datetime.now(NY).date())
            start_utc, end_utc = _fx_reset_window_for_date(date_str)
            start_iso, end_iso = start_utc.isoformat(), end_utc.isoformat()

            # 2) load data
            df = _to_utc_ts(load_tob_range(symbol, start_iso, end_iso))
            if not df.empty:
                df["mid"] = (df["bid"] + df["ask"]) / 2.0

            df_exec = _to_utc_ts(_load_bt_exec_range(symbol, start_iso, end_iso))
            # normalize sides
            if not df_exec.empty and "side" in df_exec:
                df_exec["side"] = df_exec["side"].str.upper().map({"BOT": "BUY", "SLD": "SELL"}).fillna(df_exec["side"])

            # 3) figure
            fig = make_subplots(
                rows=1, cols=1, shared_xaxes=True,
                vertical_spacing=0.06, row_heights=[1.0],
                subplot_titles=(f"[Backtest] {symbol} Price & Executions ({date_str})",),
            )

            if not df.empty:
                fig.add_trace(go.Scatter(x=df["ts"], y=df["bid"], name="Bid", mode="lines"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df["ts"], y=df["ask"], name="Ask", mode="lines"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df["ts"], y=df["mid"], name="Mid", mode="lines"), row=1, col=1)

            if not df_exec.empty:
                buys = df_exec[df_exec["side"].str.upper() == "BUY"]
                sells = df_exec[df_exec["side"].str.upper() == "SELL"]
                if not buys.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=buys["ts"], y=buys["price"], mode="markers",
                            name="Buy fills", marker=dict(symbol="triangle-up", size=10),
                            hovertemplate="BUY<br>%{x|%Y-%m-%d %H:%M:%S}<br>px=%{y:.6f}<extra></extra>",
                        ),
                        row=1, col=1
                    )
                if not sells.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sells["ts"], y=sells["price"], mode="markers",
                            name="Sell fills", marker=dict(symbol="triangle-down", size=10),
                            hovertemplate="SELL<br>%{x|%Y-%m-%d %H:%M:%S}<br>px=%{y:.6f}<extra></extra>",
                        ),
                        row=1, col=1
                    )

            fig.update_layout(
                margin=dict(l=40, r=10, t=40, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
            )
            fig.update_yaxes(title_text=f"{symbol} Price", row=1, col=1)
            fig.update_xaxes(title_text="Time (UTC)", row=1, col=1)

            if not df.empty:
                fig.update_xaxes(range=[df["ts"].min(), df["ts"].max()], row=1, col=1)

            return fig
        except Exception:
            import traceback;
            traceback.print_exc()
            return go.Figure()
