from datetime import datetime, timezone, timedelta

import pandas as pd
import plotly.graph_objects as go
import pytz
from dash import Input, Output
from plotly.subplots import make_subplots

from dashboard.utils import fifo_realized_unrealized, compute_pnl_curve
from database.db_loader import load_tob_range, load_fills_range, load_commissions_range, ORDERS_DB
from utils.utils_dt import _to_utc_ts

NY = pytz.timezone("America/New_York")


def fx_reset_window_for_date(date_str: str):
    """
    For a YYYY-MM-DD date (NY), return (start_utc, end_utc) where the FX day is
    16:55 NY on that date -> 16:55 NY next date.
    """
    # parse selected date as NY-local midnight of that day
    y, m, d = map(int, date_str.split("-"))
    day_ny = NY.localize(datetime(y, m, d))
    end_ny = day_ny.replace(hour=23, minute=58, second=0, microsecond=0)
    # end_ny = day_ny.replace(hour=16, minute=55, second=0, microsecond=0)
    start_ny = end_ny + timedelta(days=-1)
    return start_ny.astimezone(timezone.utc), end_ny.astimezone(timezone.utc)


def fx_reset_start_utc(now_utc: datetime | None = None) -> datetime:
    """
    IBKR reset for FX: 16:55 America/New_York every trading day.
    Returns that instant in UTC for the current 'today' boundary.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(NY)
    # today 16:55 NY
    reset_today_ny = now_ny.replace(hour=16, minute=55, second=0, microsecond=0)
    if now_ny >= reset_today_ny:
        reset_ny = reset_today_ny
    else:
        reset_ny = reset_today_ny - timedelta(days=1)
    return reset_ny.astimezone(timezone.utc)


def register_callbacks(app):
    @app.callback(
        Output("refresh", "disabled"),
        Input("stop-refresh", "value")
    )
    def toggle_refresh(stop_values):
        return "stop" in stop_values  # True disables the interval

    @app.callback(
        Output("tob-graph", "figure"),
        Output("pnl-table", "data"),
        Input("refresh", "n_intervals"),
        Input("symbol-dropdown", "value"),
        Input("lookback-min", "value"),
        Input("trades-date", "date"),
    )
    def update_price_pnl(n_intervals, symbol, lookback_min, trades_date):
        is_today = datetime.today().strftime('%Y-%m-%d') == trades_date

        lookback_min = int(lookback_min or 30)
        lookback_iso = (datetime.now(timezone.utc) - timedelta(minutes=lookback_min)).isoformat()

        start_utc, end_utc = fx_reset_window_for_date(trades_date or str(datetime.now(NY).date()))
        start_iso = start_utc.isoformat()
        end_iso = end_utc.isoformat()
        # print(f'{start_iso} - {end_iso} | lookback: {lookback_iso}')

        # Load Data SQL
        df_mid = _to_utc_ts(load_tob_range(symbol, start_iso, end_iso))
        df_exec = _to_utc_ts(load_fills_range(symbol, start_iso, end_iso, ORDERS_DB))
        df_comm = _to_utc_ts(load_commissions_range(symbol, start_iso, end_iso, ORDERS_DB))

        pnl_rows = [{
            "ts": "TOTAL",
            "side": "", "usd": 0.0, "qty": 0.0, "price": "", "mid_exec": "",
            "order_type": "", "liq": "",
            "spread_pnl": 0.0, "market_pnl": 0.0, "commission": 0.0, "row_total": 0.0,
        }]
        if df_mid.empty:
            return go.Figure(), pnl_rows

        # ---- figure with shared x-axis
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.06, row_heights=[0.6, 0.20, 0.20],
            specs=[[{}], [{"secondary_y": True}], [{}]],
            subplot_titles=(f"{symbol} Price & Executions", "Activity", "PnL"),
        )

        # ----- Figure Row 1: Price
        PIP = 1e-4
        df_mid_lookback = df_mid[df_mid['ts'] >= lookback_iso].copy() if is_today else df_mid
        df_mid_lookback["spread"] = (df_mid_lookback["ask"] - df_mid_lookback["bid"]) / PIP
        df_mid_lookback["d_mid"] = df_mid_lookback["mid"].diff() / PIP

        res = (
            df_mid_lookback.set_index("ts")
            .resample("30s")  # 1 minute
            .agg(
                tick_count=("mid", "size"),  # number of TOB updates
                avg_spread=("spread", "mean"),  # average spread (price)
                # realized vol in pips within the minute (std of 1s deltas)
                vol=("d_mid", "std"),
            )
            .reset_index()
        ).fillna(0.0)

        fig.add_trace(go.Scatter(x=df_mid_lookback["ts"], y=df_mid_lookback["bid"], name="Bid", mode="lines"), row=1,
                      col=1)
        fig.add_trace(go.Scatter(x=df_mid_lookback["ts"], y=df_mid_lookback["ask"], name="Ask", mode="lines"), row=1,
                      col=1)
        fig.add_trace(go.Scatter(x=df_mid_lookback["ts"], y=df_mid_lookback["mid"], name="Mid", mode="lines",
                                 visible="legendonly"), row=1,
                      col=1)

        fig.add_trace(
            go.Bar(
                x=res["ts"], y=res["tick_count"], name="Ticks/min",
                hovertemplate="Ticks=%{y:.0f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
            ),
            row=2, col=1, secondary_y=False
        )

        # avg spread (line, pips)
        fig.add_trace(
            go.Scatter(
                x=res["ts"], y=res["avg_spread"], mode="lines", name="Avg spread (pips)",
                hovertemplate="Spread=%{y:.2f} pips<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
            ),
            row=2, col=1, secondary_y=True
        )

        # realized vol (line, pips)
        fig.add_trace(
            go.Scatter(
                x=res["ts"], y=res["vol"], mode="lines", name="Vol (pips, 1‑min σ)",
                hovertemplate="Vol=%{y:.2f} pips<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
            ),
            row=2, col=1, secondary_y=True
        )

        # ---- layout / axes
        fig.update_layout(
            margin=dict(l=40, r=10, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        )
        fig.update_yaxes(title_text=f"Price", row=1, col=1)
        fig.update_yaxes(title_text="Ticks", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Pips", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="PnL", row=3, col=1)
        fig.update_xaxes(title_text="Time (UTC)", row=3, col=1)

        x0, x1 = df_mid_lookback["ts"].min(), df_mid_lookback["ts"].max()
        for r in (1, 2, 3):
            fig.update_xaxes(range=[x0, x1], row=r, col=1)

        if df_exec is None or df_exec.empty:
            return fig, pnl_rows

        # ----- Figure Row 1: Executions
        df_exec_lookback = df_exec[df_exec['ts'] >= lookback_iso] if is_today else df_exec
        df_buy = df_exec_lookback[df_exec_lookback["side"].str.upper().isin(["BUY", "BOT"])]
        df_sell = df_exec_lookback[df_exec_lookback["side"].str.upper().isin(["SELL", "SLD"])]

        if not df_buy.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_buy["ts"], y=df_buy["price"], mode="markers",
                    name="Buy fills", marker=dict(symbol="triangle-up", size=10, color='black'),
                    hovertemplate="BUY<br>%{x|%Y-%m-%d %H:%M:%S}<br>px=%{y:.6f}<extra></extra>",
                ),
                row=1, col=1
            )
        if not df_sell.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_sell["ts"], y=df_sell["price"], mode="markers",
                    name="Sell fills", marker=dict(symbol="triangle-down", size=10, color='black'),
                    hovertemplate="SELL<br>%{x|%Y-%m-%d %H:%M:%S}<br>px=%{y:.6f}<extra></extra>",
                ),
                row=1, col=1
            )

        ####################### PNL CALCULATION ##############################

        # 1) Join mid at fill
        df_exec = pd.merge_asof(df_exec, df_mid[["ts", "bid", "mid", "ask"]], on="ts", direction="backward")

        # 2) Commissions by exec_id (fallback to zero if missing)
        comm_map = df_comm.set_index("order_id")["commission"].to_dict()
        df_exec["fee"] = df_exec["order_id"].map(comm_map).fillna(0.0).astype(float)

        # 3) sign: BUY=+1, SELL=-1
        side_u = df_exec["side"].str.upper().map({"BOT": "BUY", "SLD": "SELL"}).fillna(df_exec["side"].str.upper())
        df_exec["sign"] = side_u.map({"BUY": 1.0, "SELL": -1.0}).astype(float)

        # 4) USD column (absolute notional in quote ccy at fill price)
        df_exec["usd"] = (df_exec["qty"].abs() * df_exec["price"]).round(0)

        # 5) Liquidity - market making/taking
        # df_exec["liq"] = df_exec["liquidity"].map({1: "MAKER", 2: "TAKER", 3: "ROUTED", 4: "AUCTION"}).fillna("")

        # 6) Spread/market PnL
        df_exec["spread_pnl"] = df_exec["sign"] * (df_exec["mid"] - df_exec["price"]) * df_exec["qty"]
        df_exec["market_pnl"] = df_exec["sign"] * (df_mid.iloc[-1]['mid'] - df_exec["mid"]) * df_exec["qty"]

        pnl_curve = compute_pnl_curve(df_mid[["ts", "mid"]], df_exec)

        # 7) PnL curve - Figure Row 2
        curve = fifo_realized_unrealized(df_exec[["ts", "price", "qty", "side", "fee", "spread_pnl"]],
                                         df_mid[["ts", "mid"]])
        fig.add_trace(go.Scatter(x=pnl_curve["ts"], y=pnl_curve["total_pnl"], name="PnL", mode="lines"), row=3, col=1)
        # optional components:
        # fig.add_trace(go.Scatter(x=curve["ts"], y=curve["realized"], name="Realized", mode="lines"), row=2, col=1)
        # fig.add_trace(go.Scatter(x=curve["ts"], y=curve["unrealized"], name="Unrealized", mode="lines"), row=2, col=1)
        # fig.add_trace(go.Scatter(x=curve["ts"], y=-curve["fees_cum"], name="Fees", mode="lines"), row=2, col=1)
        # fig.add_trace(go.Scatter(x=curve["ts"], y=curve["impact_cum"], name="Impact", mode="lines"), row=2, col=1)

        # 8) Per-row TOTAL = spread + market_pnl - commission
        df_exec["row_total"] = (df_exec["spread_pnl"] + df_exec["market_pnl"] - df_exec["fee"])

        # 9) Roundings:
        df_exec["price"] = df_exec["price"].round(6)
        df_exec["mid"] = df_exec["mid"].round(6)
        for c in ["spread_pnl", "fee", "market_pnl", "row_total"]:
            df_exec[c] = df_exec[c].round(2)

        # 10) Build TOTAL (Today) row
        tot_spread = float(df_exec["spread_pnl"].sum())
        tot_mtm = float(df_exec["market_pnl"].sum())
        tot_comm = float(df_exec["fee"].sum())

        total_row = {
            "ts": "TOTAL",
            "side": "",
            "usd": round(float(df_exec["usd"].sum()), 0),
            "qty": round(float(df_exec["qty"].sum()), 0),
            "price": "",
            "mid": "",
            "order_type": "",
            "liq": "",
            "spread_pnl": round(tot_spread, 2),
            "market_pnl": round(tot_mtm, 2),
            "fee": round(tot_comm, 2),
            "row_total": round(tot_spread + tot_mtm - tot_comm, 2),
        }

        # 12) Select/display latest 30 rows, newest first, and prepend TOTAL row
        view = df_exec[["ts", "side", "usd", "qty", "price", "bid", "mid", "ask", "order_type",  # "liq",
                        "spread_pnl", "market_pnl", "fee", "row_total"]]
        view = view.sort_values("ts", ascending=False)
        view["ts"] = view["ts"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        pnl_rows = [total_row] + view.to_dict("records")

        return fig, pnl_rows
