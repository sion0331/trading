from datetime import datetime, timezone, timedelta
from math import isnan

import pandas as pd
import plotly.graph_objects as go
import pytz
from dash import Input, Output
from plotly.subplots import make_subplots

from data import load_tob_since, load_tob_range
from data_orders import (
    load_executions_since,
    load_executions_raw_since,
    load_commissions_since,
    compute_pnl_curve, load_executions_range, load_commissions_range,
)
from utils import fifo_realized_unrealized, _to_utc_ts

NY = pytz.timezone("America/New_York")


def fx_reset_window_for_date(date_str: str):
    """
    For a YYYY-MM-DD date (NY), return (start_utc, end_utc) where the FX day is
    16:55 NY on that date -> 16:55 NY next date.
    """
    # parse selected date as NY-local midnight of that day
    y, m, d = map(int, date_str.split("-"))
    day_ny = NY.localize(datetime(y, m, d))
    start_ny = day_ny.replace(hour=16, minute=55, second=0, microsecond=0)
    end_ny = start_ny + timedelta(days=1)
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
        Output("tob-graph", "figure"),
        Output("pnl-table", "data"),
        Input("refresh", "n_intervals"),
        Input("symbol-dropdown", "value"),
        Input("lookback-min", "value"),
        Input("trades-date", "date"),
    )
    def update_price_pnl(n_intervals, symbol, lookback_min, trades_date):
        lookback_min = int(lookback_min or 30)

        # ---- load data
        df = _to_utc_ts(load_tob_since(symbol, lookback_minutes=lookback_min))
        df_exec = _to_utc_ts(load_executions_since(symbol, lookback_minutes=lookback_min))
        df_exec_raw = _to_utc_ts(load_executions_raw_since(symbol, lookback_minutes=lookback_min))
        df_comm = _to_utc_ts(load_commissions_since(lookback_min, symbol=symbol), col="ts")

        start_utc, end_utc = fx_reset_window_for_date(trades_date or str(datetime.now(NY).date()))
        start_iso = start_utc.isoformat()
        end_iso = end_utc.isoformat()

        df_mid_tbl = _to_utc_ts(load_tob_range(symbol, start_iso, end_iso))  # for mid@fill in table
        df_exec_tbl = _to_utc_ts(load_executions_range(symbol, start_iso, end_iso))
        df_comm_tbl = _to_utc_ts(load_commissions_range(symbol, start_iso, end_iso), col="ts")

        mid_for_merge = df[["ts", "mid"]].copy() if not df.empty else pd.DataFrame(columns=["ts", "mid"])

        exec_aligned = pd.merge_asof(
            df_exec_raw.sort_values("ts"),
            mid_for_merge.sort_values("ts"),
            on="ts",
            direction="backward"
        ).rename(columns={"mid": "mid_exec"})

        pnl_curve = compute_pnl_curve(
            df[["ts", "mid"]].copy() if not df.empty else df,
            df_exec if df_exec is not None else df_exec
        )

        # per-exec fee: join commissions by exec_id; fallback by (order_id, nearest ts within 2s)
        fees = {}
        if not df_comm.empty:
            # primary map by exec_id
            fees.update(
                {k: float(v) for k, v in
                 df_comm.dropna(subset=["exec_id"]).set_index("exec_id")["commission"].items()})

        exec_aligned["fee"] = exec_aligned["exec_id"].map(fees).astype(float).fillna(0.0)

        # impact (spread/marketable capture) per exec
        sign = exec_aligned["side"].str.upper().map({"BUY": 1.0, "BOT": 1.0, "SELL": -1.0, "SLD": -1.0}).fillna(0.0)
        exec_aligned["impact"] = sign * (exec_aligned["mid_exec"] - exec_aligned["price"]) * exec_aligned[
            "qty"].astype(
            float)

        # curve with decomposition
        curve = fifo_realized_unrealized(
            df_mid=df[["ts", "mid"]].copy() if not df.empty else pd.DataFrame(columns=["ts", "mid"]),
            df_exec=exec_aligned[
                ["ts", "price", "qty", "side", "fee", "impact"]].copy() if not exec_aligned.empty else None
        )

        # ---- figure with shared x-axis
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.06, row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol} Price & Executions", "PnL"),
        )

        # ----- Row 1: Price + executions
        if not df.empty:
            fig.add_trace(go.Scatter(x=df["ts"], y=df["bid"], name="Bid", mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["ts"], y=df["ask"], name="Ask", mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["ts"], y=df["mid"], name="Mid", mode="lines"), row=1, col=1)

        if df_exec is not None and not df_exec.empty:
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

        # optional: show most recent limit within window
        # lmt = load_latest_limit_since(symbol, lookback_minutes=lookback_min)
        # if lmt is not None:
        #     fig.add_hline(y=lmt, line_dash="dot", annotation_text=f"Limit {lmt}", row=1, col=1)

        # ----- Row 2: PnL
        if not curve.empty:
            fig.add_trace(go.Scatter(x=curve["ts"], y=curve["pnl_total"], name="PnL", mode="lines"), row=2, col=1)
            # optional components:
            # fig.add_trace(go.Scatter(x=curve["ts"], y=curve["realized"], name="Realized", mode="lines"), row=2, col=1)
            # fig.add_trace(go.Scatter(x=curve["ts"], y=curve["unrealized"], name="Unrealized", mode="lines"), row=2, col=1)
            # fig.add_trace(go.Scatter(x=curve["ts"], y=-curve["fees_cum"], name="Fees", mode="lines"), row=2, col=1)
            # fig.add_trace(go.Scatter(x=curve["ts"], y=curve["impact_cum"], name="Impact", mode="lines"), row=2, col=1)

        # ---- layout / axes
        fig.update_layout(
            margin=dict(l=40, r=10, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        )
        fig.update_yaxes(title_text=f"{symbol} Price", row=1, col=1)
        fig.update_yaxes(title_text="PnL (quote ccy)", row=2, col=1)
        fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)

        # lock both to same visible range from market data (nice initial alignment)
        if not df.empty:
            x0, x1 = df["ts"].min(), df["ts"].max()
            fig.update_xaxes(range=[x0, x1], row=1, col=1)
            fig.update_xaxes(range=[x0, x1], row=2, col=1)

        # ---- Table: PnL per trade (consistent with MTM)
        pnl_rows = []

        # need mid curve for MTM sampling even if lookback is short
        reset_utc = fx_reset_start_utc()

        if (df_exec_tbl is not None) and (not df_exec_tbl.empty):
            # 1) Join mid@fill for accurate spread calc
            mid_for_merge = df_mid_tbl[["ts", "mid"]].copy() if not df_mid_tbl.empty else pd.DataFrame(
                columns=["ts", "mid"])
            exec_aligned = pd.merge_asof(
                df_exec_tbl.sort_values("ts"),
                mid_for_merge.sort_values("ts"),
                on="ts", direction="backward"
            ).rename(columns={"mid": "mid_exec"})

            # 2) Commissions by exec_id (fallback to zero if missing)
            comm_map = {}
            if (df_comm_tbl is not None) and (not df_comm_tbl.empty) and ("exec_id" in df_comm_tbl):
                comm_map = df_comm_tbl.set_index("exec_id")["commission"].to_dict()
            exec_aligned["commission"] = exec_aligned["exec_id"].map(comm_map).fillna(0.0).astype(float)

            # 3) Normalize sides, compute spread (impact) vs mid
            side_u = exec_aligned["side"].str.upper().map({"BOT": "BUY", "SLD": "SELL"}).fillna(
                exec_aligned["side"].str.upper())
            qty = exec_aligned["qty"].astype(float)
            px = exec_aligned["price"].astype(float)
            mid = exec_aligned["mid_exec"].astype(float)

            # sign: BUY=+1, SELL=-1
            sign = side_u.map({"BUY": 1.0, "SELL": -1.0}).astype(float)

            # Spread/impact PnL: positive if better than mid at fill
            spread_pnl = sign * (mid - px) * qty
            exec_aligned["spread_pnl"] = spread_pnl

            # 4) USD column (absolute notional in quote ccy at fill price)
            exec_aligned["usd"] = (qty.abs() * px).round(0)  # show as integer USD

            # 5) Maker/Taker label from liquidity if available
            if "liquidity" in exec_aligned.columns:
                exec_aligned["liq"] = exec_aligned["liquidity"].map(
                    {1: "MAKER", 2: "TAKER", 3: "ROUTED", 4: "AUCTION"}).fillna("")
            else:
                exec_aligned["liq"] = ""

            # 6) Sample MTM (pnl_total) at each exec time (we need per-row increment)
            mtm_snap = None
            if (pnl_curve is not None) and (not pnl_curve.empty):
                snap = pd.merge_asof(
                    exec_aligned[["ts"]].sort_values("ts"),
                    pnl_curve[["ts", "pnl"]].sort_values("ts"),
                    on="ts", direction="backward"
                )
                mtm_snap = snap["pnl"].astype(float).values
            else:
                mtm_snap = [None] * len(exec_aligned)

            exec_aligned["mtm_snap"] = mtm_snap

            # 7) Filter to FX reset window (today)

            exec_today = exec_aligned[exec_aligned["ts"] >= reset_utc].copy()

            # 8) Compute per-row MTM *increment* (change in curve between fills)
            #    First increment is from reset baseline to first trade; later is diff vs previous trade.
            mtm_incr = []
            if not exec_today.empty and (pnl_curve is not None) and (not pnl_curve.empty):
                # baseline at/before reset
                base_row = pnl_curve[pnl_curve["ts"] <= reset_utc].tail(1)
                mtm_base = float(base_row["pnl"].values[0]) if not base_row.empty else float(
                    pnl_curve["pnl"].iloc[0])

                prev = mtm_base
                for val in exec_today["mtm_snap"].tolist():
                    if val is None or (isinstance(val, float) and isnan(val)):
                        mtm_incr.append(0.0)
                    else:
                        inc = float(val) - prev
                        mtm_incr.append(inc)
                        prev = float(val)
            else:
                mtm_incr = [0.0] * len(exec_today)

            exec_today["mtm_incr"] = mtm_incr

            # 9) Per-row TOTAL = spread + mtm_incr + commission
            exec_today["row_total"] = (exec_today["spread_pnl"].astype(float)
                                       + exec_today["mtm_incr"].astype(float)
                                       + exec_today["commission"].astype(float))

            # 10) Roundings:
            exec_today["price"] = exec_today["price"].astype(float).round(6)
            exec_today["mid_exec"] = exec_today["mid_exec"].astype(float).round(6)
            for c in ["spread_pnl", "commission", "mtm_incr", "row_total"]:
                exec_today[c] = exec_today[c].astype(float).round(2)

            # 11) Build TOTAL (Today) row
            tot_spread = float(exec_today["spread_pnl"].sum()) if not exec_today.empty else 0.0
            tot_comm = float(exec_today["commission"].sum()) if not exec_today.empty else 0.0

            # MTM delta today = current curve - baseline at reset
            mtm_last = float(pnl_curve["pnl"].iloc[-1]) if (pnl_curve is not None and not pnl_curve.empty) else 0.0
            base_row = pnl_curve[pnl_curve["ts"] <= reset_utc].tail(1) if (
                    pnl_curve is not None and not pnl_curve.empty) else None
            mtm_base = float(base_row["pnl"].values[0]) if (base_row is not None and not base_row.empty) else 0.0
            mtm_today = round(mtm_last - mtm_base, 2)

            total_row = {
                "ts": "TOTAL (FX day)",
                "side": "",
                "usd": round(float(exec_today["usd"].sum()) if not exec_today.empty else 0.0, 0),
                "qty": round(float(exec_today["qty"].sum()) if not exec_today.empty else 0.0, 0),
                "price": "",
                "mid_exec": "",
                "order_type": "",
                "liq": "",
                "spread_pnl": round(tot_spread, 2),
                "commission": round(tot_comm, 2),
                "mtm_incr": mtm_today,  # show daily MTM change here
                "row_total": round(tot_spread + mtm_today + tot_comm, 2),
            }

            # 12) Select/display latest 30 rows, newest first, and prepend TOTAL row
            if not exec_today.empty:
                view = exec_today[["ts", "side", "usd", "qty", "price", "mid_exec", "order_type", "liq",
                                   "spread_pnl", "commission", "mtm_incr", "row_total"]].copy()
                view = view.sort_values("ts", ascending=False)
                # format ts as readable strings
                view["ts"] = view["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
                pnl_rows = [total_row] + view.to_dict("records")
            else:
                pnl_rows = [total_row]

        else:
            # No executions: show zero totals row
            pnl_rows = [{
                "ts": "TOTAL (FX day)",
                "side": "", "usd": 0.0, "qty": 0.0, "price": "", "mid_exec": "",
                "order_type": "", "liq": "",
                "spread_pnl": 0.0, "commission": 0.0, "mtm_incr": 0.0, "row_total": 0.0,
            }]
        return fig, pnl_rows

    @app.callback(
        Output("refresh", "disabled"),
        Input("stop-refresh", "value")
    )
    def toggle_refresh(stop_values):
        return "stop" in stop_values  # True disables the interval
