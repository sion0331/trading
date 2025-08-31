import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fifo_realized_unrealized(df_exec: pd.DataFrame, df_mid: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a time series aligned to df_mid['ts'] with columns:
    position, cash, realized, unrealized, fees_cum, impact_cum, pnl_total
    df_exec columns: ts, price, qty, side('BUY'/'SELL'), fee (per exec), impact (per exec)
    """
    if df_mid.empty:
        return pd.DataFrame(
            columns=["ts", "position", "cash", "realized", "unrealized", "fees_cum", "impact_cum", "pnl_total"])

    # ensure sorting
    m = df_mid[["ts", "mid"]].dropna().sort_values("ts").reset_index(drop=True)
    if df_exec is None or df_exec.empty:
        out = m.copy()
        out["position"] = 0.0
        out["cash"] = 0.0
        out["realized"] = 0.0
        out["unrealized"] = 0.0
        out["fees_cum"] = 0.0
        out["impact_cum"] = 0.0
        out["pnl_total"] = 0.0
        return out

    e = df_exec.copy().sort_values("ts").reset_index(drop=True)
    e["side"] = e["side"].str.upper().map({"BOT": "BUY", "SLD": "SELL"}).fillna(e["side"])
    sign = e["side"].map({"BUY": 1.0, "SELL": -1.0}).astype(float)
    e["net_qty"] = sign * e["qty"].astype(float)
    e["cash_flow"] = - e["net_qty"] * e["price"].astype(float)
    e["fee"] = e.get("fee", 0.0).astype(float)
    e["impact"] = e.get("spread_pnl", 0.0).astype(float)

    # FIFO inventory of (qty, price)
    lots = []
    realized = 0.0
    fees_cum = 0.0
    impact_cum = 0.0

    # build cumulative state at each exec ts
    rows = []
    pos = 0.0
    cash = 0.0
    for _, r in e.iterrows():
        qty = float(r["qty"])
        px = float(r["price"])
        s = r["side"]
        # fees & impact at this fill
        fees_cum += float(r["fee"])
        impact_cum += float(r["impact"])

        if s == "BUY":
            # add lot (increase negative cash)
            lots.append([qty, px])
            pos += qty
            cash -= qty * px
        else:  # SELL
            q_to_match = qty
            while q_to_match > 0 and lots:
                lot_qty, lot_px = lots[0]
                take = min(q_to_match, lot_qty)
                realized += (px - lot_px) * take
                lot_qty -= take
                q_to_match -= take
                pos -= take
                cash += take * px
                if lot_qty <= 1e-9:
                    lots.pop(0)
                else:
                    lots[0][0] = lot_qty
            # if we sold more than long position (go short), create short lots
            if q_to_match > 1e-9:
                # short lot at sell price
                lots.insert(0, [-q_to_match, px])  # negative qty lot for short
                pos -= q_to_match
                cash += q_to_match * px

        rows.append({"ts": r["ts"], "position": pos, "cash": cash, "realized": realized,
                     "fees_cum": fees_cum, "impact_cum": impact_cum})

    state = pd.DataFrame(rows)
    aligned = pd.merge_asof(m, state.sort_values("ts"), on="ts", direction="backward")

    aligned[["position", "cash", "realized", "fees_cum", "impact_cum"]] = \
        aligned[["position", "cash", "realized", "fees_cum", "impact_cum"]].fillna(0.0)

    # compute unrealized from remaining lots vs mid
    # to do this per timestamp, we need weighted avg cost of OPEN lots
    # Approximate by tracking avg cost using position & cash from FIFO above:
    # cash = -sum(lot_qty*lot_px) for longs + sum(|lot_qty|*lot_px) for shorts
    # When position != 0, avg_cost = -cash/position for long; for short it's -cash/position as well (sign handles)
    eps = 1e-12
    avg_cost = -aligned["cash"] / aligned["position"].replace(0, pd.NA)
    pos = pd.to_numeric(aligned["position"], errors="coerce").fillna(0.0)
    mid = pd.to_numeric(aligned["mid"], errors="coerce")
    cost = pd.to_numeric(avg_cost, errors="coerce").fillna(0.0)

    unrealized = pos * (mid - cost)
    aligned["unrealized"] = unrealized.fillna(0.0)

    aligned["pnl_total"] = aligned["realized"] + aligned["unrealized"] - aligned["fees_cum"] + aligned["impact_cum"]
    return aligned[["ts", "position", "cash", "realized", "unrealized", "fees_cum", "impact_cum", "pnl_total"]]


def compute_pnl_curve(df_mid, df_exec) -> pd.DataFrame:
    """
    df_mid: columns ['ts', 'mid']  (already sorted ascending, UTC)
    df_exec: columns ['ts', 'price', 'qty', 'side'] with side in {'BUY','SELL'}
    Returns DataFrame with ['ts','position','cash','pnl'] aligned to df_mid['ts'].
    """
    df_mid = df_mid[["ts", "mid"]].dropna().sort_values("ts").reset_index(drop=True)

    e = df_exec.copy()
    e["net_qty"] = e["sign"] * e["qty"].astype(float)
    e["cash_flow"] = - e["net_qty"] * e["price"].astype(float)

    e = e.sort_values("ts").reset_index(drop=True)
    e["cum_pos"] = e["net_qty"].cumsum()
    e["cum_cash"] = e["cash_flow"].cumsum()
    e["cum_fee"] = e["fee"].cumsum()

    # align cumulative state to each market timestamp
    state = e[["ts", "cum_pos", "cum_cash", "cum_fee"]]
    aligned = pd.merge_asof(df_mid, state, on="ts", direction="backward")

    aligned["cum_pos"] = aligned["cum_pos"].fillna(0.0)
    aligned["cum_cash"] = aligned["cum_cash"].fillna(0.0)
    aligned["cum_fee"] = aligned["cum_fee"].fillna(0.0)
    aligned["position"] = aligned["cum_pos"]
    aligned["cash"] = aligned["cum_cash"]
    aligned["fee"] = aligned["cum_fee"]
    aligned["total_pnl"] = aligned["cash"] + aligned["position"] * aligned["mid"] - aligned["fee"]
    return aligned[["ts", "position", "cash", "total_pnl"]]


def plot_market_executions(symbol, df_mid, df_exec, df_comm, lookback_iso=None, pip=1e-4, horizons=(5, 15, 30, 60),
                           pip_col_prefix="pips_", weight_for_totals="usd"):
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
    df_mid_lookback = df_mid if lookback_iso is None else df_mid[df_mid['ts'] >= lookback_iso].copy()
    df_mid_lookback["spread"] = (df_mid_lookback["ask"] - df_mid_lookback["bid"]) / pip
    df_mid_lookback["d_mid"] = df_mid_lookback["mid"].diff() / pip

    res = (
        df_mid_lookback.set_index("ts")
        .resample("30s")
        .agg(
            tick_count=("mid", "size"),  # number of TOB updates
            avg_spread=("spread", "mean"),  # average spread (price)
            # realized vol in pips within the minute (std of 1s deltas)
            vol=("d_mid", "std"),
        )
        .reset_index()
        .fillna(0.0)
    )

    fig.add_trace(go.Scatter(x=df_mid_lookback["ts"], y=df_mid_lookback["bid"], name="Bid", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_mid_lookback["ts"], y=df_mid_lookback["ask"], name="Ask", mode="lines"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df_mid_lookback["ts"], y=df_mid_lookback["mid"], name="Mid", mode="lines", visible="legendonly"),
        row=1, col=1)

    # Row 2: bars/lines
    fig.add_trace(
        go.Bar(
            x=res["ts"], y=res["tick_count"], name="Ticks/30s",
            hovertemplate="Ticks=%{y:.0f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
        ),
        row=2, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=res["ts"], y=res["avg_spread"], mode="lines", name="Avg spread (pips)",
            hovertemplate="Spread=%{y:.2f} pips<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
        ),
        row=2, col=1, secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=res["ts"], y=res["vol"], mode="lines", name="Vol (pips, 30s σ)",
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
    df_exec_lookback = df_exec if lookback_iso is None else df_exec[df_exec['ts'] >= lookback_iso]
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

    # ===================== PNL & HORIZON PIPS =====================

    # 1) Join mid at fill
    df_exec = pd.merge_asof(df_exec, df_mid[["ts", "bid", "mid", "ask"]], on="ts", direction="backward")

    # 2) Commissions by exec_id (fallback to zero if missing)
    comm_map = df_comm.set_index("order_id")["commission"].to_dict()
    df_exec["fee"] = df_exec["order_id"].map(comm_map).fillna(0.0).astype(float)

    # 3) sign: BUY=+1, SELL=-1
    side_u = df_exec["side"].str.upper().map({"BOT": "BUY", "SLD": "SELL"}).fillna(df_exec["side"].str.upper())
    df_exec["sign"] = side_u.map({"BUY": 1.0, "SELL": -1.0}).astype(float)

    # 4) USD notional (quote ccy) at fill
    df_exec["usd"] = (df_exec["qty"].abs() * df_exec["price"]).astype(float)

    # 5) Spread/market PnL
    df_exec["spread_pnl"] = df_exec["sign"] * (df_exec["mid"] - df_exec["price"]) * df_exec["qty"]
    df_exec["market_pnl"] = df_exec["sign"] * (df_mid.iloc[-1]['mid'] - df_exec["mid"]) * df_exec["qty"]

    pnl_curve = compute_pnl_curve(df_mid[["ts", "mid"]], df_exec)
    fig.add_trace(go.Scatter(x=pnl_curve["ts"], y=pnl_curve["total_pnl"], name="PnL", mode="lines"), row=3, col=1)

    # 7) PnL curve - Figure Row 2
    # curve = fifo_realized_unrealized(df_exec[["ts", "price", "qty", "side", "fee", "spread_pnl"]],
    #                                  df_mid[["ts", "mid"]])

    # optional components:
    # fig.add_trace(go.Scatter(x=curve["ts"], y=curve["realized"], name="Realized", mode="lines"), row=2, col=1)
    # fig.add_trace(go.Scatter(x=curve["ts"], y=curve["unrealized"], name="Unrealized", mode="lines"), row=2, col=1)
    # fig.add_trace(go.Scatter(x=curve["ts"], y=-curve["fees_cum"], name="Fees", mode="lines"), row=2, col=1)
    # fig.add_trace(go.Scatter(x=curve["ts"], y=curve["impact_cum"], name="Impact", mode="lines"), row=2, col=1)

    # 8) Per-row TOTAL = spread + market_pnl - commission
    df_exec["row_total"] = (df_exec["spread_pnl"] + df_exec["market_pnl"] - df_exec["fee"])

    # 8) Horizon pip changes (signed so SELL gets positive if price falls)
    #    Precompute mid at (ts + h) for each horizon and add columns pips_{h}s
    df_mid_slim = df_mid[["ts", "mid"]].sort_values("ts")
    for h in horizons:
        future_times = df_exec[["ts"]].copy()
        future_times["ts"] = future_times["ts"] + pd.to_timedelta(h, unit="s")
        merged = pd.merge_asof(
            future_times.sort_values("ts"),
            df_mid_slim,
            on="ts",
            direction="backward"
        )  # take the latest mid before or at (t+h)
        mid_future = merged["mid"].astype(float).values
        mid_now = df_exec["mid"].astype(float).values
        signed_pips = df_exec["sign"].values * (mid_future - mid_now) / pip
        df_exec[f"{pip_col_prefix}{h}s"] = signed_pips.round(2)

    # 9) Rounding & formatting
    df_exec["price"] = df_exec["price"].round(6)
    df_exec["bid"] = df_exec["bid"].round(6)
    df_exec["mid"] = df_exec["mid"].round(6)
    df_exec["ask"] = df_exec["ask"].round(6)
    for c in ["spread_pnl", "fee", "market_pnl", "row_total"]:
        df_exec[c] = df_exec[c].round(2)

    # 10) TOTAL row aggregates
    tot_spread = float(df_exec["spread_pnl"].sum())
    tot_mtm = float(df_exec["market_pnl"].sum())
    tot_comm = float(df_exec["fee"].sum())
    n_trades = int(len(df_exec))

    # --- weighted averages for TOTAL Bid/Mid/Ask based on executions ---
    wcol = weight_for_totals if weight_for_totals in df_exec.columns else "usd"

    buy_mask = df_exec["sign"] > 0  # BUY executions
    sell_mask = df_exec["sign"] < 0  # SELL executions

    def _wavg(series, weights):
        w = np.asarray(weights, dtype=float)
        s = np.asarray(series, dtype=float)
        denom = np.nansum(w)
        return float(np.nansum(s * w) / denom) if denom > 0 else np.nan

    avg_bid_exec = _wavg(df_exec.loc[buy_mask, "price"], df_exec.loc[buy_mask, "usd"])
    avg_ask_exec = _wavg(df_exec.loc[sell_mask, "price"], df_exec.loc[sell_mask, "usd"])

    if np.isnan(avg_bid_exec) and not np.isnan(avg_ask_exec):
        avg_mid_exec = avg_ask_exec
    elif np.isnan(avg_ask_exec) and not np.isnan(avg_bid_exec):
        avg_mid_exec = avg_bid_exec
    elif not np.isnan(avg_bid_exec) and not np.isnan(avg_ask_exec):
        avg_mid_exec = 0.5 * (avg_bid_exec + avg_ask_exec)
    else:
        avg_mid_exec = np.nan  # no executions

    # rounded strings for TOTAL row (empty if NaN)
    total_bid_str = "" if np.isnan(avg_bid_exec) else round(avg_bid_exec, 6)
    total_ask_str = "" if np.isnan(avg_ask_exec) else round(avg_ask_exec, 6)
    total_mid_str = "" if np.isnan(avg_mid_exec) else round(avg_mid_exec, 6)

    # last trade price (time ascending → last row)
    last_px = float(df_mid["mid"].iloc[-1])

    # Weighted averages for pip columns
    weights = df_exec[weight_for_totals].astype(float).values if weight_for_totals in df_exec else np.ones(n_trades)
    w_sum = weights.sum() if n_trades > 0 else 1.0

    total_row = {
        "ts": "TOTAL",
        "side": f"{n_trades}",
        "usd": f"{float(df_exec['usd'].sum()):,.0f}",
        "qty": f"{float(df_exec['qty'].sum()):,.0f}",
        "price": (round(last_px, 6) if last_px is not None else ""),
        "bid": total_bid_str,
        "mid": total_mid_str,
        "ask": total_ask_str,
        "order_type": "",
        "spread_pnl": round(tot_spread, 2),
        "market_pnl": round(tot_mtm, 2),
        "fee": round(tot_comm, 2),
        "row_total": round(tot_spread + tot_mtm - tot_comm, 2),
    }
    # add weighted averages for each pip horizon to TOTAL
    for h in horizons:
        col = f"{pip_col_prefix}{h}s"
        if col in df_exec:
            vals = df_exec[col].astype(float).values
            wavg = float((vals * weights).sum() / (w_sum if w_sum != 0 else 1.0))
            total_row[col] = round(wavg, 2)

    # 11) Build view rows (include new pip columns)
    pip_cols = [f"{pip_col_prefix}{h}s" for h in horizons]
    # Ensure new columns exist in DataTable; if they aren’t in layout, add them there once.
    for c in pip_cols:
        if c not in df_exec.columns:
            df_exec[c] = np.nan

    view_cols = [
        "ts", "side", "usd", "qty", "price", "bid", "mid", "ask", "order_type",
        "spread_pnl", "market_pnl", "fee", "row_total", *pip_cols
    ]

    # nice formatting for USD/Qty with commas
    df_exec["usd"] = df_exec["usd"].apply(lambda v: f"{v:,.0f}")
    df_exec["qty"] = df_exec["qty"].apply(lambda v: f"{v:,.0f}")

    # sort newest first
    df_exec = df_exec.sort_values("ts", ascending=False)
    df_exec["ts"] = df_exec["ts"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    pnl_rows = [total_row] + df_exec[view_cols].to_dict("records")

    return fig, pnl_rows
