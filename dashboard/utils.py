import pandas as pd


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
