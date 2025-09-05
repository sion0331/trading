# tools/fx_scan_intraday.py
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
# If using ib_async (your import), keep this:
from ib_async import IB, Forex, Ticker

# ---------------- Config ----------------
RAW_PAIRS = [
    "USDCHF", "CHFUSD", "USDHKD", "HKDUSD", "USDJPY", "JPYUSD", "USDAED", "AEDUSD", "USDBGN", "BGNUSD", "USDBHD",
    "BHDUSD", "USDCAD",
    "CADUSD", "USDCNH", "CNHUSD", "USDCZK", "CZKUSD", "USDDKK", "DKKUSD", "USDHUF", "HUFUSD", "USDIDR", "IDRUSD",
    "USDILS", "ILSUSD",
    "USDKRW", "KRWUSD", "USDKWD", "KWDUSD", "USDMXN", "MXNUSD", "USDNOK", "NOKUSD", "USDOMR", "OMRUSD", "USDPLN",
    "PLNUSD", "USDQAR",
    "QARUSD", "USDRON", "RONUSD", "USDSAR", "SARUSD", "USDSEK", "SEKUSD", "USDSGD", "SGDUSD", "USDTRY", "TRYUSD",
    "USDZAR", "ZARUSD"
]

WINDOW_SECS = 120  # how long to collect quotes
DATA_TYPE = 1  # 1=real-time, 3=delayed
MIN_TICKS = 30  # minimum quote updates required to keep a pair
COMMISSION_PER_MILLION = 20.0  # optional: add $/MM to cost model


# --------------- Helpers ----------------
def split(sym: str) -> Tuple[str, str]:
    return sym[:3], sym[3:]


def pip_size(symbol: str) -> float:
    """0.01 when quote is JPY, else 0.0001."""
    _, quote = split(symbol)
    return 0.01 if quote == "JPY" else 0.0001


def canonicalize(pairs: List[str]) -> List[str]:
    """
    Dedup reciprocals, prefer USD as QUOTE (XXXUSD) when possible.
    E.g. keep JPYUSD instead of USDJPY so quote=USD form is used for spread bps comparability.
    """
    seen = set()
    out: List[str] = []
    for s in pairs:
        b, q = split(s)
        # choose a key that makes USD-quote preferred
        if q == "USD":
            key = b + "USD"
            pick = s
        elif b == "USD":
            key = q + "USD"
            pick = q + "USD"
        else:
            key = "".join(sorted([b, q]))
            pick = s
        if key not in seen:
            seen.add(key)
            out.append(pick)
    return out


def usd_pip_value_per_mm(symbol: str, mid: float, spot_cache: Dict[str, float]) -> float:
    """
    $ value per 1 pip for $1mm base notional, with proper USD conversion.
    """
    base, quote = split(symbol)
    ps = pip_size(symbol)
    if quote == "USD":
        return ps * 1_000_000.0
    if base == "USD":
        # need quote->USD; use quoteUSD from cache or invert USDquote
        conv = spot_cache.get(quote + "USD")
        if conv is None:
            inv = spot_cache.get("USD" + quote)
            conv = 1.0 / inv if inv else mid
        return ps * 1_000_000.0 * conv
    # cross: pip is in quote; convert quote->USD
    conv = spot_cache.get(quote + "USD")
    if conv is None:
        inv = spot_cache.get("USD" + quote)
        conv = 1.0 / inv if inv else mid
    return ps * 1_000_000.0 * conv


@dataclass
class Row:
    pair: str
    entitled: bool
    entitlement_msg: str
    mid: Optional[float]
    spread_bps_med: Optional[float]
    spread_pips_med: Optional[float]
    sigma_1s_bps: Optional[float]
    sigma_60s_bps: Optional[float]
    tick_rate_per_min: Optional[float]
    est_cost_per_mm: Optional[float]
    score_vol_per_cost: Optional[float]


# --------------- Core ----------------
async def intraday_scan(host='127.0.0.1', port=7497, client_id=9) -> pd.DataFrame:
    ib = IB()
    await ib.connectAsync(host, port, clientId=client_id)
    ib.reqMarketDataType(DATA_TYPE)

    pairs = canonicalize(RAW_PAIRS)
    rows: List[Row] = []
    spot_cache: Dict[str, float] = {}

    for sym in pairs:
        contract = Forex(sym)
        try:
            ticker: Ticker = ib.reqMktData(contract, '', False, False)
        except Exception as e:
            rows.append(Row(sym, False, f"mktdata error: {e}", None, None, None, None, None, 0.0, None, None))
            continue

        samples = []
        last_bid = last_ask = None
        start = time.time()

        # collect ticks; IMPORTANT: use asyncio.sleep in async code
        while time.time() - start < WINDOW_SECS:
            await asyncio.sleep(0.05)
            bid = ticker.bid
            ask = ticker.ask
            if bid is None or ask is None:
                continue
            if bid != last_bid or ask != last_ask:
                mid = (bid + ask) / 2.0 if (bid and ask) else float('nan')
                samples.append((time.time(), bid, ask, mid))
                last_bid, last_ask = bid, ask

        ib.cancelMktData(contract)

        if not samples:
            rows.append(Row(sym, False, "No bid/ask during window", None, None, None, None, None, 0.0, None, None))
            continue

        arr = np.array(samples, dtype=float)  # ts, bid, ask, mid
        ts = arr[:, 0]
        bids = arr[:, 1]
        asks = arr[:, 2]
        mids = arr[:, 3]

        # spreads
        spread_bps_vec = (asks - bids) / ((asks + bids) / 2.0) * 1e4
        spread_pips_vec = (asks - bids) / pip_size(sym)

        # volatility from tick-by-tick mids
        valid = np.isfinite(mids) & (mids > 0)
        mids = mids[valid]
        ts = ts[valid]

        sigma_1s_bps = sigma_60s_bps = None
        if len(mids) >= 5:
            lr = np.diff(np.log(mids))
            dt = np.diff(ts)  # seconds
            per_sec_var = np.nanmean((lr ** 2) / np.clip(dt, 1e-6, None))
            sigma_1s = math.sqrt(max(per_sec_var, 0.0))
            sigma_1s_bps = sigma_1s * 1e4
            sigma_60s_bps = sigma_1s_bps * math.sqrt(60.0)

        tick_rate = float(len(mids) / max(WINDOW_SECS, 1.0) * 60.0)

        # cache for USD conversions
        base, quote = split(sym)
        med_mid = float(np.nanmedian(mids)) if len(mids) else None
        if med_mid:
            if quote == "USD":
                spot_cache[sym] = med_mid
            elif base == "USD":
                spot_cache["USD" + quote] = med_mid

        # cost per $1mm: spread leg (via pip value) + commission
        est_cost = None
        score = None
        spread_pips_med = float(np.nanmedian(spread_pips_vec))
        spread_bps_med = float(np.nanmedian(spread_bps_vec))
        if med_mid and np.isfinite(spread_pips_med):
            pv = usd_pip_value_per_mm(sym, med_mid, spot_cache)
            spread_cost_usd = pv * spread_pips_med
            est_cost = spread_cost_usd + COMMISSION_PER_MILLION
            if sigma_60s_bps and est_cost > 0:
                # mixed units but good comparator: more per-minute vol per $ cost
                score = float(sigma_60s_bps / est_cost)

        rows.append(Row(
            pair=sym,
            entitled=True,
            entitlement_msg="",
            mid=med_mid,
            spread_bps_med=round(spread_bps_med, 3) if np.isfinite(spread_bps_med) else None,
            spread_pips_med=round(spread_pips_med, 3) if np.isfinite(spread_pips_med) else None,
            sigma_1s_bps=None if sigma_1s_bps is None else float(round(sigma_1s_bps, 3)),
            sigma_60s_bps=None if sigma_60s_bps is None else float(round(sigma_60s_bps, 3)),
            tick_rate_per_min=round(tick_rate, 2),
            est_cost_per_mm=None if est_cost is None else round(est_cost, 2),
            score_vol_per_cost=None if score is None else round(score, 6),
        ))

    ib.disconnect()

    df = pd.DataFrame([r.__dict__ for r in rows])
    # basic liquidity filter
    df = df[df["tick_rate_per_min"] >= (MIN_TICKS * 60.0 / WINDOW_SECS)]
    # keep only rows with spreads computed
    df = df[df["spread_bps_med"].notna()]
    # rank best first
    df = df.sort_values(
        ["score_vol_per_cost", "sigma_60s_bps", "tick_rate_per_min"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    return df


# --------------- Runner ----------------
if __name__ == "__main__":
    # Use asyncio.run; do NOT call ib.sleep/util.startLoop here.
    df = asyncio.run(intraday_scan())
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False))
