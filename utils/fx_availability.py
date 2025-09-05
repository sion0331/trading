# tools/fx_scan.py
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from ib_async import IB, Forex, util, Ticker, BarData

# ---------- Config ----------
# Pairs to test (majors + a few minors; adjust freely)

PAIRS = [
    "USDCHF", "CHFUSD", "USDHKD", "HKDUSD", "USDJPY", "JPYUSD", "USDAED", "AEDUSD", "USDBGN", "BGNUSD", "USDBHD",
    "BHDUSD", "USDCAD",
    "CADUSD", "USDCNH", "CNHUSD", "USDCZK", "CZKUSD", "USDDKK", "DKKUSD", "USDHUF", "HUFUSD", "USDIDR", "IDRUSD",
    "USDILS", "ILSUSD",
    "USDKRW", "KRWUSD", "USDKWD", "KWDUSD", "USDMXN", "MXNUSD", "USDNOK", "NOKUSD", "USDOMR", "OMRUSD", "USDPLN",
    "PLNUSD", "USDQAR",
    "QARUSD", "USDRON", "RONUSD", "USDSAR", "SARUSD", "USDSEK", "SEKUSD", "USDSGD", "SGDUSD", "USDTRY", "TRYUSD",
    "USDZAR", "ZARUSD"
]

# PAIRS = [
#     "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD","USDKRW",
#     "EURGBP","EURJPY","GBPJPY","AUDJPY","CHFJPY","EURCHF","EURNZD","GBPAUD"
# ] # "USDCLP","USDCOP","USDPEN","USDKRO","USDBRL",

# Historical settings
BAR_SIZE = '1 min'  # '1 min', '5 mins', etc.
DURATION = '3 D'  # 3 days of 1-mins is usually enough for a quick vol read
WHAT_TO_SHOW = 'MIDPOINT'  # for FX, MIDPOINT is common; TRADES not available for spot

# Commission modeling (update to YOUR schedule)
# E.g., $20 per $1mm notional
COMMISSION_PER_MILLION = 20.0

# Snapshot dwell time to accumulate quotes (seconds)
SNAPSHOT_SECS = 5.0


# ---------- Helpers ----------
def pip_size(symbol: str) -> float:
    """
    Return pip size (price delta) for a pair.
    Most FX pairs: 0.0001, JPY crosses: 0.01.
    """
    if symbol.endswith("JPY"):
        return 0.01
    return 0.0001


def pip_value_per_million(symbol: str, price: float) -> float:
    """
    Approximate USD value per 1 pip for $1mm notional.
    For USD as quote currency (XXXUSD), 1 pip per $1mm is ~ $100 (for 0.0001) or $1000 (for 0.01 on JPY).
    For crosses, we convert roughly using quote in USD via the price itself if USD is quote;
    otherwise this is a rough heuristic; refine if you need exact conversion.
    """
    ps = pip_size(symbol)
    # Value per pip per 1 unit = ps (in quote ccy).
    # For $1mm notional (base ccy), notional in quote ≈ price * 1,000,000.
    # So approx pip value per $1mm = ps * 1,000,000 in quote currency.
    # If quote currency is USD (e.g., EURUSD), this is already USD.
    # If not USD, this is an approximation; you can add a secondary conversion if needed.
    return ps * 1_000_000  # in quote currency; assume quote≈USD or close enough for ranking


def annualized_vol(close_prices: np.ndarray) -> float:
    """
    Realized vol from log returns of closes; annualized (sqrt(252*390) for 1-min bars is equity-ish).
    For FX, 24x5 trading ~ 7200 mins/week; use ~ 60*24*252 = 362,880 mins/year.
    """
    if len(close_prices) < 20:
        return float('nan')
    rets = np.diff(np.log(close_prices))
    # 1-min bars -> ~ 60*24*252 = 362,880 periods/year
    ann_factor = math.sqrt(60 * 24 * 252)
    return float(np.std(rets, ddof=1) * ann_factor)


@dataclass
class PairScanResult:
    symbol: str
    entitled: bool
    entitlement_msg: Optional[str]
    avg_spread_pips: Optional[float]
    mid_price: Optional[float]
    ann_vol: Optional[float]
    cost_per_million: Optional[float]
    score_vol_over_cost: Optional[float]


# ---------- Core ----------
async def scan_pairs(host: str = '127.0.0.1', port: int = 7497, client_id: int = 7) -> pd.DataFrame:
    ib = IB()
    await ib.connectAsync(host, port, clientId=client_id)

    results: List[PairScanResult] = []
    error_bucket: Dict[str, str] = {}

    def on_error(reqId, errorCode, errorString, contract):
        # Collect entitlement/permission errors
        if errorCode in (354, 10167, 10168, 10190, 10090):
            error_bucket[str(reqId)] = f"{errorCode}: {errorString}"

    ib.errorEvent += on_error

    # Use real-time streaming (frozen=0). You can switch to delayed if needed: ib.reqMarketDataType(3)
    ib.reqMarketDataType(1)

    # Subscribe and gather spreads
    for pair in PAIRS:
        contract = Forex(pair)
        try:
            ticker: Ticker = ib.reqMktData(contract, '', False, False)
        except:  # Error as e:
            print("error2")
            # results.append(PairScanResult(pair, False, f"{e.code}: {e.message}", None, None, None, None, None))
            continue

        # Wait a bit to collect bid/ask
        await asyncio.sleep(SNAPSHOT_SECS)
        # Access collected data
        # bids = [q.price for q in ticker.bidSizes if q.price is not None]  # not great; better use last bid/ask
        # Simpler: just use ticker.bid/ticker.ask arrays if present
        bid = ticker.bid
        ask = ticker.ask
        mid = None
        avg_spread_pips = None
        entitled = True
        msg = None

        # Check entitlement via error bucket or missing quotes
        if (bid is None or ask is None) and error_bucket:
            # Try to pick the last error for this reqId if we had one
            # ib_insync does not expose reqId easily; fallback: if no quotes after wait -> likely not entitled
            entitled = False
            # Compose message from any collected entitlement-like error
            msg = "; ".join(set(error_bucket.values()))
        elif bid is None or ask is None:
            # No explicit error but no data came in: mark as not entitled or feed unavailable
            entitled = False
            msg = "No bid/ask received (possible entitlement or closed market)."
        else:
            mid = (bid + ask) / 2.0
            spread = (ask - bid)
            pips = spread / pip_size(pair)
            avg_spread_pips = float(pips)

        # Pull recent history for realized vol if entitled
        ann_vol = None
        if entitled:
            try:
                bars: List[BarData] = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime='',
                    durationStr=DURATION,
                    barSizeSetting=BAR_SIZE,
                    whatToShow=WHAT_TO_SHOW,
                    useRTH=False,
                    formatDate=1,
                    keepUpToDate=False
                )
                closes = np.array([b.close for b in bars], dtype=float)
                ann_vol = annualized_vol(closes)
            except:  # Error as e:
                print("error1")
                # If history is gated differently from L1
                # msg = (msg + " | " if msg else "") + f"HistErr {e.code}: {e.message}"

        # Cost model
        cost_per_million = None
        score = None
        if entitled and avg_spread_pips is not None and mid is not None:
            # Spread cost per $1mm (USD-approx)
            spread_cost = pip_value_per_million(pair, mid) * avg_spread_pips
            cost_per_million = spread_cost + COMMISSION_PER_MILLION
            if ann_vol and cost_per_million > 0:
                score = ann_vol / cost_per_million

        results.append(PairScanResult(
            symbol=pair,
            entitled=entitled,
            entitlement_msg=msg,
            avg_spread_pips=avg_spread_pips if entitled else None,
            mid_price=mid if entitled else None,
            ann_vol=ann_vol if entitled else None,
            cost_per_million=cost_per_million if entitled else None,
            score_vol_over_cost=score if entitled else None
        ))

        # Cancel stream to be polite
        ib.cancelMktData(contract)

    ib.errorEvent -= on_error
    ib.disconnect()

    # Format a table
    rows = []
    for r in results:
        print(r)
        rows.append({
            "pair": r.symbol,
            "entitled": r.entitled,
            "entitlement_msg": r.entitlement_msg or "",
            "mid": None if r.mid_price is None else round(r.mid_price, 6),
            "avg_spread_pips": None if r.avg_spread_pips is None else round(r.avg_spread_pips, 2),
            "ann_vol": None if r.ann_vol is None else round(r.ann_vol, 4),
            "est_cost_per_$1mm": None if r.cost_per_million is None else round(r.cost_per_million, 2),
            "score_vol/cost": None if r.score_vol_over_cost is None else round(r.score_vol_over_cost, 6),
        })
    df = pd.DataFrame(rows)
    # Put entitled first and sort by score desc
    df = df.sort_values(
        by=["entitled", "score_vol/cost"],
        ascending=[False, False],
        na_position="last"
    ).reset_index(drop=True)
    return df


if __name__ == "__main__":
    util.startLoop()  # allow nested asyncio on Windows
    df = asyncio.run(scan_pairs())
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False))

    # from ib_insync import *
    # import pandas as pd
    #
    # ib = IB()
    # ib.connect("127.0.0.1", 7497, clientId=42)
    #
    # pairs = [
    #     "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "USDKRW",
    #     "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CHFJPY", "EURCHF", "EURNZD", "GBPAUD"
    # ]
    # results = []
    #
    # for sym in pairs:
    #     contract = Forex(sym)
    #
    #     for notional in [10_000, 1_000_000]:  # small and large
    #         action = "SELL"
    #         qty = notional  # IBKR uses units of base currency for FX
    #
    #         order = MarketOrder(action, qty)
    #         trade = ib.placeOrder(contract, order)
    #
    #         # Wait until fully filled and commission report arrives
    #         trade = ib.waitOnUpdate(timeout=10) or trade
    #         # print(trade)
    #         # while not trade.commissionReport:
    #         #     ib.waitOnUpdate(timeout=1)
    #         #
    #         # cr = trade.commissionReport[-1]  # last commission report
    #         # commission = cr.commission
    #         # currency = cr.currency
    #         #
    #         # results.append({
    #         #     "pair": sym,
    #         #     "notional": notional,
    #         #     "commission": commission,
    #         #     "currency": currency,
    #         #     "bps_equiv": (commission / notional) * 1e4
    #         # })
    #
    # df = pd.DataFrame(results)
    # print(df)
