# backtest/replay.py
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest.broker import SimIB
from marketData.data_types import Tob  # your existing class

DB_PATH = (Path(__file__).resolve().parents[1] / "data" / "db" / "market.db").resolve()


def load_tob(symbol: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    with sqlite3.connect(str(DB_PATH)) as con:
        df = pd.read_sql_query(
            """
            SELECT ts, bid, ask, bid_size AS bidSize, ask_size AS askSize
            FROM tob
            WHERE symbol = ? AND ts >= ? AND ts < ?
            ORDER BY ts ASC
            """, con, params=(symbol, start_iso, end_iso)
        )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).reset_index(drop=True)
    return df


def replay_tob(
        df: pd.DataFrame,
        strategy,
        broker: SimIB,
        symbol: str,
):
    """
    Drive your live strategy with saved TOB.
    """
    for _, r in df.iterrows():
        ts: datetime = r["ts"].to_pydatetime()
        bid = float(r["bid"])
        ask = float(r["ask"])
        tob = Tob(symbol, bid, ask, float(r.get("bidSize", 0.0) or 0.0), float(r.get("askSize", 0.0) or 0.0), ts)
        # feed strategy first (so it may place/cancel)
        strategy.on_market_data(tob)
        # then let broker attempt matching fills at this tick
        broker.on_tick(ts, bid, ask)
