# dash/data.py
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "db" / "trading.db"


def load_tob_since(symbol: str, lookback_minutes: int):
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)).isoformat()
    with sqlite3.connect(DB_PATH) as con:  # no detect_types; we'll parse in pandas
        df = pd.read_sql_query(
            """
            SELECT ts, bid, ask, bid_size AS bidSize, ask_size AS askSize
            FROM tob
            WHERE symbol = ? AND ts >= ?
            ORDER BY ts ASC
            """,
            con, params=(symbol, cutoff_iso)
        )
    # print("load_tob_since", symbol, cutoff_iso, df)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    df["spread_bps"] = (df["spread"] / df["mid"]) * 1e4
    return df.reset_index(drop=True)
#
# def load_latest_tob(symbol: str, limit: int = 3000):
#     with sqlite3.connect(DB_PATH) as con:  # <-- removed detect_types
#         df = pd.read_sql_query(
#             """
#             SELECT
#                 ts,               -- ISO8601 string
#                 bid,
#                 ask,
#                 bid_size AS bidSize,
#                 ask_size AS askSize
#             FROM tob
#             WHERE symbol = ?
#             ORDER BY ts DESC
#             LIMIT ?
#             """,
#             con, params=(symbol, limit),
#         )
#
#     if df.empty:
#         return df
#
#     df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
#     df = df.sort_values("ts").reset_index(drop=True)
#     df["mid"] = (df["bid"] + df["ask"]) / 2.0
#     df["spread"] = df["ask"] - df["bid"]
#     df["spread_bps"] = (df["spread"] / df["mid"]) * 1e4
#     return df
