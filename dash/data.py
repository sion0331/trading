import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "db" / "market.db"


def load_tob_range(symbol: str, start_utc: str, end_utc: str, db_path=DB_PATH):
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(
            """
            SELECT ts, bid, ask, bid_size AS bidSize, ask_size AS askSize,
                   (bid+ask)/2.0 AS mid
            FROM tob
            WHERE symbol = ? AND ts >= ? AND ts < ?
            ORDER BY ts ASC
            """,
            con, params=(symbol, start_utc, end_utc)
        )
    if df is None:
        return pd.DataFrame()
    # df["spread"] = df["ask"] - df["bid"]
    # df["spread_bps"] = (df["spread"] / df["mid"]) * 1e4
    return df.reset_index(drop=True)
