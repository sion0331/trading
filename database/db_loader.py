import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from utils.utils_dt import _to_utc_ts

ORDERS_DB = Path(__file__).resolve().parents[1] / "data" / "db" / "orders.db"
MARKETS_DB = Path(__file__).resolve().parents[1] / "data" / "db" / "market.db"
BACKTEST_DB = Path(__file__).resolve().parents[1] / "data" / "db" / "orders_backtest.db"
HISTORY_DB = Path(__file__).resolve().parents[1] / "data" / "db" / "history.db"


def _conn():
    return sqlite3.connect(ORDERS_DB)


def load_tob_range(symbol: str, start_utc: str, end_utc: str, db_path=MARKETS_DB, pip=1e-4):
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
    df = _to_utc_ts(df)
    df["spread"] = (df["ask"] - df["bid"]) / pip
    # df["spread_bps"] = (df["spread"] / df["mid"]) * 1e4
    return df.reset_index(drop=True)


def load_executions_range(symbol: str, start_utc: str, end_utc: str):
    with sqlite3.connect(ORDERS_DB) as con:
        df = pd.read_sql_query(
            """
            SELECT exec_id, ts, price, qty, side, order_id, order_type, liquidity
            FROM executions
            WHERE symbol = ? AND ts >= ? AND ts < ?
            ORDER BY ts ASC
            """,
            con, params=(symbol, start_utc, end_utc)
        )
    return df


def load_fills_range(symbol: str, start_utc: str, end_utc: str, path: str):
    try:
        with sqlite3.connect(path) as con:
            df = pd.read_sql_query(
                """
                SELECT ts, avg_fill_price AS price, qty, action AS side, order_id, order_type
                FROM orders
                WHERE symbol = ? AND status = 'Filled' AND ts >= ? AND ts < ?
                    AND remaining_qty < 0.1
                ORDER BY ts ASC
                """,
                con, params=(symbol, start_utc, end_utc))
        return df
    except Exception:
        return None


def load_commissions_range(symbol: str, start_utc: str, end_utc: str, path: str):
    try:
        with sqlite3.connect(path) as con:
            df = pd.read_sql_query(
                """
                SELECT exec_id, order_id, symbol, ts, commission, currency, realized_pnl
                FROM commissions
                WHERE symbol = ? AND ts >= ? AND ts < ?
                ORDER BY ts ASC
                """,
                con, params=(symbol, start_utc, end_utc)
            )
        return df
    except Exception:
        return None


def load_commissions_since(lookback_minutes: int, symbol: str | None = None):
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)).isoformat()
    q = "SELECT exec_id, order_id, symbol, ts, commission, currency, realized_pnl FROM commissions WHERE ts >= ?"
    params = [cutoff]
    if symbol:
        q += " AND symbol = ?"
        params.append(symbol)
    with _conn() as con:
        df = pd.read_sql_query(q, con, params=params)
    return df if not df.empty else pd.DataFrame(
        columns=["exec_id", "order_id", "symbol", "ts", "commission", "currency", "realized_pnl"])


def load_executions_raw_since(symbol: str, lookback_minutes: int):
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)).isoformat()
    with _conn() as con:
        df = pd.read_sql_query("""
            SELECT e.exec_id, e.ts, e.price, e.qty, e.side, e.order_id,
                   COALESCE(e.order_type, (SELECT order_type FROM orders o
                                           WHERE o.order_id = e.order_id
                                           ORDER BY ts DESC LIMIT 1)) AS order_type,
                   e.liquidity
            FROM executions e
            WHERE e.symbol = ? AND e.ts >= ?
            ORDER BY e.ts ASC
        """, con, params=(symbol, cutoff))
    return df


def load_executions_since(symbol: str, lookback_minutes: int):
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)).isoformat()
    with _conn() as con:
        df = pd.read_sql_query(
            """
            SELECT ts, price, qty, side
            FROM executions
            WHERE symbol = ? AND ts >= ?
            ORDER BY ts ASC
            """,
            con, params=(symbol, cutoff_iso)
        )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "side" in df.columns:
        df["side"] = df["side"].str.upper().map({"BOT": "BUY", "SLD": "SELL"}).fillna(df["side"])
    return df.reset_index(drop=True)
