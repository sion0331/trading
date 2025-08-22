# dash/data_orders.py
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ORDERS_DB = Path(__file__).resolve().parents[1] / "data" / "db" / "orders.db"


def _conn():
    return sqlite3.connect(ORDERS_DB)


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


def load_fills_range(symbol: str, start_utc: str, end_utc: str):
    with sqlite3.connect(ORDERS_DB) as con:
        df = pd.read_sql_query(
            """
            SELECT ts, avg_fill_price AS price, qty, action AS side, order_id, order_type
            FROM orders
            WHERE symbol = ? AND status = 'Filled' AND ts >= ? AND ts < ?
                AND remaining_qty < 0.1
            ORDER BY ts ASC
            """,
            con, params=(symbol, start_utc, end_utc)
        )
    return df


def load_commissions_range(symbol: str, start_utc: str, end_utc: str):
    with sqlite3.connect(ORDERS_DB) as con:
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
