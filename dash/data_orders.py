# dash/data_orders.py
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

EXEC_DB = Path(__file__).resolve().parents[1] / "data" / "db" / "orders.db"


def _conn():
    return sqlite3.connect(EXEC_DB)


def load_executions_raw_since(symbol: str, lookback_minutes: int):
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)).isoformat()
    with _conn() as con:
        df = pd.read_sql_query(
            """
            SELECT exec_id, ts, price, qty, side
            FROM executions
            WHERE symbol = ? AND ts >= ?
            ORDER BY ts ASC
            """,
            con, params=(symbol, cutoff_iso)
        )
    if df.empty: return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["side"] = df["side"].str.upper().map({"BOT": "BUY", "SLD": "SELL"}).fillna(df["side"])
    return df.reset_index(drop=True)


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


# def load_latest_limit_since(symbol: str, lookback_minutes: int):
#     cutoff_iso = (datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)).isoformat()
#     with _conn() as con:
#         df = pd.read_sql_query(
#             """
#             SELECT lmt_price
#             FROM orders
#             WHERE symbol = ? AND ts >= ? AND lmt_price IS NOT NULL
#             ORDER BY ts DESC
#             LIMIT 1
#             """,
#             con, params=(symbol, cutoff_iso)
#         )
#     if df.empty: return None
#     return float(df.loc[0, "lmt_price"])
#
# def load_symbol_executions(symbol: str, lookback_minutes: int = 120) -> pd.DataFrame:
#     """
#     Returns executions for a symbol within lookback window.
#     Columns: ts, price, qty, side
#     """
#     since = (datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)).isoformat()
#     with _conn() as con:
#         df = pd.read_sql_query(
#             """
#             SELECT ts, price, qty, side
#             FROM executions
#             WHERE symbol = ?
#               AND ts >= ?
#             ORDER BY ts ASC
#             """,
#             con, params=(symbol, since)
#         )
#     print(df)
#     if df.empty:
#         return df
#     df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
#     print("Loading executions: ", df)
#     return df
#
# def load_symbol_latest_limit(symbol: str) -> float | None:
#     """
#     (Optional) Get most recent limit price from orders for this symbol (if any).
#     """
#     with _conn() as con:
#         df = pd.read_sql_query(
#             """
#             WITH latest AS (
#               SELECT *
#               FROM orders
#               WHERE symbol = ?
#               ORDER BY ts DESC
#               LIMIT 1
#             )
#             SELECT lmt_price FROM latest
#             """, con, params=(symbol,)
#         )
#     if df.empty or pd.isna(df.loc[0, "lmt_price"]):
#         return None
#     return float(df.loc[0, "lmt_price"])

def compute_pnl_curve(df_mid: pd.DataFrame, df_exec: pd.DataFrame) -> pd.DataFrame:
    """
    df_mid: columns ['ts', 'mid']  (already sorted ascending, UTC)
    df_exec: columns ['ts', 'price', 'qty', 'side'] with side in {'BUY','SELL'}
    Returns DataFrame with ['ts','position','cash','pnl'] aligned to df_mid['ts'].
    """
    if df_mid.empty:
        return pd.DataFrame(columns=["ts", "position", "cash", "pnl"])

    df_mid = df_mid[["ts", "mid"]].dropna().sort_values("ts").reset_index(drop=True)

    if df_exec.empty:
        out = df_mid.copy()
        out["position"] = 0.0
        out["cash"] = 0.0
        out["pnl"] = 0.0
        return out

    e = df_exec.copy()
    e["side"] = e["side"].str.upper().map({"BOT": "BUY", "SLD": "SELL"}).fillna(e["side"])
    sign = e["side"].map({"BUY": 1.0, "SELL": -1.0}).astype(float)
    e["net_qty"] = sign * e["qty"].astype(float)
    e["cash_flow"] = - e["net_qty"] * e["price"].astype(float)

    e = e.sort_values("ts").reset_index(drop=True)
    e["cum_pos"] = e["net_qty"].cumsum()
    e["cum_cash"] = e["cash_flow"].cumsum()

    # align cumulative state to each market timestamp
    state = e[["ts", "cum_pos", "cum_cash"]]
    aligned = pd.merge_asof(df_mid, state, on="ts", direction="backward")

    aligned["cum_pos"] = aligned["cum_pos"].fillna(0.0)
    aligned["cum_cash"] = aligned["cum_cash"].fillna(0.0)
    aligned["position"] = aligned["cum_pos"]
    aligned["cash"] = aligned["cum_cash"]
    aligned["pnl"] = aligned["cash"] + aligned["position"] * aligned["mid"]
    return aligned[["ts", "position", "cash", "pnl"]]


def load_commissions_since(lookback_minutes: int):
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)).isoformat()
    with _conn() as con:
        df = pd.read_sql_query(
            """
            SELECT exec_id, commission
            FROM commissions
            WHERE ts >= ?
            """,
            con, params=(cutoff_iso,)
        )
    if df.empty:
        return pd.DataFrame(columns=["exec_id", "commission"])
    return df
