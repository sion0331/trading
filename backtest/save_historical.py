# save as e.g. tools/download_history.py

import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytz
from ib_async import IB, Forex

# DB_PATH = "../data/db/history.db"
DB_PATH = "../data/db/jpy.db"


def _ensure(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript("""
    PRAGMA journal_mode=WAL;
    PRAGMA synchronous=NORMAL;

    CREATE TABLE IF NOT EXISTS tob (
        symbol    TEXT NOT NULL,
        ts        TEXT NOT NULL,   -- ISO8601
        bid       REAL NOT NULL,
        ask       REAL NOT NULL,
        bid_size  REAL,
        ask_size  REAL,
        PRIMARY KEY (symbol, ts)
    );
    CREATE INDEX IF NOT EXISTS idx_tob_symbol_ts ON tob(symbol, ts);

    CREATE TABLE IF NOT EXISTS tape (
        symbol   TEXT NOT NULL,
        ts       TEXT NOT NULL,    -- ISO8601
        price    REAL NOT NULL,
        size     REAL,
        PRIMARY KEY (symbol, ts)
    );
    CREATE INDEX IF NOT EXISTS idx_tape_symbol_ts ON tape(symbol, ts);
    """)


def _ins_tob(conn, rows):
    conn.executemany("""
        INSERT OR IGNORE INTO tob(symbol, ts, bid, ask, bid_size, ask_size)
        VALUES (?, ?, ?, ?, ?, ?)
    """, rows)


def _ins_tape(conn, rows):
    conn.executemany("""
        INSERT OR IGNORE INTO tape(symbol, ts, price, size)
        VALUES (?, ?, ?, ?)
    """, rows)


def _page_history(
        ib: IB,
        con: sqlite3.Connection,
        contract,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        what_to_show: str,  # "Bid_Ask" or "Trades"
        page: int = 1000,
        step_s: int = 60,  # advance after a page with data
        empty_step_s: int = 300,  # advance when no data returned
        pace_sleep_s: float = 0.25,  # small sleep between requests
):
    """Walk forward from start_dt -> end_dt, inserting rows and never breaking on empty pages."""
    tz = "UTC"
    dt = start_dt.astimezone(timezone.utc)

    while dt < end_dt:
        s = dt.strftime("%Y%m%d %H:%M:%S") + f" {tz}"
        e = end_dt.astimezone(timezone.utc).strftime("%Y%m%d %H:%M:%S") + f" {tz}"
        print(symbol, what_to_show, s, e)

        if what_to_show == "Bid_Ask":
            ticks = ib.reqHistoricalTicks(
                contract, s, e, numberOfTicks=page,
                whatToShow="Bid_Ask", useRth=False, ignoreSize=True
            )
        else:  # "Last"
            ticks = ib.reqHistoricalTicks(
                contract, s, e, numberOfTicks=page,
                whatToShow="Trades", useRth=False, ignoreSize=False
            )

        rows = []
        last_ts = None

        for t in ticks or []:
            ts = getattr(t, "time", None)

            if what_to_show == "Bid_Ask":
                bid = getattr(t, "priceBid", None)
                ask = getattr(t, "priceAsk", None)
                bs = getattr(t, "sizeBid", None)
                asz = getattr(t, "sizeAsk", None)
                if ts and bid and ask and bid > 0 and ask > 0:
                    ts_iso = ts.astimezone(timezone.utc).isoformat() if isinstance(ts, datetime) \
                        else datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
                    rows.append((symbol, ts_iso, float(bid), float(ask), float(bs or 0), float(asz or 0)))
                    last_ts = ts
            else:
                px = getattr(t, "price", None)
                sz = getattr(t, "size", None)
                if ts and px and px > 0:
                    ts_iso = ts.astimezone(timezone.utc).isoformat() if isinstance(ts, datetime) \
                        else datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
                    rows.append((symbol, ts_iso, float(px), float(sz or 0)))
                    last_ts = ts

        # insert if any
        if rows:
            if what_to_show == "Bid_Ask":
                _ins_tob(con, rows)
            else:
                _ins_tape(con, rows)

        # Advance cursor:
        if last_ts is not None:
            # we got data for this window; continue just past the last tick
            if isinstance(last_ts, datetime):
                dt = last_ts + timedelta(seconds=step_s)
            else:
                dt = datetime.fromtimestamp(int(last_ts) + step_s, tz=timezone.utc)
        else:
            # empty page: jump forward a bigger step to avoid getting stuck
            dt = dt + timedelta(seconds=empty_step_s)

        # never overshoot end
        if dt > end_dt:
            dt = end_dt

        # small pacing delay to avoid hammering IB
        if pace_sleep_s:
            time.sleep(pace_sleep_s)


def download(
        ib: IB,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        page=1000,
        step_s=60,
        empty_step_s=300,
):
    con = sqlite3.connect(DB_PATH, isolation_level=None, check_same_thread=False)
    _ensure(con)

    # For now always EURUSD; customize mapping here if needed
    c = Forex(symbol)

    # Bid/Ask stream
    _page_history(
        ib, con, c, symbol, start_dt, end_dt,
        what_to_show="Bid_Ask",
        page=page, step_s=step_s, empty_step_s=empty_step_s,
    )

    # Trades stream
    # _page_history(
    #     ib, con, c, symbol, start_dt, end_dt,
    #     what_to_show="Trades",
    #     page=page, step_s=step_s, empty_step_s=empty_step_s,
    # )

    con.close()


if __name__ == "__main__":
    # sym = "EURUSD"
    sym = "USDJPY"
    sd = datetime(2025, 6, 1, tzinfo=pytz.UTC)
    ed = datetime(2025, 8, 31, tzinfo=pytz.UTC)
    # ed = sd + timedelta(days=1)

    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=7)
    try:
        download(ib, sym, sd, ed, page=1000, step_s=60, empty_step_s=300)  # e.g., jump 15m on empty pages
    finally:
        ib.disconnect()
