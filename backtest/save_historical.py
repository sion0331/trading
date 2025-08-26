# Uses ib_async to pull EURUSD TOB & trades into your existing trading.db schema

import sqlite3
from datetime import datetime, timedelta, timezone

import pytz
from ib_async import IB, Forex

DB_PATH = "../data/db/trading.db"


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


def download(ib: IB, symbol: str, start_dt: datetime, end_dt: datetime, page=1000, step_s=60):
    con = sqlite3.connect(DB_PATH, isolation_level=None, check_same_thread=False)
    _ensure(con)

    c = Forex("EURUSD") if symbol.upper() == "EUR" else Forex("EURUSD")  # keep default EUR
    tz = "UTC"
    dt = start_dt.astimezone(timezone.utc)

    while dt < end_dt:
        s = dt.strftime("%Y%m%d %H:%M:%S") + f" {tz}"
        e = end_dt.astimezone(timezone.utc).strftime("%Y%m%d %H:%M:%S") + f" {tz}"

        # Bid/Ask page
        bas = ib.reqHistoricalTicks(c, s, e, numberOfTicks=page, whatToShow="Bid_Ask", useRth=False, ignoreSize=True)
        rows = []
        last_ts = None
        for t in bas or []:
            ts = getattr(t, "time", None)
            bid = getattr(t, "priceBid", None)
            ask = getattr(t, "priceAsk", None)
            bs = getattr(t, "sizeBid", None)
            asz = getattr(t, "sizeAsk", None)
            if ts and bid and ask and bid > 0 and ask > 0:
                if isinstance(ts, datetime):
                    ts_iso = ts.astimezone(timezone.utc).isoformat()
                else:
                    ts_iso = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
                rows.append((symbol, ts_iso, float(bid), float(ask), float(bs or 0), float(asz or 0)))
                last_ts = ts
        if rows:
            _ins_tob(con, rows)

        if not bas or len(bas) < page or last_ts is None:
            break
        dt = (last_ts + timedelta(seconds=step_s)) if isinstance(last_ts, datetime) \
            else datetime.fromtimestamp(int(last_ts) + step_s, tz=timezone.utc)

    # Trades
    dt = start_dt.astimezone(timezone.utc)
    while dt < end_dt:
        s = dt.strftime("%Y%m%d %H:%M:%S") + f" {tz}"
        e = end_dt.astimezone(timezone.utc).strftime("%Y%m%d %H:%M:%S") + f" {tz}"
        lst = ib.reqHistoricalTicks(c, s, e, numberOfTicks=page, whatToShow="Last", useRth=False, ignoreSize=False)
        rows = []
        last_ts = None
        for t in lst or []:
            ts = getattr(t, "time", None)
            px = getattr(t, "price", None)
            sz = getattr(t, "size", None)
            if ts and px and px > 0:
                if isinstance(ts, datetime):
                    ts_iso = ts.astimezone(timezone.utc).isoformat()
                else:
                    ts_iso = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
                rows.append((symbol, ts_iso, float(px), float(sz or 0)))
                last_ts = ts
        if rows:
            _ins_tape(con, rows)

        if not lst or len(lst) < page or last_ts is None:
            break
        dt = (last_ts + timedelta(seconds=step_s)) if isinstance(last_ts, datetime) \
            else datetime.fromtimestamp(int(last_ts) + step_s, tz=timezone.utc)

    con.close()


if __name__ == "__main__":
    sym = "EUR"
    sd = datetime(2025, 8, 8, tzinfo=pytz.UTC)
    ed = sd + timedelta(days=1)

    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=7)
    download(ib, sym, sd, ed)
    ib.disconnect()
