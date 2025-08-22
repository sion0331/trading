import sqlite3
from datetime import datetime, timedelta, timezone

import pytz
from ib_async import *

from marketData.contracts import BTC

DB_PATH = "../data/db/market.db"


def get_conn(db_path=DB_PATH):
    con = sqlite3.connect(db_path, isolation_level=None, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS tob (
        symbol    TEXT NOT NULL,
        ts        TEXT NOT NULL,   -- ISO8601 string
        bid       REAL NOT NULL,
        ask       REAL NOT NULL,
        bid_size  REAL,
        ask_size  REAL,
        PRIMARY KEY (symbol, ts)
    );
    CREATE INDEX IF NOT EXISTS idx_tob_symbol_ts ON tob(symbol, ts);
    """)
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS tape (
            symbol   TEXT NOT NULL,
            ts       TEXT NOT NULL,   -- ISO8601 string
            price    REAL NOT NULL,
            size     REAL,
            PRIMARY KEY (symbol, ts)
        );
        CREATE INDEX IF NOT EXISTS idx_tape_symbol_ts ON tape(symbol, ts);
    """)
    return con


def save_bidask_ticks_to_db(ib: IB, contract, start_dt: datetime, end_dt: datetime, symbol: str, query_s=60,
                            db_path=DB_PATH):
    con = get_conn(db_path)
    cur = con.cursor()

    timezone_str = "UTC"
    page_size = 1000
    dt = start_dt.astimezone(timezone.utc)

    while dt < end_dt:
        dt_str = dt.strftime("%Y%m%d %H:%M:%S") + " " + timezone_str
        ed_str = end_dt.astimezone(timezone.utc).strftime("%Y%m%d %H:%M:%S") + " " + timezone_str

        print("Downloading TOB...", dt_str)

        # Historical bid/ask ticks (IB returns up to 1000)
        ticks = ib.reqHistoricalTicks(
            contract,
            startDateTime=dt_str,
            endDateTime=ed_str,
            numberOfTicks=page_size,
            whatToShow="Bid_Ask",
            useRth=False,
            ignoreSize=True
        )

        if not ticks:
            break

        # Convert to tuples for SQLite
        rows = []
        last_ts = None
        for t in ticks:
            # For HistoricalTickBidAsk, field names are typically:
            #  t.time (datetime), t.priceBid, t.priceAsk, t.sizeBid, t.sizeAsk
            ts = getattr(t, "time", None)
            bid = getattr(t, "priceBid", None)
            ask = getattr(t, "priceAsk", None)
            bid_sz = getattr(t, "sizeBid", None)
            ask_sz = getattr(t, "sizeAsk", None)

            # IB sometimes returns placeholder -1.0; skip those
            if ts and bid and ask and bid > 0 and ask > 0:
                if isinstance(ts, datetime):
                    ts_iso = ts.astimezone(timezone.utc).isoformat()
                else:
                    # If integer epoch seconds
                    ts_iso = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
                rows.append((symbol, ts_iso, float(bid), float(ask), bid_sz, ask_sz))
                last_ts = ts

        if rows:
            cur.executemany("""
                INSERT OR IGNORE INTO tob(symbol, ts, bid, ask, bid_size, ask_size)
                VALUES (?, ?, ?, ?, ?, ?)
            """, rows)

        # If we didn’t fill the page, we’re done
        if len(ticks) < page_size or last_ts is None:
            break

        # Advance the cursor to just after the last tick to avoid duplicates
        if isinstance(last_ts, datetime):
            dt = last_ts + timedelta(seconds=query_s)
        else:
            dt = datetime.fromtimestamp(int(last_ts) + query_s, tz=timezone.utc)

    con.close()


def save_trade_ticks_to_db(ib: IB, contract, start_dt: datetime, end_dt: datetime, symbol: str, query_s=60,
                           db_path=DB_PATH):
    con = get_conn(db_path)
    cur = con.cursor()

    timezone_str = "UTC"
    page_size = 1000
    dt = start_dt.astimezone(timezone.utc)

    while dt < end_dt:
        dt_str = dt.strftime("%Y%m%d %H:%M:%S") + " " + timezone_str
        ed_str = end_dt.astimezone(timezone.utc).strftime("%Y%m%d %H:%M:%S") + " " + timezone_str
        print("Downloading Tape...", dt_str)

        # Historical trades (Last)
        ticks = ib.reqHistoricalTicks(
            contract,
            startDateTime=dt_str,
            endDateTime=ed_str,
            numberOfTicks=page_size,
            whatToShow="Last",
            useRth=False,
            ignoreSize=False,
        )

        if not ticks:
            break

        rows = []
        last_ts = None
        for t in ticks:
            # HistoricalTickLast typically exposes: time, price, size
            ts = getattr(t, "time", None)
            price = getattr(t, "price", None)
            size = getattr(t, "size", None)

            if ts and price and price > 0:
                if isinstance(ts, datetime):
                    ts_iso = ts.astimezone(timezone.utc).isoformat()
                else:
                    ts_iso = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
                rows.append((symbol, ts_iso, float(price), size))
                last_ts = ts

        if rows:
            cur.executemany("""
                INSERT OR IGNORE INTO tape(symbol, ts, price, size)
                VALUES (?, ?, ?, ?)
            """, rows)

        if len(ticks) < page_size or last_ts is None:
            break

        if isinstance(last_ts, datetime):
            dt = last_ts + timedelta(seconds=query_s)
        else:
            dt = datetime.fromtimestamp(int(last_ts) + query_s, tz=timezone.utc)

    con.close()


from ib_async import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

symbol = "BTC"
contract = BTC  # EURUSD

sd = datetime(2025, 8, 8, tzinfo=pytz.UTC)
ed = sd + timedelta(days=1)

save_bidask_ticks_to_db(ib, contract, sd, ed, symbol=symbol, query_s=60, db_path="../data/db/market.db")
save_trade_ticks_to_db(ib, contract, sd, ed, symbol=symbol, query_s=60, db_path="../data/db/market.db")
