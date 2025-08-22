import sqlite3
from pathlib import Path


def get_conn(db_path="data/market.db"):
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_schema(conn):
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS tob (
        symbol     TEXT NOT NULL,
        ts         TIMESTAMP NOT NULL,
        bid        REAL NOT NULL,
        ask        REAL NOT NULL,
        bid_size   REAL,
        ask_size   REAL,
        PRIMARY KEY (symbol, ts)
    );

    CREATE INDEX IF NOT EXISTS idx_tob_symbol_ts ON tob(symbol, ts);

    CREATE TABLE IF NOT EXISTS tape (
        symbol   TEXT NOT NULL,
        ts       TIMESTAMP NOT NULL,
        price    REAL NOT NULL,
        size     REAL,
        PRIMARY KEY (symbol, ts)
    );

    CREATE INDEX IF NOT EXISTS idx_tape_symbol_ts ON tape(symbol, ts);
    """)
