class DBWriter:
    def __init__(self, conn):
        self.conn = conn

    def insert_tob_many(self, rows):
        # rows: list of tuples (symbol, ts, bid, ask, bid_size, ask_size)
        if not rows: return
        self.conn.executemany("""
            INSERT OR REPLACE INTO tob(symbol, ts, bid, ask, bid_size, ask_size)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)

    def insert_tape_many(self, rows):
        # rows: list of tuples (symbol, ts, price, size)
        if not rows: return
        self.conn.executemany("""
            INSERT OR REPLACE INTO tape(symbol, ts, price, size)
            VALUES (?, ?, ?, ?)
        """, rows)
