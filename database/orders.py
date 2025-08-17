# order_logger.py
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------- SQLite helpers ----------
def _norm_side(side):
    if not side: return None
    s = side.upper()
    return {"BOT": "BUY", "SLD": "SELL"}.get(s, s)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_column(con: sqlite3.Connection, table: str, col: str, coltype: str):
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = {r[1] for r in cur.fetchall()}  # names
    if col not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
        con.commit()


def _ensure_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id        INTEGER NOT NULL,
        perm_id         INTEGER,
        client_id       INTEGER,
        ts              TEXT NOT NULL,          -- event time (ISO8601)
        symbol          TEXT,
        action          TEXT,
        order_type      TEXT,
        qty             REAL,
        lmt_price       REAL,
        aux_price       REAL,
        status          TEXT,                   -- PendingSubmit/Submitted/Cancelled/Filled/etc.
        filled_qty      REAL,
        remaining_qty   REAL,
        avg_fill_price  REAL,
        why_held        TEXT,
        mkt_cap_price   REAL,
        note            TEXT,                   -- free-form: "intent: send", "intent: cancel", "event: orderStatus"
        PRIMARY KEY(order_id, ts)
    );

    CREATE INDEX IF NOT EXISTS idx_orders_orderid_ts ON orders(order_id, ts);
    CREATE INDEX IF NOT EXISTS idx_orders_symbol_ts  ON orders(symbol, ts);

    CREATE TABLE IF NOT EXISTS executions (
        exec_id     TEXT PRIMARY KEY,
        ts          TEXT NOT NULL,              -- event time (ISO8601)
        order_id    INTEGER,
        perm_id     INTEGER,
        symbol      TEXT,
        side        TEXT,
        qty         REAL,
        price       REAL,
        exchange    TEXT
    );

    CREATE TABLE IF NOT EXISTS commissions (
        exec_id     TEXT PRIMARY KEY,
        ts          TEXT NOT NULL,
        commission  REAL,
        currency    TEXT,
        realized_pnl REAL
    );
    """)
    _ensure_column(con, "executions", "order_type", "TEXT")
    _ensure_column(con, "executions", "liquidity", "INTEGER")
    con.close()


def _connect(db_path: Path):
    con = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    return con


# ---------- The logger ----------

class OrderLogger:
    """
    Hook into ib_async.IB events and persist order intents, status updates,
    executions, and commissions to a SQLite database.
    """

    def __init__(self, ib, db_path: str | Path = "data/db/orders.db"):
        self.ib = ib
        self.db_path = Path(db_path)
        _ensure_db(self.db_path)

        # register event handlers
        # These event names are aligned with ib_insync style and many ib_async forks.
        # If your package exposes different names, tweak here.
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_exec_details
        # Commission reports may be optional
        if hasattr(self.ib, "commissionReportEvent"):
            self.ib.commissionReportEvent += self._on_commission_report

    # --------- Public helpers to log INTENT when you send/cancel ---------

    def log_send_intent(self, contract, order):
        """
        Call this right before ib.placeOrder(contract, order)
        """
        self._insert_order_row(
            order_id=getattr(order, "orderId", None),
            perm_id=getattr(order, "permId", None),
            client_id=getattr(order, "clientId", None),
            symbol=getattr(contract, "symbol", None),
            action=getattr(order, "action", None),
            order_type=getattr(order, "orderType", None),
            qty=getattr(order, "totalQuantity", None),
            lmt_price=getattr(order, "lmtPrice", None),
            aux_price=getattr(order, "auxPrice", None),
            status="Intent",
            filled_qty=None,
            remaining_qty=None,
            avg_fill_price=None,
            why_held=None,
            mkt_cap_price=None,
            note="intent: send"
        )

    def log_cancel_intent(self, order_id: int):
        """
        Call this right before ib.cancelOrder(orderId)
        """
        self._insert_order_row(
            order_id=order_id,
            perm_id=None,
            client_id=None,
            symbol=None,
            action=None,
            order_type=None,
            qty=None,
            lmt_price=None,
            aux_price=None,
            status="IntentCancel",
            filled_qty=None,
            remaining_qty=None,
            avg_fill_price=None,
            why_held=None,
            mkt_cap_price=None,
            note="intent: cancel"
        )

    # --------- Event handlers ---------

    def _on_order_status(self, *args, **kwargs):
        """
        Handles:
          - kwargs: order=..., status=..., filled=..., remaining=..., avgFillPrice=...
          - positional: (order, status, filled, remaining, avgFillPrice, permId, ..., clientId, whyHeld, mktCapPrice)
          - Trade object: has .order, .orderStatus, .contract
        """
        order = kwargs.get("order")
        status = kwargs.get("status")
        filled = kwargs.get("filled")
        remaining = kwargs.get("remaining")
        avg_fill_price = kwargs.get("avgFillPrice", kwargs.get("avg_fill_price"))
        perm_id = kwargs.get("permId")
        client_id = kwargs.get("clientId")
        why_held = kwargs.get("whyHeld", kwargs.get("why_held"))
        mkt_cap_price = kwargs.get("mktCapPrice", kwargs.get("mkt_cap_price"))
        symbol = None
        action = None
        order_type = None
        qty = None
        lmt_price = None
        aux_price = None
        order_id = kwargs.get("orderId")

        # --- Trade shape ---
        if len(args) == 1 and hasattr(args[0], "orderStatus") and hasattr(args[0], "order"):
            trade = args[0]
            os = getattr(trade, "orderStatus", None)
            od = getattr(trade, "order", None)
            ct = getattr(trade, "contract", None)

            # from orderStatus
            if os is not None:
                status = getattr(os, "status", status)
                filled = getattr(os, "filled", filled)
                remaining = getattr(os, "remaining", remaining)
                avg_fill_price = getattr(os, "avgFillPrice", avg_fill_price)
                perm_id = getattr(os, "permId", perm_id)
                client_id = getattr(os, "clientId", client_id)
                why_held = getattr(os, "whyHeld", why_held)
                mkt_cap_price = getattr(os, "mktCapPrice", mkt_cap_price)
                order_id = getattr(os, "orderId", order_id)

            # from order
            if od is not None:
                order = od
                order_id = getattr(od, "orderId", order_id)
                action = getattr(od, "action", action)
                order_type = getattr(od, "orderType", order_type)
                qty = getattr(od, "totalQuantity", qty)
                lmt_price = getattr(od, "lmtPrice", lmt_price)
                aux_price = getattr(od, "auxPrice", aux_price)
                perm_id = getattr(od, "permId", perm_id)
                client_id = getattr(od, "clientId", client_id)

            # from contract
            if ct is not None:
                symbol = getattr(ct, "symbol", symbol)

        # --- classic positional shape ---
        elif len(args) >= 2 and order is None and status is None:
            try:
                order = args[0]
                status = args[1]
                filled = args[2] if len(args) > 2 else filled
                remaining = args[3] if len(args) > 3 else remaining
                avg_fill_price = args[4] if len(args) > 4 else avg_fill_price
                perm_id = args[5] if len(args) > 5 else perm_id
                client_id = args[8] if len(args) > 8 else client_id

                order_id = getattr(order, "orderId", order_id)
                action = getattr(order, "action", action)
                order_type = getattr(order, "orderType", order_type)
                qty = getattr(order, "totalQuantity", qty)
                lmt_price = getattr(order, "lmtPrice", lmt_price)
                aux_price = getattr(order, "auxPrice", aux_price)
                symbol = getattr(order, "symbol", symbol)  # some libs set symbol on order
            except Exception:
                pass

        # --- enrich from kwargs.order if present ---
        if order is not None:
            order_id = getattr(order, "orderId", order_id)
            action = getattr(order, "action", action)
            order_type = getattr(order, "orderType", order_type)
            qty = getattr(order, "totalQuantity", qty)
            lmt_price = getattr(order, "lmtPrice", lmt_price)
            aux_price = getattr(order, "auxPrice", aux_price)
            client_id = getattr(order, "clientId", client_id)
            perm_id = getattr(order, "permId", perm_id)

        # If we still don't have order_id, don't insert (avoid NOT NULL violation)
        if order_id is None:
            print("WARN: orderStatus without order_id; skipping insert.", args, kwargs)
            return

        self._insert_order_row(
            order_id=order_id,
            perm_id=perm_id,
            client_id=client_id,
            symbol=symbol,
            action=action,
            order_type=order_type,
            qty=qty,
            lmt_price=lmt_price,
            aux_price=aux_price,
            status=status,
            filled_qty=filled,
            remaining_qty=remaining,
            avg_fill_price=avg_fill_price,
            why_held=why_held,
            mkt_cap_price=mkt_cap_price,
            note="event: orderStatus"
        )

    def _on_exec_details(self, *args, **kwargs):
        """
        Accepts multiple shapes and writes to executions/commissions tables:
        - (contract, execution)
        - Fill object with .contract, .execution, .commissionReport
        - Trade object with .fills list of Fill
        - kwargs: contract=..., execution=..., commissionReport=...
        """
        fills = []  # list of tuples: (contract, execution, commissionReport)

        # --- kwargs shape --------------------------------------------------------
        k_contract = kwargs.get("contract")
        k_execution = kwargs.get("execution")
        k_comm = kwargs.get("commissionReport")
        if k_contract is not None and k_execution is not None:
            fills.append((k_contract, k_execution, k_comm))

        # --- positional shapes ---------------------------------------------------
        for a in args:
            # Fill-like: has .contract and .execution
            if hasattr(a, "execution") and hasattr(a, "contract"):
                fills.append((getattr(a, "contract"), getattr(a, "execution"), getattr(a, "commissionReport", None)))
                continue

            # Trade-like: has .fills iterable of Fill
            if hasattr(a, "fills"):
                try:
                    for f in (a.fills or []):
                        fills.append((getattr(f, "contract", None), getattr(f, "execution", None),
                                      getattr(f, "commissionReport", None)))
                except Exception:
                    pass
                continue

            # Classic tuple: (contract, execution)
            # If args came as (contract, execution) without attributes like .fills/.execution
            if k_contract is None and k_execution is None and len(args) >= 2:
                c0, e1 = args[0], args[1]
                if hasattr(c0, "symbol") and hasattr(e1, "price"):
                    fills.append((c0, e1, None))
                    break

        if not fills:
            print("WARN: _on_exec_details could not parse event:", args, kwargs)
            return
        # --- write all parsed fills ---------------------------------------------
        ts_now = _utc_now_iso()
        with _connect(self.db_path) as con:
            for contract, execution, comm in fills:
                if execution is None or contract is None:
                    continue

                exec_id = getattr(execution, "execId", None)
                order_id = getattr(execution, "orderId", None)
                perm_id = getattr(execution, "permId", None)
                side = _norm_side(getattr(execution, "side", None))
                qty = getattr(execution, "shares", getattr(execution, "qty", None))
                price = getattr(execution, "price", None)
                exchange = getattr(execution, "exchange", None)
                symbol = getattr(contract, "symbol", None)
                liquidity = getattr(execution, "lastLiquidity", None)  # IB enums: 1 added, 2 removed ..
                ts_iso = getattr(execution, "time", None).isoformat()

                # todo - use from message
                order_type = None
                try:
                    got = con.execute("""
                        SELECT order_type FROM orders
                        WHERE order_id = ? ORDER BY ts DESC LIMIT 1
                    """, (order_id,)).fetchone()
                    if got:
                        order_type = got[0]
                except Exception:
                    pass

                # Debug print to verify extraction
                print("EXEC ->",
                      (exec_id, ts_iso, order_id, perm_id, symbol, side, qty, price, exchange, order_type, liquidity))

                # Write execution
                con.execute("""
                    INSERT OR REPLACE INTO executions
                    (exec_id, ts, order_id, perm_id, symbol, side, qty, price, exchange, order_type, liquidity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (exec_id, ts_iso, order_id, perm_id, symbol, side, qty, price, exchange, order_type, liquidity))

                # Optional: write commission
                if comm is not None:
                    c_exec_id = getattr(comm, "execId", None) or exec_id
                    commission = getattr(comm, "commission", None)
                    currency = getattr(comm, "currency", None)
                    realized = getattr(comm, "realizedPNL", getattr(comm, "realizedPnl", None))
                    con.execute("""
                            INSERT INTO commissions
                            (exec_id, ts, commission, currency, realized_pnl, order_id, symbol)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (c_exec_id, ts_iso, commission, currency, realized, order_id, symbol))

    def _on_commission_report(self, *args, **kwargs):
        """
        Typically signature (commissionReport)
        """
        cr = kwargs.get("commissionReport", None)
        if cr is None and len(args) >= 1:
            cr = args[0]

        exec_id = getattr(cr, "execId", None)
        commission = getattr(cr, "commission", None)
        currency = getattr(cr, "currency", None)
        realized_pnl = getattr(cr, "realizedPNL", getattr(cr, "realizedPnl", None))

        ts = _utc_now_iso()
        with _connect(self.db_path) as con:
            con.execute("""
                INSERT OR REPLACE INTO commissions
                (exec_id, ts, commission, currency, realized_pnl)
                VALUES (?, ?, ?, ?, ?)
            """, (exec_id, ts, commission, currency, realized_pnl))

    # ---------- low-level insert ----------

    def _insert_order_row(
            self,
            *,
            order_id: Optional[int],
            perm_id: Optional[int],
            client_id: Optional[int],
            symbol: Optional[str],
            action: Optional[str],
            order_type: Optional[str],
            qty: Optional[float],
            lmt_price: Optional[float],
            aux_price: Optional[float],
            status: Optional[str],
            filled_qty: Optional[float],
            remaining_qty: Optional[float],
            avg_fill_price: Optional[float],
            why_held: Optional[str],
            mkt_cap_price: Optional[float],
            note: Optional[str]
    ):
        ts = _utc_now_iso()
        with _connect(self.db_path) as con:
            con.execute("""
                INSERT OR REPLACE INTO orders
                (order_id, perm_id, client_id, ts, symbol, action, order_type, qty, lmt_price, aux_price,
                 status, filled_qty, remaining_qty, avg_fill_price, why_held, mkt_cap_price, note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (order_id, perm_id, client_id, ts, symbol, action, order_type, qty, lmt_price, aux_price,
                  status, filled_qty, remaining_qty, avg_fill_price, why_held, mkt_cap_price, note))

    # ---------- optional: unregister ----------

    def close(self):
        try:
            self.ib.orderStatusEvent -= self._on_order_status
        except Exception:
            pass
        try:
            self.ib.execDetailsEvent -= self._on_exec_details
        except Exception:
            pass
        if hasattr(self.ib, "commissionReportEvent"):
            try:
                self.ib.commissionReportEvent -= self._on_commission_report
            except Exception:
                pass
