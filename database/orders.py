# order_logger.py
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------- SQLite helpers ----------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    def __init__(self, ib, db_path: str | Path = "data/db/execution_log.db"):
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
        Compatible with common IB event signatures. We try to detect both
        (order, status, ...) and keyword-style.
        """
        # Try kwargs first (some libs pass named args)
        status = kwargs.get("status", None)

        order = kwargs.get("order", None)
        order_id = getattr(order, "orderId", None) if order is not None else kwargs.get("orderId", None)
        perm_id = getattr(order, "permId", None) if order is not None else kwargs.get("permId", None)
        client_id = getattr(order, "clientId", None) if order is not None else kwargs.get("clientId", None)

        symbol = getattr(order, "symbol", None)
        action = getattr(order, "action", None)
        order_type = getattr(order, "orderType", None)
        qty = getattr(order, "totalQuantity", None)
        lmt_price = getattr(order, "lmtPrice", None)
        aux_price = getattr(order, "auxPrice", None)

        filled = kwargs.get("filled", None)
        remaining = kwargs.get("remaining", None)
        avg_fill_price = kwargs.get("avgFillPrice", kwargs.get("avg_fill_price", None))
        why_held = kwargs.get("whyHeld", kwargs.get("why_held", None))
        mkt_cap_price = kwargs.get("mktCapPrice", kwargs.get("mkt_cap_price", None))

        # Some ib_async variants pass positional args like:
        # (order, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
        if status is None and len(args) >= 2:
            try:
                order = args[0]
                status = args[1]
                filled = args[2] if len(args) > 2 else None
                remaining = args[3] if len(args) > 3 else None
                avg_fill_price = args[4] if len(args) > 4 else None
                perm_id = args[5] if len(args) > 5 else perm_id
                client_id = args[8] if len(args) > 8 else client_id

                order_id = getattr(order, "orderId", order_id)
                symbol = getattr(order, "symbol", symbol)
                action = getattr(order, "action", action)
                order_type = getattr(order, "orderType", order_type)
                qty = getattr(order, "totalQuantity", qty)
                lmt_price = getattr(order, "lmtPrice", lmt_price)
                aux_price = getattr(order, "auxPrice", aux_price)
            except Exception:
                pass

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
        Typically signature (contract, execution)
        """
        contract = kwargs.get("contract", None)
        execution = kwargs.get("execution", None)

        if execution is None and len(args) >= 2:
            contract, execution = args[0], args[1]

        exec_id = getattr(execution, "execId", None)
        order_id = getattr(execution, "orderId", None)
        perm_id = getattr(execution, "permId", None)
        side = getattr(execution, "side", None)
        qty = getattr(execution, "shares", getattr(execution, "qty", None))
        price = getattr(execution, "price", None)
        exchange = getattr(execution, "exchange", None)
        symbol = getattr(contract, "symbol", None)

        ts = _utc_now_iso()
        with _connect(self.db_path) as con:
            con.execute("""
                INSERT OR REPLACE INTO executions
                (exec_id, ts, order_id, perm_id, symbol, side, qty, price, exchange)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (exec_id, ts, order_id, perm_id, symbol, side, qty, price, exchange))

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
