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
        version     INTEGER NOT NULL DEFAULT 1,
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
        symbol      TEXT,
        order_id    INTEGER,
        commission  REAL,
        currency    TEXT,
        realized_pnl REAL
    );
    
    CREATE INDEX IF NOT EXISTS idx_executions_execid ON executions(exec_id);
    """)
    _ensure_column(con, "executions", "order_type", "TEXT")
    _ensure_column(con, "executions", "liquidity", "INTEGER")
    _ensure_column(con, "commissions", "symbol", "TEXT")
    _ensure_column(con, "commissions", "order_id", "INTEGER")

    con.close()


def _connect(db_path: Path):
    con = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    return con


# ---------- The logger ----------

def _next_version(con, table: str, exec_id: str) -> int:
    row = con.execute(
        f"SELECT COALESCE(MAX(version), 0) FROM {table} WHERE exec_id = ?",
        (exec_id,)
    ).fetchone()
    return int(row[0]) + 1


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
        self.ib.connectedEvent += self._on_connected_event
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_exec_details
        self.ib.commissionReportEvent += self._on_commission_report

        self.is_connected = False

    # --------- Public helpers to log INTENT when you send/cancel ---------

    def log_send_intent(self, contract, order, ts):
        """
        Call this right before ib.placeOrder(contract, order)
        """
        self._insert_order_row(
            ts=ts,
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

    def log_cancel_intent(self, order_id: int, ts: str):
        """
        Call this right before ib.cancelOrder(orderId)
        """
        self._insert_order_row(
            ts=ts,
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

    def _on_connected_event(self):
        self.is_connected = True

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
        if (len(args) == 1 and hasattr(args[0], "orderStatus") and hasattr(args[0], "order")
                and hasattr(args[0], "contract") and hasattr(args[0], "log")):
            trade = args[0]
            os = getattr(trade, "orderStatus", None)
            od = getattr(trade, "order", None)
            ct = getattr(trade, "contract", None)
            log = getattr(trade, "log", None)

            # from orderStatus
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
            symbol = getattr(ct, "symbol", symbol)

            for l in log:
                # print(getattr(l, "status", None), status)
                if getattr(l, "status", None) == status:
                    ts_iso = getattr(l, "time").astimezone(timezone.utc).isoformat()
                    # print("Order _on_order_status:", order_id, getattr(l, "status", ""))
                    self._insert_order_row(
                        ts=ts_iso,
                        order_id=order_id,
                        perm_id=perm_id,
                        client_id=client_id,
                        symbol=symbol,
                        action=action,
                        order_type=order_type,
                        qty=qty,
                        lmt_price=lmt_price,
                        aux_price=aux_price,
                        status=getattr(l, "status", ""),
                        filled_qty=filled,
                        remaining_qty=remaining,
                        avg_fill_price=avg_fill_price,
                        why_held=why_held,
                        mkt_cap_price=mkt_cap_price,
                        note="event: orderStatus"
                    )

        # --- classic positional shape ---
        else:
            print("### ERROR | Order Status: ", args)

    def _on_exec_details(self, *args, **kwargs):
        """
        Accepts multiple shapes and writes to executions/commissions tables:
        - (contract, execution)
        - Fill object with .contract, .execution, .commissionReport
        - Trade object with .fills list of Fill
        - kwargs: contract=..., execution=..., commissionReport=...
        """
        fills = []  # list of tuples: (contract, execution, ts)
        if hasattr(args[0], "fills"):
            for f in args[0].fills:
                fills.append((getattr(f, "contract", None), getattr(f, "execution", None), getattr(f, "time", None)))
        else:
            fills.append((getattr(args[1], "contract", None), getattr(args[1], "execution", None),
                          getattr(args[1], "time", None)))

        if not fills:
            print("WARN: _on_exec_details could not parse event:", args, kwargs)
            return
        # --- write all parsed fills ---------------------------------------------
        with _connect(self.db_path) as con:
            for contract, execution, ts in fills:
                if execution is None or contract is None or ts is None:
                    print("ERROR | Missing execution message: ", contract, execution)
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
                ver = _next_version(con, "executions", exec_id)
                ts_iso = ts.isoformat()

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

                # Write execution
                con.execute("""
                    INSERT OR REPLACE INTO executions
                    (exec_id, version, ts, order_id, perm_id, symbol, side, qty, price, exchange, order_type, liquidity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    exec_id, ver, ts_iso, order_id, perm_id, symbol, side, qty, price, exchange, order_type, liquidity))

                print(
                    f"### {ts_iso} | FILLED | {symbol} {side} {order_type} {qty} @ {price} | {order_id} {perm_id} {exec_id}")

    def _on_commission_report(self, *args, **kwargs):
        """
        Handles multiple shapes:
          - commissionReportEvent( Trade(...), Fill(...), CommissionReport(...) )
          - commissionReportEvent( CommissionReport(...) )
          - commissionReportEvent( commissionReport=... )
        We treat this as the single source of truth for fees/realized PnL.
        """
        if not self.is_connected:
            return
        # Collect tuples of (commissionReport, execution?, contract?, order?)
        items = []

        # 2) positional forms
        for a in args:
            if hasattr(a, "fills"):
                order = getattr(a, "order", None)
                for f in (a.fills or []):
                    cr = getattr(f, "commissionReport", None)
                    ex = getattr(f, "execution", None)
                    ct = getattr(f, "contract", None)
                    ts = getattr(f, "time", None)
                    if cr is not None:
                        # print("here #2")
                        items.append((cr, ex, ct, order, ts))
                continue

            # Fill(...) like object
            # if hasattr(a, "commissionReport") and (hasattr(a, "execution") or hasattr(a, "contract")):
            #     print("here #3")
            #     items.append((getattr(a, "commissionReport", None),
            #                   getattr(a, "execution", None),
            #                   getattr(a, "contract", None),
            #                   None))
            #     continue

        if not items:
            print("### ERROR | #1 Parsing Commission: ", args)
            # Nothing parseable—just bail quietly
            return

        with _connect(self.db_path) as con:
            for cr, ex, ct, od, ts in items:
                if cr is None or ex is None or ct is None or ts is None:
                    print("### ERROR | #2 Parsing Commission: ", args)
                    continue

                ts_iso = ts.isoformat()
                exec_id = getattr(cr, "execId", None)
                commission = getattr(cr, "commission", None)
                currency = getattr(cr, "currency", None)
                realized = getattr(cr, "realizedPNL", getattr(cr, "realizedPnl", None))

                # enrich from siblings if present
                order_id = getattr(ex, "orderId", None) if ex is not None else None
                symbol = getattr(ct, "symbol", None) if ct is not None else None

                # UPSERT by exec_id (treat as source of truth)
                con.execute("""
                    INSERT INTO commissions (exec_id, ts, commission, currency, realized_pnl, order_id, symbol)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(exec_id) DO UPDATE SET
                        ts            = excluded.ts,
                        commission    = excluded.commission,
                        currency      = excluded.currency,
                        realized_pnl  = excluded.realized_pnl,
                        order_id      = COALESCE(commissions.order_id, excluded.order_id),
                        symbol        = COALESCE(commissions.symbol,   excluded.symbol)
                """, (exec_id, ts_iso, commission, currency, realized, order_id, symbol))

                print(
                    f"### {ts_iso} | COMMISSION | {symbol} {currency} | commission:{commission} realized:{realized} | {order_id} {exec_id}")

    # ---------- low-level insert ----------

    def _insert_order_row(
            self,
            *,
            ts: Optional[str],
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
        # if hasattr(self.ib, "commissionReportEvent"):
        #     try:
        #         self.ib.commissionReportEvent -= self._on_commission_report
        #     except Exception:
        #         pass
