import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from database.orders import OrderLogger
from orders.order_fx import create_fx_order


@dataclass
class _OpenLmt:
    order_id: int
    symbol: str
    side: str  # BUY / SELL
    qty: float  # base qty (e.g., EUR)
    lmt_price: float
    status: str  # Submitted / PreSubmitted / ApiPending / PendingSubmit / Inactive
    order_obj: Optional[Any] = None


class OrderHandler:
    def __init__(self, ib_connection):
        self.ib_connection = ib_connection
        logging.basicConfig(level=logging.INFO)
        self.logger = OrderLogger(ib_connection.ib, db_path="./data/db/orders.db")

        self._open_lmts: Dict[str, List[_OpenLmt]] = {}
        self._cancel_events: dict[int, threading.Event] = {}  # orderId -> Event

        self._pending_replace: dict[tuple[str, str], dict] = {}  # key: (symbol, side) -> {"contract":..., "params":...}

    def create_events(self):
        self.ib_connection.ib.orderStatusEvent += self._on_order_status
        self.ib_connection.ib.newOrderEvent += self._on_new_order

        self.ib_connection.ib.openOrderEvent += self._on_open_order
        self.ib_connection.ib.orderModifyEvent += self._on_order_modify
        self.ib_connection.ib.cancelOrderEvent += self._on_order_cancel

    def send_order(self, contract, order):
        logging.info(f"Sending order: {order}")
        self.logger.log_send_intent(contract, order)
        self.ib_connection.ib.placeOrder(contract, order)

    def request_replace_limit(
            self,
            contract,
            side: str,
            *,
            usd_notional: float,
            limit_price: float,
            tif: str = "DAY",
            cancel_opposite: bool = False,
    ):
        """
        Cancel all same-side LMTs and remember the replacement.
        When we later observe all of them Cancelled, we send the new LMT.
        """
        symbol = getattr(contract, "symbol", None)
        if not symbol:
            return

        side_u = side.upper()
        same_side = list(self.open_limits(symbol, side_u))
        # If nothing to cancel, PLACE IMMEDIATELY (initial placement)
        if not same_side:
            order = self.create_order(
                side_u,
                order_type="LMT",
                usd_notional=float(usd_notional),
                limit_price=float(limit_price),
                tif=tif,
            )
            self.send_order(contract, order)
            return

        # remember what to place after cancel completes
        self._pending_replace[(symbol, side.upper())] = {
            "contract": contract,
            "params": {
                "side": side.upper(),
                "order_type": "LMT",
                "usd_notional": float(usd_notional),
                "limit_price": float(limit_price),
                "tif": tif,
            }
        }

        # optionally clean opposite side right away (non-blocking)
        if cancel_opposite:
            self.cancel_all_for(symbol, "BUY" if side.upper() == "SELL" else "SELL")

        # kick off same-side cancels (non-blocking)
        for o in same_side:
            self.cancel_order(o.order_id)
        # for o in list(self.open_limits(symbol, side)):
        #     self.cancel_order(o.order_id)

    def cancel_side_and_wait(self, symbol: str, side: str, timeout_s: float = 2.0) -> bool:
        """Cancel all open LMTs on (symbol, side) and wait until we receive 'Cancelled' (or timeout)."""
        side = side.upper()
        orders = list(self.open_limits(symbol, side))
        if not orders:
            print("### nothing to cancel")
            return True

        events: list[threading.Event] = []
        for o in orders:
            ev = self.cancel_order(o.order_id)
            if ev:
                events.append(ev)

        # Wait for all to signal, up to timeout
        if not events:
            return False
        t_end = time.time() + timeout_s
        for ev in events:
            remaining = t_end - time.time()
            if remaining <= 0:
                return False
            ok = ev.wait(timeout=remaining)
            print("### remaining:", remaining, ok)
            if not ok:
                return False

        print("open limits len: ", len(self.open_limits(symbol, side)))
        return len(self.open_limits(symbol, side)) == 0
        # return True

    def cancel_all_for(self, symbol: str, side: Optional[str] = None):
        for o in self.open_limits(symbol, side):
            self.cancel_order(o.order_id)

    def cancel_order(self, order_id):
        logging.info(f"Cancelling order ID: {order_id}")
        self.logger.log_cancel_intent(order_id)

        o = self._resolve_order_obj(order_id)
        if o is None:
            logging.warning(f"Could not resolve Order object for orderId={order_id}")
            return None

        ev = threading.Event()
        self._cancel_events[order_id] = ev

        print("Cancelling order: ", o)
        self.ib_connection.ib.cancelOrder(o)  # API expects an Order object
        return ev

    def check_order_status(self, order_id):
        order = self.ib_connection.ib.openOrders()
        status = next((o for o in order if o.orderId == order_id), None)
        logging.info(f"Order status: {status}")
        return status

    def create_order(
            self,
            side: str,
            qty: float | None = None,  # base qty (EUR)
            order_type: str = 'MKT',
            *,
            usd_notional: float | None = None,  # preferred for EURUSD
            limit_price: float | None = None,
            ref_price: float | None = None,
            tif: str = "DAY",
    ):
        return create_fx_order(
            side,
            order_type=order_type,
            qty=qty,
            usd_notional=usd_notional,
            ref_price=ref_price,
            limit_price=limit_price,
            tif=tif,
        )

    def open_limits(self, symbol: str, side: Optional[str] = None) -> List[_OpenLmt]:
        items = self._open_lmts.get(symbol, [])
        return [o for o in items if side is None or o.side == side.upper()]

    def pending_notional_usd(self, symbol: str, side: str, ref_price: float) -> float:
        # value resting base qty in USD using the current ref price
        return sum(o.qty * float(ref_price) for o in self.open_limits(symbol, side))

    def best_limit(self, symbol: str, side: str) -> Optional[_OpenLmt]:
        ords = self.open_limits(symbol, side)
        if not ords:
            return None
        if side.upper() == "BUY":
            return max(ords, key=lambda o: o.lmt_price)
        return min(ords, key=lambda o: o.lmt_price)

    def _resolve_order_obj(self, order_id: int):
        """Find an Order object for this orderId, or synthesize a minimal one."""
        # 1) Try from our cache
        for lst in self._open_lmts.values():
            for rec in lst:
                if rec.order_id == order_id and rec.order_obj is not None:
                    return rec.order_obj

        # 2) Try from ib.openOrders()
        try:
            for o in self.ib_connection.ib.openOrders():
                if getattr(o, "orderId", None) == order_id:
                    return o
        except Exception:
            pass

        # 3) Fallback: create a minimal Order with just orderId set
        try:
            from ib_async import Order as _Order
            o = _Order()
            o.orderId = int(order_id)
            return o
        except Exception:
            return None

    def _on_order_status(self, trade):
        """orderStatusEvent -> Trade(...) shape in your logs"""
        # print(f"### Order Status: {trade}")
        try:
            od = getattr(trade, "order", None)
            os = getattr(trade, "orderStatus", None)
            ct = getattr(trade, "contract", None)
            if not (od and os and ct):
                return
            if not hasattr(od, "lmtPrice"):
                # we only track LMTs in this cache
                return

            oid = getattr(od, "orderId", None)
            status_txt = str(getattr(os, "status", ""))

            # keep cache in sync (now PendingCancel stays live, Cancelled drops)
            self._upsert_open_lmt(
                symbol=getattr(ct, "symbol", None),
                order_id=oid,
                side=getattr(od, "action", None),
                qty=getattr(od, "totalQuantity", None),
                lmt_price=getattr(od, "lmtPrice", None),
                status=status_txt,
                order_obj=od,
            )

            # signal any synchronous waiters only on true Cancelled
            if status_txt == "Cancelled" and oid is not None:
                print("_on_order_status: ", status_txt)
                ev = self._cancel_events.pop(oid, None)
                if ev:
                    ev.set()

            # ---- PLACE REPLACEMENT ONLY AFTER Cancelled ----
            if status_txt == "Cancelled":
                symbol = getattr(ct, "symbol", None)
                side = getattr(od, "action", None)
                if symbol and side:
                    key = (symbol, side.upper())
                    # Still waiting on other same-side orders? then return
                    if self.open_limits(symbol, side):
                        return
                    # All same-side cleared, place the queued replacement (if any)
                    payload = self._pending_replace.pop(key, None)
                    if payload:
                        contract = payload["contract"]
                        p = payload["params"]
                        order = self.create_order(
                            p["side"],
                            order_type=p["order_type"],
                            usd_notional=p["usd_notional"],
                            limit_price=p["limit_price"],
                            tif=p["tif"],
                        )
                        self.send_order(contract, order)

        except Exception:
            pass

    def _on_new_order(self, trade):
        """newOrderEvent -> Trade(...) shape"""
        print(f"### New Order: {trade}")
        try:
            od = getattr(trade, "order", None)
            os = getattr(trade, "orderStatus", None)
            ct = getattr(trade, "contract", None)
            if not (od and ct):
                return
            if not hasattr(od, "lmtPrice"):
                return
            self._upsert_open_lmt(
                symbol=getattr(ct, "symbol", None),
                order_id=getattr(od, "orderId", None),
                side=getattr(od, "action", None),
                qty=getattr(od, "totalQuantity", None),
                lmt_price=getattr(od, "lmtPrice", None),
                status=getattr(os, "status", "Submitted") if os else "Submitted",
                order_obj=od,
            )
        except Exception:
            print("ERROR | new_order: ", trade)
            pass

    def _on_open_order(self, *args, **kwargs):
        """
        openOrderEvent often comes as (orderId, contract, order, orderState)
        We ingest it if it's a LMT.
        """
        try:
            if len(args) >= 3:
                orderId, contract, order = args[0], args[1], args[2]
                orderState = args[3] if len(args) > 3 else None
            else:
                orderId = kwargs.get("orderId")
                contract = kwargs.get("contract")
                order = kwargs.get("order")
                orderState = kwargs.get("orderState")

            if not (order and contract):
                return
            if not hasattr(order, "lmtPrice"):
                return
            status = getattr(orderState, "status", "Submitted") if orderState else "Submitted"
            self._upsert_open_lmt(
                symbol=getattr(contract, "symbol", None),
                order_id=orderId or getattr(order, "orderId", None),
                side=getattr(order, "action", None),
                qty=getattr(order, "totalQuantity", None),
                lmt_price=getattr(order, "lmtPrice", None),
                status=status,
            )
        except Exception:
            pass

    def _on_order_modify(self, *args, **kwargs):
        """orderModifyEvent: treat same as open order update if LMT."""
        try:
            order = kwargs.get("order", None)
            contract = kwargs.get("contract", None)
            status = kwargs.get("status", "Submitted")
            if len(args) >= 1 and not order:
                order = args[0]
            if len(args) >= 2 and not contract:
                contract = args[1]
            if len(args) >= 3 and not status:
                status = args[2]

            if order and contract and hasattr(order, "lmtPrice"):
                self._upsert_open_lmt(
                    symbol=getattr(contract, "symbol", None),
                    order_id=getattr(order, "orderId", None),
                    side=getattr(order, "action", None),
                    qty=getattr(order, "totalQuantity", None),
                    lmt_price=getattr(order, "lmtPrice", None),
                    status=status,
                )
        except Exception:
            pass

    def _on_order_cancel(self, *args, **kwargs):
        """cancelOrderEvent: remove id from cache if provided."""
        try:
            # print("_on_order_cancel", args, kwargs)
            pass
            # order_id = kwargs.get("orderId")
            # if order_id is None and len(args) >= 1:
            #     order_id = args[0]
            # if order_id is not None:
            #     self._remove_from_book(int(order_id))
        except Exception:
            pass

    @staticmethod
    def _is_live(status: Optional[str]) -> bool:
        return str(status) in {"Submitted", "PreSubmitted", "ApiPending", "PendingSubmit", "Inactive", "PendingCancel"}

    def _upsert_open_lmt(
            self, *, symbol: str, order_id: int, side: str, qty: float, lmt_price: float, status: str, order_obj=None
    ):
        if not symbol or not order_id:
            return
        book = self._open_lmts.setdefault(symbol, [])
        rec = _OpenLmt(
            order_id=int(order_id),
            symbol=symbol,
            side=str(side).upper(),
            qty=float(qty or 0.0),
            lmt_price=float(lmt_price or 0.0),
            status=str(status or ""),
            order_obj=order_obj,  # <-- keep it
        )
        if not self._is_live(rec.status):
            self._open_lmts[symbol] = [o for o in book if o.order_id != rec.order_id]
            return
        for i, o in enumerate(book):
            if o.order_id == rec.order_id:
                # preserve order_obj if the new event didn’t include it
                if rec.order_obj is None:
                    rec.order_obj = o.order_obj
                book[i] = rec
                break
        else:
            book.append(rec)

    def _remove_from_book(self, order_id: int):
        for sym, lst in list(self._open_lmts.items()):
            self._open_lmts[sym] = [o for o in lst if o.order_id != order_id]
