from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

from backtest.sim_events import Event


# --- light contract & order shells (compatible fields the strategy/handler read) ---
@dataclass
class Contract:
    symbol: str
    exchange: str = "SIM"
    currency: str = "USD"


@dataclass
class Order:
    orderId: int
    clientId: int = 1
    permId: int = 0
    action: str = "BUY"  # BUY/SELL
    totalQuantity: float = 0.0  # base qty (e.g., EUR)
    orderType: str = "MKT"  # "MKT" or "LMT"
    lmtPrice: float = 0.0
    auxPrice: float = 0.0
    tif: str = "DAY"
    transmit: bool = True


@dataclass
class OrderStatus:
    orderId: int
    status: str  # PendingSubmit/Submitted/Cancelled/Filled/...
    filled: float = 0.0
    remaining: float = 0.0
    avgFillPrice: float = 0.0
    permId: int = 0
    parentId: int = 0
    lastFillPrice: float = 0.0
    clientId: int = 1
    whyHeld: str = ""
    mktCapPrice: float = 0.0


@dataclass
class TradeLogEntry:
    time: datetime
    status: str
    message: str = ""
    errorCode: int = 0


@dataclass
class Trade:
    contract: Contract
    order: Order
    orderStatus: OrderStatus
    fills: list = field(default_factory=list)
    log: list = field(default_factory=list)
    advancedError: str = ""

    def isDone(self):
        return self.orderStatus.status in {"Filled", "Cancelled", "Inactive"}


# Execution / Fill shells
@dataclass
class Execution:
    execId: str
    time: datetime
    acctNumber: str = "SIM"
    exchange: str = "SIM"
    side: str = "BOT"  # BOT/SLD
    shares: float = 0.0
    price: float = 0.0
    permId: int = 0
    clientId: int = 1
    orderId: int = 0
    liquidation: int = 0
    cumQty: float = 0.0
    avgPrice: float = 0.0
    orderRef: str = ""
    evRule: str = ""
    evMultiplier: float = 0.0
    modelCode: str = ""
    lastLiquidity: int = 0


@dataclass
class CommissionReport:
    execId: str
    commission: float
    currency: str = "USD"
    realizedPNL: float = 0.0


@dataclass
class Fill:
    contract: Contract
    execution: Execution
    commissionReport: CommissionReport | None
    time: datetime


# --- Simple positions (what your strategy needs) ---
@dataclass
class PositionSimple:
    symbol: str
    size: float = 0.0  # base qty


class BacktestPositions:
    """Expose .positions dict like your PositionHandler does."""

    def __init__(self):
        self.positions: Dict[str, PositionSimple] = {}

    def apply_fill(self, symbol: str, side: str, qty: float):
        sign = 1.0 if side.upper() in {"BUY", "BOT"} else -1.0
        pos = self.positions.setdefault(symbol, PositionSimple(symbol, 0.0))
        pos.size += sign * float(qty)


# --- The SIM IB core ---
class SimIB:
    """
    Minimal broker sim with ib_async-like surface:
      - .newOrderEvent, .orderStatusEvent, .openOrderEvent, .cancelOrderEvent, .execDetailsEvent, .commissionReportEvent
      - .placeOrder(contract, order), .cancelOrder(order)
      - .openOrders()
    Fill logic:
      - MKT fills immediately at touch (BUY@ask, SELL@bid)
      - LMT fills when price crosses/touches (BUY if ask<=lmt, SELL if bid>=lmt)
    """

    def __init__(self):
        # events your code already uses
        self.newOrderEvent = Event("newOrderEvent")
        self.orderStatusEvent = Event("orderStatusEvent")
        self.openOrderEvent = Event("openOrderEvent")
        self.orderModifyEvent = Event("orderModifyEvent")
        self.cancelOrderEvent = Event("cancelOrderEvent")
        self.execDetailsEvent = Event("execDetailsEvent")
        self.commissionReportEvent = Event("commissionReportEvent")

        self._next_id = 1
        self._trades: Dict[int, Trade] = {}  # orderId -> Trade
        self._acct = "DU-SIM"

    # --- API the OrderHandler expects ---
    def placeOrder(self, contract: Contract, order: Order) -> Trade:
        if order.orderId is None or order.orderId == 0:
            order.orderId = self._alloc_id()
        os = OrderStatus(
            orderId=order.orderId, status="PendingSubmit",
            filled=0.0, remaining=order.totalQuantity, avgFillPrice=0.0,
            permId=order.permId or order.orderId, clientId=order.clientId
        )
        trade = Trade(contract=contract, order=order, orderStatus=os, fills=[])
        trade.log.append(TradeLogEntry(datetime.now(timezone.utc), "PendingSubmit"))
        self._trades[order.orderId] = trade

        # emit "new order"
        self.newOrderEvent.emit(trade)

        # consider it "Submitted" instantly in sim
        os.status = "Submitted"
        trade.log.append(TradeLogEntry(datetime.now(timezone.utc), "Submitted"))
        self.orderStatusEvent.emit(trade)
        # openOrderEvent shape in ib_async: (orderId, contract, order, orderState)
        self.openOrderEvent.emit(order.orderId, contract, order, None)
        return trade

    def cancelOrder(self, order: Order):
        oid = getattr(order, "orderId", None)
        tr = self._trades.get(oid)
        if not tr:
            return None
        # pending cancel
        tr.orderStatus.status = "PendingCancel"
        self.orderStatusEvent.emit(tr)
        # immediate cancel in sim
        tr.orderStatus.status = "Cancelled"
        self.orderStatusEvent.emit(tr)
        self.cancelOrderEvent.emit(oid)
        return tr

    def openOrders(self) -> List[Order]:
        out = []
        for tr in self._trades.values():
            if tr.orderStatus.status in {"Submitted", "PreSubmitted", "PendingSubmit", "ApiPending", "Inactive"}:
                out.append(tr.order)
        return out

    # --- Matching / filling called by the replayer each tick ---
    def on_tick(self, ts: datetime, bid: float, ask: float):
        """
        Try to fill any live orders at this tick.
        Simple rules:
          - MKT BUY: fill @ask, MKT SELL: fill @bid
          - LMT BUY: if ask <= lmt -> fill @min(lmt, ask)
            LMT SELL: if bid >= lmt -> fill @max(lmt, bid)
        """
        for tr in list(self._trades.values()):
            os = tr.orderStatus
            od = tr.order
            if os.status not in {"Submitted", "PreSubmitted"}:
                continue

            fill_px = None
            if od.orderType == "MKT":
                fill_px = ask if od.action.upper() == "BUY" else bid
            else:  # LMT
                if od.action.upper() == "BUY" and ask <= od.lmtPrice:
                    fill_px = min(od.lmtPrice, ask)
                elif od.action.upper() == "SELL" and bid >= od.lmtPrice:
                    fill_px = max(od.lmtPrice, bid)

            if fill_px is None:
                continue

            # full fill (you can extend to partials)
            qty = od.totalQuantity
            os.filled = qty
            os.remaining = 0.0
            os.avgFillPrice = fill_px
            os.lastFillPrice = fill_px
            os.status = "Filled"

            # emit execution & optional commission
            exec_id = f"SIM.{od.orderId}.{int(ts.timestamp())}"
            execution = Execution(
                execId=exec_id,
                time=ts,
                acctNumber=self._acct,
                exchange="SIM",
                side="BOT" if od.action.upper() == "BUY" else "SLD",
                shares=qty,
                price=fill_px,
                permId=od.permId or od.orderId,
                clientId=od.clientId,
                orderId=od.orderId,
                cumQty=qty, avgPrice=fill_px, lastLiquidity=1
            )
            # tiny flat commission model (override as you like)
            commission = max(0.0, 0.0000025 * qty * fill_px)  # 0.25 bps notionally
            comm = CommissionReport(execId=exec_id, commission=commission, currency="USD", realizedPNL=0.0)
            fill = Fill(contract=tr.contract, execution=execution, commissionReport=comm, time=ts)
            tr.fills.append(fill)

            # events your logger already records
            self.execDetailsEvent.emit(tr.contract, execution)
            self.commissionReportEvent.emit(comm)
            self.orderStatusEvent.emit(tr)  # to send "Filled"

    # helpers
    def _alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i


# --- tiny connection shell so OrderHandler sees .ib ---
class FakeConnection:
    def __init__(self, ib: SimIB):
        self.ib = ib


# --- subclass OrderHandler only to redirect to a separate backtest DB (optional) ---
from orders.order_manager import OrderHandler as LiveOrderHandler  # your existing handler
from database.orders import OrderLogger


class BacktestOrderHandler(LiveOrderHandler):
    def __init__(self, ib_connection: FakeConnection, orders_db_path: str = "./data/db/orders_backtest.db"):
        # call parent but immediately swap logger to write into a separate DB
        super().__init__(ib_connection)
        self.logger = OrderLogger(ib_connection.ib, db_path=orders_db_path)
        print("[BT] OrderLogger DB:", orders_db_path)
