from datetime import timezone

from database.db import get_conn, init_schema
from database.writer import DBWriter
from marketData.data_types import Tob, Tape
from utils.utils_cmp import is_pos, is_equal


class MarketDataHandler:
    def __init__(self, ib_conn, db_path):
        self.ib = ib_conn.ib
        self.conn = get_conn(db_path)
        init_schema(self.conn)
        self.writer = DBWriter(self.conn)

        self.subscribed_contracts = []
        self.subscribers = []

        self.last = {}
        self.price_eps = 1e-6
        self.size_eps = 1e-6

    def create_events(self):
        self.ib.pendingTickersEvent += self.handle_msg

    def subscribe(self, contract):
        if contract not in self.subscribed_contracts:
            self.ib.reqMktData(contract)
            self.subscribed_contracts.append(contract)

    def add_callback(self, callback_fn):
        self.subscribers.append(callback_fn)

    def handle_msg(self, msgs):
        for msg in msgs:
            sym = msg.contract.symbol
            ts = msg.time.astimezone(timezone.utc).isoformat()

            bid = getattr(msg, "bid", None)
            ask = getattr(msg, "ask", None)
            bidSize = getattr(msg, "bidSize", None)
            askSize = getattr(msg, "askSize", None)
            last = getattr(msg, "last", None)
            lastSize = getattr(msg, "lastSize", None)

            prev = self.last.setdefault(sym, {})

            tob_changed = False
            if is_pos(bid) and is_pos(bidSize) and is_pos(ask) and is_pos(askSize):
                tob_changed = (
                        not is_equal(bid, prev.get("bid"), self.price_eps) or
                        not is_equal(ask, prev.get("ask"), self.price_eps) or
                        not is_equal(bidSize, prev.get("bidSize"), self.size_eps) or
                        not is_equal(askSize, prev.get("askSize"), self.size_eps)
                )
                if tob_changed:
                    tob = Tob(msg.contract.symbol, msg.bid, msg.ask, msg.bidSize, msg.askSize, ts)
                    prev.update({"bid": bid, "ask": ask, "bidSize": bidSize, "askSize": askSize})

                    for callback in self.subscribers:
                        callback(tob)

                    row = (tob.symbol, ts, tob.bid, tob.ask, tob.bidSize, tob.askSize)
                    self.writer.insert_tob_many([row])

            ### TODO - does not capture subsequent trades with same price & size
            tape_changed = False
            if is_pos(last):  # and is_pos(lastSize):
                tape_changed = (
                        not is_equal(last, prev.get("last"), self.price_eps) or
                        not is_equal(lastSize, prev.get("lastSize"), self.size_eps)
                )
                if tape_changed:
                    tape = Tape(
                        symbol=msg.contract.symbol,
                        price=msg.last,
                        size=getattr(msg, "lastSize", None),
                        ts=ts
                    )
                    prev.update({"last": last, "lastSize": lastSize})

                    for callback in self.subscribers:
                        callback(tape)

                    row = (tape.symbol, ts, tape.price, tape.size)
                    self.writer.insert_tape_many([row])

            if not tape_changed and not tob_changed:
                # print("### Unknown Msg: ", msg)
                continue
