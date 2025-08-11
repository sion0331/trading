from datetime import datetime, timezone

from database.db import get_conn, init_schema
from database.writer import DBWriter
from marketData.data_types import Tob, Tape


# class MarketDataHandler:
#     def __init__(self, ib_connection):
#         self.ib_connection = ib_connection
#         self.ticker_data = []
#
#     def create_events(self):
#         self.ib_connection.ib.pendingTickersEvent += self.on_tick
#
#     def on_tick(self, tickers):
#         for ticker in tickers:
#             if ticker is not None:
#                 tob = Tob(ticker.bid, ticker.ask, ticker.bidSize, ticker.askSize, ticker.time)
#                 self.ticker_data.append(ticker)
#                 tob.log()


# TODO GPT

class MarketDataHandler:
    def __init__(self, ib_conn, db_path):
        self.ib = ib_conn.ib
        self.conn = get_conn(db_path)
        init_schema(self.conn)
        self.writer = DBWriter(self.conn)

        self.subscribed_contracts = []
        self.subscribers = []

    def subscribe(self, contract):
        if contract not in self.subscribed_contracts:
            self.ib.reqMktData(contract)
            # self.ib.reqMktData(contract, '', False, False)
            self.subscribed_contracts.append(contract)

    def add_subscriber(self, callback_fn):
        self.subscribers.append(callback_fn)

    def handle_msg(self, msgs):
        for msg in msgs:
            print(msg)

            if msg.bid > 0 and msg.ask > 0:
                tob = Tob(msg.contract.symbol, msg.bid, msg.ask, msg.bidSize, msg.askSize, msg.time)
                #
                # q = QuoteMsg(
                #     symbol=(t.contract.symbol + getattr(t.contract, "currency", "")),
                #     contract=t.contract,
                #     bid=t.bid, ask=t.ask,
                #     bidSize=getattr(t, "bidSize", None),
                #     askSize=getattr(t, "askSize", None),
                #     ts=(t.time or datetime.now(timezone.utc))
                # )
                for callback in self.subscribers:
                    callback(tob)

                row = (tob.symbol, str(tob.ts), tob.bid, tob.ask, tob.bidSize, tob.askSize)
                print("inserting: ", row)
                self.writer.insert_tob_many([row])

            if msg.last > 0:
                tape = Tape(
                    symbol=msg.contract.symbol,
                    contract=msg.contract,
                    price=msg.last,
                    size=getattr(msg, "lastSize", None),
                    ts=(msg.time or datetime.now(timezone.utc))
                )
                for callback in self.subscribers:
                    callback(tape)

                self.writer.insert_tape_many([tape])
