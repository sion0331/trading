from marketData.data_types import Tob, Tape
from datetime import datetime, timezone

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
    def __init__(self, ib_conn):
        self.ib = ib_conn.ib
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

            # Emit QuoteMsg if we have a real BBO
            if msg.bid is not None and msg.ask is not None:
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

            # Emit TradeMsg if we have a last trade
            if msg.last is not None:
                tape = Tape(
                    symbol=msg.contract.symbol,
                    contract=msg.contract,
                    price=msg.last,
                    size=getattr(msg, "lastSize", None),
                    ts=(msg.time or datetime.now(timezone.utc))
                )
                for callback in self.subscribers:
                    callback(tape)

            #
            #
            # if ticker.bid and ticker.ask and ticker.bidSize > 0 and ticker.askSize > 0:
            #     tob = Tob(ticker.contract.symbol, ticker.bid, ticker.ask, ticker.bidSize, ticker.askSize, ticker.time)
            #     for callback in self.subscribers:
            #         callback(tob)
            # else:
            #     print(f'ERROR | {ticker.contract.symbol} | {ticker.bidSize} {ticker.bid} <> {ticker.ask} {ticker.askSize}')
