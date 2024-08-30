from marketData.data_types import Tob


class MarketDataHandler:
    def __init__(self, ib_connection):
        self.ib_connection = ib_connection
        self.ticker_data = []

    def create_events(self):
        self.ib_connection.ib.pendingTickersEvent += self.on_tick

    def on_tick(self, tickers):
        for ticker in tickers:
            if ticker is not None:
                tob = Tob(ticker.bid, ticker.ask, ticker.bidSize, ticker.askSize, ticker.time)
                self.ticker_data.append(ticker)
                tob.log()
