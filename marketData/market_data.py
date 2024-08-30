from marketData.data_types import Tob


class MarketDataHandler:
    def __init__(self):
        self.ticker_data = []

    @staticmethod
    def on_tick(tickers):
        for ticker in tickers:
            if ticker is not None:
                tob = Tob(ticker.bid, ticker.ask, ticker.bidSize, ticker.askSize, ticker.time)
                tob.log()
