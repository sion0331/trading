# trade_handler.py
import logging


class TradeHandler:
    def __init__(self, ib_connection):
        self.ib_connection = ib_connection
        self.trades = []
        logging.basicConfig(level=logging.INFO)

    def create_events(self):
        self.ib_connection.ib.execDetailsEvent += self.on_trade

    def on_trade(self, trades):
        for trade in trades:
            if trade is not None:
                self.process_trade(trade)

    def process_trade(self, trade):
        logging.info(f"Processing trade: {trade}")
        self.trades.append(trade)
