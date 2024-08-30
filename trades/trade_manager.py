# trade_handler.py
import logging


class TradeHandler:
    def __init__(self):
        self.trades = []
        logging.basicConfig(level=logging.INFO)

    def on_trade(self, trades):
        for trade in trades:
            if trade is not None:
                self.process_trade(trade)

    def process_trade(self, trade):
        logging.info(f"Processing trade: {trade}")
        self.trades.append(trade)
        # Further processing, such as updating positions or triggering alerts