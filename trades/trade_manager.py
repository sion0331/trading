# trade_handler.py
import logging


class TradeHandler:
    def __init__(self, ib_connection):
        self.ib_connection = ib_connection
        self.trades = []
        logging.basicConfig(level=logging.DEBUG)

    def create_events(self):
        self.ib_connection.ib.execDetailsEvent += self.process_trade

    def process_trade(self, trade, fill):
        print(f"### Process trade: {trade} ")
        print(f"### Process fill: {fill}")
        self.trades.append(trade)
