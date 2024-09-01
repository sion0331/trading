# trade_handler.py
import logging


class TradeHandler:
    def __init__(self, ib_connection):
        self.ib_connection = ib_connection
        self.trades = []
        logging.basicConfig(level=logging.DEBUG)

    def create_events(self):
        self.ib_connection.ib.execDetailsEvent += self.process_trade

    def load_trades(self):
        self.ib_connection.ib.reqExecutions()
        # self.ib_connection.ib.execDetails()

        print("load trades")
        # print(self.ib_connection.get_executions())
        # self.trades = self.ib_connection.ib.get_executions()

    def process_trade(self, trade, fill):
        print(f"### Process trade: {trade} ")
        print(f"### Process fill: {fill}")
        self.trades.append(trade)
