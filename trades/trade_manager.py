import csv
import logging
from datetime import datetime

from trades.trade import Trade


class TradeHandler:
    def __init__(self, ib_connection):
        self.ib_connection = ib_connection
        self.trades = []
        self.executions = []
        logging.basicConfig(level=logging.INFO)

    def create_events(self):
        self.ib_connection.ib.execDetailsEvent += self.process_trade
        # self.ib_connection.ib.commissionReportEvent += self.process_commission

    def load_trades(self):
        print(f"### Load Trades S ###")
        start_time = datetime.now()
        # todo - handle orderStats
        # todo sort?
        tdy = datetime.today().strftime('%Y.%m.%d')

        self.trades = []
        with open('data/trades/uat/' + tdy + '.csv', newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                self.trades.append(Trade.from_data(row))

        trades_ib = []
        for trade in self.ib_connection.ib.trades():
            for fill in trade.fills:
                trades_ib.append(Trade.from_ib(fill))

        # todo fix ts
        set1 = set(
            (x.ts, x.execId, x.symbol, x.side, x.shares, x.price, x.commission, x.realizedPNL) for x in self.trades)
        difference = [x for x in trades_ib if
                      (x.ts, x.execId, x.symbol, x.side, x.shares, x.price, x.commission, x.realizedPNL) not in set1]
        print(f'difference: {difference}')

        # set1 = set((x.ts, x.execId, x.symbol, x.side, x.shares, x.price, x.commission, x.realizedPNL) for x in self.trades)
        # difference = [x for x in trades_ib if (x.ts, x.execId, x.symbol, x.side, x.shares, x.price, x.commission, x.realizedPNL) not in set1]
        # set1 = set(self.trades)
        # print(set1)
        # set2 = set(trades_ib)
        # print(set2)
        # # todo exit program
        # if len(list(set1 - set2)) > 0:
        #     print(f'Algo Missing Trade: ')
        #     for t in list(set1 - set2):
        #         print(t)
        # if len(set2 - set1) > 0:
        #     print(f'Data Missing Trade: ')
        #     for t in list(set2 - set1):
        #         print(t)

        self.log_trades()
        print(f"### Load Trades E | {(datetime.now() - start_time).total_seconds()}s ###")

    def save_trades(self):
        if len(self.trades) > 0:
            tdy = datetime.today().strftime('%Y.%m.%d')
            with open('data/trades/uat/' + tdy + '.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                print(self.trades[0].__dict__.keys())
                writer.writerow(self.trades[0].__dict__.keys())
                for trade in self.trades:
                    writer.writerow(trade.to_csv())

    def log_trades(self):
        for trade in self.trades:
            trade.log()

    def process_trade(self, reqId, execution):
        # print(f"### Process execution: {execution} ")
        self.executions.append(execution)

    def process_commission(self, trade, fill, report):
        print(f"### Process commission S ###")
        print(f"### Trade: {trade} ###")
        print(f"### Fill: {fill} ###")
        print(f"### Report: {report} ###")
        print(f"### Process commission E ###")
