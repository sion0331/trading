import logging

from ib_async import IB

logging.basicConfig(level=logging.INFO)


class IBConnection:
    def __init__(self, host, port, client_id, market_data_handler, trade_handler, position_handler):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.market_data_handler = market_data_handler
        self.trader_handler = trade_handler
        self.position_handler = position_handler

    def connect(self):
        try:
            logging.info(f'### Connecting IB # HOST: {self.host} | PORT: {self.port} | clientID: {self.client_id}')
            self.ib.connect(self.host, self.port, self.client_id)
            self.ib.reqMarketDataType(1)  # 1 for live, 4 for delayed
            self.ib.pendingTickersEvent += self.market_data_handler.on_tick
            self.ib.execDetailsEvent += self.trader_handler.on_trade
            self.ib.positionEvent += self.position_handler.process_position
        except Exception as e:
            print(f"Failed to connect: {e}")

    def subscribe_market_data(self, contract):
        self.ib.reqMktData(contract, '', False, False)

    def run(self):
        self.ib.run()
