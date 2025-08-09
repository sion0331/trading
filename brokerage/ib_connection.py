import logging

from ib_async import IB

logging.basicConfig(level=logging.INFO)


class IBConnection:
    def __init__(self, host, port, client_id):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id

    def connect(self):
        try:
            print(f'### Connecting IB # HOST: {self.host} | PORT: {self.port} | clientID: {self.client_id}')
            self.ib.connect(self.host, self.port, self.client_id)
            self.ib.reqMarketDataType(4)  # 1 for live, 4 for delayed
        except Exception as e:
            print(f"Failed to connect: {e}")

    def subscribe_market_data(self, contract):
        self.ib.reqMktData(contract, '', False, False)

    def run(self):
        self.ib.run()