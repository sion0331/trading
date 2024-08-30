import logging


class PositionHandler:
    def __init__(self, ib_connection):
        self.ib_connection = ib_connection
        self.positions = []
        self.portfolios = []
        logging.basicConfig(level=logging.INFO)

    def create_events(self):
        self.ib_connection.ib.positionEvent += self.process_position
        self.ib_connection.ib.updatePortfolioEvent += self.process_portfolio

    def process_position(self, position):
        print(f"### Process position: {position}")
        self.positions.append(position)

    def process_portfolio(self, portfolio):
        print(f"### Process portfolio: {portfolio}")
        self.portfolios.append(portfolio)