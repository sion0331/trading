import logging

from position.portfolio import Portfolio
from position.position import Position


class PositionHandler:
    def __init__(self, ib_connection):
        self.ib_connection = ib_connection
        self.positions = {}
        self.portfolios = {}
        logging.basicConfig(level=logging.INFO)

    def create_events(self):
        self.ib_connection.ib.positionEvent += self.process_position
        self.ib_connection.ib.updatePortfolioEvent += self.process_portfolio

    def process_position(self, position):
        if position.contract.symbol not in self.positions.keys():
            self.positions[position.contract.symbol] = Position(position)
        else:
            self.positions[position.contract.symbol].update_position(position)

    def process_portfolio(self, portfolio):
        if portfolio.contract.symbol not in self.portfolios.keys():
            self.portfolios[portfolio.contract.symbol] = Portfolio(portfolio)
        else:
            self.portfolios[portfolio.contract.symbol].update_portfolio(portfolio)

    def log_portfolio(self):
        print(f"### Log Position #")
        for key, value in self.positions.items(): value.log()
        print(f"### Log Portfolio #")
        for key, value in self.portfolios.items(): value.log()
