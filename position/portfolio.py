class Portfolio:
    def __init__(self, portfolio):
        self.symbol = portfolio.contract.symbol
        self.size = portfolio.position
        self.cost = portfolio.averageCost
        self.price = portfolio.marketValue #todo divide
        self.realized = portfolio.realizedPNL
        self.unrealized = portfolio.unrealizedPNL
        print(
            f"### New Portfolio: {self.symbol} | {self.size} @ {self.cost} <> {self.price}| {self.unrealized} {self.realized}")
        # ts??

    def update_portfolio(self, portfolio):
        if self.size != portfolio.position or self.cost != portfolio.averageCost:
            print(
                f"### Updating Position: {self.symbol} | size: {self.size}->{portfolio.position} | cost: {self.cost}->{portfolio.averageCost}")
        if self.price != portfolio.marketValue or self.realized != portfolio.realizedPNL or self.unrealized != portfolio.unrealizedPNL:  # TODO DOUBLE
            print(
                f"### Updating PnL: {self.symbol} | price: {self.price}->{portfolio.marketValue} | realized: {self.realized}->{portfolio.realizedPNL} | unrealized: {self.unrealized}->{portfolio.unrealizedPNL}")
        self.size = portfolio.position
        self.cost = portfolio.averageCost
        self.price = portfolio.marketValue
        self.realized = portfolio.realizedPNL
        self.unrealized = portfolio.unrealizedPNL

    def log(self):
        print(f'{self.symbol} | {self.size} @ {self.cost} <> {self.price}| {self.unrealized} {self.realized}"')
