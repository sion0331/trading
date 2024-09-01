class Position:
    def __init__(self, position):
        self.symbol = position.contract.symbol
        self.size = position.position
        self.price = position.avgCost
        print(f"### New Position: {self.symbol} | {self.size} @ {self.price}")
        # ts??

    def update_position(self, position):
        print(f"### Updating Position Before: {self.symbol} | {self.size} @ {self.price}")
        self.size = position.position
        self.price = position.avgCost
        print(f"### Updating Position After: {self.symbol} | {self.size} @ {self.price}")

    def log(self):
        print(f'{self.symbol} | {self.size} @ {self.price}')
