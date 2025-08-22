from datetime import datetime, timezone

from utils.utils_cmp import is_equal


class Position:
    def __init__(self, position):
        self.symbol = position.contract.symbol
        self.size = position.position
        self.price = position.avgCost
        print(f"### New Position: {self.symbol} | {self.size} @ {self.price}")
        # ts??

    def update_position(self, position):
        if not is_equal(self.size, position.position) or not is_equal(self.price, position.avgCost):
            print(
                f"### {datetime.now(timezone.utc).isoformat()} | POSITION | {self.symbol} {self.size} -> {position.position} @ {self.price} -> {position.avgCost}")
            self.size = position.position
            self.price = position.avgCost

    def log(self):
        print(f'{self.symbol} | {self.size} @ {self.price}')
