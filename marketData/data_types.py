from dataclasses import dataclass
from datetime import datetime


@dataclass
class Tob:
    def __init__(self, symbol, bid, ask, bidSize, askSize, ts):
        self.symbol = symbol
        self.bid = bid
        self.ask = ask
        self.bidSize = bidSize
        self.askSize = askSize
        self.ts = ts

    def mid(self):
        return 0.5 * (self.bid + self.ask)

    def spread(self):
        return self.ask - self.bid

    def log(self):
        print(
            f'{self.symbol} | {self.mid()} | {self.bidSize} {self.bid} <{self.spread()}> {self.ask} {self.askSize} | {self.ts}')


@dataclass
class Tape:
    symbol: str
    contract: object
    price: float
    size: float | None
    ts: datetime
