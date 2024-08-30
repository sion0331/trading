class Tob:
    def __init__(self, bid, ask, bidSize, askSize, ts):
        self.bid = bid
        self.ask = ask
        self.bidSize = bidSize
        self.askSize = askSize
        self.ts = ts

    def log(self):
        print(f'{self.bidSize} {self.bid} <> {self.ask} {self.askSize} | {self.ts}')
