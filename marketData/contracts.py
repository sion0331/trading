from ib_async import Stock, Forex, Crypto

# Define Stock contracts
NFLX = Stock('NFLX', 'SMART', 'USD')
AAPL = Stock('AAPL', 'SMART', 'USD')
GOOG = Stock('GOOG', 'SMART', 'USD')

# Define Forex contracts
EURUSD = Forex('EURUSD')
USDJPY = Forex('USDJPY')
GBPUSD = Forex('GBPUSD')

# Define Bitcoin contracts
BTC = Crypto('BTC', 'PAXOS', 'USD')
ETH = Crypto('ETH', 'PAXOS', 'USD')
