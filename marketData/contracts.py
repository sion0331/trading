from ib_async import Stock, Forex, Crypto

# Define Stock contracts
NFLX = Stock('NFLX', 'SMART', 'USD')
AAPL = Stock('AAPL', 'SMART', 'USD')

# Define Forex contracts
EURUSD = Forex('EURUSD')
USDJPY = Forex('USDJPY')

# Define Bitcoin contracts
BTC = Crypto('BTC', 'PAXOS', 'USD')
