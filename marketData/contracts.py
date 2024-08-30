from ib_async import Stock, Forex

# Define Stock contracts
NFLX = Stock('NFLX', 'SMART', 'USD')
AAPL = Stock('AAPL', 'SMART', 'USD')

# Define Forex contracts
EURUSD = Forex('EURUSD')
USDJPY = Forex('USDJPY')
