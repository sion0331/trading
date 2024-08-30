from ib_async import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

ib.reqMarketDataType(4)  # Use free, delayed, frozen data
contract = Forex('EURUSD')
bars = ib.reqHistoricalData(
    contract, endDateTime='', durationStr='30 D',
    barSizeSetting='1 hour', whatToShow='MIDPOINT', useRTH=True)

# convert to pandas dataframe (pandas needs to be installed):
df = util.df(bars)
print(df)

#####
nflx_contract = Stock('NFLX', 'SMART', 'USD')

historical_data_nflx = ib.reqHistoricalData(
    nflx_contract,
    '',
    barSizeSetting='15 mins',
    durationStr='2 D',
    whatToShow='MIDPOINT',
    useRTH=True
    )
print(util.df(historical_data_nflx))

nflx_order = MarketOrder('SELL', 100)
trade = ib.placeOrder(nflx_contract, nflx_order)
print(trade.log)