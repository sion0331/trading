from ib_insync import *

ib = IB()
ib.connect(host='127.0.0.1', port=7497, clientId=1)

nflx_contract = Stock('NFLX', 'SMART', 'USD')
ib.qualifyContracts(nflx_contract)

data = ib.reqMktData(nflx_contract)
data.marketPrice()

eur_usd_contract = Forex('EURUSD', 'IDEALPRO')
ib.qualifyContracts(eur_usd_contract)

# btc_fut_contract = Future('BRR', '20201224', 'CMECRYPTO')
# ib.qualifyContracts(btc_fut_contract)

historical_data_nflx = ib.reqHistoricalData(
    nflx_contract,
    '',
    barSizeSetting='15 mins',
    durationStr='2 D',
    whatToShow='MIDPOINT',
    useRTH=True
    )

print(historical_data_nflx)
# historical_data_nflx[-1]