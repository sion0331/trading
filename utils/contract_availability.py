# entitlement_probe.py
import pandas as pd
from ib_insync import *


def mk_contracts():
    return {
        "EURUSD@IDEALPRO": Forex('EURUSD'),
        "BTCUSD@PAXOS": Crypto('BTC', 'ZEROHASH'),
        "AAPL@SMART": Stock('AAPL', 'SMART', 'USD'),
        "JPM@SMART": Stock('JPM', 'SMART', 'USD'),
        "SPY@ARCA": Stock('SPY', 'ARCA', 'USD'),
    }


MDT_MAP = {1: "REALTIME", 2: "FROZEN", 3: "DELAYED", 4: "DELAYED_FROZEN"}


def probe(host='127.0.0.1', port=7497, clientId=99, timeout=2.0):
    ib = IB()
    ib.connect(host, port, clientId=clientId)

    results = []
    contracts = mk_contracts()

    # First try real-time
    ib.reqMarketDataType(1)  # 1=real-time
    for name, c in contracts.items():
        ib.qualifyContracts(c)
        t = ib.reqMktData(c, '', False, False)
        ib.sleep(timeout)
        results.append({
            'symbol': name,
            'secType': c.secType,
            'exchange': c.exchange,
            'marketDataType': MDT_MAP.get(getattr(t, 'marketDataType', None), 'UNKNOWN'),
            'bid': t.bid, 'ask': t.ask, 'last': t.last,
            'note': ''
        })
        ib.cancelMktData(c)

    # If something isn’t realtime, optionally try delayed streaming
    ib.reqMarketDataType(4)  # 3=delayed streaming (if enabled on your account)
    for row in results:
        if row['marketDataType'] != 'REALTIME':
            c = contracts[row['symbol']]
            t = ib.reqMktData(c, '', False, False)
            ib.sleep(timeout)
            if getattr(t, 'marketDataType', None) in (3, 4) or any([t.last, t.bid, t.ask]):
                row['marketDataType'] = MDT_MAP.get(t.marketDataType, row['marketDataType'])
                row['note'] = 'delayed available'
            else:
                row['note'] = 'no delayed data'
            ib.cancelMktData(c)

    ib.disconnect()
    df = pd.DataFrame(results,
                      columns=['symbol', 'secType', 'exchange', 'marketDataType', 'bid', 'ask', 'last', 'note'])
    return df


if __name__ == '__main__':
    df = probe()
    print(df.to_string(index=False))
