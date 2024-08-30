from datetime import datetime, timedelta

import pandas as pd
from ib_async import *
import pytz

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

ccy = 'EURUSD'
ib.reqMarketDataType(4)  # Use free, delayed, frozen data
contract = Forex(ccy)

timezone = 'UTC'
sd = datetime(2024, 8, 15, tzinfo=pytz.UTC)
ed = datetime(2024, 8, 15, tzinfo=pytz.UTC)
ed = ed + timedelta(days=1)
sd_str = sd.strftime("%Y%m%d %H:%M:%S") + " " + timezone
ed_str = ed.strftime("%Y%m%d %H:%M:%S") + " " + timezone

ticks_df = pd.DataFrame()
dt = sd
while dt < ed:
    dt_str = dt.strftime("%Y%m%d %H:%M:%S") + " " + timezone
    ticks = util.df(ib.reqHistoricalTicks(contract, startDateTime=dt_str, endDateTime=ed_str
                                          , numberOfTicks=1000, whatToShow='Bid_Ask', useRth=False, ignoreSize=True))

    print(len(ticks), dt, ticks.iloc[-1]['time'], ed)
    ticks_df = pd.concat([ticks_df, ticks], ignore_index=True)
    if len(ticks) < 1000: break
    dt = ticks.iloc[-1]['time']

len(ticks_df)
ticks_df.to_csv('ticks_test.csv')

