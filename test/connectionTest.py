from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.ticktype import TickTypeEnum

import pandas

import threading
import time

class IBapi(EWrapper, EClient):
     def __init__(self):
         EClient.__init__(self, self)
         self.data = []  # Initialize variable to store candle

     def historicalData(self, reqId, bar):
         print(f'Time: {bar.date} Close: {bar.close}')
         self.data.append([bar.date, bar.close])

     # def tickPrice(self, reqId, tickType, price, attrib):
     #     if tickType == 2 and reqId == 1:
     #         print('The current ask price is: ', price)
     #     if tickType == 1 and reqId == 1:
     #         print('The current bid price is: ', price)
     #     if tickType == 2 and reqId == 2:
     #         print('The current ask price is: ', price)
     #     if tickType == 1 and reqId == 2:
     #         print('The current bid price is: ', price)
     #     if tickType == 2 and reqId == 3:
     #         print('The current ask price is: ', price)
     #     if tickType == 1 and reqId == 3:
     #         print('The current bid price is: ', price)
def run_loop():
	app.run()

# for i in range(91):
# 	print(TickTypeEnum.to_str(i), i)

app = IBapi()
app.connect('127.0.0.1', 7497, 123)

#Start the socket in a thread
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

time.sleep(1) #Sleep interval to allow time for connection to server

#Create contract object
apple_contract = Contract()
apple_contract.symbol = 'AAPL'
apple_contract.secType = 'STK'
apple_contract.exchange = 'SMART'
apple_contract.currency = 'USD'
# app.reqMktData(1, apple_contract, '', False, False, [])

#Create contract object
eurusd_contract = Contract()
eurusd_contract.symbol = 'EUR'
eurusd_contract.secType = 'CASH'
eurusd_contract.exchange = 'IDEALPRO'
eurusd_contract.currency = 'USD'
# app.reqMktData(2, eurusd_contract, '', False, False, [])

# Create contract object
BTC_futures__contract = Contract()
BTC_futures__contract.symbol = 'BRR'
BTC_futures__contract.secType = 'FUT'
BTC_futures__contract.exchange = 'CMECRYPTO'
BTC_futures__contract.lastTradeDateOrContractMonth  = '202408'
# app.reqMktData(3, BTC_futures__contract, '', False, False, [])

XAUUSD_contract = Contract()
XAUUSD_contract.symbol = 'XAUUSD'
XAUUSD_contract.secType = 'CMDTY'
XAUUSD_contract.exchange = 'SMART'
XAUUSD_contract.currency = 'USD'
# app.reqMktData(3, XAUUSD_contract, '', False, False, [])

# historical
app.reqHistoricalData(1, eurusd_contract, '', '2 D', '1 hour', 'BID', 0, 2, False, [])


time.sleep(2)

df = pandas.DataFrame(app.data, columns=['DateTime', 'Close'])
df['DateTime'] = pandas.to_datetime(df['DateTime'],unit='s')
# df.to_csv('EURUSD_Hourly.csv')

print(df)

df['20SMA'] = df['Close'].rolling(20).mean()
print(df.tail(10))

total = 0
for i in app.data[-20:]:
    total += float(i[1])

print('20SMA =', round(total/20, 5))

time.sleep(10) #Sleep interval to allow time for incoming price data
app.disconnect()