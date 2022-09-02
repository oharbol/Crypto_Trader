# import json
import config, requests
# import alpaca_trade_api as tradeapi
# from alpaca_trade_api.stream import Stream
# from datetime import datetime
# #from Stock_Trader import *
# import time
# import pandas as pd
# from stock_indicators.indicators.common import Quote
# from stock_indicators import indicators


# #get testing data
# # bar_url = 'https://data.alpaca.markets/v2/stocks/{}/bars?start={}&end={}&timeframe=1Min'.format("AAPL",ctime_start,ctime_end)
# # r = requests.get(bar_url, headers=config.HEADERS)

# # print("Hello")

# api = tradeapi.REST(config.API_KEY, config.SECRET_KEY, config.BASE_URL)
# account = api.get_account()
# #print(account)

# # clock = api.get_clock()
# # t = clock.timestamp
# # print(t)
# # print(type(t.to_pydatetime()))


# symbol = "BTCUSD"
# timeframe = "5Min"
# start = "2022-08-12"
# end = "2022-08-13"
# # start = "2022-08-13T18:07:04.451420928-04:00"
# # end =   "2022-08-13T18:07:04.451420928-04:00"

# btc_bars = api.get_crypto_bars(symbol, timeframe, start, end).df
# #btc_bars['new_col'] = range(1, len(btc_bars) + 1)
# #btc_bars.set_index("new_col")
# btc_bars = btc_bars.reset_index(drop=False)
# btc_bars = btc_bars.drop(btc_bars[btc_bars.exchange != "FTXU"].index)
# #print(btc_bars[btc_bars.exchange == "FTXU"].index)
# btc_bars = btc_bars.reset_index(drop=True)
# btc_bars = btc_bars.drop(columns=["vwap", "trade_count", "exchange"])
# btc_bars = btc_bars.rename(columns={"timestamp": "date"})
# print(btc_bars)
# # print(btc_bars.loc[0])
# #btc_bars.to_csv("out.csv", index=False)

# #print(btc_bars.iloc[0].date)

# # t = btc_bars.iloc[0].date
# # print(t)
# # t = t.to_pydatetime()
# # print(type(t))
# # btc_bars.at[0,"date"] = t
# # print(type(btc_bars.iloc[0].date))
# #datetime.fromtimestamp(timestamp)

# quotes_list = [
#     Quote(d,o,h,l,c,v) 
#     for d,o,h,l,c,v 
#     in zip(btc_bars['date'].apply(lambda x: datetime.strptime(str(x)[0:19], "%Y-%m-%d %H:%M:%S")), btc_bars['open'], btc_bars['high'], btc_bars['low'], btc_bars['close'], btc_bars['volume'])
# ]


# # calculate 20-period SMA
# ema_50 = indicators.get_ema(quotes_list, 50)
# ema_200 = indicators.get_ema(quotes_list, 200)
# heikin_ashi = indicators.get_heikin_ashi(quotes_list)
# adx = indicators.get_adx(quotes_list, 14)

# #time.sleep(10)

# print("date: {}, open: {}, high:{}, low:{}, close:{}".format(heikin_ashi[-1].date, round(heikin_ashi[-1].open, 2), round(heikin_ashi[-1].high, 2), round(heikin_ashi[-1].low, 2), round(heikin_ashi[-1].close, 2)))

# from alpaca_trade_api.common import URL
# from alpaca_trade_api.stream import Stream

# open, high, low, close = 0,0,0,0
# temp = 0
# interval = 5

# async def bar_callback(bar):
#     if(bar.exchange == "CBSE"):
#         return
#     global temp, open, high, low, close, interval
#     temp += 1



#     #print(bar)
#     # if(temp == 1):
#     #     open = bar.open
#     #     high = bar.high
#     #     low = bar.low
    
#     # else:
#     #     high = bar.high if bar.high > high else high
#     #     low = bar.low if bar.low < low else low
#     #     if(temp % 5 == 0):
#     #         temp = 0
#     #         close = bar.close
#     #         print_bars({"time": datetime.fromtimestamp(int(bar.timestamp)[0:10]), "open": open, "high": high, "low": low, "close": close})
 
# def print_bars(stuff):
#     direction = ""
#     if(stuff["open"] == stuff["low"]):
#         direction = "green"
#     elif(stuff["open"] == stuff["high"]):
#         direction = "red"
#     elif(stuff["open"] - stuff["close"] < 0):
#         direction = "green doji"
#     else:
#         direction = "red doji"
#     stuff["direction"] = direction
#     print("quote", stuff)


# # Initiate Class Instance
# stream = Stream(config.API_KEY,
#                 config.SECRET_KEY,
#                 base_url=config.BASE_URL,
#                 data_feed='iex')  # <- replace to 'sip' if you have PRO subscription

# # subscribing to event
# stream.subscribe_crypto_bars(bar_callback, 'BTCUSD')

# #stream.run()

# #print(datetime.fromtimestamp(1660414320))

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# no keys required for crypto data
client = CryptoHistoricalDataClient()
t = TimeFrame(5, TimeFrameUnit.Minute)
request_params = CryptoBarsRequest(
                        symbol_or_symbols=["BTC/USD"],
                        timeframe=t,
                        start="2022-08-01"
                 )

bars = client.get_crypto_bars(request_params)

#print(bars.df.iloc[-1])
#print(bars.df.head)


trading_client = TradingClient(config.API_KEY, config.SECRET_KEY)

pos = trading_client.get_all_positions()
#trading_client.close_position("ETHUSD")
# preparing order data
market_order_data = MarketOrderRequest(
                      symbol="ETHUSD",
                      notional=100,
                      side=OrderSide.BUY,
                      time_in_force=TimeInForce.GTC
                  )

# # Market order
market_order = trading_client.submit_order(
                order_data=market_order_data
                )
trading_client.get_account().non_marginable_buying_power
print(trading_client.get_open_position("ETHUSD"))