import config, requests
import alpaca_trade_api as tradeapi
import datetime
#from Stock_Trader import *
import time
import pandas as pd
from stock_indicators.indicators.common import Quote
from stock_indicators import indicators
from alpaca_trade_api.common import URL
from alpaca_trade_api.stream import Stream

#create the connection to the api
api = tradeapi.REST(config.API_KEY, config.SECRET_KEY, config.BASE_URL)
account = api.get_account()

#variables for historical data
symbols = ["BTCUSD"]
timeframe = "5Min"
# start = (datetime.datetime.now()+ datetime.timedelta(days=-1)).strftime("%Y-%m-%d") #"2022-08-12"
# end = datetime.datetime.now().strftime("%Y-%m-%d") #"2022-08-13"

#global variables for stream
#open, high, low, close = 0,0,0,0
temp = 0
interval = 5
prev_vol = 0


#grab historical data
def get_hist(symbol):
    start, end = get_time()
    btc_bars = api.get_crypto_bars(symbol, timeframe, start, end).df
    btc_bars = btc_bars.reset_index(drop=False)
    btc_bars = btc_bars.drop(btc_bars[btc_bars.exchange != "FTXU"].index)
    btc_bars = btc_bars.reset_index(drop=True)
    btc_bars = btc_bars.drop(columns=["vwap", "trade_count", "exchange"])
    btc_bars = btc_bars.rename(columns={"timestamp": "date"})
    #btc_bars[["open", "high", "low", "close"]] = btc_bars[["open", "high", "low", "close"]].apply(lambda x: round(x, 2))
    return btc_bars


#gets the current day and previous day for indicators
def get_time():
    start = (datetime.datetime.now()+ datetime.timedelta(days=-1)).strftime("%Y-%m-%d")
    end = datetime.datetime.now().strftime("%Y-%m-%d")
    return start, end


#converts the df format into the quote format to create indicators
def convert_bars(btc_bars):
    quotes_list = [
    Quote(d,o,h,l,c,v) 
    for d,o,h,l,c,v 
    in zip(btc_bars['date'].apply(lambda x: datetime.datetime.strptime(str(x)[0:19], "%Y-%m-%d %H:%M:%S")), btc_bars['open'], btc_bars['high'], btc_bars['low'], btc_bars['close'], btc_bars['volume'])
    ]
    return quotes_list


#generate indicators
def get_indicators(quotes_list):
    heikin_ashi = indicators.get_heikin_ashi(quotes_list)

    #create new quote list for heikin_ashi bars
    h_a_quotes = []
    for i in heikin_ashi:
        h_a_quotes.append(Quote(i.date, i.open, i.high, i.low, i.close, i.volume))

    ema_50 = indicators.get_ema(h_a_quotes, 50)
    ema_200 = indicators.get_ema(h_a_quotes, 200)
    adx = indicators.get_adx(h_a_quotes, 14)

    return {"date": adx[-1].date, "ema_50": ema_50[-1].ema, "ema_200": ema_200[-1].ema, "open": heikin_ashi[-1].open, "high": heikin_ashi[-1].high, "low": heikin_ashi[-1].low, "close": heikin_ashi[-1].close, "adx": adx[-1].adx, "volume": heikin_ashi[-1].volume}


#Buy crypto with x amount of dollars
def trade_buy(dollars, symbol):
    api.submit_order(
        symbol=symbol,
        notional= dollars,
        side="buy",
        type='market',
        time_in_force='gtc',
    )
    print("Bought ${} of {}!\n".format(dollars, symbol))

#Sell crypto of x qty
def trade_sell(unrealized_pl, symbol):
    api.close_position(symbols[0])
    print("Sold ${} of {}!\n".format(unrealized_pl, symbol))

#at the start of a 5 minute bar
async def bar_callback(bar):
    if(bar.exchange == "CBSE"):
        return
    global temp, interval
    temp += 1
    if(temp % interval == 0): #
        temp = 0
        btc_bars = get_hist(bar.symbol)
        print(btc_bars)
        quotes_list = convert_bars(btc_bars)
        ind = get_indicators(quotes_list)
        # ind = get_indicators(convert_bars(get_hist()))
        print_bars(ind, bar.symbol)

#print and buy
def print_bars(stuff, symbol):
    global prev_vol
    direction = ""
    if(stuff["open"] == stuff["low"]):
        direction = "green"
    elif(stuff["open"] == stuff["high"]):
        direction = "red"
    elif(stuff["open"] < stuff["close"]):
        direction = "green doji"
    else:
        direction = "red doji"
    
    #holding crypto
    if(len(api.list_positions()) > 0):
        #sell
        if(direction == "red"):
            trade_sell(api.list_positions()[0].unrealized_pl, symbol)
    #buy
    #rule:
    #ADX above 25
    #EMA_50 above EMA_200
    #Bullish candle
    #or.......
    #ADX above 25
    #Volume increase from previous tick
    #Bullish candle
    elif(stuff["adx"] >= 25 and direction == "green" and stuff["ema_50"] > stuff["ema_200"]):
        trade_buy(api.get_account().equity, symbol)
    #otherwise hold
    else:
        print("Hold!!!!!!!!!!!")
    

    #used for debugging
    print("quote: {} - {}\nopen: {}, high: {}, low: {}, close: {}\ndirection: {}\nema_50: {}, ema_200: {}\nadx: {}\nvolume: {}\n".format(
        symbol, stuff["date"], round(stuff["open"],2), round(stuff["high"],2), round(stuff["low"],2), round(stuff["close"],2), direction, round(stuff["ema_50"],2), round(stuff["ema_200"],2), round(stuff["adx"],2), round(stuff["volume"],2)))

    #reset previous volume
    prev_vol = stuff["volume"]


# Initiate Class Instance
stream = Stream(config.API_KEY,
                config.SECRET_KEY,
                base_url=config.BASE_URL,
                data_feed='iex')  # <- replace to 'sip' if you have PRO subscription

# subscribing to event
for i in symbols:
    stream.subscribe_crypto_bars(bar_callback, i)

stream.run()