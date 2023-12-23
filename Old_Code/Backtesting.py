import config
import alpaca_trade_api as tradeapi
import datetime
import pandas as pd
import time
from stock_indicators.indicators.common import Quote
from stock_indicators import indicators
import csv

#Create the connection to the api
api = tradeapi.REST(config.API_KEY, config.SECRET_KEY, config.BASE_URL)
timeframe = "1Hour"
ticker = "MATICUSD"
graph_bank = 2000
bank = 2000
holding = False
cost = 0
shares = 0
buy_amount = 0
prev_vol = 0


#Get historical data
def get_hist(symbol):
    start, end = get_time()
    bars = api.get_crypto_bars(symbol, timeframe, start, end).df
    bars = bars.reset_index(drop=False)
    bars = bars.drop(bars[bars.exchange != "FTXU"].index)
    bars = bars.reset_index(drop=True)
    bars = bars.drop(columns=["vwap", "trade_count", "exchange"])
    bars = bars.rename(columns={"timestamp": "date"})
    return bars


#Gets the current day and previous day for indicators
def get_time():
    start = "2021-07-01" #(datetime.datetime.now() + datetime.timedelta(days=-30)).strftime("%Y-%m-%d") #"2021-07-01"
    end = "2022-08-20" #datetime.datetime.now().strftime("%Y-%m-%d")  #"2022-08-20"
    return start, end

def convert_bars(bars):
    quotes_list = [
    Quote(d,o,h,l,c,v) 
    for d,o,h,l,c,v 
    in zip(bars['date'].apply(lambda x: datetime.datetime.strptime(str(x)[0:19], "%Y-%m-%d %H:%M:%S")), bars['open'], bars['high'], bars['low'], bars['close'], bars['volume'])
    ]
    return quotes_list

def get_indicators(quotes_list):
    heikin_ashi = indicators.get_heikin_ashi(quotes_list)

    #Create new quote list for heikin_ashi bars
    h_a_quotes = []
    for i in heikin_ashi:
        h_a_quotes.append(Quote(i.date, i.open, i.high, i.low, i.close, i.volume))

    ema_50 = indicators.get_ema(h_a_quotes, 50)
    ema_200 = indicators.get_ema(h_a_quotes, 200)
    adx = indicators.get_adx(h_a_quotes, 14)

    for index, i in enumerate(adx):
        if(i.adx != None):
            adx[index] = round(float(i.adx), 3)
        else:
            adx[index] = None
    
    for index, i in enumerate(ema_50):
        if(i.ema != None):
            ema_50[index] = round(float(i.ema), 3)
        else:
            ema_50[index] = None
    
    for index, i in enumerate(ema_200):
        if(i.ema != None):
            ema_200[index] = round(float(i.ema), 3)
        else:
            ema_200[index] = None
    
    for index, i in enumerate(heikin_ashi):
            heikin_ashi[index] = (round(i.open,3), round(i.high,3), round(i.low,3), round(i.close,3))

    return adx, ema_50, ema_200, heikin_ashi

bars = get_hist(ticker)
quotes_list = convert_bars(bars)
ind_1, ind_2, ind_3, ind_4 = get_indicators(quotes_list)
bars = bars.assign(adx=ind_1, ema_50=ind_2, ema_200=ind_3, heikin_ashi=ind_4)
bars = bars.dropna()
bars = bars.reset_index(drop=True)

length = len(bars)

with open("data.csv", 'a', newline= '') as csvfile:
    writer = csv.writer(csvfile)
    for index, row in bars.iterrows():
        #determine direction
        direction = ""
        if(row["heikin_ashi"][0] == row["heikin_ashi"][2]):
            direction = "green"
        elif(row["heikin_ashi"][0] == row["heikin_ashi"][1]):
            direction = "red"
        elif(row["heikin_ashi"][0] < row["heikin_ashi"][3]):
            direction = "green doji"
        else:
            direction = "red doji"
        #determine if holding crypto
        #check to sell
        if(holding and direction == "red" and index != length-1):
            sell_cost = bars.iloc[index+1]["open"]
            bank = shares * sell_cost
            price_gain = bank - buy_amount
            graph_bank += price_gain
            shares = 0
            holding = False
            writer.writerow([row["date"], graph_bank, price_gain])
        #if not, check to buy
        elif(not(holding) and direction == "green" and (row["adx"] >= 25 and row["ema_50"] > row["ema_200"] and row["volume"] > prev_vol)):
            buy_cost = bars.iloc[index+1]["open"]
            shares = (bank * 0.997) / buy_cost
            buy_amount = bank * 0.997
            bank = 0
            holding = True
        
        prev_vol = row["volume"]