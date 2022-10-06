#Trading and bars libraries
from multiprocessing.reduction import steal_handle
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from stock_indicators.indicators.common import Quote
from stock_indicators import indicators
import datetime
import pandas as pd
import numpy as np
#import tensorflow as tf

TICKER = "BTC/USD"
TICKER_NAME = "BTC"
TIMEFRAME = TimeFrame(1, TimeFrameUnit.Hour)

#Gets the current day and previous day for indicators
def get_time():
    start = "2018-01-01"
    end = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    return start, end

#Get historical data
def get_hist(symbol):
    client = CryptoHistoricalDataClient()
    #t = TimeFrame(5, TimeFrameUnit.Minute)
    start, end = get_time()
    request_params = CryptoBarsRequest(
                        symbol_or_symbols=[symbol],
                        timeframe=TIMEFRAME,
                        start=start,
                        end=end
                 )
    bars = bars = client.get_crypto_bars(request_params)
    #bars = api.get_crypto_bars(symbol, timeframe, start, end).df
    bars = bars.df
    bars = bars.reset_index(drop=False)
    #print(bars)
    #bars = bars.drop(bars[bars.exchange != "FTXU"].index)
    bars = bars.reset_index(drop=True)
    bars = bars.drop(columns=["vwap", "trade_count"])
    bars = bars.rename(columns={"timestamp": "date"})
    return bars

#Converts the df format into the quote format to create indicators
def convert_bars(bars):
    quotes_list = [
    Quote(d,o,h,l,c,v) 
    for d,o,h,l,c,v 
    in zip(bars['date'].apply(lambda x: datetime.datetime.strptime(str(x)[0:19], "%Y-%m-%d %H:%M:%S")), bars['open'], bars['high'], bars['low'], bars['close'], bars['volume'])
    ]
    return quotes_list

#Generate indicators
def get_indicators(quotes_list):
    heikin_ashi = indicators.get_heikin_ashi(quotes_list)

    #Create new quote list for heikin_ashi bars
    h_a_quotes = []
    open_ha, high_ha, low_ha, close_ha = [], [], [], []
    for i in heikin_ashi:
        open_ha.append(i.open)
        high_ha.append(i.high)
        low_ha.append(i.low)
        close_ha.append(i.close)
        h_a_quotes.append(Quote(i.date, i.open, i.high, i.low, i.close, i.volume))


    ema_50 = indicators.get_ema(h_a_quotes, 50)
    for index, i in enumerate(ema_50):
        ema_50[index] = i.ema
    ema_200 = indicators.get_ema(h_a_quotes, 200)
    for index, i in enumerate(ema_200):
        ema_200[index] = i.ema
    adx = indicators.get_adx(h_a_quotes, 14)
    for index, i in enumerate(adx):
        adx[index] = i.adx

    return adx, ema_50, ema_200, open_ha, high_ha, low_ha, close_ha


def ema_cross(bar):
    if bar["ema_50"] > bar["ema_200"]:
        return 0 
    return 1

def ha_candle(bur):
    if(bur["open_ha"] == bur["low_ha"]):
        return 0
    elif(bur["open_ha"] == bur["high_ha"]):
        return 1
    elif(bur["open_ha"] < bur["close_ha"]):
        return 2
    else:
        return 3


#create data set
bars = get_hist(TICKER)
quotes_list = convert_bars(bars)
#remove high and low columns
bars = bars.drop(columns=["high", "low", "symbol"])
#add data to df in new columns
bars["adx"], bars["ema_50"], bars["ema_200"], bars["open_ha"], bars["high_ha"], bars["low_ha"], bars["close_ha"] = get_indicators(quotes_list)
#remove all NaN 
bars = bars.dropna()

#write to csv
bars.to_csv("Data_Raw_{}.csv".format(TICKER_NAME), index=False, header=False)

#data = pd.read_csv("Data.csv", index_col=False)
#onehot data
data_onehot = pd.DataFrame()

data_onehot["open"] = bars["open"]
data_onehot["ema_cross"] = bars[["ema_50", "ema_200"]].apply(lambda x: ema_cross(x) , axis=1)
data_onehot["ha_candle"] = bars[["open_ha", "high_ha", "low_ha", "close_ha"]].apply(lambda x: ha_candle(x) , axis=1)
#optional
#["volume"] = data[["volume"]].apply(lambda x: ha_candle(x) , axis=1)
data_onehot["adx"] = bars["adx"].apply(lambda x: 0 if x > 25 else 1)

#write to csv
data_onehot.to_csv("Data_OneHot_{}.csv".format(TICKER_NAME), index=False, header=False)

#Trying to figure out new config function

# EMA_INPUT_SIZE = 2
# HA_INPUT_SIZE = 4
# ADX_INPUT_SIZE = 2

# data = open("Data_OneHot_BTC.csv")
# state = next(data)

# state = state.split(",")
# state = [np.float64(i) for i in state]
# state_onehot = tf.keras.utils.to_categorical(state[0], num_classes=EMA_INPUT_SIZE).reshape((2,))
# state_onehot = np.concatenate((tf.keras.utils.to_categorical(state[1], num_classes=HA_INPUT_SIZE).reshape((4,)), state_onehot), axis=None)
# state_onehot = np.concatenate((tf.keras.utils.to_categorical(state[2], num_classes=ADX_INPUT_SIZE).reshape((2,)), state_onehot), axis=None)
# return_thingy = [state[0]] + state_onehot.tolist()
# np.array(return_thingy)