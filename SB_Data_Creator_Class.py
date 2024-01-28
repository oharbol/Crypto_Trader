#Trading and bars libraries
#from multiprocessing.reduction import steal_handle
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from stock_indicators.indicators.common import Quote
from stock_indicators import indicators
from datetime import datetime
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler 

# Const variables to change for data creation
# TIME_VALUE = 15
# TICKER_NAME = "ETH"
# TIME_FRAME_UNIT = TimeFrameUnit.Minute

# # Naming for CSV file
# TICKER = "ETH/USD"
# TIMEFRAME = f"{TIME_VALUE}Min"

# START_TIME = datetime(2016, 1, 1)
# END_TIME = datetime.now()
# TIME_FRAME = TimeFrame(TIME_VALUE, TimeFrameUnit.Minute)
# Min - 5, 15, 30 , 45
# Hour - 1, 2, 3, 4

prev_indicator = None

# Get historical data
def get_hist(ticker, time_frame, start_time, end_time): #ticker : list, start_time : datetime, end_time : datetime, timeframe : TimeFrame
    client = CryptoHistoricalDataClient()
    #t = TimeFrame(5, TimeFrameUnit.Minute)
    request_params = CryptoBarsRequest(
                        symbol_or_symbols= [ticker + "/USD"],
                        timeframe= time_frame,
                        start= start_time,
                        end= end_time
                )
    
    bars = client.get_crypto_bars(request_params)
    #bars = api.get_crypto_bars(symbol, timeframe, start, end).df
    bars = bars.df
    bars = bars.reset_index(drop=False)
    #bars = bars.drop(bars[bars.exchange != "FTXU"].index)
    bars = bars.reset_index(drop=True)
    bars = bars.drop(columns=["vwap", "trade_count"])
    bars = bars.rename(columns={"timestamp": "date"})
    bars = bars.round(2)
    return bars

# Converts the df format into the quote format to create indicators
def convert_bars(bars):
    bars["date"] = bars['date'].apply(lambda x: datetime.strptime(str(x)[0:19], "%Y-%m-%d %H:%M:%S"))
    quotes_list = [
    Quote(d,o,h,l,c,v) 
    for d,o,h,l,c,v 
    in zip(bars['date'], bars['open'], bars['high'], bars['low'], bars['close'], bars['volume'])
    ]
    return quotes_list


# OSCILLATORS

# Generate RSI data
def get_RSI(quotes_list):
    rsi = indicators.get_rsi(quotes_list)
    # Convert indicator object data to raw rsi
    for index, i in enumerate(rsi):
        rsi[index] = i.rsi
    return rsi
    
# Convert raw RSI into onehot data
# Use in Lambda Function
def get_onehot_RSI(rsi):
    # BUY (< 30)
    if(rsi < 30):
        return 0
    # SELL (> 70)
    elif(rsi > 70):
        return 2
    # NEUTRAL (30 < rsi < 70)
    else:
        return 1

# Generate Stochastic data
def get_Stoch(quotes_list):
    stoch = indicators.get_stoch(quotes_list)

    # Convert indicator object data to raw stoch
    for index, i in enumerate(stoch):
        stoch[index] = (i.oscillator, i.signal)

    stoch1 = []
    stoch2 = []

    for index, i in enumerate(stoch):
        stoch1.append(i[0]) 
        stoch2.append(i[1])
    return stoch1, stoch2

# Convert raw Stoachastic data into onehot data
# Usein Lambda Function
def get_onehot_Stoch(stoch):
    global prev_indicator
    stoch_return = None
    
    # BUY
    if(stoch[0] < 20 and (prev_indicator[0] < prev_indicator[1] and stoch[0] > stoch[1])):
        stoch_return = 0
    # SELL
    elif(stoch[0] > 80 and (prev_indicator[0] > prev_indicator[1] and stoch[0] < stoch[1])):
        stoch_return = 2
    # NEUTRAL
    else:
        stoch_return = 1
    
    # Return and update stoch_return
    prev_indicator = (stoch[0], stoch[1])
    return stoch_return

# Generate MACD data
def get_MACD(quotes_list):
    macd = indicators.get_macd(quotes_list)
    # Convert indicator object data into raw MACD
    for index, i in enumerate(macd):
        macd[index] = i.histogram
    return macd

# Convert raw MACD into onehot data
# Use in Lambda Function
def get_onehot_MACD(histogram):
    # BUY
    if(histogram > 0):
        return 0
    # SELL
    elif(histogram < 0):
        return 2
    # NEUTRAL
    else:
        return 1

# Generate Ultimate data
def get_Ultimate(quotes_list):
    ultimate = indicators.get_ultimate(quotes_list)
    # Convert indicator object data into raw Ultimate
    for index, i in enumerate(ultimate):
        ultimate[index] = i.ultimate
    return ultimate

# Convert raw MACD into onehot data
# Use in Lambda Function
def get_onehot_Ultimate(ultimate):
    # BUY
    if(ultimate > 70):
        return 0
    # SELL
    elif(ultimate < 30):
        return 2
    # NEUTRAL
    else:
        return 1
    
# Generate Momentum data
def get_Momentum(quotes_list):
    mom = indicators.get_roc(quotes_list, lookback_periods=10)
    # Convert indicator object data into raw Ultimate
    for index, i in enumerate(mom):
        mom[index] = i.roc
    return mom

# Convert raw Momentum data into onehot data
# Use in Lambda Function
def get_onehot_Momentum(mom):
    global prev_indicator
    mom_return = None

    # BUY
    if(mom > prev_indicator):
        mom_return = 0
    # SELL
    elif(mom < prev_indicator):
        mom_return = 2
    # NEUTRAL
    else:
        mom_return = 1
    
    # Return and update global
    prev_indicator = mom
    return mom_return

# Generate CCI data
def get_CCI(quotes_list):
    cci = indicators.get_cci(quotes_list)
    # Convert indicator object data into raw Ultimate
    for index, i in enumerate(cci):
        cci[index] = i.cci
    return cci

# Convert raw CCI into onehot data
# Use in Lambda Function
def get_onehot_CCI(cci):
    global prev_indicator
    cci_return = None

    # BUY
    if(cci < -100 and cci > prev_indicator):
        cci_return = 0
    # SELL
    elif(cci > 100 and cci < prev_indicator):
        cci_return = 2
    # NEUTRAL
    else:
        cci_return = 1

    # Return and update global
    prev_indicator = cci
    return cci_return


# MOVING AVERAGES

# Exponential Moving Average
def get_ema(quotes_list, look_back):
    ema = indicators.get_ema(quotes_list, look_back)
    # Convert indicator object data to raw ema
    for index, i in enumerate(ema):
        ema[index] = i.ema
    return ema

# Simple Moving Average
def get_sma(quotes_list, look_back):
    sma = indicators.get_sma(quotes_list, look_back)
    # Convert indicator object data to raw sma
    for index, i in enumerate(sma):
        sma[index] = i.sma
    return sma

# Volume Weighted Moving Average
def get_vwma(quotes_list, look_back):
    vwma = indicators.get_vwma(quotes_list, look_back)
    # Convert indicator object data to raw vwma
    for index, i in enumerate(vwma):
        vwma[index] = i.vwma
    return vwma

# Hull Moving Average
def get_hma(quotes_list, look_back):
    hma = indicators.get_hma(quotes_list, look_back)
    # Convert indicator object data to raw hma
    for index, i in enumerate(hma):
        hma[index] = i.hma
    return hma

# Convert raw Moving Average data into onehot data
# Use in Lambda Function
def get_onehot_ma(row, name):
    # BUY
    if(row[name] < row["close"]):
        return 0
    # SELL
    elif(row[name] > row["close"]):
        return 2
    # NEUTRAL
    else:
        return 1


class DataCreator():

    # Create data set
    def __get_RawData(self, ticker, start_time, end_time, time_value, timeframe_unit):
        bars = get_hist(ticker, TimeFrame(time_value, timeframe_unit), start_time, end_time)

        quotes_list = convert_bars(bars)

        # Remove high and low columns
        bars = bars.drop(columns=["high", "low", "symbol", "date", "open", "volume"])

        return bars, quotes_list

    # Oscillators
    def __add_Oscillators(self, bars, quotes_list):
        bars["rsi"] = get_RSI(quotes_list)
        # bars["stoch"] = get_stoch(quotes_list)
        bars["macd"] = get_MACD(quotes_list)
        bars["ultimate"] = get_Ultimate(quotes_list)
        bars["mom"] = get_Momentum(quotes_list)
        bars["stoch1"], bars["stoch2"] = get_Stoch(quotes_list)
        bars["cci"] = get_CCI(quotes_list)
        return bars

    # Moving Averages
    def __add_MA(self, bars, quotes_list):
        bars["ema10"] = get_ema(quotes_list, 10)
        bars["sma10"] = get_sma(quotes_list, 10)

        bars["ema20"] = get_ema(quotes_list, 20)
        bars["sma20"] = get_sma(quotes_list, 20)

        bars["ema30"] = get_ema(quotes_list, 30)
        bars["sma30"] = get_sma(quotes_list, 30)

        bars["ema50"] = get_ema(quotes_list, 50)
        bars["sma50"] = get_sma(quotes_list, 50)

        bars["ema100"] = get_ema(quotes_list, 100)
        bars["sma100"] = get_sma(quotes_list, 100)

        bars["ema200"] = get_ema(quotes_list, 200)
        bars["sma200"] = get_sma(quotes_list, 200)

        bars["vwma20"] = get_vwma(quotes_list, 20)
        bars["hma9"] = get_hma(quotes_list, 9)
        return bars


    # Create data for MA
    def get_MA(self, ticker, start_time = datetime(2016, 1, 1), end_time = datetime.now(), time_value = 30, timeframe_unit = TimeFrame.Minute, to_csv = True, normalize = True):
        bars, quotes_list = self.__get_RawData(ticker, start_time, end_time, time_value, timeframe_unit)
        bars = self.__add_MA(bars, quotes_list)
        # Remove all NaN
        bars = bars.dropna()

        bars = bars.round(2)


        # Write raw data
        #TODO Create switch case to handle min and hr labeling
        if(to_csv):
            self.__write_datadir()
            bars.to_csv("Data/Data_Raw_MA_{}_{}.csv".format(ticker, time_value + "Min"), index=False, header=True)

        if(normalize):
            self.__write_datadir()
            self.__Normalize(ticker, bars, ticker, time_value, ma=True)
        
        # Return last row (utilized for live data)
        return bars.iloc[-1]


    # Create data for MA
    def get_Oscillators(self, ticker, start_time = datetime(2016, 1, 1), end_time = datetime.now(), time_value = 30, timeframe_unit = TimeFrame.Minute, to_csv = True, normalize = True):
        bars, quotes_list = self.__get_RawData(ticker, start_time, end_time, time_value, timeframe_unit)
        bars = self.__add_Oscillators(bars, quotes_list)
        # Remove all NaN
        bars = bars.dropna()

        bars = bars.round(2)


        # Write raw data
        #TODO Create switch case to handle min and hr labeling
        if(to_csv):
            self.__write_datadir()
            bars.to_csv("Data/Data_Raw_Oscillator_{}_{}.csv".format(ticker, time_value + "Min"), index=False, header=True)
        
        if(normalize):
            self.__write_datadir()
            self.__Normalize(ticker, bars, ticker, time_value, oscillator=True)

        # Return last row (utilized for live data)
        return bars.iloc[-1]

    # Create data for OMA
    def get_OMA(self, ticker, start_time = datetime(2016, 1, 1), end_time = datetime.now(), time_value = 30, timeframe_unit = TimeFrame.Minute, to_csv = True, normalize = True):
        bars, quotes_list = self.__get_RawData(ticker, start_time, end_time, time_value, timeframe_unit)
        bars = self.__add_MA(bars, quotes_list)
        bars = self.__add_Oscillators(bars, quotes_list)
        # Remove all NaN
        bars = bars.dropna()

        bars = bars.round(2)


        # Write raw data
        #TODO Create switch case to handle min and hr labeling
        if(to_csv):
            self.__write_datadir()
            bars.to_csv("Data/Data_Raw_OMA_{}_{}.csv".format(ticker, time_value + "Min"), index=False, header=True)
        
        if(normalize):
            self.__write_datadir()
            self.__Normalize(ticker, bars, ticker, time_value, ma=True, oscillator=True)
        
        # Return last row (utilized for live data)
        return bars.iloc[-1]

    # Normalized Variables
    def __Normalize(self, bars, ticker, time_value, ma = False, oscillator = False):
        df_normalized = bars.copy()

        if(oscillator):
            df_normalized["rsi"] = MinMaxScaler().fit_transform(np.array(df_normalized["rsi"]).reshape(-1,1))
            df_normalized["macd"] = MinMaxScaler().fit_transform(np.array(df_normalized["macd"]).reshape(-1,1))
            df_normalized["ultimate"] = MinMaxScaler().fit_transform(np.array(df_normalized["ultimate"]).reshape(-1,1))
            df_normalized["mom"] = MinMaxScaler().fit_transform(np.array(df_normalized["mom"]).reshape(-1,1))
            df_normalized["stoch1"] = MinMaxScaler().fit_transform(np.array(df_normalized["stoch1"]).reshape(-1,1))
            df_normalized["stoch2"] = MinMaxScaler().fit_transform(np.array(df_normalized["stoch2"]).reshape(-1,1))
            df_normalized["cci"] = MinMaxScaler().fit_transform(np.array(df_normalized["cci"]).reshape(-1,1))

        if(ma):
            df_normalized["ema10"] = MinMaxScaler().fit_transform(np.array(df_normalized["ema10"]).reshape(-1,1))
            df_normalized["sma10"] = MinMaxScaler().fit_transform(np.array(df_normalized["sma10"]).reshape(-1,1))

            df_normalized["ema20"] = MinMaxScaler().fit_transform(np.array(df_normalized["ema20"]).reshape(-1,1))
            df_normalized["sma20"] = MinMaxScaler().fit_transform(np.array(df_normalized["sma20"]).reshape(-1,1))

            df_normalized["ema30"] = MinMaxScaler().fit_transform(np.array(df_normalized["ema30"]).reshape(-1,1))
            df_normalized["sma30"] = MinMaxScaler().fit_transform(np.array(df_normalized["sma30"]).reshape(-1,1))

            df_normalized["ema50"] = MinMaxScaler().fit_transform(np.array(df_normalized["ema50"]).reshape(-1,1))
            df_normalized["sma50"] = MinMaxScaler().fit_transform(np.array(df_normalized["sma50"]).reshape(-1,1))

            df_normalized["ema100"] = MinMaxScaler().fit_transform(np.array(df_normalized["ema100"]).reshape(-1,1))
            df_normalized["sma100"] = MinMaxScaler().fit_transform(np.array(df_normalized["sma100"]).reshape(-1,1))

            df_normalized["ema200"] = MinMaxScaler().fit_transform(np.array(df_normalized["ema200"]).reshape(-1,1))
            df_normalized["sma200"] = MinMaxScaler().fit_transform(np.array(df_normalized["sma200"]).reshape(-1,1))

            df_normalized["vwma20"] = MinMaxScaler().fit_transform(np.array(df_normalized["vwma20"]).reshape(-1,1))
            df_normalized["hma9"] = MinMaxScaler().fit_transform(np.array(df_normalized["hma9"]).reshape(-1,1))

        df_normalized.to_csv("Data/Data_Normalized_OMA_{}_{}.csv".format(ticker, time_value + "Min"), index=False, header=True)
    
        def __write_datadir(self):
            # Create "Data" directory
            if not os.path.exists("Data"):
                os.makedirs("Data")