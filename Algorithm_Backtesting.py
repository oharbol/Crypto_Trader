# Algorithm Backtesting

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import numpy as np
import urllib
import datetime as dt
import csv
import pandas as pd
import mplfinance as mpf

# Backtest Variables
TIME_VALUE = 15
TIME_UNIT = "Min"
TICKER = "FNGU" #"ETH"
# Percent of captial risked per trade (0-1)
#PERCENT_RISK = 0.02
# Risk Reward Ratio (Integers > 0)
RRR = 20

ohlc_df = pd.read_csv(f"Data/Data_Raw_OHLC_{TICKER}_{TIME_VALUE}{TIME_UNIT}.csv")
has_df = pd.read_csv(f"Data/Data_Raw_HAS_{TICKER}_{TIME_VALUE}{TIME_UNIT}.csv")

# Constants
LENGTH = has_df.__len__()
INITIAL_CAPITAL = 1000
# Percent of commission taken per trade
COMMISSION = 0

# Manipulated Variables
wins = 0
total_cash_win = 0
losses = 0
total_cash_loss = 0
holding_price = 0
cash = INITIAL_CAPITAL
stoploss = -1
target_price = 0

# df1 = pd.read_csv("Data/Data_Raw_HAS_ETH_1Min.csv", index_col=0, parse_dates=True)
# df2 = pd.read_csv("Data/Data_Raw_OHLC_ETH_1Min.csv", index_col=0, parse_dates=True)
# df3 = pd.read_csv("Data/Data_Raw_HA_ETH_1Min.csv")

#ap = mpf.make_addplot(df1, type='candle')
#aap = mpf.make_addplot(df3, type='line')

# mpf.plot(df2,type='candle', style="yahoo", returnfig=True, addplot=ap)
# mpf.show()
# mpf.plot(df3,type='line')
# mpf.show()


# Strategy utilizing Smoothed Heikin Ashi (HAS) candle sticks
# Long Position: When HAS is trending down and actual price engulfs HAS candle with close higher than the HAS close
# Stop Loss: When HAS is trending up and actual price engulfs HAS candle with close lower than HAS close
def HAS_Strategy(open, close, has_open, has_close, macd, holding, stoploss):
    has_istrendup = False

    # Determine if HAS is trending upwards
    if(has_open < close):
        has_istrendup = True
    else:
        has_istrendup = False
    
    # BUY = 0
    # SELL = 1
    # HOLD = 2
    # Determine if to take long position
    if(not(holding) and open < has_close and close > has_close):
        return 0

    # Determine to sell long position
    elif(holding and (close < has_close or close > target_price)):                 #holding and (close < has_close or close > target_price or close < stoploss
        return 1

    # Otherwise hold
    else:
        return 2
    
# Stop loss rule associated with HAS Strategy
def HAS_Stoploss(open, high, low, close, holding):

    if(stoploss > 0 and (low > stoploss)):
        return stoploss
    
    # Determine if new low is lowest low from bearish candles
    return low



# Test read values from CSV file
# CSV: date, open, high, low, close, volume
# file = open("{}.csv".format("Data/Data_Raw_HAS_ETH_1Min"))
# reader = csv.reader(file)
# Remove labels
# next(reader)

holding = False
list_cash = []

# Loop through all OHLC data
for i in range(LENGTH):
    # Get current stoploss value
    stoploss = HAS_Stoploss(ohlc_df["open"][i], ohlc_df["high"][i], ohlc_df["low"][i], ohlc_df["close"][i], holding)

    # Get action from dedicated strategy
    action = HAS_Strategy(ohlc_df["open"][i], ohlc_df["high"][i], has_df["open"][i], has_df["close"][i], has_df["MACD"][i], holding, stoploss)

    # Buy
    if(action == 0):
        holding_price = ohlc_df["close"][i]
        holding = True

        # Calculate target price
        target_price = holding_price * (1 + ((holding_price - stoploss) * RRR) / stoploss)###########################
        #print("BUY: ", ohlc_df["date"][i])
    # SELL
    elif(action == 1):
        # Update price win/loss realized ratio
        price_ratio = ohlc_df["close"][i] / holding_price
        # New total cash ammount
        new_cash = price_ratio * cash - (cash * COMMISSION) # Will need to change second cash statement in this line to the risked money

        # Determine agerage gain/loss
        # Gain
        if(price_ratio > 1):
            total_cash_win += new_cash - cash
            wins += 1
        else:
            total_cash_loss += new_cash - cash
            losses += 1

        cash = new_cash
        holding = False
        # Reset stoploss
        stoploss = -1
        #print("SELL: ", ohlc_df["date"][i])
    
    # ELSE HOLD
            
    
    list_cash.append(cash)

# Print backtest summary
print(f"Summary: {TICKER} {TIME_VALUE} {TIME_UNIT}")
# Print wins
print("Wins: ", wins)
# Print losses
print("Losses: ", losses)
# Print W/L rate
print(f"Win/Loss Rate: {round(wins / (wins + losses), 2) * 100}%")
# Avg Win Ammount
print(f"Average Win Per Trade: ", round(total_cash_win / wins, 2))
# Avg Loss Ammount
print(f"Average Loss Per Trade: ", round(total_cash_loss / losses, 2))
# Ending cash ammount
print("Ending cash: ", round(cash, 2))

# Build dataframe for cash ammounts
df = pd.DataFrame()
df["date"] = ohlc_df["date"]
df["cash"] = list_cash

# Plot backtest results
df.plot.line(x="date", y="cash")
plt.show()