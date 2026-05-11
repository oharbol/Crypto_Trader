import config
import datetime
import time
import pandas
from stock_indicators.indicators.common import Quote
from stock_indicators import indicators, CandlePart

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


#Create the connection to the api
# api = tradeapi.REST(config.API_KEY, config.SECRET_KEY, config.BASE_URL)
# account = api.get_account()
trading_client = TradingClient(config.API_KEY, config.SECRET_KEY)

#Variables for historical data

#Add or remove crypto to trade from symbols
symbols = ["SPY", "AAPL", "MSFT"]
timeframe = TimeFrame(5, TimeFrameUnit.Minute)
risk = 1 / len(symbols)
round_var = 2

#Get historical data
def get_hist(symbol):
    client = StockHistoricalDataClient()
    #t = TimeFrame(5, TimeFrameUnit.Minute)
    start, end = get_time()
    request_params = StockBarsRequest(
                        symbol_or_symbols=[symbol],
                        timeframe=timeframe,
                        start=start,
                        end=end
                 )
    
    bars = bars = client.get_stock_bars(request_params)
    #bars = api.get_crypto_bars(symbol, timeframe, start, end).df
    bars = bars.df
    bars = bars.reset_index(drop=False)
    #print(bars)
    #bars = bars.drop(bars[bars.exchange != "FTXU"].index)
    bars = bars.reset_index(drop=True)
    bars = bars.drop(columns=["vwap", "trade_count"])
    bars = bars.rename(columns={"timestamp": "date"})
    return bars


#Gets the current day and previous day for indicators
def get_time():
    start = (datetime.datetime.now() + datetime.timedelta(days=-10)).strftime("%Y-%m-%d")
    end = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    return start, end


#Converts the df format into the quote format to create indicators
def convert_bars(bars):
    quotes_list = [
    Quote(d,o,h,l,c,v) 
    for d,o,h,l,c,v 
    in zip(bars['date'].apply(lambda x: datetime.datetime.strptime(str(x)[0:19], "%Y-%m-%d %H:%M:%S")), bars['open'], bars['high'], bars['low'], bars['close'], bars['volume'])
    ]
    return quotes_list

def get_HA(quotes_list):
    ha = indicators.get_heikin_ashi(quotes_list)
    n = len(quotes_list)
    open = [0 for i in range(n)] 
    high = [0 for i in range(n)] 
    low = [0 for i in range(n)] 
    close = [0 for i in range(n)] 
    # Convert indicator object data to candle stick ohlc heikin ashi
    for index, i in enumerate(ha):
        open[index] = i.open
        high[index] = i.high
        low[index] = i.low
        close[index] = i.close
    return open, high, low, close

# Generate MACD data
def get_MACD(quotes_list):
    macd = indicators.get_macd(quotes_list)
    # Convert indicator object data into raw MACD
    for index, i in enumerate(macd):
        macd[index] = i.histogram
    return macd

# Simple Moving Average
def get_sma(quotes_list, look_back, candle_part = CandlePart.CLOSE):
    sma = indicators.get_sma(quotes_list, look_back, candle_part)
    # Convert indicator object data to raw sma
    for index, i in enumerate(sma):
        sma[index] = i.sma
    return sma


#Generate indicators
def get_indicators(quotes_list):
    #Create new quote list for heikin_ashi bars
    bars = pandas.DataFrame()
    bars["open"], bars["high"], bars["low"], bars["close"] = get_HA(quotes_list)
    ha = convert_bars()

    bars["open"] = get_sma(ha, 20, CandlePart.OPEN)
    bars["high"] = get_sma(ha, 20, CandlePart.HIGH)
    bars["low"] = get_sma(ha, 20, CandlePart.LOW)
    bars["close"] = get_sma(ha, 20, CandlePart.CLOSE)
    bars["MACD"] = get_MACD(quotes_list)

    return {"date": bars[-1].date, "ema_50": bars[-1].ema, "ema_200": bars[-1].ema, "open": heikin_ashi[-1].open, "high": heikin_ashi[-1].high, "low": heikin_ashi[-1].low, "close": heikin_ashi[-1].close, "adx": adx[-1].adx, "volume": heikin_ashi[-1].volume}


#Buy crypto with risked portion of account
def trade_buy(equity, symbol):
    #Used for submitting orders
    dollars = 0

    #Settled funds in account
    settled = float(trading_client.get_account().non_marginable_buying_power)

    #Remove unrealized gains from total equity
    for i in trading_client.get_all_positions():
        equity -= abs(float(i.unrealized_pl))
    #Percent of total equity is settled funds
    percent_left = settled / equity
    
    #Buy with all of settled funds
    if(percent_left < risk):
        dollars = settled
    #Otherwise buy risk amount
    else:
        dollars = equity * risk
    
    #API buy order
    # api.submit_order(
    #     symbol=symbol,
    #     notional= dollars,
    #     side="buy",
    #     type='market',
    #     time_in_force='gtc',
    # )
    market_order_data = MarketOrderRequest(
                    symbol=symbol,
                    notional=dollars,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                )
    trading_client.submit_order(
                order_data=market_order_data
                )
    print("\nBought ${} of {}! At {}\n".format(dollars, symbol, datetime.datetime.now()))

#Sell entire position of crypto
def trade_sell(unrealized_pl, symbol):
    #Sell incremental amount of assets
    
    trading_client.close_position(symbol)
    print("\nSold profit ${} of {}! At {}\n".format(unrealized_pl, symbol, datetime.datetime.now()))

#print and buy
def print_bars(stuff, symbol):
    #Get all cypto held
    positions = [i.symbol for i in trading_client.get_all_positions()]

    # Check if holding given symbol
    # SELL
    if(symbol in positions):
        #Is account holding a sellable amount of crypto?
        if(close < has_close or close > target_price or macd < 0): #0.0001
                trade_sell(trading_client.get_open_position(symbol).unrealized_pl, symbol)

    # Only used once when no positions are held
    # BUY
    elif(open < has_close and close > has_close):
        trade_buy(float(trading_client.get_account().equity), symbol)

    #Used for debugging
    # print("quote: {} - {}\nopen: {}, high: {}, low: {}, close: {}\ndirection: {}\nema_50: {}, ema_200: {}\nadx: {}\nvolume: {}\n".format(
    #     symbol, stuff["date"], round(stuff["open"],round_var), round(stuff["high"],round_var), round(stuff["low"],round_var), round(stuff["close"],round_var), direction, round(stuff["ema_50"],round_var), round(stuff["ema_200"],round_var), round(stuff["adx"],round_var), round(stuff["volume"],round_var)))


#Trading Loop
while(True):
    #Determine minutes and seconds until next 15 minutes
    tn = datetime.datetime.now()
    time_min = 14 - tn.minute
    time_sec = 62 - tn.second
    #For debugging purposes only
    print("{}.{}\n".format(time_min, time_sec))
    #Sleep until 2 seconds after 15 minute mark
    time.sleep(time_min * 60 + time_sec)

    #Loop through all desired symbols
    for ticker in symbols:
        bars = get_hist(ticker)
        quotes_list = convert_bars(bars)
        ind = get_indicators(quotes_list)
        print_bars(ind, ticker)