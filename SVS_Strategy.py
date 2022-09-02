import config
#import alpaca_trade_api as tradeapi <---- Depricated
import datetime
import time
from stock_indicators.indicators.common import Quote
from stock_indicators import indicators

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


#Create the connection to the api
# api = tradeapi.REST(config.API_KEY, config.SECRET_KEY, config.BASE_URL)
# account = api.get_account()
trading_client = TradingClient(config.API_KEY, config.SECRET_KEY)


#Backtest Results from TV 1H for 8 months with 100% risk:
#Note: Will need to create my own backtests as TV results and trades are different than mine
#AAVEUSD  - 126,116.98 Acc: 55.43%
#BTCUSD   -   2,284.66 Acc: 41.48%
#DOGEUSD  -   9,681.68 Acc: 46.18%
#ETHUSD   -  12,429.29 Acc: 47.33%
#NEARUSD  -   8,183.74 Acc: 48.25%
#MATICUSD -  34,082.42 Acc: 50.92%

#Variables for historical data

#Add or remove crypto to trade from symbols
symbols = ["BTC/USD"]
timeframe = TimeFrameUnit.Hour#"1Hour" #1Hour
risk = 1 / len(symbols)
round_var = 3

#Dictionary of the minimum quantities required to sell
#This could be found in the alpaca API but I can't find it at the moment
sellable = {
    "AAVEUSD": 0.01,
    "ALGOUSD": 1,
    "BATUSD" : 1,
    "BTCUSD" : 0.0001,
    "BCHUSD" : 0.001,
    "LINKUSD": 0.1,
    "DAIUSD" : 0.1,
    "DOGEUSD" : 1,
    "ETHUSD" : 0.001,
    "GRTUSD" : 1,
    "LTCUSD" : 0.01,
    "MKRUSD" : 0.001,
    "MATICUSD" : 10, #<- you are pain
    "NEARUSD" : 0.1,
    "PAXGUSD" : 0.0001,
    "SHIBUSD" : 0.1,
    "SOLUSD" : 0.1,
    "SUSHIUSD" : 100000, #<- you are also pain
    "USDTUSD" : 0.01,
    "TRXUSD" : 1,
    "UNIUSD" : 0.1,
    "WBTCUSD" : 0.0001,
    "YFIUSD" : 0.001
}

#Get historical data
def get_hist(symbol):
    client = CryptoHistoricalDataClient()
    #t = TimeFrame(5, TimeFrameUnit.Minute)
    start, end = get_time()
    request_params = CryptoBarsRequest(
                        symbol_or_symbols=[symbol],
                        timeframe=timeframe,
                        start=start,
                        end=end
                 )
    bars = bars = client.get_crypto_bars(request_params)
    #bars = api.get_crypto_bars(symbol, timeframe, start, end).df
    bars = bars.df
    bars = bars.reset_index(drop=False)
    bars = bars.drop(bars[bars.exchange != "FTXU"].index)
    bars = bars.reset_index(drop=True)
    bars = bars.drop(columns=["vwap", "trade_count", "exchange"])
    bars = bars.rename(columns={"timestamp": "date"})
    return bars


#Gets the current day and previous day for indicators
def get_time():
    start = (datetime.datetime.now() + datetime.timedelta(days=-10)).strftime("%Y-%m-%d")
    end = datetime.datetime.now().strftime("%Y-%m-%d")
    return start, end


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
    for i in heikin_ashi:
        h_a_quotes.append(Quote(i.date, i.open, i.high, i.low, i.close, i.volume))

    ema_50 = indicators.get_ema(h_a_quotes, 50)
    ema_200 = indicators.get_ema(h_a_quotes, 200)
    adx = indicators.get_adx(h_a_quotes, 14)

    return {"date": adx[-1].date, "ema_50": ema_50[-1].ema, "ema_200": ema_200[-1].ema, "open": heikin_ashi[-1].open, "high": heikin_ashi[-1].high, "low": heikin_ashi[-1].low, "close": heikin_ashi[-1].close, "adx": adx[-1].adx, "volume": heikin_ashi[-1].volume}


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
    if(symbol == "MATICUSD" or symbol == "SUSHIUSD"):
        #API sell order
    #     api.submit_order(
    #         symbol=symbol,
    #         qty= (float(api.get_position(symbol).qty) // sellable[symbol]) * sellable[symbol],
    #         side="sell",
    #         type='market',
    #         time_in_force='gtc',
    #     )
        market_order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty= (float(trading_client.get_open_position(symbol).qty) // sellable[symbol]) * sellable[symbol],
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC,
                    )
        trading_client.submit_order(
                    order_data=market_order_data
                    )
    #Sell all sellable assets
    else:
        trading_client.close_position(symbol)
    print("\nSold profit ${} of {}! At {}\n".format(unrealized_pl, symbol, datetime.datetime.now()))

#print and buy
def print_bars(stuff, symbol):
    #Determine direction of heikin ashi candle stick
    direction = ""
    if(stuff["open"] == stuff["low"]):
        direction = "green"
    elif(stuff["open"] == stuff["high"]):
        direction = "red"
    elif(stuff["open"] < stuff["close"]):
        direction = "green doji"
    else:
        direction = "red doji"
    
    #Get all cypto held
    positions = [i.symbol for i in trading_client.get_all_positions()]
    
    #Check if holding given symbol
    if(symbol in positions):
        #Is account holding a sellable amount of crypto?
        if(float(trading_client.get_open_position(symbol).qty) >= sellable[symbol]): #0.0001
            #Sell
            if(direction == "red"):
                trade_sell(trading_client.get_open_position(symbol).unrealized_pl, symbol)

        #Account not holding a sellable amount of crypto
        elif(stuff["adx"] >= 20 and direction == "green" and stuff["ema_50"] > stuff["ema_200"]):
            #Buy
            trade_buy(float(trading_client.get_account().equity), symbol) 

    #Only used once when no positions are held
    elif(stuff["adx"] >= 20 and direction == "green" and stuff["ema_50"] > stuff["ema_200"]):
        trade_buy(float(trading_client.get_account().equity), symbol)

    #Used for debugging
    print("quote: {} - {}\nopen: {}, high: {}, low: {}, close: {}\ndirection: {}\nema_50: {}, ema_200: {}\nadx: {}\nvolume: {}\n".format(
        symbol, stuff["date"], round(stuff["open"],round_var), round(stuff["high"],round_var), round(stuff["low"],round_var), round(stuff["close"],round_var), direction, round(stuff["ema_50"],round_var), round(stuff["ema_200"],round_var), round(stuff["adx"],round_var), round(stuff["volume"],round_var)))


#Trading Loop
while(True):
    #Determine minutes and seconds until next hour
    tn = datetime.datetime.now()
    time_min = 59 - tn.minute
    time_sec = 62 - tn.second
    #For debugging purposes only
    print("{}.{}\n".format(time_min, time_sec))
    #Sleep until 2 seconds after hour
    time.sleep(time_min * 60 + time_sec)

    #Loop through all desired symbols
    for ticker in symbols:
        bars = get_hist(ticker)
        quotes_list = convert_bars(bars)
        ind = get_indicators(quotes_list)
        print_bars(ind, ticker)