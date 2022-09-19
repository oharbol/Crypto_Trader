from tradingview_ta import TA_Handler, Interval, Exchange
import csv
import time

#turn buy, neutral, sell to 0, 1, 2 respectively
def Translate_Recomendation(recomendation):
    if(recomendation == "BUY"):
        return 0
    if(recomendation == "NEUTRAL"):
        return 1
    return 2

#write analysis to csv
def WritetoCSV(interval):
    #crease handler
    btc = TA_Handler(
        symbol="BTCUSD",
        screener="CRYPTO",
        exchange="BINANCE",
        interval=interval,
    )

    #get buy/sell analysis from moving averages and oscillators
    osc = btc.get_analysis().oscillators["COMPUTE"]
    mov = btc.get_analysis().moving_averages["COMPUTE"]

    #get closing price
    close = btc.get_indicators()['close']

    #create list with close price
    analysis = [close]

    #add analysis data to list
    for i in osc:
        analysis.append(Translate_Recomendation(osc[i]))

    for i in mov:
        analysis.append(Translate_Recomendation(mov[i]))

    #write list to specific csv file
    with open("TV_BTC_Analysis_{}.csv".format(interval), "a", newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(analysis)

    print("Wrote to {}".format(interval))

while(True):
    #get time as int
    sleep_time = int(time.time())
    time.sleep(60 - (sleep_time % 60))
    time_int = int(time.time())
    time_1m = time_int % 60
    time_5m = time_int % 300
    time_1h = time_int % 3600
    time_4h = time_int % 14400

    #get one minute

    WritetoCSV(Interval.INTERVAL_1_MINUTE)

    #get five minute
    if(time_5m == 0):
        WritetoCSV(Interval.INTERVAL_5_MINUTES)

    #get 1 hour
    if(time_1h == 0):
        WritetoCSV(Interval.INTERVAL_1_HOUR)

    #get 4 hour
    if(time_4h == 0):
        WritetoCSV(Interval.INTERVAL_4_HOURS)