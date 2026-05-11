import mplfinance as mpf
import pandas as pd
import numpy as np
import math

# Establish Constants 
SWING_BUFFER = 8
HIGH_ARROW_OFFSET = 1.01
LOW_ARROW_OFFSET = 0.99
ARROW_SIZE = 125
EMPTY = np.nan
PERCENT_BARRIER = 0.01
TREND_LINE = True
TREND_UP = 1
TREND_DOWN = 2
PERCENT_INCREASE = 0.05
STOCK = "NRGV"

trends = []

# Get Data
df = pd.read_csv(f'Data/Data_Raw_OHLC_{STOCK}_2Hour.csv')

LENGTH = len(df)
LAST_DATE = pd.Timestamp(df.iloc[LENGTH-1].name)

# Convert Data
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df.drop(columns='volume')

# Set variables for Trendline
initial_data = df.iloc[0]
swing_high = initial_data["high"]
swing_low = initial_data['low']

# (Price, index)
local_high = (swing_high, 0)
local_low = (swing_low, 0)
valid_high = 0
valid_low = 0

marker_lows = []
marker_highs = []

# 0 - initial
# 1 - bullish (up)
# 2 - bearish (down)
trend = 0

index = 0
low_input = -1
high_input = -1


# Helper Functions
def percent_change(new_val, old_val):
    # Return decemal percent value
    return (new_val - old_val) / old_val

# Iterate DataFrame for swing highs/lows
for row in df.itertuples(index=False):
    match trend:
        # Initial setup
        case 0:
            marker_highs.append(EMPTY)
            marker_lows.append(EMPTY)
            # Chart Assumed Up Trend
            if(row.high > swing_high and row.low > swing_low):
                marker_highs[-1] = swing_high * HIGH_ARROW_OFFSET
                marker_lows[local_low[1]] = local_low[0] * LOW_ARROW_OFFSET
                valid_low = local_low[0]
                local_low = (row.high, index)
                trend = TREND_UP
                high_input = index
                swing_high = row.high
            # Chart Assumed Down Trend
            elif(row.low < swing_low and row.high < swing_high):
                marker_lows[-1] = swing_low * LOW_ARROW_OFFSET
                marker_highs[local_high[1]] = local_high[0] * HIGH_ARROW_OFFSET
                valid_high = local_high[0]
                local_high = (row.low, index)
                trend = TREND_DOWN
                low_input = index
                swing_low = row.low
            # Sideways Motion
            # Update Highs and Lows
            else:
                if(row.high > local_high[0]):
                    local_high = (row.high, index)
                if(row.low < local_low[0]):
                    local_low = (row.low, index)

        # Up Trend
        case 1:
            marker_lows.append(EMPTY)

            # Check if new local low
            if(row.low < local_low[0]):
                local_low = (row.low, index)

            # Check if new high
            if(row.high > swing_high):
                swing_high = row.high
                valid_high = row.high
                marker_highs.append(swing_high * HIGH_ARROW_OFFSET)

                # Check if within recent high buffer
                if(index - high_input < SWING_BUFFER):
                    marker_highs[high_input] = EMPTY
                # Otherwise validate new local lows
                else:  
                    low_input = local_low[1]
                    valid_low = local_low[0]
                    marker_lows[low_input] = valid_low * LOW_ARROW_OFFSET
                
                local_low = (row.high, index) 
                high_input = index
            # No New Highs
            else:
                marker_highs.append(EMPTY)
            
            # Determine if reversal is present
            if(valid_low * (1 - PERCENT_BARRIER) > row.low):
                trend = TREND_DOWN
                local_high = (row.low, index)
                swing_low = row.low
                marker_lows[-1] = swing_low * LOW_ARROW_OFFSET
                low_input = index

        # Down Trend
        case 2:
            marker_highs.append(EMPTY)

            # Check if new local high
            if(row.high > local_high[0]):
                local_high = (row.high, index)

            # Check if new low
            if(row.low < swing_low):
                swing_low = row.low
                valid_low = row.low
                marker_lows.append(swing_low * LOW_ARROW_OFFSET)

                # Check if within recent low buffer
                if(index - low_input < SWING_BUFFER):
                    marker_lows[low_input] = EMPTY
                # Validate new local high
                else:
                    high_input = local_high[1]
                    valid_high = local_high[0]
                    marker_highs[high_input] = valid_high * HIGH_ARROW_OFFSET

                local_high = (row.low, index)
                low_input = index
            else:
                marker_lows.append(EMPTY)
            
            # Determine if reversal is present
            if(valid_high * (1 + PERCENT_BARRIER) < row.high):
                trend = TREND_UP
                local_low = (row.high, index)
                swing_high = row.high
                marker_highs[-1] = swing_high * HIGH_ARROW_OFFSET
                high_input = index
            
        case _:
            print("Unknown Input")

    index += 1
    trends.append(trend)

# Set list of indexs of Fair Value Gaps
fair_indexs = []

# Set variables for Fair Value Gap
fair_value_1 = df.iloc[0] # Only use row.high
fair_value_2 = df.iloc[1] # var 1 and 3 high/low is inbetween open and close
fair_value_3 = df.iloc[2] # Only use row.low
fair_values = [EMPTY, EMPTY]
index = 2

# Iterate DataFrame for fair value gaps
# TODO: Add to swing high/low loop and make its own function
for row in df.iloc[2:].itertuples(index=False):
    # Append to end of list
    fair_values.append(EMPTY)
    # Set fair value 3
    fair_value_3 = row
    # compare high and low, and check center candle is green
    if(fair_value_3.low - fair_value_1.high > 0 and fair_value_2.open < fair_value_2.close and 
       (fair_value_3.low <= fair_value_2.close and fair_value_1.high >= fair_value_2.open)):
        fair_values[index - 1] = fair_value_2.low * 0.97
        # Add Index
        fair_indexs.append(index - 1)
    
    # Set values
    fair_value_1 = fair_value_2
    fair_value_2 = fair_value_3
    
    # Increment index
    index += 1


# Set list of demand locations
demands = [EMPTY] * LENGTH
last_high = math.inf # Index of last swing high value
ignore_fair = 0

# Determine Demand Areas
# Loop through marker_highs and fair_indexes 
for index, (fair_value, swing_high) in enumerate(zip(fair_values, marker_highs)):
    # Check if new swing high present
    if(not(swing_high is EMPTY)):
        last_high = swing_high
    # TODO: Simplify this boolean zen
    # Check if we are at a fair value
    # Check if values exist after 20 candles
    # Check if fair value is within a trend up (within 20 spaces)
    # Check if candles are larger than last swing high in 5/10 candles
    # Check if we need to ignore any fair value gaps
    # Check if there are candles after 10 candles (implied true due to fair values)
    # Check if 10% increase after 5 or 10 candles
    if(not(fair_value is EMPTY) and 
       index + 20 < LENGTH and
       ignore_fair <= 0 and
       trends[index] == TREND_UP and
       (df.iloc[index+9]["close"] > last_high or df.iloc[index+4]["close"] > last_high)): 
        if(percent_change(df.iloc[index+9]["close"], df.iloc[index-1]["open"]) > PERCENT_INCREASE or percent_change(df.iloc[index+4]["close"], df.iloc[index-1]["open"]) > PERCENT_INCREASE):
            demands[index-1] = df.iloc[index-1]["high"] * 1.03
            # Add function to add dict to graph to create demand area
            ignore_fair = 10
    
    ignore_fair -= 1


# Create additional plots
additional_graph = []

if(TREND_LINE):
    if(len(set(fair_values)) == 1):
        additional_graph = [mpf.make_addplot(marker_lows,type='scatter', markersize=ARROW_SIZE, marker='^', color="r", panel = 0),
                            mpf.make_addplot(marker_highs,type="scatter", markersize=ARROW_SIZE, marker='v', color='g', panel = 0),
                            mpf.make_addplot(trends, type='bar', panel=1)
                            ]
    elif(len(set(demands)) == 1):
        additional_graph = [mpf.make_addplot(marker_lows,type='scatter', markersize=ARROW_SIZE, marker='^', color="r", panel = 0),
                            mpf.make_addplot(marker_highs,type="scatter", markersize=ARROW_SIZE, marker='v', color='g', panel = 0),
                            mpf.make_addplot(trends, type='bar', panel=1)
                            #mpf.make_addplot(fair_values, type='scatter', markersize=ARROW_SIZE, marker='^', color='b', panel = 0)
                            ]
    else:
        additional_graph = [mpf.make_addplot(marker_lows,type='scatter', markersize=ARROW_SIZE, marker='^', color="r", panel = 0),
                            mpf.make_addplot(marker_highs,type="scatter", markersize=ARROW_SIZE, marker='v', color='g', panel = 0),
                            mpf.make_addplot(trends, type='bar', panel=1),
                            #mpf.make_addplot(fair_values, type='scatter', markersize=ARROW_SIZE, marker='^', color='b', panel = 0),
                            mpf.make_addplot(demands, type="scatter", markersize=ARROW_SIZE, marker='v', color="lime", panel = 0)
                            ]
else:
    additional_graph = [mpf.make_addplot(marker_lows,type='scatter', markersize=ARROW_SIZE, marker='^', color="r"),
                        mpf.make_addplot(marker_highs,type="scatter", markersize=ARROW_SIZE, marker='v', color='g')
                        ]

# create empty list to hold dicts
fill_areas = []

# Loop over demand areas 
for index, i in enumerate(demands):
# Check if not null
    if(not(i is EMPTY)):
        # Set Variables for fill area plot
        dates_df     = pd.DataFrame(df.index)
        buy_date     = pd.Timestamp(df.iloc[index].name)
        sell_date    = pd.Timestamp(df.iloc[LENGTH-1].name)

        where_values = pd.notnull(dates_df[ (dates_df>=buy_date) & (dates_df <= sell_date) ])['date'].values

        # Append dict
        fill_areas.append(dict(y1=df.iloc[index]["high"],y2=df.iloc[index]["low"], where=where_values, alpha=0.2, color="g"))


# Fill between outline
"""
fill_between=dict(y1=value(s),y2=0,where=None,kwargs)
where is a series of true/false same length of graph
"""
# Fill Plot
mpf.plot(df, type='candle', volume=False, style='yahoo', fill_between=fill_areas, addplot=additional_graph, warn_too_much_data=LENGTH+1)
# Simple candlestic
# mpf.plot(df, type='candle', volume=False, style='yahoo', addplot=additional_graph, warn_too_much_data=LENGTH+1)