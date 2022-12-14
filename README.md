# Crypto_Trader
See the [Machine Learning](https://github.com/oharbol/Crypto_Trader/tree/main/Machine%20Learning) portion of this project.
## Requirements
- Pandas
- ~~[Alpaca Trade API](https://github.com/alpacahq/alpaca-trade-api-python)~~ <-- Depricated
- [Alpaca-py](https://github.com/alpacahq/alpaca-py)
- [Stock.Indicators.Python](https://github.com/DaveSkender/Stock.Indicators.Python) (created by DaveSkender)

## Trading Rule

This trading bot follows the [Simple Volume Strategy](https://www.youtube.com/watch?v=ydolTWrM5vc). The strategy follows these rules to enter long:
- EMA 50 is greater than EMA 200
- ADX is greater than 20 (edited from default 25)
- Volume is greater than previous tick (Alpaca historical volume isn't reliable so this is scratched)
- Bullish Heikin Ashi candle stick

### Optimizations
- Wait for price to touch the 50 EMA
- Use Wave Trend Indicator
- Use RSI to gauge pullbacks
- View higher time frame
- Trade when volume is above its own MA

## How To Use Code
Go into config.py and set the variables "API_KEY" and "SECRET_KEY" to your account's information. Once that is set, run:

    python SVS_Strategy.py


Default outputs (1Hour ticker):

1-Hour Bar Example:
```
quote: BTCUSD - 2022-08-15 19:00:00
open: 24146.28, high: 24158.00, low: 23980.00, close: 24056.00
direction: red doji
ema_50: 24338.52, ema_200: 23960.51
adx: 20.73
volume: 84.05
```

Buy/Sell Example:
```
Bought $3289.6896440499995 of BTC/USD! At 2022-09-09 11:00:04.374117
```

## Overview and Strategy
As of August 20th 2022, backtesting has shown that the Simple Volume Strategy (linked below) is somewhat profitable during extremely bullish market periods, while generally horizontal movements and bearish treands will yield unprofitable results.

### Issues
The biggest issues that arise with creating a trading bot are:
- Achieving a high level of winning trades
- Gaining above the 0.3% Alpaca crypto commission on every trade

The Simple Volume Strategy is predicted to average a 52% win rate on a 5-Minute time frame. Utilizing the exclusive script by the author on Trading View, I have found that only the 1-Hour time frame can can produce enough consistent profits above the 0.3% Alpaca commission. While the strategy tester has shown a general uptrend of profits, my backtesting did not yield the same results. This is because my strategy has followed the trading strategy exactly while the one on Trading View forgoes the EMA crossing rule. 

### Potential Fixes
To make this trading strategy profitable, I need to find a balance between creating high levels of accuracy (winning trades) and trades that yield profits above the Alpaca commission rates during horizontal moving and bearish markets. 
- __Utilizing a Deep Q Neural Network:__ I have hopes that a neural network might be able to figure out a better way to trade than the hard coded bot.
    - The machine learning portion of this project can be viewed here: [Machine Learning](https://github.com/oharbol/Crypto_Trader/tree/main/Machine%20Learning)
- __Addiding Optimizations and Fine Tuning Strategy:__ Adding the given optimizations and then playing around with different variables could lead to greater accuracy/profit.

## Alpaca Commission Fees

The fees that alpaca places on total currency traded is somewhat cutthroat. This forces me to create a strategy that is not only profitable, but also gains above the breakeven amount of the commission fee.

| Tier | 30d vol (USD) | Rate   |
| ---- | ------------- | ------ |
|  1   | 0             | 0.30%  |
|  2   | >500,000      | 0.28%  |
|  3   | >1,000,000    | 0.25%  |
|  4   | >5,000,000    | 0.20%  |
|  5   | >10,000,000   | 0.18%  |
|  6   | >25,000,000   | 0.15%  |
|  7   | >50,000,000   | 0.125% |
|  8   | >100,000,000  | 0.125%  |

Source: [Updated Price for Crypto Trading on Trading API](https://alpaca.markets/blog/updated-pricing-for-crypto-trading-on-trading-api/)

## 1-Hour Bot Results
<img src="data_BTC.PNG" width="350">

## TODO
- [x] Allow trading multiple crypto currencies
- [x] Add backtesting feature
- [ ] Add optimizations to bot version
- [x] Add DQN and train models
- [ ] Compare RL agent against bot

# Disclaimer
__Use this library at your own risk.__ Trading is inherently risky and could lead to financial loss especially with automatated trading. Never run automated trading code without thorough testing and proper supervision to prevent financial losses. I make no promises that this repository will guarantee profits. I am not responsible for any monetary losses.