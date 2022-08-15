# Crypto_Trader

## How To Use Code
Go into config.py and set the variables "API_KEY" and "SECRET_KEY" to your account's information. Once that is set, run:

    python SVS_Strategy.py


Default outputs (1Hour ticker):

```
quote: BTCUSD - 2022-08-15 19:00:00
open: 24146.28, high: 24158.00, low: 23980.00, close: 24056.00
direction: red doji
ema_50: 24338.52, ema_200: 23960.51
adx: 20.73
volume: 84.05
```



## Trading Rule

This trading bot follows the [Simple Volume Strategy](https://www.youtube.com/watch?v=ydolTWrM5vc). 

The strategy follows these rules to enter long:
- EMA 50 is greater than EMA 200
- ADX is greater than 20 (edited from default 25)
- Volume is greater than previous tick (Alpaca historical volume isn't reliable so this is scratched)
- Bullish Heikin Ashi candle stick

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

Source [Updated Price for Crypto Trading on Trading API](https://alpaca.markets/blog/updated-pricing-for-crypto-trading-on-trading-api/)

## Strategy

1. Code the stuff
2. Test it out
3. Profit???


## TODO
- [x] Allow trading multiple crypto currencies
- [ ] Add backtesting feature
- [ ] Allow testing on mulitiple time intervals