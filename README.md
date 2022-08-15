# Crypto_Trader

## TODO
- Add backtesting feature
- Allow trading multiple crypto currencies
- Allow testing on mulitiple time intervals

## Rules 

This trading bot follows the [Simple Volume Strategy](https://www.youtube.com/watch?v=ydolTWrM5vc). 

The strategy follows these rules to enter long:
- EMA 50 is greater than EMA 200
- ADX is greater than 25
- Volume is greater than previous tick
- Bullish Heikin Ashi candle stick

## Alpaca Fees

| Tier | 30d vol (USD) | Rate   |
| ---- | ------------- | ------ |
|  1   | 0             | 0.30%  |
|  2   | >500,000      | 0.28%  |
|  3   | >1,00,000     | 0.25%  |
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