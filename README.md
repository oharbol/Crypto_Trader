# Crypto_Trader

Man-Hours worked on project (After 14DEC2023): 
- Coding: 22 hours
- Manual Backtesting: 4.5 hours
- Research: 3.5 hours

Total Man-Hours: 29.5 hours

Note: Total Man-Hours does not reflect model training time and automated forward testing which would contribute 300+ hours alone at this point.

## Overview and Strategy
This repo utilizes the machine learning models from Stable Baselines 3 in an attempt to beat the cryptocurrency market. The main models utilized are DQN, PPO and RecurrantPPO. The models are feed with a selection of raw market data, market indicators, and various custom datapoints to give the model an understanding of the current market trend. The model will take these observations and translate them into BUY, SELL, or HOLD actions. The model will not make decisions on the amount of crypto traded. Risk will be determined by the human.

See the TODO list for the current features being developed.

## Requirements
- Pandas, Numpy, Tensorflow/Keras, matplotlib
- [Alpaca-py](https://github.com/alpacahq/alpaca-py)
- [Stock.Indicators.Python](https://github.com/DaveSkender/Stock.Indicators.Python) (created by DaveSkender)
- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)
- [Open AI Gymnasium](https://gymnasium.farama.org/index.html) (Formerly known as "Gym")


## How To Use Code
#### Data creation
To create the csv data for your models to utilize, run:
```
python SB_Data_Creator
```
To get different time intervals or crytpo currency, change these three variables at the top of the file
```python
# Const variables to change for data creation
TIME_VALUE = 30
TICKER_NAME = "ETH"
TIME_FRAME_UNIT = TimeFrameUnit.Minute
```

#### Test Models
To train a given model, run:
```
python SB_Crypto_Test
```

These parameters at the top of SB_Crypto_Test.py are utilized for fine tuning the model. More ways to fine tune will be available in the future as development continues.

```python
# Global consts for training
SAVE_MODEL = True
TIMESTEPS = 53290
DATA_CSV = "Data/Data_Raw_OMA_ETH_30Min"
SCORE = 50
```

To change the model, you will have to manually comment and uncomment the selected model.
```python
# Models
# model = A2C("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
# model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
model = DQN("MlpPolicy", env, verbose=0, exploration_fraction=0.9, exploration_final_eps=0.001, batch_size=256, tensorboard_log=logdir)
# model = RecurrentPPO("MlpLstmPolicy", env, verbose=0, tensorboard_log=logdir)
# model = QRDQN("MlpPolicy", env, verbose=0, exploration_fraction=0.5, batch_size=128, tensorboard_log=logdir)
```

#### Backtesting
The backtesting class allows you to visibly see the profits overtime and allows you to save the data into a csv file. The example below is the Visual_Backtest.py file that shows the graph at the end but does not prompt to save a csv file:
```python
from Backtester import Backtester

# Global const
DATA_CSV = "Data/Data_Raw_OMA_ETH_30Min"
TIMESTEPS = 53290

MODEL_NAME = "PPO_ETH_sh23_30Min_OMARaw_Reward6_obslevel_score20_2"
MODEL_ZIP = "PPO_ETH_sh23_30Min_OMARaw_Reward6_obslevel_score20_2_2860000"
SCORE = 20

# Create Backtester class
tester = Backtester(DATA_CSV, TIMESTEPS, True, write_data=False)

tester.Backtest(MODEL_NAME, MODEL_ZIP, score=SCORE)
```

## PPO vs. DQN vs. RecurrantPPO
TODO

## Challenges
The biggest issues that arise with creating a trading model are:
- Achieving a high level of winning trades
- Obtaining high profitability during bull markets and reduce losses during bear markets
- Gaining above the 0.4% Alpaca crypto commission on every trade
- During the training process, ensuring the model's reward can relate to the increase in profit and the win/loss ratio

## Alpaca Commission Fees

The fees that alpaca places on total currency traded is somewhat cutthroat. This forces me to create a strategy that is not only profitable, but also gains above the breakeven amount of the commission fee. The worst fee that can be accumulated is approximately 0.4% or 0.40 cents for every 100 dollars used. This means every winning trade needs to be a little above 0.4%.

| Tier | 30D vol (USD)            | Maker  | Take  |
| ---- | -------------            | ------ | ----  |
|  1   | 0 - 100,000              | 0.15%  | 0.25% |
|  2   | 100,000 - 500,000        | 0.28%  | 0.22% |
|  3   | 500,000 - 1,000,000      | 0.10%  | 0.20% |
|  4   | 1,000,000 - 10,000,000   | 0.08%  | 0.18% |
|  5   | 10,000,000 - 25,000,000  | 0.05%  | 0.15% |
|  6   | 25,000,000 - 50,000,000  | 0.02%  | 0.13% |
|  7   | 50,000,000 - 100,000,000 | 0.02%  | 0.12% |
|  8   | 100,000,000+             | 0.00%  | 0.10% |

Source: [Alpaca Crypto Fees](https://docs.alpaca.markets/docs/crypto-fees#:~:text=While%20Alpaca%20stock%20trading%20remains,two%20parties%2C%20buyers%20and%20sellers.)


## TODO
- [x] Allow trading multiple crypto currencies
- [x] Add backtesting feature
- [ ] Create overload to backtesting feature
- [ ] Test various DQN models against various score goals
- [ ] Add command line arguments to DataCreator class
- [ ] Fine tune models by removing unnecessary observations
- [ ] Explore different observations and reward systems
- [ ] Create way to conduct paper/live testing
- [ ] Look into normalizing data
- [ ] Cry 

# Disclaimer
__Use this library at your own risk.__ Trading is inherently risky and could lead to financial loss especially with automatated trading. Never run automated trading code without thorough testing and proper supervision to prevent financial losses. I make no promises that this repository will guarantee profits. I am not responsible for any monetary losses.