# Crypto Machine Learning Model

Code is a modified version of Sentdex's [Deep Q Networks (DQN) - Reinforcement Learning](https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/) tailored to crypto trading.

You can learn more at: https://pythonprogramming.net/

## Requirements
- Tensorflow
- Keras
- Numpy
- Collections (deque)
- TQDM

TradingView data collection provided by brian-the-dev's [python-tradingview-ta](https://github.com/brian-the-dev/python-tradingview-ta)

## How to Use Code
To optomize the time it takes the model to learn, I utilize Google Collab to run all of the machine learning code. I have been unsuccessful in running it on my own computer so Collab works perfectly. Save the Crypto_DQN.ipynb file to collab and add the appropriate .csv file from the data creator scripts. Here you should be able to run the cells in the jupyter notbook on Collab.

## Overview and Strategy

### What is a DQN?
A Deep Q Network (DQN) is a reinforcement learning model that utilizes the concepts from the [markov decision process](https://www.geeksforgeeks.org/markov-decision-process/) (MDP). MDPs are composed of large tables called Q-Tables that store the Q values for each action the agent can make. The higher the Q value, the more likely the agent will take that associated action. The DQN modifies this concept by replacing the large Q-Tables with a neural network. Each node in the output layer of the neural network represents an action that the agent can take. The agent picks the action by selecting the node that has the highest value after processing the current state. During training, there is an epsilon value that gradually decreases from 1.0 to 0.0 each episode. This value represents the percent chance for the agent to take a random action. This ensures that the agent is able to explore all outcomes to seek out optimal and less optimal decisions.

### Modifications
While most MDPs have states that represents a positive and negative reward at the terminal state, our DQN will utilize the realized gain and loss from selling as the terminal state respectively. Each action that the DQN takes is stored in a "replay memory" that will be learned upon by a smaller batch once the minimum number of steps have been take.

### Data
__Trading View Technical Analysis Data:__ The initial data that I am utilizing to train the model comes from the tradingview technical analysis as stated above. The technical data returend involves 26 different indicators: 11 Oscillators and 15 Moving Averages. Each indicator returns a BUY, SELL, or NEUTRAL signal in the form of 0, 1, and 2. These can be easily translated into 3-bit one-hot encoded values for the model to learn a lot quicker. 

|Pros|Cons|
|----|----|
|One-hot encoded|Need to collect more data|
|A lot of data to use| Need to conduct a lot of tests by dropping certain signals|

__Simple Volume Strategy Data:__ The raw data needed for the SVS strategy is the 50 EMA, 200 EMA, ADX, Volume, and Heikin-Ashi candle sticks. This data is converted into onehot encodings for each of the triggers for a long entry. 

Note: I have coded the Heikin-Ashi candles for a onehot encoding of 4 bits to represent the four different types of candles while all others have 2 bits. 

Note: Volume has not been added to the onehot encoding as of now.

|Pros|Cons|
|----|----|
|Data is tailored to a strategy|Need to heavily modify raw data|
|Can get historical data|Less uses for one-hot encoding|

### Optimizations
- Changing the shape and size of the model
- Convert compatable data to onehot encoding

## Results
It ain't looking good fam
## TODO
- [x] Cry
- [ ] Do more research on reinforcement optimizations
- [ ] Send issue request to Alpaca regarding historical crypto data
- [ ] Get more data for historical crypto bars
- [ ] Train 2 differnet models for each crypto ticker
- [ ] Cry even more

# Disclaimer
__Use this library at your own risk.__ Trading is inherently risky and could lead to financial loss especially with automatated trading. Never run automated trading code without thorough testing and proper supervision to prevent financial losses. I make no promises that this repository will guarantee profits. I am not responsible for any monetary losses.