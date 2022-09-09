import numpy as np
#import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
#from keras.optimizers import Adam
from keras.optimizer_v2.adam import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
import keras
import datetime

#Trading and bars libraries
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from stock_indicators.indicators.common import Quote
from stock_indicators import indicators

LOAD_MODEL = None#"models/7210ep__1X64_2X64_stock_1____29.47max___22.12avg__221.21min.model"#"models/3000ep__2X64_stock_0_____0.92max____0.91avg____0.90min.model" #None #Or None

TICKER = "BTC/USD"
SKIP = 50000
#0      : Min
#100000 : 
#200000 : 
#402500 : 
#400119 : Max
timeframe = TimeFrame(1, TimeFrameUnit.Hour)

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 10_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 500  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 256  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 1  # Terminal states (end of episodes)
MODEL_NAME = '1X64_2X64_stock_0'
MIN_REWARD = 1000  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 3000
# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99 #0.99975
MIN_EPSILON = 0.01

#  Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes

#stock_norm2
# MAX_VAL = 327.85
# MIN_VAL = 89.47
# RANGE = 238.38

#stock_norm3
MAX_VAL = 156.3
MIN_VAL = 89.47
RANGE = 66.83

def convert(input):
  time, open_f, high_f, low_f, close_f, lips, teeth, jaw, rsi, macd = input.split(",")
  state = [np.float(time), np.float(open_f), np.float(high_f), np.float(low_f), np.float(close_f), np.float(lips), np.float(teeth), np.float(jaw), np.float(rsi), np.float(macd), 0, 0]
  return np.array(state)
  

class StockEnv:
    #TODO: Need to test different weights on Buying and Selling
    #   Add reward for buying action, make selling = reward + buy_reward
    #     - Need this to incentivise the AI to buy more often. Currently taking way too long
    #   Try making loss and gain a fixed reward
    #     - This would make any loss dramaticly bad and any gain significantly good
    #   Add a holding penalty so that the AI doesn't hold a stock for long???
    #     - This could be unnecissary. Should keep note in the event that this occurs.
    #     - Could force the AI to be a high frequency trading bot
    ACTION_SPACE = 3
    HOLD_PENALTY = 1.5
    GAIN_MULT = 250
    LOSS_MULT = 1000
    holding = 0
    holding_price = 0

    #Fixed Rewards
    BUY_REWARD = 30
    GAIN_REWARD = 500
    LOSS_PENALTY = 500

    # Worked Well
    # BUY_REWARD = 30
    # GAIN_REWARD = 15
    # LOSS_PENALTY = 600
    # HOLD_PENALTY = 1.5

    #Default
    # BUY_REWARD = 30
    # GAIN_REWARD = 15
    # LOSS_PENALTY = 500
    
    # def reset(self):
    #     with open("./data/{}_callibration.csv".format(STOCK), newline= '') as f:
    #         reader = csv.reader(f, delimiter=' ')
    #         open_f, high_f, low_f, close_f, lips, teeth, jaw, rsi = reader[0].split(",")
    #     return (float(open_f), float(high_f), float(low_f), float(close_f), float(lips), float(teeth), float(jaw), float(rsi))


    #Evaluates each action and returns the new state
    def step(self, action, line):
      # 0 = buy
      # 1 = sell
      # 2 = hold
      terminal_state = False
      gain_loss = 0
      reward = 0
      
      time, open_f, high_f, low_f, close_f, lips, teeth, jaw, rsi, macd = line.split(",")
      if(action == 0 and self.holding == 0):
          self.holding = 1
          self.holding_price = float(open_f)
          reward = self.BUY_REWARD
      elif(action == 1 and self.holding == 1):
          terminal_state = True
          #When using normalized data must unnormalize the given data to get actual gian_loss value
          gain_loss = (float(open_f) * RANGE + MIN_VAL) - (self.holding_price * RANGE + MIN_VAL)
          #gain_loss = float(open_f) - self.holding_price
          if(gain_loss > 0):
            #reward = gain_loss * self.GAIN_MULT
            reward = self.GAIN_REWARD * gain_loss
          else:
            #reward = gain_loss * self.LOSS_MULT
            reward = self.LOSS_PENALTY * gain_loss
          self.holding = 0
          self.holding_price = 0
      #apply hold penalty
      elif(action == 2 and self.holding == 1):
          reward = -self.HOLD_PENALTY
      
      #gain_loss = gain_loss * RANGE + MIN_VAL
      state = [np.float(time), np.float(open_f), np.float(high_f), np.float(low_f), np.float(close_f), np.float(lips), np.float(teeth), np.float(jaw), np.float(rsi), np.float(macd), self.holding, self.holding_price]
      return np.array(state), reward, terminal_state, gain_loss


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter 

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    # Added because of version
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()
        #self.model.summary()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):

        if LOAD_MODEL is not None:
          print(f"loading {LOAD_MODEL}")
          model = load_model(LOAD_MODEL)
          print(f"Model {LOAD_MODEL} loaded!")

        else:
          model = Sequential()
          model.add(Dense(64, input_shape=(12,))) #Change activation space to be (8) ohlc, lips, teeth, jaw, rsi
          model.add(Activation('relu'))
          model.add(Dropout(0.2))


          model.add(Dense(64))
          model.add(Activation('relu'))
          model.add(Dropout(0.2))

          model.add(Dense(64))
          model.add(Activation('relu'))
          model.add(Dropout(0.2))

          #buy, hold, sell
          model.add(Dense(3, activation='linear'))
          model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        # print("mini")
        # print(minibatch)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch], dtype=np.object)
    
        #Need to do this to prevent errors FUCK!!
        current_states = np.asarray(current_states).astype(np.float)


        current_qs_list = self.model.predict(current_states)
        #print(current_qs_list)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])

        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index]) 
                new_q = reward + DISCOUNT * max_future_q #current_qs_list[index][action] + reward + DISCOUNT * (max_future_q - current_qs_list[index][action])
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            #print(current_qs)
            #print(current_qs)
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)

            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]



#Gets the current day and previous day for indicators
def get_time():
    start = (datetime.datetime.now() + datetime.timedelta(days=-10)).strftime("%Y-%m-%d")
    end = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    return start, end

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
    #print(bars)
    #bars = bars.drop(bars[bars.exchange != "FTXU"].index)
    bars = bars.reset_index(drop=True)
    bars = bars.drop(columns=["vwap", "trade_count"])
    bars = bars.rename(columns={"timestamp": "date"})
    return bars

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
    open_ha, high_ha, low_ha, close_ha = [], [], [], []
    for i in heikin_ashi:
        open_ha.append(i.open)
        high_ha.append(i.high)
        low_ha.append(i.low)
        close_ha.append(i.close)
        h_a_quotes.append(Quote(i.date, i.open, i.high, i.low, i.close, i.volume))


    ema_50 = indicators.get_ema(h_a_quotes, 50)
    for index, i in enumerate(ema_50):
        ema_50[index] = i.ema
    ema_200 = indicators.get_ema(h_a_quotes, 200)
    for index, i in enumerate(ema_200):
        ema_200[index] = i.ema
    adx = indicators.get_adx(h_a_quotes, 14)
    for index, i in enumerate(adx):
        adx[index] = i.adx

    return adx, ema_50, ema_200, open_ha, high_ha, low_ha, close_ha

####################
# START OF TRANING #
####################

env = StockEnv()
#state:
#Open, High, Low, Close, Lips, Teeth, Jaw, Ris
agent = DQNAgent()
ep_rewards = [0]
ep_gain = [0]

# open csv of normalized data
#old
state_file = open("./data/{}_norm3.csv".format(STOCK))
#new
bars = get_hist(TICKER)
quotes_list = convert_bars(bars)
ind = get_indicators(quotes_list)
#TODO: bars drop high and low columns
bars["adx"] = ind[0]
bars["ema_50"] = ind[1]
bars["ema_200"] = ind[2]
bars["open_ha"] = ind[3]
bars["high_ha"] = ind[4]
bars["low_ha"] = ind[5]
bars["close_ha"] = ind[6]
#TODO: remove all NaN 

#TODO: delete this and add 
for i in range(0,SKIP):
    current_state = next(state_file)
current_state = convert(next(state_file)) #env.reset()

days = 0
step = 1
profit = 0
num_gains = 0
num_losses = 0
# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    episode_gain = 0

    # Reset flag and start iterating until episode ends
    done = False
    while not done:#step != 390:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
            
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE)

        #get the next ohlc, alligator, and rsi
        line = next(state_file)

        #force sell at end of day
        # if current_state[0] == 1 or current_state[0] == 0.998410174880763:
        #   action = 1
          #print("HERE")
        #add gain_loss variable
        new_state, reward, done, gain_loss = env.step(action, line)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward
        episode_gain += gain_loss
        profit += gain_loss

        if gain_loss > 0:
          num_gains += 1
        else:
          num_losses += 1
        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done)

        current_state = new_state
        step += 1
        if(step == 390):
          step = 1
          days += 1
          print(days) #comment out while training

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    ep_gain.append(episode_gain)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        average_gain = sum(ep_gain[-AGGREGATE_STATS_EVERY:])
        min_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(epsilon=epsilon, gain_loss=average_gain, profit=profit, min_reward=min_reward)

        # Save model, but only when min reward is greater or equal a set value
        if average_gain >= MIN_REWARD:
            agent.model.save(f'models/{episode}ep__{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min.model')
        print(days)

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    else:
        epsilon = 0

state_file.close() 
print()
print(days)
print()
print("# Wins: {}".format(num_gains))
print("# Loss: {}".format(num_losses))
print("% Win-Rate: {}".format(num_gains/(num_gains+num_losses)))
