import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import keras

import csv

# IDEA CORNER
# Create a reward for buying after selling to mimic a short trade
# * Set a reward with the observer level observation if else chain - tried
# Add observation for holding (and number of steps held)
# Remove -0.4 commission, have win/loss reflect commission values, have reward for observer level change for 0 - 0.4 gain
# Have entire environment keep track of the profit, win/loss, or any other trends to help with trades
# OneHot encoding: Have 

# Doing Now


DATA_CSV = "Data/Data_Raw_OMA_BTC_5Min"
TIMESTEPS = 293380
# TIMESERIES = "1Hour"
SHAPE = 22
CASH = 100
REWARD_MULT = 1

OBS_LEVEL = True
CLASSIFICATIONS = 7

# Convert Observation space into floats
# Return as np array
def str_to_float(data_list):
    return np.array([float(i) for i in data_list])

class CryptoEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.truncated = False
        self.file = open("{}.csv".format(DATA_CSV))
        self.reader = csv.reader(self.file)
        self.profit = 0
        self.done = False
        self.steps = 0
        self.hold_steps = 0

        self.wins = 1
        self.losses = 1
        self.avg_win = 0
        self.avg_loss = 0
        self.avg_win_total = 0
        self.avg_loss_total = 0
        self.num_trades = 1

        self.wait_buy = 0
        #self.remove_int = remove_int
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-700, high=70000,
                                            shape=(SHAPE,), dtype=np.float32)
        

    def step(self, action):
        info = {}
        self.reward = 0
        self.steps += 1

        # Calculate Reward
        # BUY
        if(action == 0 and not self.holding):
            self.buy_price = self.previous_price
            self.holding = True
            self.amount_bought = CASH / self.buy_price

            self.wait_buy = 0
        # SELL
        elif(action == 2 and self.holding):
            self.num_trades += 1
            self.hold_steps = 0

            # Sell reward
            gain_loss = self.previous_price - self.buy_price
            realized_gl = (gain_loss * self.amount_bought) - 0.4
            self.buy_price = 0
            self.holding = False
            self.done = True

            # REWARDS

            # Reward 1: Control
            # self.reward = self.previous_price - self.buy_price
            # self.buy_price = 0
            # self.holding = False
            # self.done = True

            # Reward 2: Reward = Gain/Loss
            # self.reward = realized_gl

            # Reward 3: Levels of earning (4 levels) yield flat reward
            # if(realized_gl < 0):
            #     self.reward = -100
            # elif(realized_gl > 0 and realized_gl < 0.7 * REWARD_MULT):
            #     self.reward = 0
            # elif(realized_gl > 0.7 * REWARD_MULT and realized_gl < 1.4 * REWARD_MULT):
            #     self.reward = 10
            # else:
            #     self.reward = 20

            # Reward 4: Levels of earning (6) yeild flat reward to incentivise 
            # Combined with reward of -0.01 every hold action
            # if(realized_gl < -1.4 * REWARD_MULT):
            #     self.reward = -150
            # elif(realized_gl < -0.7 * REWARD_MULT and realized_gl > -1.4 * REWARD_MULT):
            #     self.reward = -100
            # elif(realized_gl < 0 and realized_gl > -0.7 * REWARD_MULT):
            #     self.reward = -50
            # elif(realized_gl > 0 and realized_gl < 0.7 * REWARD_MULT):
            #     self.reward = 10
            # elif(realized_gl > 0.7 * REWARD_MULT and realized_gl < 1.4 * REWARD_MULT):
            #     self.reward = 25
            # else:
            #     self.reward = 50
            
            # Reward 5: Levels of earning (6) yeild flat reward to incentivise 
            # Combined with reward of -0.01 every hold action
            # Worse profits yield lower reward, all positive gains are same reward
            if(realized_gl < -1.4 * REWARD_MULT):
                self.reward = -150
            elif(realized_gl < -0.7 * REWARD_MULT and realized_gl > -1.4 * REWARD_MULT):
                self.reward = -100
            elif(realized_gl < 0 and realized_gl > -0.7 * REWARD_MULT):
                self.reward = -50
            elif(realized_gl > 0 and realized_gl < 0.7 * REWARD_MULT):
                self.reward = 10
            elif(realized_gl > 0.7 * REWARD_MULT and realized_gl < 1.4 * REWARD_MULT):
                self.reward = 10
            else:
                self.reward = 10


            # Add realized gains to total profit
            self.profit += realized_gl


            # Add to wins or losses
            if(realized_gl > 0):
                self.avg_win = (self.avg_win_total / self.wins) + (realized_gl / self.wins)
                self.avg_win_total += realized_gl
                self.wins += 1
            else:
                self.avg_loss = (self.avg_loss_total / self.losses) + (realized_gl / self.losses)
                self.avg_loss_total += realized_gl
                self.losses += 1
        
        # HOLD
        if(self.holding):
            self.hold_steps += 1
            # gain_loss = self.previous_price - self.buy_price
            # realized_gl = (gain_loss * self.amount_bought) - 0.4

            # For some reason this value is needed to create somewhat positive results
            self.reward = -0.01 #* self.hold_steps

        # Do Nothing
        # else:
        #     self.wait_buy += 1
        #     self.reward = -0.001 #* self.wait_buy
            

        # Get next set of data
        if(self.steps >= TIMESTEPS):
            self.file.close()
            self.file = open("{}.csv".format(DATA_CSV))
            self.reader = csv.reader(self.file)
            self.steps = 0
            self.holding = False
            self.done = True
            self.reward = 0
            next(self.reader)

        # Get observation for reward 3
        # if(OBS_LEVEL):
        #     self.gl_level = 0
        #     if(self.holding):
        #         current_gain_loss = self.previous_price - self.buy_price
        #         realized_gl = (current_gain_loss * self.amount_bought) - 0.4
        #         if(realized_gl < 0.4):
        #             self.gl_level = 0
        #         elif(realized_gl > 0.4 and realized_gl < 0.7):
        #             self.gl_level = 1
        #         elif(realized_gl > 0.7 and realized_gl < 1.4):
        #             self.gl_level = 2
        #         else:
        #             self.gl_level = 3
        
        # Get observation for reward 4
        # Observer Level 1
        if(OBS_LEVEL):
            # Reset to neutral gl observation level
            self.gl_level = 3
            if(self.holding):
                # Calculate realized gain/loss
                current_gain_loss = self.previous_price - self.buy_price
                realized_gl = (current_gain_loss * self.amount_bought) - 0.4
                # Determine gl observation level
                if(realized_gl < -1.4 * REWARD_MULT):
                    self.gl_level = 0
                elif(realized_gl < -0.7 * REWARD_MULT and realized_gl > -1.4 * REWARD_MULT):
                    self.gl_level = 1
                elif(realized_gl < 0 and realized_gl > -0.7 * REWARD_MULT):
                    self.gl_level = 2
                elif(realized_gl > 0 and realized_gl < 0.7 * REWARD_MULT):
                    self.gl_level = 4
                elif(realized_gl > 0.7 * REWARD_MULT and realized_gl < 1.4 * REWARD_MULT):
                    self.gl_level = 5
                else:
                    self.gl_level = 6
        
        # Observer Level 2
        # if(OBS_LEVEL):
        #     # Reset to neutral gl observation level
        #     self.gl_level = 3
        #     if(self.holding):
        #         # Calculate realized gain/loss
        #         current_gain_loss = self.previous_price - self.buy_price
        #         realized_gl = (current_gain_loss * self.amount_bought) - 0.4
        #         # Determine gl observation level
        #         if(realized_gl < -1.4):
        #             self.gl_level = 0
        #             self.reward -= 10
        #         elif(realized_gl < -0.7 and realized_gl > -1.4):
        #             self.gl_level = 1
        #             self.reward -= 5
        #         elif(realized_gl < 0 and realized_gl > -0.7):
        #             self.gl_level = 2
        #             self.reward -= 1
        #         elif(realized_gl > 0 and realized_gl < 0.7):
        #             self.gl_level = 4
        #             self.reward += 1
        #         elif(realized_gl > 0.7 and realized_gl < 1.4):
        #             self.gl_level = 5
        #             self.reward += 1
        #         else:
        #             self.gl_level = 6
        #             self.reward += 1
        

        # Get next set of data
        self.observation = next(self.reader)

        # Remove current price
        self.previous_price = float(self.observation.pop(0))

        # Remove ma50 and ema50
        # self.observation.pop(self.remove_int)
        # self.observation.pop(self.remove_int)

        # Add current gain/loss level
        if(OBS_LEVEL):
            self.observation.append(self.gl_level)

        # # Add Hold to observation
        # if(self.holding):
        #     self.observation.append(1)
        # else:
        #     self.observation.append(0)

        # # Add number of steps held
        # self.observation.append(self.hold_steps)
            
        # Add wait buy and hold steps to observation
        # self.observation.append(self.wait_buy)
        # self.observation.append(self.hold_steps)
            
        # Add gain loss of observation
        # gain_loss = self.previous_price - self.buy_price
        # realized_gl = (gain_loss * self.amount_bought) - 0.4
        # self.observation.append(realized_gl)

        # Convert data to float
        self.observation = str_to_float(self.observation)

        # Convert to Onehot encoding
        # self.observation = keras.utils.to_categorical(self.observation, num_classes=CLASSIFICATIONS).reshape((SHAPE,))

        return self.observation, self.reward, self.done, self.truncated, info

    def reset(self, seed=None, options=None):
        info = {}
        #self.done = False
        self.buy_price = 0
        self.holding = False
        self.reward = 0
        self.amount_bought = 0

        
        # Get next set of data
        if(not self.done):
            next(self.reader)
            self.observation = next(self.reader)

            # Remove current price
            self.previous_price = float(self.observation.pop(0))

            # Remove ma50 and ema50
            # self.observation.pop(self.remove_int)
            # self.observation.pop(self.remove_int)

            # # Add current gain/loss level
            if(OBS_LEVEL):
                self.gl_level = 3
                self.observation.append(self.gl_level)

            # # Add Hold to observation
            # self.observation.append(0)

            # # Add number of steps held
            # self.observation.append(self.hold_steps)
                
            # Add gain loss of observation
            # gain_loss = self.previous_price - self.buy_price
            # realized_gl = (gain_loss * self.amount_bought) - 0.4
            # self.observation.append(realized_gl)
                
            # Add wait buy and hold steps to observation
            # self.observation.append(self.wait_buy)
            # self.observation.append(self.hold_steps)

            # Convert data to float
            self.observation = str_to_float(self.observation)

            # self.observation = keras.utils.to_categorical(self.observation, num_classes=CLASSIFICATIONS).reshape((SHAPE,))
        
        self.done = False

        return self.observation, info

    # Close File
    def close(self):
        self.file.close()