import gymnasium as gym
import numpy as np
from gymnasium import spaces
# from collections import deque

# Only used for OneHot encoding
# import keras

import csv

# IDEA CORNER
# Create a reward for buying after selling to mimic a short trade
# Increase the number of thresholds for observation levels

# Global consts
SHAPE = 24
CASH = 100
REWARD_MULT = 3

OBS_LEVEL = True
CLASSIFICATIONS = 7

# Convert Observation space into floats
def str_to_float(data_list):
    return np.array([float(i) for i in data_list])


# Custom Environment Class
class CryptoEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, data_csv, timesteps, score = 20):
        super().__init__()
        # Const data points
        self.SCORE = score
        self.TIMESTEPS = timesteps #61500
        self.DATA_CSV = data_csv

        self.truncated = False
        self.file = open("{}.csv".format(self.DATA_CSV))
        self.reader = csv.reader(self.file)
        self.profit = 0
        self.done = False
        self.steps = timesteps
        self.step_scorereset = 0
        self.hold_steps = 0

        # Log related code
        self.wins = 1
        self.losses = 1
        self.avg_win = 0
        self.avg_loss = 0
        self.avg_win_total = 0
        self.avg_loss_total = 0
        self.num_trades = 1
        self.score = 0
        self.restart = 0
        self.score_wins = 0
        self.score_losses = 0

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-1, high=1,
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

        # SELL
        elif(action == 2 and self.holding):
            self.num_trades += 1
            #self.hold_steps = 0

            # Sell reward
            gain_loss = self.previous_price - self.buy_price
            realized_gl = (gain_loss * self.amount_bought) - 0.4
            self.buy_price = 0
            self.holding = False


            # REWARDS

            # Reward 6: Levels of earning (6) yeild flat reward to incentivise 
            # Combined with reward of -0.01 every hold action
            # Worse profits yield lower reward, all positive gains are same reward

            #TODO: Gabe suggestion: Increase thresholds
            # if(realized_gl < -1.4 * REWARD_MULT):
            #     self.reward = -15
            #     self.score -= 1
            # elif(realized_gl < -0.7 * REWARD_MULT and realized_gl > -1.4 * REWARD_MULT):
            #     self.reward = -10
            #     self.score -= 1
            # elif(realized_gl < 0 and realized_gl > -0.7 * REWARD_MULT):
            #     self.reward = -5
            #     self.score -= 1
            # elif(realized_gl > 0 and realized_gl < 0.7 * REWARD_MULT):
            #     self.reward = 1
            #     self.score += 1
            # elif(realized_gl > 0.7 * REWARD_MULT and realized_gl < 1.4 * REWARD_MULT):
            #     self.reward = 2
            #     self.score += 1
            # else:
            #     self.reward = 3
            #     self.score += 1

            # Reward 7
            if(realized_gl < -1.4 * REWARD_MULT):
                self.reward = -30
                self.score -= 3
            elif(realized_gl < -0.7 * REWARD_MULT and realized_gl > -1.4 * REWARD_MULT):
                self.reward = -20
                self.score -= 2
            elif(realized_gl < 0 and realized_gl > -0.7 * REWARD_MULT):
                self.reward = -10
                self.score -= 1
            elif(realized_gl > 0 and realized_gl < 0.7 * REWARD_MULT):
                self.reward = 5
                self.score += 1
            elif(realized_gl > 0.7 * REWARD_MULT and realized_gl < 1.4 * REWARD_MULT):
                self.reward = 10
                self.score += 2
            else:
                self.reward = 15
                self.score += 3


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
        # if(self.holding):
        #     self.hold_steps += 1
        #     # gain_loss = self.previous_price - self.buy_price
        #     # realized_gl = (gain_loss * self.amount_bought) - 0.4

        #     # For some reason this value is needed to create somewhat positive results
        #     self.reward = -0.01 * self.hold_steps
        
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
        
        # End game after end of data or set time
        if(self.steps >= self.TIMESTEPS or self.steps - self.step_scorereset >= 15375): #self.TIMESTEPS
            self.done = True
            
            # Set reward
            self.reward = self.score * 100

            self.score = 0

            # Update score reset value
            self.step_scorereset = self.steps

        # Check if game is won
        if(self.score >= 15): #self.SCORE
            self.done = True
            self.reward = 100 * self.SCORE

            # Update score reset value
            self.step_scorereset = self.steps

            # Used for logging wins from scores
            self.score_wins += 1
        elif(self.score <= -self.SCORE):
            self.done = True
            self.reward = -100 * self.SCORE

            # Update score reset value
            self.step_scorereset = self.steps

            # Used for logging losing from scores
            self.score_losses += 1

        
        # Get next set of data
        self.observation = next(self.reader)

        # Remove current price
        self.previous_price = float(self.observation.pop(0))

        # Add current gain/loss level
        if(OBS_LEVEL):
            self.observation.append(self.gl_level)

        # Add score
        self.observation.append(self.score)

        # Add Hold to observation
        if(self.holding):
            self.observation.append(1)
        else:
            self.observation.append(0)

        # Add time remaining

        # Convert data to float
        self.observation = str_to_float(self.observation)

        # Saved code for fine tuning and testing different observations

        # Remove ma50 and ema50
        # self.observation.pop(self.remove_int)
        # self.observation.pop(self.remove_int)

        # # Add number of steps held
        # self.observation.append(self.hold_steps)
            
        # Add wait buy and hold steps to observation
        # self.observation.append(self.wait_buy)
        # self.observation.append(self.hold_steps)
            
        # Add gain loss of observation
        # gain_loss = self.previous_price - self.buy_price
        # realized_gl = (gain_loss * self.amount_bought) - 0.4
        # self.observation.append(realized_gl)

        # Convert to Onehot encoding
        # self.observation = keras.utils.to_categorical(self.observation, num_classes=CLASSIFICATIONS).reshape((SHAPE,))

        return self.observation, self.reward, self.done, self.truncated, info

    def reset(self, seed=None, options=None):
        #TODO: remove the need to create a new observation during reset to make backtesting more accurate
        info = {}
        self.done = False
        self.buy_price = 0
        self.holding = False
        self.reward = 0
        self.amount_bought = 0
        
        self.score = 0
        self.hold_steps = 0
        self.restart += 1

        #TODO: Change this to only close and reopen file when end is reached.
        # Still give reward for winning or losing by score
        # End game after set time
        if(self.steps >= self.TIMESTEPS):
            # Reset step counters
            self.steps = 0
            self.step_scorereset = 0

            # Reset data file
            self.file.close()
            self.file = open("{}.csv".format(self.DATA_CSV))
            self.reader = csv.reader(self.file)
            next(self.reader)

        self.observation = next(self.reader)
        self.steps += 1

        # Remove current price
        self.previous_price = float(self.observation.pop(0))

        # Add current gain/loss level
        if(OBS_LEVEL):
            self.gl_level = 3
            self.observation.append(self.gl_level)
            
        # Add score
        self.observation.append(self.score)

        # Add Hold to observation
        if(self.holding):
            self.observation.append(1)
        else:
            self.observation.append(0)

        # Convert data to float
        self.observation = str_to_float(self.observation)
        
        self.done = False

        # Saved code for fine tuning and testing different observations

        # Remove ma50 and ema50
        # self.observation.pop(self.remove_int)
        # self.observation.pop(self.remove_int)

        

        # # Add number of steps held
        # self.observation.append(self.hold_steps)
            
        # Add wait buy and hold steps to observation
        # self.observation.append(self.wait_buy)
        # self.observation.append(self.hold_steps)
            
        # Add gain loss of observation
        # gain_loss = self.previous_price - self.buy_price
        # realized_gl = (gain_loss * self.amount_bought) - 0.4
        # self.observation.append(realized_gl)

        # Convert to Onehot encoding
        # self.observation = keras.utils.to_categorical(self.observation, num_classes=CLASSIFICATIONS).reshape((SHAPE,))

        return self.observation, info

    # Close File
    def close(self):
        self.file.close()