import pandas as pd
import numpy as np
import os
from datetime import datetime
import csv

from stable_baselines3 import PPO

# from SB_Crypto_env import CryptoEnv
from SB_Crypto_env_new import CryptoEnv

DATA_CSV = "Data/Data_Raw_OMA_BTC_30Min"
TIME_STEPS = 52000
CASH = 100

# Convert Observation space into floats
# Return as np array
def str_to_float(data_list):
    return np.array([float(i) for i in data_list])

# Observer Level 1
def obs_level(holding, realized_gl):
    # Reset to neutral gl observation level
    gl_level = 3
    if(holding):
        # Determine gl observation level
        if(realized_gl < -1.4):
            gl_level = 0
        elif(realized_gl < -0.7  and realized_gl > -1.4):
            gl_level = 1
        elif(realized_gl < 0 and realized_gl > -0.7):
            gl_level = 2
        elif(realized_gl > 0 and realized_gl < 0.7):
            gl_level = 4
        elif(realized_gl > 0.7 and realized_gl < 1.4):
            gl_level = 5
        else:
            gl_level = 6
    
    return gl_level

models_dir = "models"
model_name = "PPO_BTC_30Min_OMARaw_Reward6_obslevel_score20_1"
model_zip = "PPO_BTC_30Min_OMARaw_Reward6_obslevel_score20_1_3949720"

env = CryptoEnv()

# Global Variables
current_price = 0
buy_price = 0
holding = False
profit = 0
amount_bought = 0


model_path = f"{models_dir}/{model_name}/{model_zip}.zip"

model = PPO.load(model_path)

file = open("{}.csv".format(DATA_CSV))
reader = csv.reader(file)
next(reader)

data = pd.DataFrame()
data["Profits"] = [0]


# Main Loop
for i in range(TIME_STEPS):

    # read line
    line = next(reader)
    current_price = float(line.pop(0))

    # Add extra observations
    gain_loss = current_price - buy_price
    realized_gl = (gain_loss * amount_bought) - 0.4
    line.append(obs_level(holding, realized_gl))
    
    # Convert csv data
    line = str_to_float(line)

    # make prediction
    action, _states = model.predict(line)

    # Evaluate action
    # Buy
    if(action == 0 and not holding):
        # - Update holding
        holding = True
        buy_price = current_price
        amount_bought = CASH / buy_price

    # Sell
    elif(action == 1 and holding):
        # - Update holding
        holding = False
        # - Update profit
        # gain_loss = current_price - buy_price
        # realized_gl = (gain_loss * amount_bought) - 0.4
        profit += realized_gl
        
    # Hold
    # - Nothing
    thing = pd.DataFrame({"Profits" : [profit]})
    data = pd.concat([data, thing], axis=0)

# Wrap it up
file.close()
data.to_csv("Backtest_Data.csv")