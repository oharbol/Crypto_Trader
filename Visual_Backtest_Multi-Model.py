from Backtester import Backtester
from SB_Crypto_env_new import CryptoEnv
from stable_baselines3 import PPO, DQN
import matplotlib.pyplot as plt

import numpy as np
from collections import deque 

# Global const
DATA_CSV = "Data/Data_Normalized_OMA_ETH_30Min"
TIMESTEPS = 53870
ACTION_TOLLERANCE = 1

MODEL_NAME = "DQN_ETH_sh24_30MinNorm_OMARaw_Mult1_Reward7_norestart_obslevel_hold_score100_1"
SCORE = 100

model_zips = ["DQN_ETH_sh24_30MinNorm_OMARaw_Mult1_Reward7_norestart_obslevel_hold_score100_1_94262",
"DQN_ETH_sh24_30MinNorm_OMARaw_Mult1_Reward7_norestart_obslevel_hold_score100_1_269320",
"DQN_ETH_sh24_30MinNorm_OMARaw_Mult1_Reward7_norestart_obslevel_hold_score100_1_350116",
"DQN_ETH_sh24_30MinNorm_OMARaw_Mult1_Reward7_norestart_obslevel_hold_score100_1_403980",
"DQN_ETH_sh24_30MinNorm_OMARaw_Mult1_Reward7_norestart_obslevel_hold_score100_1_2545074",
"DQN_ETH_sh24_30MinNorm_OMARaw_Mult1_Reward7_norestart_obslevel_hold_score100_1_2572006",
"DQN_ETH_sh24_30MinNorm_OMARaw_Mult1_Reward7_norestart_obslevel_hold_score100_1_2598938",
"DQN_ETH_sh24_30MinNorm_OMARaw_Mult1_Reward7_norestart_obslevel_hold_score100_1_2625870",
"DQN_ETH_sh24_30MinNorm_OMARaw_Mult1_Reward7_norestart_obslevel_hold_score100_1_2639336",
"DQN_ETH_sh24_30MinNorm_OMARaw_Mult1_Reward7_norestart_obslevel_hold_score100_1_2693200",
"DQN_ETH_sh24_30MinNorm_OMARaw_Mult1_Reward7_norestart_obslevel_hold_score100_1_2760530",
"DQN_ETH_sh24_30MinNorm_OMARaw_Mult1_Reward7_norestart_obslevel_hold_score100_1_2787462"]

# Plot variables:
x = np.arange(0, TIMESTEPS, 1)
y = deque([])

plt.ylabel("Profit")
plt.xlabel("Timesteps")

env = CryptoEnv(DATA_CSV, TIMESTEPS, score = SCORE)

# Initialize environment
observation, info = env.reset()

# Create Backtester class
# tester = Backtester(DATA_CSV, TIMESTEPS, True, write_data=False)

# Load all models into list
models = []

for i in model_zips:
    # Handle getting trained model
    model_path = f"models/{MODEL_NAME}/{i}.zip"
    models.append(DQN.load(model_path))

action_pool = 0

# Cycle through all timesteps
for j in range(TIMESTEPS):
    
    for k in models:
        temp = k.predict(observation, deterministic=True)
        # BUY
        if(temp == 0):
            action_pool += 1
        # SELL
        elif(temp == 2):
            action_pool -= 1
    
    # Handle creating action
    # BUY
    if(action_pool >= ACTION_TOLLERANCE):
        action = 0
    # SELL
    elif(action_pool <= ACTION_TOLLERANCE):
        action = 2
    # HOLD
    else:
        action = 1

    # Take step
    #profit, observation = tester.Single_Backtest(env, action)
    observation, reward, done, truncated, info = env.step(action)

    # Set graphing variable
    y.append(env.profit)

    # Reset variabels
    print(observation)
    action_pool = 0

    # Debugging time to complete
    if(j % 1000 == 0):
        print(j)

# Graph data
plt.plot(x, y)
plt.show()