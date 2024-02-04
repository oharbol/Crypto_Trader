from Backtester import Backtester
from SB_Crypto_env_new import CryptoEnv
from stable_baselines3 import PPO, DQN
import matplotlib.pyplot as plt

import numpy as np
from collections import deque 

# Global const
AUTO_CYCLE = True
DATA_CSV = "Data/Data_Raw_OMA_ETH_30Min"
TIMESTEPS = 53290
ACTION_TOLLERANCE = 1

MODEL_NAME = "DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6"
SCORE = 100

model_zips = ["DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_53288",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_79932",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_2744332",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_2917518",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_2944162",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_3303856",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_3397110",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_3410432",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_3623584",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_3903346",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_4689344",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_4862530",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_4915818",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_4969106",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_5128970",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_5262190",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_5275512",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_5328800",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_5342122",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_5488664",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_5515308",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_5555274",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_5661850",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_5954934",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_6114798",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_6407882",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_6434526",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_6474492",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_6554424",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_6594390",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_6621034",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_6634356",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_6647678",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_6687644",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_6820864",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_6940762",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_7100626",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_7140592",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_7180558",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_7193880",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_7207202",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_7247168",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_7300456",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_7340422",
"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_7353744"]

# Plot variables:
x = np.arange(0, TIMESTEPS, 1)
y = deque([])

plt.ylabel("Profit")
plt.xlabel("Timesteps")

env = CryptoEnv(DATA_CSV, TIMESTEPS, score = SCORE)

# Initialize environment
observation, info = env.reset()

# Create Backtester class
tester = Backtester(DATA_CSV, TIMESTEPS, True, write_data=False)

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
    profit, observation = tester.Single_Backtest(env, action)

    # Set graphing variable
    y.append(profit)

    # Reset variabels
    action_pool = 0

    if(j % 1000 == 0):
        print(j)

# Graph data
plt.plot(x, y)
plt.show()