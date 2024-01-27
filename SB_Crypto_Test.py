import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from sb3_contrib import RecurrentPPO, QRDQN, ARS
import os
# from SB_Crypto_env import CryptoEnv
from SB_Crypto_env_new import CryptoEnv

import random

import tensorflow as tf
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        trades = model.env.get_attr("num_trades").pop()
        score = model.env.get_attr("score").pop()
        restart = model.env.get_attr("restart").pop()

        profit = model.env.get_attr("profit").pop()
        self.logger.record("Data/profit", profit)

        win = model.env.get_attr("wins").pop()
        loss = model.env.get_attr("losses").pop()
        self.logger.record("Data/WL Ratio", win / trades)

        avg_win = model.env.get_attr("avg_win").pop()
        avg_loss = model.env.get_attr("avg_loss").pop()
        self.logger.record("Data/avg_win", avg_win)
        self.logger.record("Data/avg_loss", avg_loss)

        
        self.logger.record("Supplementary Data/trades", trades)
        self.logger.record("Supplementary Data/num_wins", win)
        self.logger.record("Supplementary Data/num_loss", loss)
        self.logger.record("Supplementary Data/score", score)
        self.logger.record("Supplementary Data/reset_nums", restart)

        return True
    

SAVE_MODEL = True

# Naming Convention
# "Model_Timeframe_data source_SHAPE_Reward Function_added observations_#itteration"
model_name = "DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward6_obslevel_score75_1"
models_dir = f"models/{model_name}"
logdir = "logs"


# Model Name creation
# Used for automating model runs
def create_model_name(model_name, iteraion):
    return model_name + str(iteraion)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create the environment
env = CryptoEnv()  # continuous: LunarLanderContinuous-v2
#load env

'''
model_path = f"{models_dir}/170000.zip"

model - PPO.load(model_path, env=env)
--------------------------------------------
'''
# models_dir_load = "models/PPO_Delta_Segments_ApplePos_Steps2_OGReward"

# model_path = f"{models_dir_load}/490000.zip"
# model = PPO.load(model_path, env=env)


# Required before you can step the environment
env.reset()

# Models
# model = A2C("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
# model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
model = DQN("MlpPolicy", env, verbose=0, exploration_fraction=0.9, exploration_final_eps=0, batch_size=128, tensorboard_log=logdir)
# model = RecurrentPPO("MlpLstmPolicy", env, verbose=0, tensorboard_log=logdir)
# model = QRDQN("MlpPolicy", env, verbose=0, exploration_fraction=0.5, batch_size=128, tensorboard_log=logdir)

# 4 Million total timesteps
TIMESTEPS = 53290
for i in range(316):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name= model_name, callback=TensorboardCallback())
    if(SAVE_MODEL):
        model.save(f"{models_dir}/{model_name}_{TIMESTEPS*i}")
#model.learn(total_timesteps=290000, reset_num_timesteps=False, tb_log_name= model_name, callback=TensorboardCallback())

env.close()

#tensorboard --logdir=logs

#cd C:\Users\oharb\OneDrive\Documents\Python Scripts\Crypto_Trader
#python SB_Crypto_Test.py