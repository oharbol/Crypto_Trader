import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from sb3_contrib import RecurrentPPO, QRDQN, ARS
import os
from SB_Crypto_env import CryptoEnv

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
        profit = model.env.get_attr("profit").pop()
        self.logger.record("Data/profit", profit)

        win = model.env.get_attr("wins").pop()
        loss = model.env.get_attr("losses").pop()
        self.logger.record("Data/WL Ratio", win / (win + loss))

        avg_win = model.env.get_attr("avg_win").pop()
        avg_loss = model.env.get_attr("avg_loss").pop()
        self.logger.record("Data/avg_win", avg_win)
        self.logger.record("Data/avg_loss", avg_loss)

        trades = model.env.get_attr("num_trades").pop()
        self.logger.record("Supplementary Data/trades", trades)

        self.logger.record_mean("Supplementary Data/avg_win", avg_win)

        return True
    

SAVE_MODEL = True
OVERALL = 3

# Naming Convention
# "Model_Timeframe_data source_Reward Function_added observations_#itteration"


# Model Name creation
# Used for automating model runs
def run_iter(model_name, model, env):

    models_dir = f"models/{model_name}"
    logdir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)



    TIMESTEPS = 51400
    for i in range(79):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name= model_name, callback=TensorboardCallback())
        if(SAVE_MODEL):
            model.save(f"{models_dir}/{model_name}_{TIMESTEPS*i}")



for i in range(7):
    model_name = f"PPO_30Min_OMARaw_Reward5_obslevel_remove{i}_{OVERALL}"
    env = CryptoEnv(7 + (i * 2))

    env.reset()
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="logs")

    run_iter(model_name, model, env)

    env.close()



#tensorboard --logdir=logs

#cd OneDrive/Documents/"Python Scripts"/Crypto
#python SB_Crypto_Test.py