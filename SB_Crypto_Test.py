from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from sb3_contrib import RecurrentPPO, QRDQN, ARS
import os

# from SB_Crypto_env import CryptoEnv
from SB_Crypto_env_new import CryptoEnv
from Backtester import Backtester

from stable_baselines3.common.callbacks import BaseCallback

# Custom TensorboardCallBack to create custom variables to track
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

        score_wins = model.env.get_attr("score_wins").pop()
        score_losses = model.env.get_attr("score_losses").pop()

        self.logger.record("Supplementary Data/score_wins", score_wins)
        self.logger.record("Supplementary Data/score_losses", score_losses)

        return True


#TODO Add backtester
    
# Global consts for training
SAVE_MODEL = True
TIMESTEPS = 344610
# 5 MIN - 344610
# 30Min - 61500
DATA_CSV = "Data/Data_Raw_OMA_ETH_5Min"
SCORE = 20

# Naming Convention
# "Model_Timeframe_data source_SHAPE_Reward Function_added observations_#itteration"
model_name = "DQN_ETH_sh24_5Min_OMARaw_Reward8_1"
models_dir = f"models/{model_name}"
logdir = "logs"


if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create the environment
env = CryptoEnv(DATA_CSV, TIMESTEPS, SCORE)

# Required before you can step the environment
env.reset()

# Models
# model = A2C("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
# model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
model = DQN("MlpPolicy", env, verbose=0, exploration_fraction=0.95, exploration_final_eps=0.0005, batch_size=512, tensorboard_log=logdir) # exploration_fraction=0.95 batch_size=256
# model = RecurrentPPO("MlpLstmPolicy", env, verbose=0, tensorboard_log=logdir)
# model = QRDQN("MlpPolicy", env, verbose=0, exploration_fraction=0.5, batch_size=128, tensorboard_log=logdir)

# 7 Million Timesteps
for i in range(10):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name= model_name, callback=TensorboardCallback())
    if(SAVE_MODEL):
        model.save(f"{models_dir}/{model_name}_{TIMESTEPS*i}")
#model.learn(total_timesteps=290000, reset_num_timesteps=False, tb_log_name= model_name, callback=TensorboardCallback())

env.close()

#tensorboard --logdir=logs

#cd C:\Users\oharb\OneDrive\Documents\Python Scripts\Crypto_Trader
#python SB_Crypto_Test.py