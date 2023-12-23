from stable_baselines3.common.env_checker import check_env
from SB_Crypto_env import CryptoEnv

env = CryptoEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)