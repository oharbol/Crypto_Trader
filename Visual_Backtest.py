from Backtester import Backtester

# Global const
DATA_CSV = "Data/Data_Raw_OMA_ETH_30Min"
TIMESTEPS = 53290

MODEL_NAME = "PPO_ETH_sh23_30Min_OMARaw_Reward6_obslevel_score20_2"
MODEL_ZIP = "PPO_ETH_sh23_30Min_OMARaw_Reward6_obslevel_score20_2_2860000"
SCORE = 20

# Create Backtester class
tester = Backtester(DATA_CSV, TIMESTEPS, True, write_data=False)

# Show graph
tester.Backtest(MODEL_NAME, MODEL_ZIP, score=SCORE)