from Backtester import Backtester

# Global const
DATA_CSV = "Data/Data_Raw_OMA_ETH_30Min"
TIMESTEPS = 53290

MODEL_NAME = "DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward6_obslevel_score75_1"
MODEL_ZIP = "DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward6_obslevel_score75_1_2078310"
SCORE = 10

# Create Backtester class
tester = Backtester(DATA_CSV, TIMESTEPS, True, write_data=False)

tester.Backtest(MODEL_NAME, MODEL_ZIP, score=SCORE)