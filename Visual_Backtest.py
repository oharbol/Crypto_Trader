from Backtester import Backtester

# Global const
AUTO_CYCLE = True
DATA_CSV = "Data/Data_Raw_OMA_ETH_30Min"
TIMESTEPS = 13322

MODEL_NAME = "DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6"
MODEL_ZIP = "DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_13322"
SCORE = 100

# Create Backtester class
tester = Backtester(DATA_CSV, TIMESTEPS, True, write_data=False)

# Auto cycle through all model zip files
if(AUTO_CYCLE):
    step = 5302156
    zip_file = f"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_{step}"
    while(True):
        
        # Show graph
        tester.Backtest(MODEL_NAME, zip_file, score=SCORE)
        print(zip_file)
        step += TIMESTEPS
        zip_file = f"DQN_ETH_sh23_30Min_OMARaw_Mult1_Reward7_norestart_obslevel_score100_6_{step}"

# Show one graph
else:
    # Show graph
    tester.Backtest(MODEL_NAME, MODEL_ZIP, score=SCORE)