from Backtester import Backtester

# Global const
AUTO_CYCLE = True
DATA_CSV = "Data/Data_Normalized_OMA_ETH_30Min"
TIMESTEPS = 13466

MODEL_NAME = "DQN_ETH_sh24_30MinNorm_OMARaw_Mult1_Reward7_norestart_obslevel_hold_score100_1"
MODEL_ZIP = "DQN_ETH_sh24_30MinNorm_OMARaw_Mult1_Reward7_norestart_obslevel_hold_score100_1_13466"
SCORE = 100

# Create Backtester class
tester = Backtester(DATA_CSV, TIMESTEPS, True, write_data=False)

# Auto cycle through all model zip files
if(AUTO_CYCLE):
    step = 11136382
    zip_file = f"{MODEL_NAME}_{step}"
    while(True):
        
        # Show graph
        tester.Backtest(MODEL_NAME, zip_file, score=SCORE)
        print(zip_file)
        step += TIMESTEPS
        zip_file = f"{MODEL_NAME}_{step}"

# Show one graph
else:
    # Show graph
    tester.Backtest(MODEL_NAME, MODEL_ZIP, score=SCORE)