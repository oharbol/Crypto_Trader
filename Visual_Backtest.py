from Backtester import Backtester

# Global const
AUTO_CYCLE = True
DATA_CSV = "Data/Data_Raw_OMA_ETH_1Hour"
TIMESTEPS = 26840

MODEL_NAME = "DQN_ETH_sh23_1Hour_OMARaw_Mult1_Reward7_norestart_obslevel_score100_1"
MODEL_ZIP = "DQN_ETH_sh23_1Hour_OMARaw_Mult1_Reward7_norestart_obslevel_score100_1_26840"
SCORE = 100

# Create Backtester class
tester = Backtester(DATA_CSV, TIMESTEPS, True, write_data=False)

# Auto cycle through all model zip files
if(AUTO_CYCLE):
    step = TIMESTEPS
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