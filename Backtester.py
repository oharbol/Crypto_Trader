import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque 

from stable_baselines3 import PPO, DQN

# from SB_Crypto_env import CryptoEnv
from SB_Crypto_env_new import CryptoEnv

# TODO: Learn about overloading in Python and add compatability to have
# Backtest method take model directory or the loaded model

# TODO: Make plt a global variable so that user can plot when desired

class Backtester():
    
    def __init__(self, data_csv, time_steps, render_graph, write_data = False, autosave_data = False, print_data = True) -> None:
        self.DATA_CSV = data_csv
        self.TIME_STEPS = time_steps
        # Used for automation and visual of data
        self.render = render_graph
        self.write_data = write_data
        self.print_data = print_data
        self.save = autosave_data
        self.wins = 0
        self.loss = 0
        self.prev_profit = 0

    # Full backtesting loop against one model
    def Backtest(self, model_name, model_zip, models_dir = "models", backtest_name = "Backtest_Data", score = 20) -> int:
        # Variables for graph
        if(self.render):
            x = np.arange(0, self.TIME_STEPS, 1)
            y = deque([])

            plt.ylabel("Profit")
            plt.xlabel("Timesteps")

        # Handle getting trained model
        model_path = f"{models_dir}/{model_name}/{model_zip}.zip"
        #TODO: Make switch case for model to load
        model = DQN.load(model_path)

        # Create environment
        env = CryptoEnv(self.DATA_CSV, self.TIME_STEPS, score=score)

        # Initialize environment
        observation, info = env.reset()

        # Create csv for collection of backtesting data
        if(self.write_data):
            f = open(f"{backtest_name}.csv", 'w')

        # Main Backtesting Loop
        for i in range(self.TIME_STEPS):
            
            # Make prediction
            action, _states = model.predict(observation, deterministic=True)

            # Get observation from step
            observation, reward, done, truncated, info = env.step(action)

            data = round(env.profit,2)

            # Gather print data
            if(self.print_data):
                # Winning trade
                if(self.prev_profit < env.profit):
                    self.wins += 1
                # Losing trade
                elif(self.prev_profit > env.profit):
                    self.loss += 1
            
            self.prev_profit = env.profit

            # Record profit data
            if(self.write_data):
                f.write(str(data) + "\n")

            # Add data to matplot for visual graph
            y.append(data)

            # Reset as needed
            if(done):
                env.reset()

        # Close file
        if(self.write_data):
            f.close()

        # Print data
        if(self.print_data):
            print(f"Win/Loss Rate: {round((self.wins / (self.wins + self.loss)) * 100, 2)}%")
            
        # Show graph
        if(self.render):
            plt.title(model_zip)
            plt.plot(x, y)
            plt.show()

            # Prompt to keep backteseting data
            if(self.write_data and not self.save):
                keep_data = str.lower(input("Keep CSV file? (Y)es/No: "))

                # Delete file
                if(len(keep_data) == 0 or keep_data[0] != 'y'):
                    print("\nDeleted")
                    os.remove(f"{backtest_name}.csv")
                # Save file
                else:
                    print("\nSaved")

        # Wrap it up
        env.close()

        # Returns last datapoint
        return y[-1]
    
    # Single step of the Backtest
    # Requires loop to be created
    def Single_Backtest(self, env, action):

        # Get observation from step
        observation, reward, done, truncated, info = env.step(action)

        data = round(env.profit,2)

        # Gather print data
        if(self.print_data):
            # Winning trade
            if(self.prev_profit < env.profit):
                self.wins += 1
            # Losing trade
            elif(self.prev_profit > env.profit):
                self.loss += 1
        
        self.prev_profit = env.profit

        # Fix:
        # Record profit data
        # if(self.write_data):
        #     f.write(str(data) + "\n")

        # Fix:
        # Add data to matplot for visual graph
        # y.append(data)

        # Reset as needed
        if(done):
            env.reset()
        
        return data, observation