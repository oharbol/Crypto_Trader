import numpy as np
#import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
#from keras.optimizers import *
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
from tqdm import tqdm
import keras
import random
import pandas as pd

def convert(input):
    state = input.split(",")
    state = [np.float64(i) for i in state]
    state_onehot = tf.keras.utils.to_categorical(state[1], num_classes=EMA_INPUT_SIZE).reshape((2,))
    state_onehot = np.concatenate((tf.keras.utils.to_categorical(state[2], num_classes=HA_INPUT_SIZE).reshape((4,)), state_onehot), axis=None)
    state_onehot = np.concatenate((tf.keras.utils.to_categorical(state[3], num_classes=ADX_INPUT_SIZE).reshape((2,)), state_onehot), axis=None)
    state_onehot = [state[0]] + state_onehot.tolist()
    return np.array(state_onehot)

# Create the connection to the api
# api = tradeapi.REST(config.API_KEY, config.SECRET_KEY, config.BASE_URL)
ticker = "BTC"
graph_bank = 1000
bank = 1000
holding = False
cost = 0
shares = 0
buy_amount = 0
#prev_vol = 0

LOAD_MODEL = "models/250ep__2X64_raw_1___310.54max__235.64avg___49.79min.model" #Or None

model = load_model(LOAD_MODEL)
model.summary()

# Open one hot data for RL model
data = pd.read_csv(f"Data_OneHot_{ticker}_Short")

# Open backtest bank data
with open(f"ML_BackTest_{ticker}.csv", 'a', newline= '') as csvfile:
    writer = csv.writer(csvfile)

    for i in data.iterrows():
        action = np.argmax(model.predict(np.array(state).reshape(-1, *state.shape)))
        writer.writerow([i["date"], graph_bank, price_gain])