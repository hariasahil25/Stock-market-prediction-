from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

def build_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(60,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    X = np.random.rand(100,60,1)
    y = np.random.rand(100,1)

    model = build_model()
    model.
  fit(X, y, epochs=2)
