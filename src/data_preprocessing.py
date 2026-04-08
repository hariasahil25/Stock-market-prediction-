import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess(data, seq_len=60):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])

    return np.array(X), np.array(y), scaler
