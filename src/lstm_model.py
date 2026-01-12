import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data
df = pd.read_csv("data/btc.csv", parse_dates=["Date"])
df = df[["Date", "Close"]]
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df.dropna(inplace=True)

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[["Close"]])

# Create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

# Reshape for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Predict
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted.reshape(-1, 1))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot
plt.figure(figsize=(12,5))
plt.plot(actual, label="Actual Price")
plt.plot(predicted, label="Predicted Price")
plt.title("Bitcoin Price Prediction using LSTM")
plt.legend()
plt.show()
