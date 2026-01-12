import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load data
df = pd.read_csv("data/btc.csv", parse_dates=["Date"])

# Keep only required columns
df = df[["Date", "Close"]]

# Convert Close to numeric (IMPORTANT)
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Drop missing values
df.dropna(inplace=True)

# Set Date as index
df.set_index("Date", inplace=True)

# Time series
series = df["Close"]

# Train-test split
train = series[:-30]
test = series[-30:]

# Train ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=30)

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(train, label="Training Data")
plt.plot(test, label="Actual Price")
plt.plot(forecast, label="Forecasted Price")
plt.title("Bitcoin Price Forecast using ARIMA")
plt.legend()
plt.show()
