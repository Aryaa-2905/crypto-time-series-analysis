import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load data
df = pd.read_csv("data/btc.csv", parse_dates=["Date"])

# Keep required columns
df = df[["Date", "Close"]]

# Rename for Prophet
df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

# Ensure numeric
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df.dropna(inplace=True)

# Initialize Prophet model
model = Prophet()
model.fit(df)

# Create future dataframe (next 30 days)
future = model.make_future_dataframe(periods=30)

# Forecast
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title("Bitcoin Price Forecast using Prophet")
plt.show()
