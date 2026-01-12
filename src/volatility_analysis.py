import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/btc.csv", parse_dates=["Date"])
df = df[["Date", "Close"]]
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df.dropna(inplace=True)
df.set_index("Date", inplace=True)

# Calculate daily returns
df["Daily_Return"] = df["Close"].pct_change()

# Rolling volatility (30-day)
df["Volatility_30"] = df["Daily_Return"].rolling(window=30).std()

# Plot volatility
plt.figure(figsize=(12,5))
plt.plot(df["Volatility_30"], label="30-Day Volatility")
plt.title("Bitcoin Price Volatility (30-Day Rolling)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.show()
