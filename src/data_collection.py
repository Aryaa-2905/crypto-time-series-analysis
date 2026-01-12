import yfinance as yf
import pandas as pd

def fetch_crypto_data(symbol="BTC-USD", start="2018-01-01"):
    df = yf.download(symbol, start=start)
    df.reset_index(inplace=True)
    return df

if __name__ == "__main__":
    btc_data = fetch_crypto_data()
    btc_data.to_csv("data/btc.csv", index=False)
    print("Bitcoin data saved successfully!")
