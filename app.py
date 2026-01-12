import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Crypto Time Series Analysis",
    layout="wide"
)

st.title("📈 Cryptocurrency Time Series Analysis")
st.write("Forecasting Bitcoin prices using ARIMA, Prophet, and LSTM")

# ---------------- DATA FUNCTIONS ----------------
def load_live_btc_data(period="5y"):
    df = yf.download("BTC-USD", period=period)

    # FIX: flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    return df


def clean_price_data(df):
    required_cols = ["Date", "Close"]

    if not all(col in df.columns for col in required_cols):
        st.error("Dataset must contain Date and Close columns")
        st.stop()

    df = df[required_cols]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(inplace=True)
    df.sort_values("Date", inplace=True)

    return df


# ---------------- SIDEBAR ----------------
st.sidebar.header("Data Source")

data_source = st.sidebar.radio(
    "Choose Data Source",
    ["Live Bitcoin Data", "Upload CSV"]
)

st.sidebar.header("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Forecasting Model",
    ["Raw Price (EDA)", "ARIMA", "Prophet", "LSTM"]
)

# ---------------- LOAD DATA ----------------
if data_source == "Live Bitcoin Data":
    st.info("Using live Bitcoin data from Yahoo Finance")
    df = load_live_btc_data()
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.warning("Please upload a CSV file")
        st.stop()
    df = pd.read_csv(uploaded_file)

df = clean_price_data(df)

# ---------------- PLOTTING ----------------
fig, ax = plt.subplots(figsize=(10, 4))

# RAW PRICE
if model_choice == "Raw Price (EDA)":
    ax.plot(df["Date"], df["Close"], label="Bitcoin Price")

# ARIMA
elif model_choice == "ARIMA":
    series = df.set_index("Date")["Close"]
    train = series[:-30]

    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)

    ax.plot(train.index, train, label="Training Data")
    ax.plot(
        pd.date_range(start=train.index[-1], periods=30, freq="D"),
        forecast,
        label="ARIMA Forecast"
    )

# PROPHET
elif model_choice == "Prophet":
    prophet_df = df.rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    ax.plot(prophet_df["ds"], prophet_df["y"], label="Actual Price")
    ax.plot(forecast["ds"], forecast["yhat"], label="Prophet Forecast")

# LSTM
elif model_choice == "LSTM":
    data = df[["Close"]].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    seq_len = 60
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)

    pred = model.predict(X_test)
    pred = scaler.inverse_transform(pred)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    ax.plot(actual, label="Actual Price")
    ax.plot(pred, label="LSTM Prediction")

ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

st.success(f"Model in use: {model_choice}")
