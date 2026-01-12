# 📈 Cryptocurrency Time Series Analysis

An end-to-end data analytics and forecasting project that analyzes and predicts cryptocurrency (Bitcoin) price trends using time series models and machine learning, deployed as an interactive web application.

---

## 🔍 Project Overview

This project focuses on analyzing historical and live Bitcoin price data to:
- Understand long-term trends and volatility
- Forecast future prices using multiple models
- Provide an interactive dashboard for users

The application is built using **Python**, **Streamlit**, and popular time-series forecasting libraries.

## 🌐 Live Demo

🚀 **Access the deployed Streamlit application here:**  
👉 https://crypto-time-series-analysis-frgajte5dlat3mdjjufb5l.streamlit.app/

### What you can do in the live app:
- View **live Bitcoin price trends**
- Upload your own **CSV cryptocurrency data**
- Compare forecasts using **ARIMA, Prophet, and LSTM**
- Analyze **price volatility** and **market sentiment**
- Interact with charts through a clean GUI


## 🚀 Features

### 1. 📊 Data Sources
- **Live Bitcoin Data** fetched automatically from Yahoo Finance
- **User-uploaded CSV files** for custom analysis

### 2. 📉 Exploratory Data Analysis (EDA)
- Price trend visualization
- Rolling volatility analysis
- Cleaned and preprocessed time-series data

### 3. 🔮 Forecasting Models
- **ARIMA** – Statistical time series forecasting
- **Facebook Prophet** – Trend + seasonality-based forecasting
- **LSTM (Deep Learning)** – Neural network-based prediction

### 4. 📈 Volatility Analysis
- Rolling 30-day volatility to identify high-risk periods

### 5. 📰 Sentiment Analysis
- NLP-based sentiment scoring of cryptocurrency-related news headlines
- Visual sentiment trend analysis

### 6. 🖥 Interactive Dashboard (GUI)
- Built with **Streamlit**
- Model selection via dropdown
- Dynamic plots and real-time updates

---

## 🧠 Technologies Used

- Python
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn
- Statsmodels (ARIMA)
- Facebook Prophet
- TensorFlow / Keras (LSTM)
- Scikit-learn
- Yahoo Finance (yfinance)
- Git & GitHub

---

## 📂 Project Structure

```text
crypto-time-series-analysis/
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
│
├── data/
│   └── btc.csv             
│
├── src/
│   ├── data_collection.py
│   ├── eda.py
│   ├── arima_model.py
│   ├── prophet_model.py
│   ├── lstm_model.py
│   ├── volatility_analysis.py
│   └── sentiment_analysis.py
│
└── notebooks/              

▶️ How to Run Locally
1. Clone the repository
git clone https://github.com/Aryaa-2905/crypto-time-series-analysis.git
cd crypto-time-series-analysis

2. Create & activate virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

3. Install dependencies
pip install -r requirements.txt

4. Run the app
streamlit run app.py

🌍 Live Deployment

The application is deployed on Streamlit Cloud and accessible via a public URL.

Users can:

View real-time Bitcoin trends

Upload their own datasets

Compare multiple forecasting models

Analyze volatility and sentiment

🎯 Real-World Use Cases

Traders analyzing price trends and risk

Investors exploring long-term forecasts

Students learning time series analysis

Recruiters evaluating end-to-end ML projects

📌 Future Enhancements

Multi-cryptocurrency support

Advanced sentiment analysis using Twitter/Reddit

Model performance comparison metrics

User authentication and saved dashboards

👩‍💻 Author

Arya Gahine
 Data Scientist | Python | Machine Learning | Time Series Analysis

GitHub: https://github.com/Aryaa-2905
