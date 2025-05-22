import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
from datetime import datetime, date
from PIL import Image
from streamlit_lightweight_charts import renderLightweightCharts
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
import matplotlib.pyplot as plt
import io

@st.cache_data(ttl=3600)
def fetch_data(ticker="BTC-USD", start_date='2021-01-01'):
    data = yf.download(tickers=ticker, start=start_date)
    return data

def forecast_bitcoin(data):
    # Prepare data
    y = data['Close'].fillna(method='ffill').values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y)

    training_data_len = int(np.ceil(len(y_scaled) * .95))
    train_data = y_scaled[0:training_data_len, :]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

    # Prepare test data
    test_data = y_scaled[training_data_len - 60:, :]
    x_test = []
    y_test = y[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predict
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

    return predictions, y_test, rmse

def main():
    st.set_page_config(layout="wide", page_title="Bitcoin DashBoard For LSTM")

    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.title('PREDIKSI HARGA BITCOIN MENGGUNAKAN LSTM')

    data = fetch_data()

    if data.empty:
        st.error("Failed to fetch data. Please check the ticker symbol or your internet connection.")
        return

    # Existing dashboard code here (metrics, sparklines, charts) ...
    # For brevity, omitted here but will remain unchanged from original home.py.py

    # Forecasting section close to the notebook style
    st.header("Bitcoin Price Forecasting (LSTM Model)")

    if st.button("Run Forecast"):
        with st.spinner("Training LSTM model and forecasting..."):
            predictions, y_test, rmse = forecast_bitcoin(data)

            # Prepare data for plotting
            valid = data.iloc[-len(y_test):].copy()
            valid['Predictions'] = predictions

            fig, ax = plt.subplots(figsize=(16,6))
            ax.set_title('Model')
            ax.set_xlabel('Date')
            ax.set_ylabel('Close Price USD ($)')
            ax.plot(data['Close'][:len(data)-len(y_test)], label='Train')
            ax.plot(valid['Close'], label='Val')
            ax.plot(valid['Predictions'], label='Predictions')
            ax.legend(loc='lower right')

            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            st.image(buf, use_container_width=True)
            buf.close()

            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")

if __name__ == '__main__':
    main()
