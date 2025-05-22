import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_lightweight_charts import renderLightweightCharts
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
import matplotlib.pyplot as plt
import io

def add_range_selector(fig):
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ]
            )
        ),
        xaxis_type='date'
    )

def clean_column_names(data, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        return data.droplevel(level=1, axis=1)
    if any(isinstance(col, tuple) for col in data.columns):
        return data.rename(columns=lambda x: x[0] if isinstance(x, tuple) else x)
    if any(ticker in str(col) for col in data.columns):
        return data.rename(columns=lambda x: str(x).replace(f"{ticker} ", "").replace(f"{ticker}_", ""))
    return data

@st.cache_data(ttl=3600)
def fetch_data(ticker="BTC-USD", start_date='2021-01-01'):
    data = yf.download(tickers=ticker, start=start_date)
    return data

def forecast_bitcoin(data, months_ahead):
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

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

    test_data = y_scaled[training_data_len - 60:, :]
    x_test = []

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    last_60_days = y_scaled[-60:].reshape(1, 60, 1)
    future_predictions = []
    for _ in range(months_ahead * 30):
        pred = model.predict(last_60_days)[0][0]
        future_predictions.append(pred)
        last_60_days = np.append(last_60_days[:,1:,:], [[[pred]]], axis=1)
    future_predictions = np.array(future_predictions).reshape(-1,1)
    future_predictions = scaler.inverse_transform(future_predictions)

    total_predictions = np.concatenate((predictions, future_predictions), axis=0)

    y_test = y[training_data_len:, :]
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

    return total_predictions, y_test, rmse

def main():
    st.set_page_config(layout="wide", page_title="Bitcoin DashBoard For LSTM")

    try:
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Using default styles.")

    st.title('PREDIKSI HARGA BITCOIN MENGGUNAKAN LSTM')

    ticker = "BTC-USD"
    data = fetch_data(ticker)

    if data.empty:
        st.error("Failed to fetch data. Please check the ticker symbol or your internet connection.")
        return

    months_ahead = st.sidebar.slider("Select forecast horizon (months)", min_value=1, max_value=12, value=3)

    st.subheader("Bitcoin Price Forecasting (LSTM Model)")

    if st.button("Run Forecast"):
        with st.spinner("Training LSTM model and forecasting..."):
            predictions, y_test, rmse = forecast_bitcoin(data, months_ahead)

            valid = data.iloc[-len(y_test):].copy()
            valid['Predictions'] = predictions[:len(y_test)]

            last_date = data.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, months_ahead*30 + 1)]

            fig, ax = plt.subplots(figsize=(16,6))
            ax.set_title('Model')
            ax.set_xlabel('Date')
            ax.set_ylabel('Close Price USD ($)')
            ax.plot(data['Close'][:len(data)-len(y_test)], label='Train')
            ax.plot(valid['Close'], label='Val')
            ax.plot(valid['Predictions'], label='Predictions')
            ax.plot(future_dates, predictions[len(y_test):], label='Future Predictions')
            ax.legend(loc='lower right')

            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            st.image(buf, use_container_width=True)
            buf.close()

            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")

if __name__ == '__main__':
    main()
