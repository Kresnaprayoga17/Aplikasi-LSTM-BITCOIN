import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from streamlit_lightweight_charts import renderLightweightCharts

# Fungsi untuk membersihkan nama kolom
def clean_column_names(data, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        return data.droplevel(level=1, axis=1)
    if any(isinstance(col, tuple) for col in data.columns):
        return data.rename(columns=lambda x: x[0] if isinstance(x, tuple) else x)
    if any(ticker in str(col) for col in data.columns):
        return data.rename(columns=lambda x: str(x).replace(f"{ticker} ", "").replace(f"{ticker}_", ""))
    return data

# Fungsi untuk membuat dataset LSTM
def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Fungsi untuk membangun model LSTM
def build_lstm_model(look_back):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Fungsi untuk prediksi harga
def predict_future_prices(model, last_sequence, future_steps, scaler):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(future_steps):
        # Prediksi nilai berikutnya
        next_pred = model.predict(current_sequence.reshape(1, look_back, 1))
        predictions.append(next_pred[0,0])
        
        # Update sequence dengan prediksi baru
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    
    # Kembalikan ke skala asli
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)

def main():
    st.set_page_config(layout="wide", page_title="Bitcoin Prediction Dashboard")
    st.title('BITCOIN PRICE PREDICTION WITH LSTM')
    
    # Sidebar untuk parameter prediksi
    st.sidebar.header("Prediction Settings")
    prediction_period = st.sidebar.selectbox(
        "Prediction Period",
        ["1 Month", "3 Months", "6 Months", "1 Year", "Custom"]
    )
    
    if prediction_period == "Custom":
        custom_months = st.sidebar.number_input("Number of Months to Predict", 1, 24, 6)
        future_steps = custom_months * 30  # Asumsi 30 hari per bulan
    else:
        period_map = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365
        }
        future_steps = period_map[prediction_period]
    
    look_back = st.sidebar.slider("Look Back Period (Days)", 30, 180, 60)
    epochs = st.sidebar.slider("Epochs", 10, 100, 50)
    batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
    
    # Download data Bitcoin
    ticker = "BTC-USD"
    try:
        data = yf.download(tickers=ticker, start='2020-01-01', progress=False)
        data = clean_column_names(data, ticker)
        
        if data.empty:
            st.error("Failed to fetch data. Please check your internet connection.")
            return
            
        # Tampilkan data historis
        st.subheader("Historical Bitcoin Price Data")
        st.dataframe(data.tail())
        
        # Persiapkan data untuk LSTM
        dataset = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset_scaled = scaler.fit_transform(dataset)
        
        # Buat dataset training
        X, y = create_dataset(dataset_scaled, look_back)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split data (80% training, 20% validation)
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Bangun dan latih model
        with st.spinner('Training LSTM model...'):
            model = build_lstm_model(look_back)
            early_stop = EarlyStopping(monitor='val_loss', patience=5)
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
        
        # Prediksi untuk data historis (hanya untuk visualisasi)
        train_predict = model.predict(X_train)
        val_predict = model.predict(X_val)
        
        # Transformasi kembali ke skala asli
        train_predict = scaler.inverse_transform(train_predict)
        y_train = scaler.inverse_transform([y_train])
        val_predict = scaler.inverse_transform(val_predict)
        y_val = scaler.inverse_transform([y_val])
        
        # Buat prediksi masa depan
        last_sequence = dataset_scaled[-look_back:]
        future_prices = predict_future_prices(model, last_sequence, future_steps, scaler)
        
        # Buat tanggal untuk prediksi masa depan
        last_date = data.index[-1]
        future_dates = [last_date + timedelta(days=x) for x in range(1, future_steps+1)]
        
        # Visualisasi hasil
        st.subheader("Price Prediction Results")
        
        # Plot training dan validasi
        fig = go.Figure()
        
        # Data aktual
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Actual Price',
            line=dict(color='blue')
        ))
        
        # Prediksi training
        train_plot = np.empty_like(dataset)
        train_plot[:, :] = np.nan
        train_plot[look_back:look_back+len(train_predict), :] = train_predict
        fig.add_trace(go.Scatter(
            x=data.index,
            y=train_plot.flatten(),
            mode='lines',
            name='Training Prediction',
            line=dict(color='green')
        ))
        
        # Prediksi validasi
        val_plot = np.empty_like(dataset)
        val_plot[:, :] = np.nan
        val_plot[split+look_back:split+look_back+len(val_predict), :] = val_predict
        fig.add_trace(go.Scatter(
            x=data.index,
            y=val_plot.flatten(),
            mode='lines',
            name='Validation Prediction',
            line=dict(color='orange')
        ))
        
        # Prediksi masa depan
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_prices.flatten(),
            mode='lines+markers',
            name='Future Prediction',
            line=dict(color='red', dash='dot')
        ))
        
        fig.update_layout(
            title='Bitcoin Price Prediction with LSTM',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tampilkan prediksi masa depan dalam tabel
        st.subheader(f"Future Price Prediction ({prediction_period})")
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': future_prices.flatten()
        })
        st.dataframe(future_df)
        
        # Tampilkan metrik prediksi
        st.subheader("Prediction Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_price = data['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:,.2f}")
        
        with col2:
            predicted_change = future_prices[-1][0] - current_price
            pct_change = (predicted_change / current_price) * 100
            st.metric(
                f"Predicted Price in {prediction_period}", 
                f"${future_prices[-1][0]:,.2f}",
                delta=f"{pct_change:.2f}%"
            )
        
        with col3:
            avg_predicted = np.mean(future_prices)
            st.metric("Average Predicted Price", f"${avg_predicted:,.2f}")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
