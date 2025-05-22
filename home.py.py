import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Set up the page
st.set_page_config(layout="wide", page_title="Bitcoin LSTM Predictor")
st.title("Bitcoin Price Prediction with LSTM")
st.write("""
This application uses Long Short-Term Memory (LSTM) neural networks to predict future Bitcoin prices.
The model is trained on historical BTC-USD price data from Yahoo Finance.
""")

# Sidebar controls
st.sidebar.header("Model Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", "BTC-USD")
start_date = st.sidebar.date_input("Start Date", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now())

prediction_days = st.sidebar.slider("Days to Predict", 7, 365, 30)
look_back = st.sidebar.slider("Look Back Period (Days)", 30, 180, 60)
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

# Model architecture
st.sidebar.subheader("Model Architecture")
lstm_units = st.sidebar.slider("LSTM Units", 32, 256, 50)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)
dense_units = st.sidebar.slider("Dense Units", 16, 128, 25)

# Training parameters
st.sidebar.subheader("Training Parameters")
epochs = st.sidebar.slider("Epochs", 10, 200, 50)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
validation_split = st.sidebar.slider("Validation Split", 0.1, 0.3, 0.2)

# Data Preparation
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("No data fetched. Please check your ticker symbol and date range.")
    st.stop()

# Show raw data
st.subheader("Raw Data")
st.dataframe(data.tail())

# Plot closing price
st.subheader("Closing Price History")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# Prepare data for LSTM
st.subheader("Data Preprocessing")

with st.expander("Show Preprocessing Steps"):
    st.write("""
    1. Extract only the 'Close' prices
    2. Scale data to range [0, 1] using MinMaxScaler
    3. Create sequences of look_back days as features and next day as target
    4. Split into training and test sets
    """)

# Create dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

X, Y = create_dataset(scaled_data, look_back)

# Split into train and test sets
train_size = int(len(X) * (1 - test_size))
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
st.subheader("Model Architecture")
model = Sequential()
model.add(LSTM(lstm_units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(dropout_rate))
model.add(LSTM(lstm_units, return_sequences=False))
model.add(Dropout(dropout_rate))
model.add(Dense(dense_units))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Display model summary
with st.expander("Show Model Summary"):
    from io import StringIO
    import sys
    buffer = StringIO()
    sys.stdout = buffer
    model.summary()
    sys.stdout = sys.__stdout__
    st.text(buffer.getvalue())

# Train the model
st.subheader("Model Training")
if st.button("Train Model"):
    with st.spinner('Training in progress...'):
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        history = model.fit(
            X_train, Y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history.history['loss'], label='Train Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_title('Model Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        st.pyplot(fig)
        
        # Save the model
        model.save('btc_lstm_model.h5')
        st.success("Model training completed!")
        
        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        # Inverse transform predictions
        train_predict = scaler.inverse_transform(train_predict)
        Y_train_inv = scaler.inverse_transform([Y_train])
        test_predict = scaler.inverse_transform(test_predict)
        Y_test_inv = scaler.inverse_transform([Y_test])
        
        # Plot predictions
        st.subheader("Model Predictions")
        
        # Create arrays for plotting
        train_predict_plot = np.empty_like(scaled_data)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict
        
        test_predict_plot = np.empty_like(scaled_data)
        test_predict_plot[:, :] = np.nan
        test_predict_plot[len(train_predict)+(look_back*2)+1:len(scaled_data)-1, :] = test_predict
        
        # Plot baseline and predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            name='Actual Price',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=train_predict_plot.flatten(),
            name='Training Prediction',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=test_predict_plot.flatten(),
            name='Testing Prediction',
            line=dict(color='red')
        ))
        fig.update_layout(
            title='Actual vs Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Future prediction
        st.subheader(f"Future {prediction_days}-Day Price Prediction")
        
        last_sequence = scaled_data[-look_back:]
        future_predictions = []
        
        for _ in range(prediction_days):
            current_pred = model.predict(last_sequence.reshape(1, look_back, 1))
            future_predictions.append(current_pred[0,0])
            last_sequence = np.append(last_sequence[1:], current_pred)
        
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = [last_date + timedelta(days=x) for x in range(1, prediction_days+1)]
        
        # Plot future predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index[-100:],  # Show last 100 days
            y=data['Close'].values[-100:],
            name='Historical Price',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions.flatten(),
            name='Future Prediction',
            line=dict(color='red', dash='dot')
        ))
        fig.update_layout(
            title=f'Next {prediction_days} Days Prediction',
            xaxis_title='Date',
            yaxis_title='Price (USD)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show future predictions in a table
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': future_predictions.flatten()
        })
        st.dataframe(future_df)
        
        # Calculate and show performance metrics
        st.subheader("Model Performance Metrics")
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        train_score = np.sqrt(mean_squared_error(Y_train_inv[0], train_predict[:,0]))
        st.write(f'Train RMSE: {train_score:.2f}')
        
        test_score = np.sqrt(mean_squared_error(Y_test_inv[0], test_predict[:,0]))
        st.write(f'Test RMSE: {test_score:.2f}')
        
        mae = mean_absolute_error(Y_test_inv[0], test_predict[:,0])
        st.write(f'Test MAE: {mae:.2f}')
        
        r2 = r2_score(Y_test_inv[0], test_predict[:,0])
        st.write(f'Test R² Score: {r2:.2f}')

# Clean up
if os.path.exists('best_model.h5'):
    os.remove('best_model.h5')
if os.path.exists('btc_lstm_model.h5'):
    os.remove('btc_lstm_model.h5')
