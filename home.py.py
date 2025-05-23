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

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(data, lookback=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    x_all = []
    y_all = []
    
    for i in range(lookback, len(scaled_data)):
        x_all.append(scaled_data[i-lookback:i, 0])
        y_all.append(scaled_data[i, 0])
    
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    
    # Reshape data for LSTM [samples, time steps, features]
    x_all = np.reshape(x_all, (x_all.shape[0], x_all.shape[1], 1))
    
    return x_all, y_all, scaler

def predict_future(model, last_sequence, scaler, n_steps):
    """
    Predict future values using the trained LSTM model
    """
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_steps):
        # Reshape for prediction
        current_sequence_reshaped = current_sequence.reshape((1, current_sequence.shape[0], 1))
        # Get prediction
        next_pred = model.predict(current_sequence_reshaped, verbose=0)
        # Add to predictions
        future_predictions.append(next_pred[0, 0])
        # Update sequence
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
        
    # Transform predictions back to original scale
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    return scaler.inverse_transform(future_predictions)

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

def main():
    st.set_page_config(layout="wide", page_title="Bitcoin DashBoard For LSTM")

    try:
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Using default styles.")

    st.title('PREDIKSI HARGA BITCOIN MENGGUNAKAN LSTM')

    ticker = "BTC-USD"
    try:
        data = yf.download(tickers=ticker, start='2021-01-01', progress=False)
        data = clean_column_names(data, ticker)
        if data.empty:
            st.error("Failed to fetch data. Please check the ticker symbol or your internet connection.")
            return
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
            return
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return

    if len(data) < 2:
        st.error("Not enough data for calculations")
        return

    try:
        latest_close = data['Close'].iloc[-1].item()
        prev_close = data['Close'].iloc[-2].item()
        close_change = latest_close - prev_close

        latest_volume = data['Volume'].iloc[-1].item()
        prev_volume = data['Volume'].iloc[-2].item()
        volume_change = latest_volume - prev_volume

        data_filtered = data
        if not data_filtered.empty and len(data_filtered) > 1:
            latest_close_price = data_filtered['Close'].iloc[-1].item()
            earliest_close_price = data_filtered['Close'].iloc[0].item()
            yearly_change = ((latest_close_price - earliest_close_price) / earliest_close_price) * 100
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return

    st.subheader("Key Metrics")
    a1, a2, a3 = st.columns(3)

    with a1:
        try:
            open_change = data['Open'].iloc[-1] - data['Open'].iloc[-2]
            st.metric("Current Open Price", f"${data['Open'].iloc[-1]:,.2f}", delta=f"{open_change:.2f}")
            fig_sparkline_open = px.line(data.tail(24), x=data.tail(24).index, y='Open', width=150, height=50)
            fig_sparkline_open.update_layout(
                plot_bgcolor="rgba(0, 0, 0, 0)",
                paper_bgcolor="rgba(0, 0, 0, 0)",
                yaxis={"visible": False},
                xaxis={"visible": False},
                showlegend=False,
                margin={"l":4,"r":4,"t":0, "b":0, "pad": 4}
            )
            st.plotly_chart(fig_sparkline_open, use_container_width=True)
            st.markdown("<div style='text-align:center; color:green;'>OPEN</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error Open Price: {str(e)}")

    with a2:
        try:
            high_change = data['High'].iloc[-1] - data['High'].iloc[-2]
            st.metric("Current High Price", f"${data['High'].iloc[-1]:,.2f}", delta=f"{high_change:.2f}")
            fig_sparkline_high = px.line(data.tail(24), x=data.tail(24).index, y='High', width=150, height=50)
            fig_sparkline_high.update_layout(
                plot_bgcolor="rgba(0, 0, 0, 0)",
                paper_bgcolor="rgba(0, 0, 0, 0)",
                yaxis={"visible": False},
                xaxis={"visible": False},
                showlegend=False,
                margin={"l":4,"r":4,"t":0, "b":0, "pad": 4}
            )
            st.plotly_chart(fig_sparkline_high, use_container_width=True)
            st.markdown("<div style='text-align:center; color:green;'>HIGH</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error High Price: {str(e)}")

    with a3:
        try:
            volume_change = data['Volume'].iloc[-1] - data['Volume'].iloc[-2]
            st.metric("Current Volume", f"{data['Volume'].iloc[-1]:,.0f}", delta=f"{volume_change:,.0f}")
            fig_sparkline_volume = px.line(data.tail(24), x=data.tail(24).index, y='Volume', width=150, height=50)
            fig_sparkline_volume.update_layout(
                plot_bgcolor="rgba(0, 0, 0, 0)",
                paper_bgcolor="rgba(0, 0, 0, 0)",
                yaxis={"visible": False},
                xaxis={"visible": False},
                showlegend=False,
                margin={"l":4,"r":4,"t":0, "b":0, "pad": 4}
            )
            st.plotly_chart(fig_sparkline_volume, use_container_width=True)
            st.markdown("<div style='text-align:center; color:green;'>VOLUME</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error Volume: {str(e)}")

    st.subheader("Historical Metrics")
    b1, b2, b3, b4 = st.columns(4)

    with b1:
        try:
            st.metric("Highest Close", f"${data['Close'].max():,.2f}")
        except Exception as e:
            st.error(f"Error Highest Close: {str(e)}")

    with b2:
        try:
            st.metric("Lowest Close", f"${data['Close'].min():,.2f}")
        except Exception as e:
            st.error(f"Error Lowest Close: {str(e)}")

    with b3:
        try:
            st.metric("Avg Volume", f"{data['Volume'].mean():,.0f}")
        except Exception as e:
            st.error(f"Error Avg Volume: {str(e)}")

    with b4:
        try:
            if not data_filtered.empty and len(data_filtered) > 1:
                change_label = "Yearly Change" if yearly_change >= 0 else "Yearly Decrease"
                st.metric(label=change_label, value=f"{yearly_change:.2f}%")
        except Exception as e:
            st.error(f"Error Yearly Change: {str(e)}")

    st.subheader("Price Analysis")
    c1, c2 = st.columns((7, 3))

    with c1:
        try:
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close']
            )])
            add_range_selector(fig)
            fig.update_layout(
                title=f"{ticker} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                xaxis_rangeslider_visible=False,
                height=500,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error Candlestick Chart: {str(e)}")

    with c2:
        try:
            st.write("Recent Data")
            st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail())
        except Exception as e:
            st.error(f"Error Data Table: {str(e)}")

    if len(data) >= 24:
        st.subheader("Recent Trends (24 periods)")
        trend_col1, trend_col2 = st.columns(2)

        with trend_col1:
            try:
                fig = px.line(data.tail(24), x=data.tail(24).index, y='Close', height=100)
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Price Trend")
            except Exception as e:
                st.error(f"Error Price Trend: {str(e)}")

        with trend_col2:
            try:
                fig = px.line(data.tail(24), x=data.tail(24).index, y='Volume', height=100)
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Volume Trend")
            except Exception as e:
                st.error(f"Error Volume Trend: {str(e)}")

    # LSTM Prediction Section
    st.subheader("LSTM Price Predictions")
    
    try:
        # Prepare data for LSTM
        x_all, y_all, scaler = prepare_data(data['Close'])
        
        # Split data into train and test
        train_size = int(len(x_all) * 0.95)
        x_train, y_train = x_all[:train_size], y_all[:train_size]
        x_test, y_test = x_all[train_size:], y_all[train_size:]
        
        # Create and train model
        model = create_lstm_model((x_train.shape[1], 1))
        with st.spinner('Training LSTM model...'):
            model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)
        
        # Make predictions
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'Date': data.index[train_size+60:],
            'Actual': actual.flatten(),
            'Predicted': predictions.flatten()
        }).set_index('Date')
        
        # Plot predictions vs actual
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Actual'], name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted'], name='Predicted', line=dict(color='red')))
        
        fig.update_layout(
            title='LSTM Model: Actual vs Predicted Bitcoin Prices',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=400,
            template="plotly_dark"
        )
        add_range_selector(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        pred_metrics = st.columns(3)
        with pred_metrics[0]:
            rmse = np.sqrt(np.mean((predictions - actual) ** 2))
            st.metric("RMSE", f"${rmse:,.2f}")
        with pred_metrics[1]:
            mae = np.mean(np.abs(predictions - actual))
            st.metric("MAE", f"${mae:,.2f}")
        with pred_metrics[2]:
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            st.metric("MAPE", f"{mape:.2f}%")

        # Future Predictions Section
        st.subheader("Future Price Predictions")
        
        # Add prediction period buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Predict Next 7 Days"):
                days = 7
                period = "7 Days"
                future_dates = [data.index[-1] + timedelta(days=x) for x in range(1, days + 1)]
        with col2:
            if st.button("Predict Next Month"):
                days = 30
                period = "1 Month"
                future_dates = [data.index[-1] + timedelta(days=x) for x in range(1, days + 1)]
        with col3:
            if st.button("Predict Next Year"):
                days = 365
                period = "1 Year"
                future_dates = [data.index[-1] + timedelta(days=x) for x in range(1, days + 1)]

        # If any button is clicked, show predictions
        if 'days' in locals():
            with st.spinner(f'Generating {period} predictions...'):
                # Get the last sequence from our data
                last_sequence = scaler.transform(data['Close'].values[-60:].reshape(-1, 1)).flatten()
                
                # Generate future predictions
                future_pred = predict_future(model, last_sequence, scaler, days)
                
                # Create DataFrame for future predictions
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted': future_pred.flatten()
                }).set_index('Date')
                
                # Plot future predictions
                fig = go.Figure()
                # Plot last 30 days of actual data
                fig.add_trace(go.Scatter(
                    x=data.index[-30:],
                    y=data['Close'].values[-30:],
                    name='Historical',
                    line=dict(color='blue')
                ))
                # Plot future predictions
                fig.add_trace(go.Scatter(
                    x=future_df.index,
                    y=future_df['Predicted'],
                    name='Future Prediction',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title=f'Bitcoin Price Prediction - Next {period}',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display prediction table
                st.write(f"Predicted prices for next {period}:")
                st.dataframe(future_df)
            
    except Exception as e:
        st.error(f"Error in LSTM Predictions: {str(e)}")

if __name__ == '__main__':
    main()
