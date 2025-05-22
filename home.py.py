import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from streamlit_lightweight_charts import renderLightweightCharts

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

def main():
    # Set page configuration
    st.set_page_config(layout="wide", page_title="Bitcoin DashBoard For LSTM")

    # Load custom styles
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Aplikasi Streamlit
    st.title('PREDIKSI HARGA BITCOIN MENGGUNAKAN LSTM')
    
    # Fetch data from Yahoo Finance for BTC-USD from 2021
    ticker = "BTC-USD"
    data = yf.download(tickers=ticker, start='2021-01-01')

    # Check if data is empty
    if data.empty:
        st.error("Failed to fetch data. Please check the ticker symbol or your internet connection.")
        return

    # Sidebar to select the start year
    start_year = st.sidebar.selectbox(
        "Periode Forecast", 
        options=range(2021, datetime.now().year + 1), 
        index=0
    )

    # Calculate metrics
    latest_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2]
    close_change = latest_close - prev_close
    
    latest_volume = data['Volume'].iloc[-1]
    prev_volume = data['Volume'].iloc[-2]
    volume_change = latest_volume - prev_volume
    
    # Filter data for yearly change calculation
    data_filtered = data[data.index.year >= start_year]
    if not data_filtered.empty:
        latest_close_price = data_filtered['Close'].iloc[-1]
        earliest_close_price = data_filtered['Close'].iloc[0]
        yearly_change = ((latest_close_price - earliest_close_price) / earliest_close_price) * 100

    # Row A: Metrics
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"${latest_close:,.2f}", 
                 delta=f"{close_change:,.2f}" if pd.notna(close_change) else None)
        
    with col2:
        st.metric("24h Volume", f"{latest_volume:,.0f}", 
                 delta=f"{volume_change:,.0f}" if pd.notna(volume_change) else None)
        
    with col3:
        if not data_filtered.empty:
            change_label = "Yearly Change" if yearly_change >= 0 else "Yearly Decrease"
            st.metric(change_label, f"{yearly_change:.2f}%")

    # Row B: Charts
    st.subheader("Price Charts")
    chart_col, table_col = st.columns([7, 3])
    
    with chart_col:
        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
        
        add_range_selector(fig)
        fig.update_layout(
            title="BTC-USD Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    with table_col:
        st.write("Recent Data")
        st.dataframe(data.tail()[['Open', 'High', 'Low', 'Close', 'Volume']])

    # Row C: Additional visualizations
    st.subheader("Detailed Analysis")
    if len(data) >= 24:
        # Sparklines for last 24 periods
        st.write("24h Trends")
        
        spark_col1, spark_col2, spark_col3 = st.columns(3)
        
        with spark_col1:
            fig = px.line(data.tail(24), x=data.tail(24).index, y='Close', 
                         height=100)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False, 
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Closing Price Trend")

        with spark_col2:
            fig = px.line(data.tail(24), x=data.tail(24).index, y='Volume', 
                         height=100)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False, 
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Volume Trend")

if __name__ == '__main__':
    main()
