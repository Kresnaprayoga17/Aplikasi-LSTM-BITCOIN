import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from PIL import Image
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
    try:
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Using default styles.")

    # Aplikasi Streamlit
    st.title('PREDIKSI HARGA BITCOIN MENGGUNAKAN LSTM')
    
    # Fetch data from Yahoo Finance for BTC-USD from 2021
    ticker = "BTC-USD"
    try:
        data = yf.download(tickers=ticker, start='2021-01-01', progress=False)
        
        # Clean column names by removing 'BTC-USD ' prefix
        data.columns = [col.replace(f"{ticker} ", "") for col in data.columns]
        
        if data.empty:
            st.error("Failed to fetch data. Please check the ticker symbol or your internet connection.")
            return
            
        # Debug: show column structure
        st.write("Struktur Kolom Data:", data.columns.tolist())
    except Exception as e:
        st.error(f"Gagal mengambil data: {str(e)}")
        return

    # Sidebar to select the start year
    start_year = st.sidebar.selectbox(
        "Periode Forecast", 
        options=range(2021, datetime.now().year + 1), 
        index=0
    )

    # Pastikan data memiliki cukup baris
    if len(data) < 2:
        st.error("Data tidak cukup untuk perhitungan perubahan")
        return

    # Ambil nilai sebagai float dengan cara yang benar
    latest_close = data['Close'].iloc[-1].item()
    prev_close = data['Close'].iloc[-2].item()
    close_change = latest_close - prev_close
    
    latest_volume = data['Volume'].iloc[-1].item()
    prev_volume = data['Volume'].iloc[-2].item()
    volume_change = latest_volume - prev_volume
    
    # Filter data for yearly change calculation
    data_filtered = data[data.index.year >= start_year]
    if not data_filtered.empty and len(data_filtered) > 1:
        latest_close_price = data_filtered['Close'].iloc[-1].item()
        earliest_close_price = data_filtered['Close'].iloc[0].item()
        yearly_change = ((latest_close_price - earliest_close_price) / earliest_close_price) * 100

    # Row A: Metrics
    st.subheader("Key Metrics")
    a1, a2, a3 = st.columns(3)
    
    with a1:
        st.metric("Highest Open Price", f"${data['Open'].max():,.2f}", 
                 delta=f"{data['Open'].max() - data['Open'].iloc[-2]:.2f}")
        
        # Sparkline for Open prices
        fig_sparkline_open = px.line(data.tail(24), x=data.tail(24).index, y='Open', 
                                   width=150, height=50)
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

    with a2:
        st.metric("Highest High Price", f"${data['High'].max():,.2f}", 
                 delta=f"{data['High'].max() - data['High'].iloc[-2]:.2f}")
        
        # Sparkline for High prices
        fig_sparkline_high = px.line(data.tail(24), x=data.tail(24).index, y='High', 
                                   width=150, height=50)
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

    with a3:
        st.metric("Highest Volume", f"{data['Volume'].max():,.0f}", 
                 delta=f"{data['Volume'].max() - data['Volume'].iloc[-2]:.0f}")
        
        # Sparkline for Volume
        fig_sparkline_volume = px.line(data.tail(24), x=data.tail(24).index, y='Volume', 
                                     width=150, height=50)
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

    # Row B: Financial metrics
    b1, b2, b3, b4 = st.columns(4)
    
    with b1:
        st.metric("Highest Close Price", f"${data['Close'].max():,.2f}", 
                 delta=f"{data['Close'].max() - data['Close'].iloc[-2]:.2f}")

    with b2:
        st.metric("Lowest Close Price", f"${data['Close'].min():,.2f}", 
                 delta=f"{data['Close'].min() - data['Close'].iloc[-2]:.2f}")

    with b3:
        st.metric("Average Daily Volume", f"{data['Volume'].mean():,.0f}", 
                 delta=f"{data['Volume'].mean() - data['Volume'].iloc[-2]:.0f}")

    with b4:
        if not data_filtered.empty and len(data_filtered) > 1:
            change_label = "Yearly Change" if yearly_change >= 0 else "Yearly Decrease"
            st.metric(label=change_label, value=f"{yearly_change:.2f}%", 
                     delta=f"{abs(yearly_change):.2f}%")

    # Row C: Main chart and data table
    c1, c2 = st.columns((7, 3))
    with c1:
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
        
        add_range_selector(fig)
        fig.update_layout(
            title="BTC-USD - Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=500,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.write("Real Data")
        st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail())

    # Row D: Additional analysis
    if len(data) >= 24:
        st.subheader("24h Trends")
        trend_col1, trend_col2 = st.columns(2)
        
        with trend_col1:
            fig = px.line(data.tail(24), x=data.tail(24).index, y='Close', 
                         height=100)
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Closing Price Trend")

        with trend_col2:
            fig = px.line(data.tail(24), x=data.tail(24).index, y='Volume', 
                         height=100)
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Volume Trend")

if __name__ == '__main__':
    main()
