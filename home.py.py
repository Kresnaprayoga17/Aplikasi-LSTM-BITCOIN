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

    start_year = st.sidebar.selectbox("Periode Forecast", options=range(2021, datetime.now().year + 1), index=0)

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

        data_filtered = data[data.index.year >= start_year]
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

if __name__ == '__main__':
    main()
