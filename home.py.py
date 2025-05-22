import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

def main():
    st.set_page_config(layout="wide", page_title="Bitcoin DashBoard For LSTM")
    st.title('PREDIKSI HARGA BITCOIN MENGGUNAKAN LSTM')
    
    # Mengambil data dengan penanganan MultiIndex
    ticker = "BTC-USD"
    try:
        data = yf.download(tickers=ticker, start='2021-01-01', group_by='ticker')
        
        # Jika data menggunakan MultiIndex, kita perlu merestrukturisasi
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten the MultiIndex
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
            # Atau bisa juga diakses dengan cara: data.xs('BTC-USD', axis=1, level=1)
        
        if data.empty:
            st.error("Data tidak ditemukan. Periksa koneksi internet atau simbol ticker.")
            return
            
    except Exception as e:
        st.error(f"Gagal mengambil data: {str(e)}")
        return

    # Pastikan kolom yang diperlukan ada
    required_columns = ['Close_BTC-USD', 'High_BTC-USD', 'Low_BTC-USD', 'Open_BTC-USD', 'Volume_BTC-USD']
    for col in required_columns:
        if col not in data.columns:
            st.error(f"Kolom {col} tidak ditemukan dalam data")
            return

    # Perhitungan metrik
    latest_close = data['Close_BTC-USD'].iloc[-1]
    prev_close = data['Close_BTC-USD'].iloc[-2]
    close_change = latest_close - prev_close
    
    # Tampilkan data
    st.subheader("Data Bitcoin (BTC-USD)")
    st.dataframe(data.tail())
    
    # Buat candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open_BTC-USD'],
        high=data['High_BTC-USD'],
        low=data['Low_BTC-USD'],
        close=data['Close_BTC-USD']
    )])
    
    fig.update_layout(
        title="BTC-USD Candlestick Chart",
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
