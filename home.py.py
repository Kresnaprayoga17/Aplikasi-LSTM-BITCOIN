import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

def get_column_name(data, column_base):
    """
    Mencari nama kolom yang sesuai dengan pola:
    1. 'Close' (format lama)
    2. ('Close', 'BTC-USD') (MultiIndex)
    3. 'Close_BTC-USD' (format gabungan)
    """
    # Coba format sederhana
    if column_base in data.columns:
        return column_base
    
    # Coba format MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        for col in data.columns:
            if col[0] == column_base:
                return col
        # Alternatif: akses langsung
        try:
            if ('Close', 'BTC-USD') in data.columns:
                return ('Close', 'BTC-USD')
        except:
            pass
    
    # Coba format gabungan
    combined_name = f"{column_base}_BTC-USD"
    if combined_name in data.columns:
        return combined_name
    
    # Jika tidak ditemukan
    raise ValueError(f"Kolom {column_base} tidak ditemukan dalam data")

def main():
    st.set_page_config(layout="wide", page_title="Bitcoin DashBoard For LSTM")
    st.title('PREDIKSI HARGA BITCOIN MENGGUNAKAN LSTM')
    
    # Mengambil data dengan berbagai opsi format
    ticker = "BTC-USD"
    try:
        # Coba beberapa metode pengambilan data
        try:
            data = yf.download(tickers=ticker, start='2021-01-01', group_by='ticker')
        except:
            data = yf.download(tickers=ticker, start='2021-01-01')
        
        if data.empty:
            st.error("Data tidak ditemukan. Periksa koneksi internet atau simbol ticker.")
            return
            
        # Debug: tampilkan struktur kolom
        st.write("Struktur Kolom Data:", data.columns)
        
        # Dapatkan nama kolom yang benar
        close_col = get_column_name(data, 'Close')
        high_col = get_column_name(data, 'High')
        low_col = get_column_name(data, 'Low')
        open_col = get_column_name(data, 'Open')
        volume_col = get_column_name(data, 'Volume')
        
        # Perhitungan metrik
        latest_close = data[close_col].iloc[-1]
        prev_close = data[close_col].iloc[-2]
        close_change = latest_close - prev_close
        
        # Tampilkan metrik utama
        st.metric("Harga Terakhir", f"${latest_close:,.2f}", 
                 delta=f"{close_change:,.2f}" if pd.notna(close_change) else None)
        
        # Buat candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data[open_col],
            high=data[high_col],
            low=data[low_col],
            close=data[close_col]
        )])
        
        fig.update_layout(
            title=f"{ticker} Candlestick Chart",
            xaxis_title="Tanggal",
            yaxis_title="Harga (USD)",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tampilkan data mentah
        st.subheader("Data Mentah")
        st.dataframe(data.tail())
        
    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")

if __name__ == '__main__':
    main()
