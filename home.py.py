import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
from datetime import datetime, date
from PIL import Image
from streamlit_lightweight_charts import renderLightweightCharts
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
from datetime import datetime, date
from PIL import Image

def main():
    # Set page configuration
    st.set_page_config(layout="wide", page_title="Bitcoin DashBoard For LSTM")

    # Load custom styles
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Aplikasi Streamlit
    st.title('PREDIKSI HARGA BITCOIN MENGGUNAKAN LSTM')
    # Fetch data from Yahoo Finance for KKGI.JK from 2021
    ticker = "BTC-USD"
    data = yf.download(tickers=ticker, start='2021-01-01')
 
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

    # Tentukan tahun akhir sebagai 2024
    end_year = datetime.now()

# Sidebar to select the start year
    start_year = st.sidebar.selectbox("Periode Forecast", options=range(2021, datetime.now().year + 1), index=0)

    # Generate sparkline data
    np.random.seed(1)
    y = data['Close'].values[-24:]  # Use the last 24 closing prices for sparkline
    x = np.arange(len(y))
    fig = px.line(x=x, y=y, width=400, height=100)
    
    xmin = x[0]
    xmax = x[-1]
    ymin = round(y[0], 1)
    ymax = round(y[-1], 1)

    layout = {
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "yaxis": {"visible": False},
        "xaxis": {
            "nticks": 2,
            "tickmode": "array",
            "tickvals": [xmin, xmax],
            "ticktext": [f"{ymin} <br> {xmin}", f"{ymax} <br> {xmax}"],
            "title_text": None
        },
        "showlegend": False,
        "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
    }
    config = {'displayModeBar': False}

    fig.update_layout(layout)

        # Row A: Logo and basic metrics
    a1, a2, a3 = st.columns(3)

        # Calculate changes
    highest_open_price_change = data['Open'].max() - data['Open'].iloc[-2]
    highest_high_price_change = data['High'].min() - data['High'].iloc[-2]
    highest_volume_change = data['Volume'].min() - data['Volume'].iloc[-2]

    # Sparkline data untuk Open
    sparkline_data_open = data['Open'].iloc[-24:]  # Mengambil 24 harga open terakhir
    x_sparkline_open = np.arange(len(sparkline_data_open))

    # Sparkline data untuk Close
    sparkline_data_high = data['High'].iloc[-24:]  # Mengambil 24 harga penutupan terakhir
    x_sparkline_high = np.arange(len(sparkline_data_high))

    # Sparkline data untuk Volume
    sparkline_data_volume = data['Volume'].iloc[-24:]  # Mengambil 24 volume terakhir
    x_sparkline_volume = np.arange(len(sparkline_data_volume))

    

        # Metrik untuk Open, Close, dan Volume
    with a1:
        if highest_open_price_change >= 2021:
            st.metric("Highest Open Price", f"${data['Open'].max():,.2f}", delta=f"+{highest_open_price_change:.2f}")
        else:
            st.metric("Highest Open Price", f"${data['Open'].max():,.2f}", delta=f"{highest_open_price_change:.2f}")
        # Generate sparkline untuk Open
        fig_sparkline_open = px.line(x=x_sparkline_open, y=sparkline_data_open, width=150, height=50)
        fig_sparkline_open.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "yaxis": {"visible": False},
                "xaxis": {"visible": False},
                "showlegend": False,
                "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
            }
        )
        st.plotly_chart(fig_sparkline_open, use_container_width=True)
        st.markdown("<div style='text-align:center; color:green;'>OPEN KKGI.JK</div>", unsafe_allow_html=True)


    with a2:
        if highest_high_price_change >= 2021:
            st.metric("Highest High Price", f"${data['High'].max():,.2f}", delta=f"+{highest_high_price_change:.2f}")
        else:
            st.metric("Highest High Price", f"${data['High'].max():,.2f}", delta=f"{highest_high_price_change:.2f}")
        # Generate sparkline untuk High
        fig_sparkline_high = px.line(x=x_sparkline_high, y=sparkline_data_high, width=150, height=50)
        fig_sparkline_high.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "yaxis": {"visible": False},
                "xaxis": {"visible": False},
                "showlegend": False,
                "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
            }
        )
        st.plotly_chart(fig_sparkline_high, use_container_width=True)
        st.markdown("<div style='text-align:center; color:green;'>HIGH KKGI.JK</div>", unsafe_allow_html=True)

    with a3:
        if highest_volume_change >= 2021:
            st.metric("Highest Volume", f"{data['Volume'].max():,.2f}", delta=f"+{highest_volume_change:.2f}")
        else:
            st.metric("Highest Volume", f"{data['Volume'].max():,.2f}", delta=f"{highest_volume_change:.2f}")
        # Generate sparkline untuk Volume
        fig_sparkline_volume = px.line(x=x_sparkline_volume, y=sparkline_data_volume, width=150, height=50)
        fig_sparkline_volume.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "yaxis": {"visible": False},
                "xaxis": {"visible": False},
                "showlegend": False,
                "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
            }
        )
        st.plotly_chart(fig_sparkline_volume, use_container_width=True)
        st.markdown("<div style='text-align:center; color:green;'>VOLUME KKGI.JK</div>", unsafe_allow_html=True)


    
    # Calculate Year-over-Year (YoY) Change
    data_filtered = data[data.index.year >= start_year]
    latest_close_price = data_filtered['Close'].iloc[-1]
    earliest_close_price = data_filtered['Close'].iloc[0]
    yearly_change = ((latest_close_price - earliest_close_price) / earliest_close_price) * 100

    # Calculate changes
    highest_close_price_change = data['Close'].max() - data['Close'].iloc[-2]
    lowest_close_price_change = data['Close'].min() - data['Close'].iloc[-2]
    average_daily_volume_change = data['Volume'].mean() - data['Volume'].iloc[-2]
    
    # Row B: Financial metrics and charts
    b1, b2, b3, b4 = st.columns(4)

    # Calculate changes
    highest_close_price_change = data['Close'].max() - data['Close'].iloc[-2]
    lowest_close_price_change = data['Close'].min() - data['Close'].iloc[-2]
    average_daily_volume_change = data['Volume'].mean() - data['Volume'].iloc[-2]

    # Sparkline data untuk perubahan harga tertinggi
    sparkline_data_b1 = data['Close'].iloc[-24:]  # Mengambil 24 harga penutupan terakhir
    x_sparkline_b1 = np.arange(len(sparkline_data_b1))

    # Sparkline data untuk perubahan harga terendah
    sparkline_data_b2 = data['Close'].iloc[-24:]  # Mengambil 24 harga penutupan terakhir
    x_sparkline_b2 = np.arange(len(sparkline_data_b2))

    # Sparkline data untuk perubahan volume harian rata-rata
    sparkline_data_b3 = data['Volume'].iloc[-24:]  # Mengambil 24 volume harian terakhir
    x_sparkline_b3 = np.arange(len(sparkline_data_b3))

    # Sparkline data untuk perubahan harga tertinggi
    sparkline_data = data['Close'].iloc[-24:]  # Mengambil 24 harga penutupan terakhir
    x_sparkline = np.arange(len(sparkline_data))

    with b1:
        if highest_close_price_change >= 2021:
            st.metric("Highest Close Price", f"${data['Close'].max():,.2f}", delta=f"+{highest_close_price_change:.2f}")
        else:
            st.metric("Highest Close Price", f"${data['Close'].max():,.2f}", delta=f"{highest_close_price_change:.2f}")
        # Generate sparkline untuk perubahan harga tertinggi
        fig_sparkline_b1 = px.line(x=x_sparkline_b1, y=sparkline_data_b1, width=150, height=50)
        fig_sparkline_b1.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "yaxis": {"visible": False},
                "xaxis": {"visible": False},
                "showlegend": False,
                "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
            }
        )
        st.plotly_chart(fig_sparkline_b1, use_container_width=True)

    with b2:
        if lowest_close_price_change >= 2021:
            st.metric("Lowest Close Price", f"${data['Close'].min():,.2f}", delta=f"+{lowest_close_price_change:.2f}")
        else:
            st.metric("Lowest Close Price", f"${data['Close'].min():,.2f}", delta=f"{lowest_close_price_change:.2f}")
        # Generate sparkline untuk perubahan harga terendah
        fig_sparkline_b2 = px.line(x=x_sparkline_b2, y=sparkline_data_b2, width=150, height=50)
        fig_sparkline_b2.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "yaxis": {"visible": False},
                "xaxis": {"visible": False},
                "showlegend": False,
                "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
            }
        )
        st.plotly_chart(fig_sparkline_b2, use_container_width=True)

    with b3:
        if average_daily_volume_change >= 2021:
            st.metric("Average Daily Volume", f"{round(data['Volume'].mean(), 2):,.2f}", delta=f"+{average_daily_volume_change:.2f}")
        else:
            st.metric("Average Daily Volume", f"{round(data['Volume'].mean(), 2):,.2f}", delta=f"{average_daily_volume_change:.2f}")
        # Generate sparkline untuk perubahan volume harian rata-rata
        fig_sparkline_b3 = px.line(x=x_sparkline_b3, y=sparkline_data_b3, width=150, height=50)
        fig_sparkline_b3.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "yaxis": {"visible": False},
                "xaxis": {"visible": False},
                "showlegend": False,
                "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
            }
        )
        st.plotly_chart(fig_sparkline_b3, use_container_width=True)


    with b4:
        if start_year > 2013:
            if yearly_change >= 0:
                yearly_change_label = "Yearly Change (Increase)"
            else:
                yearly_change_label = "Yearly Change (Decrease)"
            st.metric(label=yearly_change_label, value=f"{yearly_change:.2f}%", delta=f"{abs(yearly_change):.2f}%")
            
            # Generate sparkline untuk perubahan tahunan
            fig_sparkline = px.line(x=x_sparkline, y=sparkline_data, width=150, height=50)
            fig_sparkline.update_layout(
                {
                    "plot_bgcolor": "rgba(0, 0, 0, 0)",
                    "paper_bgcolor": "rgba(0, 0, 0, 0)",
                    "yaxis": {"visible": False},
                    "xaxis": {"visible": False},
                    "showlegend": False,
                    "margin": {"l":4,"r":4,"t":0, "b":0, "pad": 4}
                }
            )
            st.plotly_chart(fig_sparkline, use_container_width=True)


    # Row C
    c1, c2 = st.columns((7, 3))
    with c1:
        
        # Buat series data dari harga penutupan
        kkgi_close_series = []
        for index, row in data.iterrows():
            kkgi_close_series.append({
                "time": index.strftime('%Y-%m-%d'),
                "value": row['Close']
            })

        priceVolumeChartOptions = {
            "height": 400,
            "rightPriceScale": {
                "scaleMargins": {
                    "top": 0.2,
                    "bottom": 0.25,
                },
                "borderVisible": False,
            },
            "overlayPriceScales": {
                "scaleMargins": {
                    "top": 0.7,
                    "bottom": 0,
                }
            },
            "layout": {
                "background": {
                    "type": 'solid',
                    "color": '#131722'
                },
                "textColor": '#d1d4dc',
            },
            "grid": {
                "vertLines": {
                    "color": 'rgba(42, 46, 57, 0)',
                },
                "horzLines": {
                    "color": 'rgba(42, 46, 57, 0.6)',
                }
            }
        }

        priceVolumeSeries = [
            {
                "type": 'Area',
                "data": kkgi_close_series,
                "options": {
                    "topColor": 'rgba(38,198,218, 0.56)',
                    "bottomColor": 'rgba(38,198,218, 0.04)',
                    "lineColor": 'rgba(38,198,218, 1)',
                    "lineWidth": 2,
                }
            },
            {
                "type": 'Histogram',
                "data": kkgi_close_series,  # Gunakan kembali data harga penutupan untuk histogram
                "options": {
                    "color": '#26a69a',
                    "priceFormat": {
                        "type": 'volume',
                    },
                    "priceScaleId": ""  # Set as an overlay setting,
                },
                "priceScale": {
                    "scaleMargins": {
                        "top": 0.7,
                        "bottom": 0,
                    }
                }
            }
        ]
        renderLightweightCharts([
            {
                "chart": priceVolumeChartOptions,
                "series": priceVolumeSeries
            }
        ], 'priceAndVolume')
    
    # The KKGI.JK table:
    with c2:
        st.write("Real Data")
        st.write(data)  # Menampilkan data KKGI.JK sebagai tabel

        # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'])])
    # Set the color from white to black on range selector buttons
    fig.update_layout(xaxis=dict(rangeselector = dict(font = dict( color = 'black'))))

    st.info('''
        
        ''', icon="🧐")


    z1, z2 = st.columns((7, 3))
    with z1:

        #Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'])])

        # Update layout
        add_range_selector(fig)
        fig.update_layout(
            title="BTC-USD - Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=500,
            template="plotly_dark"
        )

        # Show the chart using Streamlit
        st.plotly_chart(fig)
    with z2:
        # Create progress bar and table for open, high, low
        st.write("Open, High, Low Table")
        st.dataframe(data[['Open', 'High', 'Low']])

    
        
if __name__ == '__main__':
    main()
