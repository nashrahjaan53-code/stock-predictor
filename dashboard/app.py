"""
Stock Predictor Dashboard
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stock_data import StockDataGenerator, StockAnalyzer
from src.predictor import SimpleLSTMPredictor

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

@st.cache_resource
def load_stock_data():
    df = StockDataGenerator.generate_realistic_stock_data('AAPL', 500)
    return StockAnalyzer(df)

analyzer = load_stock_data()

st.title("📈 Stock Price Prediction Dashboard")

tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🔮 Forecast", "📈 Indicators"])

with tab1:
    st.header("Stock Analysis")
    
    stats = analyzer.get_statistics()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${stats['current_price']:.2f}", delta=f"+{stats['total_return']:.1f}%")
    with col2:
        st.metric("52W High", f"${stats['high_52w']:.2f}")
    with col3:
        st.metric("52W Low", f"${stats['low_52w']:.2f}")
    with col4:
        st.metric("Volatility", f"{stats['volatility']:.2f}%")
    
    st.markdown("---")
    
    # Historical price
    df = analyzer.df
    fig = px.line(df, x='Date', y='Close', title='Historical Price',
                 color_discrete_sequence=['#1f77b4'], markers=True)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("30-Day Forecast")
    
    predictor = SimpleLSTMPredictor()
    prices = analyzer.df['Close'].values
    predictions = predictor.predict_next_30_days(prices)
    
    # Create forecast dataframe
    last_date = analyzer.df['Date'].max()
    forecast_df = pd.DataFrame({
        'Day': pd.date_range(start=last_date, periods=30, freq='D')[1:],
        'Predicted_Price': predictions
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(forecast_df, x='Day', y='Predicted_Price',
                     title='Price Forecast',
                     color_discrete_sequence=['#2ca02c'],
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Current Price", f"${prices[-1]:.2f}")
        st.metric("Forecast (30d)", f"${predictions[-1]:.2f}", 
                 delta=f"{(predictions[-1]-prices[-1])/prices[-1]*100:.1f}%")
        st.metric("Best Case", f"${max(predictions):.2f}")
        st.metric("Worst Case", f"${min(predictions):.2f}")

with tab3:
    st.header("Technical Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        df_recent = analyzer.df.tail(100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_recent['Date'], y=df_recent['Close'],
                                name='Close', mode='lines'))
        fig.add_trace(go.Scatter(x=df_recent['Date'], y=df_recent['SMA_20'],
                                name='SMA 20', mode='lines'))
        fig.add_trace(go.Scatter(x=df_recent['Date'], y=df_recent['SMA_50'],
                                name='SMA 50', mode='lines'))
        fig.update_layout(title='Moving Averages', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(df_recent, x='Date', y='RSI',
                     title='RSI Indicator',
                     color_discrete_sequence=['#ff7f0e'])
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        st.plotly_chart(fig, use_container_width=True)
