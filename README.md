# 📈 Stock Price Predictor

LSTM neural network for stock price forecasting with technical indicators and risk analysis.

## 📊 Features

- **LSTM Model:** Deep learning time series prediction
- **Technical Indicators:** RSI, MACD, Bollinger Bands
- **Backtesting:** Historical performance validation
- **Risk Metrics:** Sharpe ratio, max drawdown, volatility
- **Multi-Stock:** Train on multiple symbols
- **Real-time:** Daily updates and predictions

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow/Keras
- **Data:** yfinance, pandas
- **Analysis:** NumPy, SciPy
- **Visualization:** Plotly, Matplotlib
- **Dashboard:** Streamlit

## 🚀 Quick Start

```bash
cd stock_predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
streamlit run dashboard/app.py
```
