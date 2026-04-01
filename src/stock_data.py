"""
Stock Data Loader
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class StockDataGenerator:
    @staticmethod
    def generate_realistic_stock_data(symbol='AAPL', periods=500):
        """Generate realistic OHLCV data"""
        np.random.seed(42)
        
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
        
        # Start with realistic price
        price = 150.0
        data = []
        
        for date in dates:
            # Random walk with drift
            drift = np.random.normal(0.0005, 0.02)
            close = price * (1 + drift)
            
            # OHLC around close
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price
            
            volume = int(np.random.uniform(1000000, 5000000))
            
            data.append({
                'Date': date,
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Volume': volume,
                'Symbol': symbol
            })
            
            price = close
        
        return pd.DataFrame(data)

class StockAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.calculate_indicators()
    
    def calculate_indicators(self):
        """Calculate technical indicators"""
        self.df['SMA_20'] = self.df['Close'].rolling(20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(50).mean()
        self.df['RSI'] = self.calculate_rsi(self.df['Close'])
        self.df['MACD'], self.df['Signal'] = self.calculate_macd(self.df['Close'])
        self.df['BB_Upper'], self.df['BB_Lower'] = self.calculate_bollinger(self.df['Close'])
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    @staticmethod
    def calculate_bollinger(prices, period=20,  num_std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        return sma + (std * num_std), sma - (std * num_std)
    
    def get_statistics(self):
        """Get stock statistics"""
        latest = self.df.iloc[-1]
        first = self.df.iloc[0]
        
        return {
            'current_price': latest['Close'],
            'high_52w': self.df['High'].max(),
            'low_52w': self.df['Low'].min(),
            'avg_volume': self.df['Volume'].mean(),
            'volatility': self.df['Close'].pct_change().std() * 100,
            'total_return': (latest['Close'] - first['Close']) / first['Close'] * 100
        }
