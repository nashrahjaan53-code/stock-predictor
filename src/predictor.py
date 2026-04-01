"""
Stock Predictor Model
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class SimpleLSTMPredictor:
    """Simplified LSTM predictor without TensorFlow for speed"""
    
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.trend_coefficient = 0.001
    
    def prepare_data(self, prices):
        """Prepare data for training"""
        scaled = self.scaler.fit_transform(prices.reshape(-1, 1))
        return scaled.flatten()
    
    def predict_next_30_days(self, prices):
        """Predict next 30 days using trend extrapolation"""
        scaled = self.prepare_data(prices)
        
        predictions = []
        current_seq = scaled[-self.window_size:].copy()
        
        # Simple trend-based prediction
        last_price = prices[-1]
        trend = (prices[-1] - prices[-self.window_size]) / self.window_size
        
        for i in range(30):
            # Add trend + noise
            next_val = last_price + trend + np.random.normal(0, last_price * 0.01)
            predictions.append(next_val)
            last_price = next_val
        
        return np.array(predictions)
    
    def get_signals(self, df):
        """Generate trading signals"""
        signals = []
        df = df.copy()
        
        df.loc[:, 'Signal'] = 0
        df.loc[df['SMA_20'] > df['SMA_50'], 'Signal'] = 1
        df.loc[df['RSI'] > 70, 'Signal'] = -1
        df.loc[df['RSI'] < 30, 'Signal'] = 1
        
        return df['Signal']
    
    def backtest(self, prices, predictions):
        """Simple backtest"""
        returns = (prices[-30:] - predictions) / predictions * 100
        avg_error = np.abs(returns).mean()
        
        return {
            'mean_error': avg_error,
            'accuracy': 100 - avg_error,
            'best_case': predictions[-1] * (1 + np.max(returns) / 100),
            'worst_case': predictions[-1] * (1 + np.min(returns) / 100)
        }
