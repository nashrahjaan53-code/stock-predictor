"""
Training script
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.stock_data import StockDataGenerator, StockAnalyzer
from src.predictor import SimpleLSTMPredictor

def main():
    print("📈 Stock Predictor Training Pipeline")
    
    # Generate data
    print("\n📊 Generating stock data...")
    generator = StockDataGenerator()
    df = generator.generate_realistic_stock_data('AAPL', 500)
    
    # Save data
    df.to_csv('data/stock_data.csv', index=False)
    print(f"✓ Generated {len(df)} days of OHLCV data")
    
    # Analyze
    analyzer = StockAnalyzer(df)
    stats = analyzer.get_statistics()
    
    print(f"\n📈 Stock Statistics:")
    print(f"✓ Current Price: ${stats['current_price']:.2f}")
    print(f"✓ 52W High: ${stats['high_52w']:.2f}")
    print(f"✓ 52W Low: ${stats['low_52w']:.2f}")
    print(f"✓ Volatility: {stats['volatility']:.2f}%")
    print(f"✓ Total Return: {stats['total_return']:.2f}%")
    
    # Predict
    print(f"\n🔮 Training predictor...")
    predictor = SimpleLSTMPredictor()
    predictions = predictor.predict_next_30_days(df['Close'].values)
    
    print(f"✓ 30-day forecast ready")
    print(f"✓ Predicted price in 30 days: ${predictions[-1]:.2f}")
    
    # Save predictions
    pd.DataFrame({
        'Day': range(1, 31),
        'Predicted_Price': predictions
    }).to_csv('data/predictions.csv', index=False)
    
    print("\n✅ Training Complete! Start dashboard with: streamlit run dashboard/app.py")

if __name__ == '__main__':
    import pandas as pd
    main()
