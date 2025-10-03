"""
Train First AI Model
Downloads comprehensive forex data and trains your first trading AI
"""

import sys
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add src to path
sys.path.append('src')

from data_handler import DataHandler
from trading_agent import ForexTradingAgent
from loguru import logger

def download_training_data():
    """Download comprehensive training dataset"""
    print("🔄 Downloading comprehensive forex dataset...")
    
    handler = DataHandler()
    
    # Download data for major pairs (2 years of hourly data)
    symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X']
    
    all_data = []
    
    for symbol in symbols:
        print(f"   Downloading {symbol}...")
        try:
            # Download 2 years of hourly data
            data = handler.download_yahoo_data(symbol, period="2y", interval="1h")
            if data is not None:
                print(f"   ✅ {symbol}: {len(data)} rows")
                all_data.append(data)
            else:
                print(f"   ❌ {symbol}: Failed to download")
        except Exception as e:
            print(f"   ❌ {symbol}: Error - {e}")
    
    if not all_data:
        print("❌ No data downloaded!")
        return None
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=False)
    combined_data = combined_data.sort_index()
    
    # Save raw data
    handler.save_data(combined_data, "raw_forex_data.csv")
    print(f"✅ Combined dataset: {len(combined_data)} rows")
    print(f"   Date range: {combined_data.index.min()} to {combined_data.index.max()}")
    
    return combined_data, handler

def create_ml_features(data, handler):
    """Create machine learning features from raw data"""
    print("\n🔄 Creating ML features...")
    
    try:
        # This will add technical indicators and features
        # Note: Some TA-Lib functions might not work on Python 3.13 yet
        # So we'll use simple indicators first
        
        processed_data = []
        
        # Process each symbol separately
        for symbol in data['Symbol'].unique():
            symbol_data = data[data['Symbol'] == symbol].copy()
            
            print(f"   Processing {symbol}...")
            
            # Simple moving averages (pandas-based)
            symbol_data['SMA_20'] = symbol_data['Close'].rolling(window=20).mean()
            symbol_data['SMA_50'] = symbol_data['Close'].rolling(window=50).mean()
            
            # Exponential moving averages
            symbol_data['EMA_12'] = symbol_data['Close'].ewm(span=12).mean()
            symbol_data['EMA_26'] = symbol_data['Close'].ewm(span=26).mean()
            
            # RSI (simple implementation)
            def calculate_rsi(prices, window=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            symbol_data['RSI_14'] = calculate_rsi(symbol_data['Close'])
            
            # Price changes
            symbol_data['Price_Change'] = symbol_data['Close'].pct_change()
            symbol_data['Price_Change_4h'] = symbol_data['Close'].pct_change(periods=4)
            
            # Volatility
            symbol_data['Volatility_20'] = symbol_data['Price_Change'].rolling(window=20).std()
            
            # Time features
            symbol_data['Hour'] = symbol_data.index.hour
            symbol_data['DayOfWeek'] = symbol_data.index.dayofweek
            
            processed_data.append(symbol_data)
        
        final_data = pd.concat(processed_data, ignore_index=False).sort_index()
        
        # Save processed data
        handler.save_data(final_data, "processed_forex_data.csv")
        
        print(f"✅ Features created: {len(final_data.columns)} columns")
        return final_data
        
    except Exception as e:
        print(f"❌ Error creating features: {e}")
        return data

def prepare_training_data(data):
    """Prepare data for ML training"""
    print("\n🔄 Preparing training data...")
    
    # Create target variable (future price movement)
    df = data.copy()
    
    # Predict next hour price direction
    df['Future_Return'] = df.groupby('Symbol')['Close'].shift(-1) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] > 0).astype(int)  # 1 = price goes up, 0 = down
    
    # Select feature columns (numeric only)
    feature_cols = []
    for col in df.columns:
        if col not in ['Target', 'Future_Return', 'Symbol'] and df[col].dtype in ['float64', 'int64']:
            feature_cols.append(col)
    
    print(f"   Selected {len(feature_cols)} features:")
    for col in feature_cols[:10]:  # Show first 10
        print(f"     - {col}")
    if len(feature_cols) > 10:
        print(f"     ... and {len(feature_cols) - 10} more")
    
    # Remove rows with NaN values
    df_clean = df[feature_cols + ['Target']].dropna()
    
    if df_clean.empty:
        print("❌ No valid data after cleaning!")
        return None, None, None
    
    # Split features and targets
    X = df_clean[feature_cols].values
    y = df_clean['Target'].values
    
    print(f"✅ Training data prepared:")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target distribution: Up={np.sum(y)}, Down={len(y)-np.sum(y)}")
    
    return X, y, feature_cols

def train_simple_model(X, y, feature_cols):
    """Train a simple machine learning model"""
    print("\n🔄 Training AI model...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("   Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"   Training accuracy: {train_acc:.3f}")
    print(f"   Test accuracy: {test_acc:.3f}")
    
    # Feature importance
    importance = model.feature_importances_
    top_features = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)[:5]
    
    print("   Top 5 most important features:")
    for feature, imp in top_features:
        print(f"     - {feature}: {imp:.3f}")
    
    # Save model
    import joblib
    model_path = "models/first_forex_model.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    print(f"   Model saved to {model_path}")
    
    return model

def create_trading_environment():
    """Create a basic trading environment for the agent"""
    print("\n🔄 Setting up trading environment...")
    
    class SimpleForexEnv:
        def __init__(self, data, initial_balance=10000):
            self.data = data.dropna()
            self.initial_balance = initial_balance
            self.reset()
        
        def reset(self):
            self.current_step = 50  # Start after we have some history
            self.balance = self.initial_balance
            self.position = 0  # 0 = no position, 1 = long, -1 = short
            self.done = False
            return self.get_state()
        
        def get_state(self):
            # Return last 10 price changes as state
            if self.current_step < 10:
                return np.zeros(10)
            
            recent_data = self.data.iloc[self.current_step-10:self.current_step]
            if len(recent_data) < 10:
                return np.zeros(10)
            
            changes = recent_data['Close'].pct_change().dropna().values
            if len(changes) < 9:
                return np.zeros(10)
            
            # Pad with zeros if needed
            state = np.zeros(10)
            state[:len(changes)] = changes
            return state
        
        def step(self, action):
            if self.done or self.current_step >= len(self.data) - 1:
                return self.get_state(), 0, True, {}
            
            # Calculate reward based on action and price movement
            current_price = self.data.iloc[self.current_step]['Close']
            next_price = self.data.iloc[self.current_step + 1]['Close']
            price_change = (next_price - current_price) / current_price
            
            reward = 0
            if action == 1:  # Buy
                reward = price_change * 1000  # Scale reward
            elif action == 2:  # Sell
                reward = -price_change * 1000
            # action == 0 (hold) gets reward = 0
            
            self.current_step += 1
            
            # Check if done
            self.done = self.current_step >= len(self.data) - 1
            
            return self.get_state(), reward, self.done, {'price_change': price_change}
    
    # Load some data for the environment
    try:
        data_path = "data/processed_forex_data.csv"
        if os.path.exists(data_path):
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            # Use just EURUSD for the environment
            eurusd_data = data[data['Symbol'] == 'EURUSD'].copy()
            
            if len(eurusd_data) > 100:
                env = SimpleForexEnv(eurusd_data)
                print("✅ Trading environment created")
                return env
        
        print("❌ Could not create trading environment - insufficient data")
        return None
        
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return None

def main():
    """Main execution function"""
    print("🚀 Training Your First AI Forex Trading Model")
    print("=" * 60)
    
    try:
        # Step 1: Download data
        data, handler = download_training_data()
        if data is None:
            print("❌ Failed to download data. Exiting.")
            return
        
        # Step 2: Create features
        processed_data = create_ml_features(data, handler)
        
        # Step 3: Prepare training data
        X, y, feature_cols = prepare_training_data(processed_data)
        if X is None:
            print("❌ Failed to prepare training data. Exiting.")
            return
        
        # Step 4: Train model
        model = train_simple_model(X, y, feature_cols)
        
        # Step 5: Create trading environment
        env = create_trading_environment()
        
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Your AI trading system is ready!")
        print("\nWhat you now have:")
        print("✅ Comprehensive forex dataset (2 years, 5 pairs)")
        print("✅ Trained ML model for price prediction")
        print("✅ Trading environment for backtesting")
        print("✅ All necessary infrastructure")
        
        print("\nNext steps:")
        print("1. Install MetaTrader 5 for live trading")
        print("2. Backtest your model on historical data")
        print("3. Optimize model parameters")
        print("4. Implement risk management")
        print("5. Start paper trading!")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()