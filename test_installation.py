"""
Test Installation Script
Verifies all packages are working and downloads sample forex data
"""

import sys
import os
from datetime import datetime, timedelta

def test_imports():
    """Test that all required packages can be imported"""
    print("=== Testing Package Imports ===")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
        
        import seaborn as sns
        print("✅ Seaborn imported successfully")
        
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
        
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__} imported successfully")
        
        import yfinance as yf
        print("✅ yfinance imported successfully")
        
        import gymnasium as gym
        print("✅ Gymnasium imported successfully")
        
        import MetaTrader5 as mt5
        print("✅ MetaTrader5 imported successfully")
        
        from loguru import logger
        print("✅ Loguru imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_download():
    """Test downloading forex data"""
    print("\n=== Testing Data Download ===")
    
    try:
        import yfinance as yf
        import pandas as pd
        
        # Download a small sample of EUR/USD data
        print("Downloading EUR/USD data...")
        ticker = yf.Ticker("EURUSD=X")
        data = ticker.history(period="5d", interval="1h")
        
        if not data.empty:
            print(f"✅ Downloaded {len(data)} rows of EUR/USD data")
            print(f"   Date range: {data.index.min()} to {data.index.max()}")
            print(f"   Latest price: ${data['Close'].iloc[-1]:.4f}")
            
            # Save sample data
            os.makedirs("data", exist_ok=True)
            data.to_csv("data/sample_eurusd.csv")
            print("   Saved to data/sample_eurusd.csv")
            
            return True
        else:
            print("❌ No data downloaded")
            return False
            
    except Exception as e:
        print(f"❌ Data download error: {e}")
        return False

def test_ml_functionality():
    """Test basic machine learning functionality"""
    print("\n=== Testing ML Functionality ===")
    
    try:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        import tensorflow as tf
        
        # Create some dummy forex-like data
        print("Creating sample trading data...")
        np.random.seed(42)
        n_samples = 1000
        
        # Simulate price features (SMA, RSI, etc.)
        features = np.random.random((n_samples, 5))
        
        # Simulate trading targets (1=buy, 0=sell)
        targets = np.random.choice([0, 1], size=n_samples)
        
        # Test scikit-learn
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"✅ Scikit-learn model trained with {accuracy:.2%} accuracy")
        
        # Test TensorFlow
        tf_model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("✅ TensorFlow model created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ ML functionality error: {e}")
        return False

def test_trading_agent_basic():
    """Test basic trading agent functionality"""
    print("\n=== Testing Trading Agent ===")
    
    try:
        # Add src to path to import our modules
        sys.path.append('src')
        
        from trading_agent import ForexTradingAgent
        
        # Create agent
        agent = ForexTradingAgent(initial_balance=10000.0)
        print("✅ Trading agent created successfully")
        
        # Test basic functionality
        import numpy as np
        market_state = np.random.random(10)
        action, confidence = agent.predict_action(market_state)
        
        print(f"   Agent action: {action} (confidence: {confidence:.2f})")
        
        # Test trade execution
        if action != "hold":
            result = agent.execute_trade("EURUSD", action, 0.1, 1.0850)
            print(f"   Trade result: {result['status']}")
        
        # Get performance metrics
        metrics = agent.get_performance_metrics()
        print(f"   Current balance: ${metrics['current_balance']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading agent error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing AI Forex Trading Agent Installation")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    # Run tests
    if test_imports():
        tests_passed += 1
        
    if test_data_download():
        tests_passed += 1
        
    if test_ml_functionality():
        tests_passed += 1
        
    if test_trading_agent_basic():
        tests_passed += 1
    
    # Results
    print("\n" + "=" * 50)
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("Your AI Forex Trading Agent is ready to use!")
        print("\nNext steps:")
        print("1. Install MetaTrader 5 (if you want live trading)")
        print("2. Download more historical data")
        print("3. Train your first AI model")
    else:
        print(f"⚠️  {tests_passed}/{total_tests} tests passed")
        print("Some components may need attention")
    
    print("=" * 50)

if __name__ == "__main__":
    main()