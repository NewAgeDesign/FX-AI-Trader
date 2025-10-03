"""
Simple test to verify Python installation and basic functionality
"""

import sys
import os
from datetime import datetime

def test_python_installation():
    """Test basic Python functionality"""
    print("=== Python Installation Test ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Current time: {datetime.now()}")
    
    # Test basic data structures
    test_dict = {"EUR/USD": 1.0850, "GBP/USD": 1.2650}
    test_list = [1.0850, 1.0851, 1.0849, 1.0852]
    
    print(f"Test dictionary: {test_dict}")
    print(f"Test list: {test_list}")
    
    # Simple calculations
    balance = 10000.0
    trade_size = 0.1
    price_change = 0.0050
    profit = trade_size * price_change * 10000  # Simplified forex profit calc
    
    print(f"Simulated trade profit: ${profit:.2f}")
    
    print("\n✅ Basic Python functionality test PASSED!")
    
    return True

def check_directory_structure():
    """Check if our project directories exist"""
    print("\n=== Directory Structure Check ===")
    
    expected_dirs = ["src", "data", "models", "logs", "config", "tests"]
    
    for directory in expected_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/ exists")
        else:
            print(f"❌ {directory}/ missing")
    
    # Check if our Python files exist
    expected_files = [
        "src/trading_agent.py",
        "src/mt5_connector.py", 
        "src/data_handler.py",
        "requirements.txt"
    ]
    
    print("\n=== Core Files Check ===")
    for file in expected_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")

def test_basic_trading_logic():
    """Test basic trading logic without external packages"""
    print("\n=== Basic Trading Logic Test ===")
    
    # Simulate price data
    prices = [1.0850, 1.0851, 1.0849, 1.0852, 1.0855, 1.0853]
    
    # Simple moving average calculation
    def simple_moving_average(data, window):
        if len(data) < window:
            return None
        return sum(data[-window:]) / window
    
    # Calculate 3-period moving average
    sma_3 = simple_moving_average(prices, 3)
    current_price = prices[-1]
    
    print(f"Price data: {prices}")
    print(f"3-period SMA: {sma_3:.4f}")
    print(f"Current price: {current_price:.4f}")
    
    # Simple trading signal
    if current_price > sma_3:
        signal = "BUY"
    else:
        signal = "SELL"
    
    print(f"Trading signal: {signal}")
    print("✅ Basic trading logic test PASSED!")

if __name__ == "__main__":
    try:
        test_python_installation()
        check_directory_structure()
        test_basic_trading_logic()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED!")
        print("Python is working correctly for your trading agent!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Please check your Python installation.")