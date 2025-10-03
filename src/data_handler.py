"""
Data Handler
Downloads, processes, and manages historical forex data for training
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import pickle
from loguru import logger
import requests
from alpha_vantage.foreignexchange import ForeignExchange
try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    ta = None
    TALIB_AVAILABLE = False


class DataHandler:
    """
    Handles downloading, processing, and storing historical forex data
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataHandler
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        self.ensure_data_dir()
        
        # Common forex pairs
        self.major_pairs = [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X',
            'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X'
        ]
        
        # Technical indicators to calculate
        self.indicators = [
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14',
            'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'ATR_14'
        ]
    
    def ensure_data_dir(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def download_yahoo_data(self, symbol: str, period: str = "2y", 
                           interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        Download forex data from Yahoo Finance
        
        Args:
            symbol: Forex symbol (e.g., 'EURUSD=X')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Downloading {symbol} data from Yahoo Finance...")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.error(f"No data received for {symbol}")
                return None
            
            # Clean up data
            data = data.dropna()
            data.columns = data.columns.str.title()  # Capitalize column names
            
            # Add symbol column
            data['Symbol'] = symbol.replace('=X', '')
            
            logger.info(f"Downloaded {len(data)} rows of {symbol} data")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading {symbol} from Yahoo Finance: {e}")
            return None
    
    def download_alpha_vantage_data(self, from_currency: str, to_currency: str,
                                   api_key: str, outputsize: str = "full") -> Optional[pd.DataFrame]:
        """
        Download forex data from Alpha Vantage
        
        Args:
            from_currency: Base currency (e.g., 'EUR')
            to_currency: Quote currency (e.g., 'USD')
            api_key: Alpha Vantage API key
            outputsize: 'compact' (last 100 points) or 'full' (20+ years)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Downloading {from_currency}/{to_currency} from Alpha Vantage...")
            
            fx = ForeignExchange(key=api_key, output_format='pandas')
            data, meta_data = fx.get_currency_exchange_daily(
                from_symbol=from_currency,
                to_symbol=to_currency,
                outputsize=outputsize
            )
            
            if data.empty:
                logger.error(f"No data received for {from_currency}/{to_currency}")
                return None
            
            # Rename columns to match our standard format
            data.columns = ['Open', 'High', 'Low', 'Close']
            data.index.name = 'Date'
            
            # Add symbol column
            data['Symbol'] = f"{from_currency}{to_currency}"
            
            logger.info(f"Downloaded {len(data)} rows of {from_currency}/{to_currency} data")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading from Alpha Vantage: {e}")
            return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for forex data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        df = data.copy()
        
        try:
            if TALIB_AVAILABLE:
                # Use TA-Lib if available
                # Simple Moving Averages
                df['SMA_20'] = ta.SMA(df['Close'], timeperiod=20)
                df['SMA_50'] = ta.SMA(df['Close'], timeperiod=50)
                
                # Exponential Moving Averages
                df['EMA_12'] = ta.EMA(df['Close'], timeperiod=12)
                df['EMA_26'] = ta.EMA(df['Close'], timeperiod=26)
                
                # RSI
                df['RSI_14'] = ta.RSI(df['Close'], timeperiod=14)
                
                # MACD
                macd, macd_signal, macd_hist = ta.MACD(df['Close'])
                df['MACD'] = macd
                df['MACD_signal'] = macd_signal
                df['MACD_hist'] = macd_hist
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = ta.BBANDS(df['Close'], timeperiod=20)
                df['BB_upper'] = bb_upper
                df['BB_middle'] = bb_middle
                df['BB_lower'] = bb_lower
                
                # Average True Range
                df['ATR_14'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
                
                logger.info("Calculated technical indicators using TA-Lib")
            else:
                # Use pandas-based calculations as fallback
                logger.info("TA-Lib not available, using pandas-based indicators")
                
                # Simple Moving Averages
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                
                # Exponential Moving Averages
                df['EMA_12'] = df['Close'].ewm(span=12).mean()
                df['EMA_26'] = df['Close'].ewm(span=26).mean()
                
                # RSI calculation
                def calculate_rsi(prices, window=14):
                    delta = prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                    rs = gain / loss
                    return 100 - (100 / (1 + rs))
                
                df['RSI_14'] = calculate_rsi(df['Close'])
                
                # MACD
                ema12 = df['Close'].ewm(span=12).mean()
                ema26 = df['Close'].ewm(span=26).mean()
                df['MACD'] = ema12 - ema26
                df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
                df['MACD_hist'] = df['MACD'] - df['MACD_signal']
                
                # Bollinger Bands
                rolling_mean = df['Close'].rolling(window=20).mean()
                rolling_std = df['Close'].rolling(window=20).std()
                df['BB_upper'] = rolling_mean + (rolling_std * 2)
                df['BB_middle'] = rolling_mean
                df['BB_lower'] = rolling_mean - (rolling_std * 2)
                
                # Average True Range (simplified)
                high_low = df['High'] - df['Low']
                high_close = np.abs(df['High'] - df['Close'].shift())
                low_close = np.abs(df['Low'] - df['Close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                df['ATR_14'] = true_range.rolling(window=14).mean()
                
                logger.info("Calculated technical indicators using pandas")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        
        return df
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for machine learning
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            
        Returns:
            DataFrame with additional features
        """
        df = data.copy()
        
        try:
            # Price changes and returns
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_1h'] = df['Close'].pct_change(periods=1)
            df['Price_Change_4h'] = df['Close'].pct_change(periods=4)
            df['Price_Change_24h'] = df['Close'].pct_change(periods=24)
            
            # Volatility measures
            df['Volatility_20'] = df['Price_Change'].rolling(window=20).std()
            df['Volatility_50'] = df['Price_Change'].rolling(window=50).std()
            
            # Price position relative to moving averages
            df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
            df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
            
            # Volume features (if available)
            if 'Volume' in df.columns:
                df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
            
            # Time-based features
            df['Hour'] = df.index.hour
            df['DayOfWeek'] = df.index.dayofweek
            df['Month'] = df.index.month
            
            # Lag features (previous values)
            for lag in [1, 2, 3, 6, 12, 24]:
                df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
                df[f'RSI_lag_{lag}'] = df['RSI_14'].shift(lag)
            
            logger.info("Created additional features for ML")
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
        
        return df
    
    def prepare_training_data(self, data: pd.DataFrame, 
                            prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning training
        
        Args:
            data: DataFrame with features
            prediction_horizon: How many periods ahead to predict
            
        Returns:
            Tuple of (features, targets)
        """
        df = data.copy()
        
        # Create target variable (future price direction)
        df['Future_Return'] = df['Close'].shift(-prediction_horizon) / df['Close'] - 1
        df['Target'] = (df['Future_Return'] > 0).astype(int)  # 1 for up, 0 for down
        
        # Select feature columns (exclude target and non-feature columns)
        feature_cols = [col for col in df.columns if col not in [
            'Target', 'Future_Return', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume'
        ]]
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if df.empty:
            logger.error("No valid data after removing NaN values")
            return None, None
        
        features = df[feature_cols].values
        targets = df['Target'].values
        
        logger.info(f"Prepared training data: {features.shape[0]} samples, {features.shape[1]} features")
        
        return features, targets
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """Save data to file"""
        filepath = os.path.join(self.data_dir, filename)
        
        if filename.endswith('.csv'):
            data.to_csv(filepath)
        elif filename.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            # Default to CSV
            data.to_csv(filepath + '.csv')
        
        logger.info(f"Saved data to {filepath}")
    
    def load_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from file"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None
        
        try:
            if filename.endswith('.csv'):
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            elif filename.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                # Try CSV first
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            logger.info(f"Loaded data from {filepath}: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            return None
    
    def download_and_prepare_dataset(self, symbols: List[str] = None, 
                                   period: str = "2y") -> pd.DataFrame:
        """
        Download and prepare a complete dataset for training
        
        Args:
            symbols: List of symbols to download (uses major pairs if None)
            period: Data period
            
        Returns:
            Combined dataset ready for training
        """
        if symbols is None:
            symbols = self.major_pairs
        
        all_data = []
        
        for symbol in symbols:
            logger.info(f"Processing {symbol}...")
            
            # Download data
            data = self.download_yahoo_data(symbol, period=period)
            if data is None:
                continue
            
            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)
            
            # Create features
            data = self.create_features(data)
            
            all_data.append(data)
        
        if not all_data:
            logger.error("No data was successfully downloaded")
            return pd.DataFrame()
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=False)
        combined_data = combined_data.sort_index()
        
        logger.info(f"Combined dataset shape: {combined_data.shape}")
        
        # Save the dataset
        self.save_data(combined_data, "forex_dataset.csv")
        
        return combined_data


# Example usage
if __name__ == "__main__":
    handler = DataHandler()
    
    # Download and prepare dataset
    dataset = handler.download_and_prepare_dataset()
    
    if not dataset.empty:
        print(f"Dataset shape: {dataset.shape}")
        print(f"Columns: {list(dataset.columns)}")
        print(f"Date range: {dataset.index.min()} to {dataset.index.max()}")
        
        # Prepare training data
        features, targets = handler.prepare_training_data(dataset)
        if features is not None:
            print(f"Training data: {features.shape[0]} samples, {features.shape[1]} features")
            print(f"Target distribution: {np.bincount(targets)}")