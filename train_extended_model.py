"""
Advanced Multi-Asset AI Training
Trains sophisticated models on Forex, Gold, Silver, Oil across multiple timeframes
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')

from data_handler import DataHandler
from trading_agent import ForexTradingAgent
from loguru import logger

def load_all_available_data():
    """Load all available datasets"""
    print("🔄 Loading All Available Data...")
    print("=" * 50)
    
    handler = DataHandler()
    datasets = []
    data_summary = []
    
    # Check for different data sources
    data_sources = {
        'Forex (Original)': 'data/processed_forex_data.csv',
        'Commodities (Yahoo)': 'data/commodities/raw_commodities_data.csv', 
        'MT5 Imports': 'data/imported/mt5_imported_data.csv',
        'Combined Dataset': 'data/combined/mega_trading_dataset.csv'
    }
    
    for source_name, file_path in data_sources.items():
        if os.path.exists(file_path):
            try:
                print(f"   Loading {source_name}...")
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                if len(data) > 0:
                    datasets.append(data)
                    data_summary.append({
                        'Source': source_name,
                        'Rows': len(data),
                        'Columns': len(data.columns),
                        'Date_Range': f"{data.index.min()} to {data.index.max()}",
                        'Symbols': data['Symbol'].nunique() if 'Symbol' in data.columns else 'Unknown'
                    })
                    print(f"   ✅ {source_name}: {len(data):,} rows, {data['Symbol'].nunique() if 'Symbol' in data.columns else '?'} symbols")
                else:
                    print(f"   ⚠️  {source_name}: File exists but empty")
                    
            except Exception as e:
                print(f"   ❌ {source_name}: Error loading - {e}")
        else:
            print(f"   ⚠️  {source_name}: File not found")
    
    if not datasets:
        print("❌ No data found! Run download scripts first.")
        return None, None
    
    # Use the largest/most comprehensive dataset
    if len(datasets) == 1:
        master_data = datasets[0]
        print(f"\n✅ Using single dataset: {len(master_data):,} rows")
    else:
        # Find the combined dataset or use the largest
        combined_idx = next((i for i, summary in enumerate(data_summary) 
                           if 'Combined' in summary['Source']), -1)
        
        if combined_idx >= 0:
            master_data = datasets[combined_idx]
            print(f"\n✅ Using combined dataset: {len(master_data):,} rows")
        else:
            # Combine all datasets
            print("\n🔄 Combining all datasets...")
            master_data = pd.concat(datasets, ignore_index=False).drop_duplicates()
            master_data = master_data.sort_index()
            print(f"✅ Combined dataset: {len(master_data):,} rows")
    
    return master_data, pd.DataFrame(data_summary)

def create_advanced_features(data):
    """Create advanced ML features for multi-asset trading"""
    print("\n🔄 Creating Advanced Features...")
    
    df = data.copy()
    processed_data = []
    
    # Process each symbol separately to avoid data leakage
    for symbol in df['Symbol'].unique():
        symbol_data = df[df['Symbol'] == symbol].copy().sort_index()
        
        if len(symbol_data) < 100:  # Skip symbols with insufficient data
            continue
            
        print(f"   Processing {symbol}... ({len(symbol_data)} rows)")
        
        try:
            # Basic price features
            symbol_data['Returns'] = symbol_data['Close'].pct_change()
            symbol_data['Log_Returns'] = np.log(symbol_data['Close'] / symbol_data['Close'].shift(1))
            
            # Multi-timeframe moving averages
            for period in [5, 10, 20, 50, 100]:
                symbol_data[f'SMA_{period}'] = symbol_data['Close'].rolling(period).mean()
                symbol_data[f'EMA_{period}'] = symbol_data['Close'].ewm(span=period).mean()
                
                # Price relative to MA
                symbol_data[f'Price_vs_SMA_{period}'] = (symbol_data['Close'] - symbol_data[f'SMA_{period}']) / symbol_data[f'SMA_{period}']
            
            # Volatility features
            for period in [10, 20, 50]:
                symbol_data[f'Volatility_{period}'] = symbol_data['Returns'].rolling(period).std()
                symbol_data[f'ATR_{period}'] = (symbol_data[['High', 'Low', 'Close']].max(axis=1) - 
                                               symbol_data[['High', 'Low', 'Close']].min(axis=1)).rolling(period).mean()
            
            # RSI for multiple periods
            for period in [14, 21]:
                delta = symbol_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss
                symbol_data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = symbol_data['Close'].ewm(span=12).mean()
            ema26 = symbol_data['Close'].ewm(span=26).mean()
            symbol_data['MACD'] = ema12 - ema26
            symbol_data['MACD_Signal'] = symbol_data['MACD'].ewm(span=9).mean()
            symbol_data['MACD_Histogram'] = symbol_data['MACD'] - symbol_data['MACD_Signal']
            
            # Bollinger Bands
            for period in [20, 50]:
                rolling_mean = symbol_data['Close'].rolling(period).mean()
                rolling_std = symbol_data['Close'].rolling(period).std()
                symbol_data[f'BB_Upper_{period}'] = rolling_mean + (rolling_std * 2)
                symbol_data[f'BB_Lower_{period}'] = rolling_mean - (rolling_std * 2)
                symbol_data[f'BB_Width_{period}'] = symbol_data[f'BB_Upper_{period}'] - symbol_data[f'BB_Lower_{period}']
                symbol_data[f'BB_Position_{period}'] = (symbol_data['Close'] - symbol_data[f'BB_Lower_{period}']) / symbol_data[f'BB_Width_{period}']
            
            # Price momentum
            for period in [1, 5, 10, 20]:
                symbol_data[f'Momentum_{period}'] = symbol_data['Close'] / symbol_data['Close'].shift(period) - 1
            
            # Volume features (if available)
            if 'Volume' in symbol_data.columns and symbol_data['Volume'].sum() > 0:
                for period in [10, 20]:
                    symbol_data[f'Volume_SMA_{period}'] = symbol_data['Volume'].rolling(period).mean()
                    symbol_data[f'Volume_Ratio_{period}'] = symbol_data['Volume'] / symbol_data[f'Volume_SMA_{period}']
            
            # Time-based features
            try:
                symbol_data['Hour'] = symbol_data.index.hour
                symbol_data['DayOfWeek'] = symbol_data.index.dayofweek
                symbol_data['Month'] = symbol_data.index.month
                symbol_data['Quarter'] = symbol_data.index.quarter
            except AttributeError:
                # Fallback if index doesn't have time attributes
                symbol_data['Hour'] = pd.to_datetime(symbol_data.index).hour
                symbol_data['DayOfWeek'] = pd.to_datetime(symbol_data.index).dayofweek
                symbol_data['Month'] = pd.to_datetime(symbol_data.index).month
                symbol_data['Quarter'] = pd.to_datetime(symbol_data.index).quarter
            
            # Lag features (previous values)
            for lag in [1, 2, 3, 5]:
                symbol_data[f'Close_lag_{lag}'] = symbol_data['Close'].shift(lag)
                symbol_data[f'Returns_lag_{lag}'] = symbol_data['Returns'].shift(lag)
                symbol_data[f'RSI_14_lag_{lag}'] = symbol_data['RSI_14'].shift(lag)
            
            processed_data.append(symbol_data)
            
        except Exception as e:
            print(f"   ❌ Error processing {symbol}: {e}")
    
    if processed_data:
        final_data = pd.concat(processed_data, ignore_index=False)
        final_data = final_data.sort_index()
        
        print(f"✅ Advanced features created: {len(final_data.columns)} total columns")
        print(f"   Processed symbols: {len(processed_data)}")
        
        return final_data
    else:
        print("❌ No data was processed!")
        return data

def prepare_multi_target_data(data):
    """Prepare data with multiple prediction targets"""
    print("\n🔄 Preparing Multi-Target Training Data...")
    
    df = data.copy()
    
    # Create multiple prediction targets
    targets_created = []
    
    for symbol in df['Symbol'].unique():
        symbol_mask = df['Symbol'] == symbol
        symbol_data = df[symbol_mask].copy()
        
        if len(symbol_data) < 50:
            continue
        
        # Target 1: Next period direction (1h ahead)
        df.loc[symbol_mask, 'Target_Direction_1h'] = (
            symbol_data['Close'].shift(-1) > symbol_data['Close']
        ).astype(int)
        
        # Target 2: Strong move (>0.5% change)
        df.loc[symbol_mask, 'Target_Strong_Move'] = (
            np.abs(symbol_data['Close'].shift(-1) / symbol_data['Close'] - 1) > 0.005
        ).astype(int)
        
        # Target 3: Direction over next 5 periods
        df.loc[symbol_mask, 'Target_Direction_5h'] = (
            symbol_data['Close'].shift(-5) > symbol_data['Close']
        ).astype(int)
        
        targets_created.append(symbol)
    
    # Select feature columns
    feature_columns = []
    for col in df.columns:
        if (col not in ['Target_Direction_1h', 'Target_Strong_Move', 'Target_Direction_5h', 'Symbol', 'Asset_Type'] and 
            df[col].dtype in ['float64', 'int64', 'float32', 'int32']):
            feature_columns.append(col)
    
    # Remove rows with NaN values
    required_columns = feature_columns + ['Target_Direction_1h', 'Target_Strong_Move', 'Target_Direction_5h']
    df_clean = df[required_columns].dropna()
    
    if df_clean.empty:
        print("❌ No valid data after cleaning!")
        return None, None, None
    
    print(f"✅ Multi-target data prepared:")
    print(f"   Samples: {len(df_clean):,}")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Symbols processed: {len(targets_created)}")
    
    # Split features and targets
    X = df_clean[feature_columns].values
    y1 = df_clean['Target_Direction_1h'].values  # 1h direction
    y2 = df_clean['Target_Strong_Move'].values   # Strong move
    y3 = df_clean['Target_Direction_5h'].values  # 5h direction
    
    return X, (y1, y2, y3), feature_columns

def train_ensemble_models(X, y_tuple, feature_columns):
    """Train ensemble of models for different targets"""
    print("\n🔄 Training Ensemble Models...")
    
    y1, y2, y3 = y_tuple
    target_names = ['Direction_1h', 'Strong_Move', 'Direction_5h']
    
    models = {}
    results = {}
    
    # Split data
    X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=42, stratify=y1)
    _, _, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=42, stratify=y2)
    _, _, y3_train, y3_test = train_test_split(X, y3, test_size=0.2, random_state=42, stratify=y3)
    
    y_trains = [y1_train, y2_train, y3_train]
    y_tests = [y1_test, y2_test, y3_test]
    
    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")
    
    for i, (target_name, y_train, y_test) in enumerate(zip(target_names, y_trains, y_tests)):
        print(f"\n   🎯 Training {target_name} models...")
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        # Gradient Boosting  
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        gb_accuracy = accuracy_score(y_test, gb_pred)
        
        # Store best model
        if rf_accuracy > gb_accuracy:
            best_model = rf_model
            best_accuracy = rf_accuracy
            best_name = "Random Forest"
        else:
            best_model = gb_model  
            best_accuracy = gb_accuracy
            best_name = "Gradient Boosting"
        
        models[target_name] = best_model
        results[target_name] = {
            'model_type': best_name,
            'accuracy': best_accuracy,
            'rf_accuracy': rf_accuracy,
            'gb_accuracy': gb_accuracy
        }
        
        print(f"     Random Forest: {rf_accuracy:.3f}")
        print(f"     Gradient Boosting: {gb_accuracy:.3f}")
        print(f"     🏆 Best: {best_name} ({best_accuracy:.3f})")
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
            top_features = sorted(zip(feature_columns, importance), key=lambda x: x[1], reverse=True)[:5]
            print(f"     Top features:")
            for feat, imp in top_features:
                print(f"       - {feat}: {imp:.3f}")
    
    # Save models
    print(f"\n💾 Saving models...")
    os.makedirs("models/ensemble", exist_ok=True)
    
    import joblib
    for target_name, model in models.items():
        model_path = f"models/ensemble/{target_name}_model.pkl"
        joblib.dump(model, model_path)
        print(f"   ✅ {target_name}: {model_path}")
    
    # Save feature columns
    with open("models/ensemble/feature_columns.txt", "w") as f:
        f.write("\\n".join(feature_columns))
    print(f"   ✅ Feature columns: models/ensemble/feature_columns.txt")
    
    return models, results

def create_performance_report(results, data_summary):
    """Create comprehensive performance report"""
    print("\n📊 Creating Performance Report...")
    
    report = f"""# 🤖 AI Forex Trading Model - Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Dataset Summary
"""
    
    if data_summary is not None:
        for _, row in data_summary.iterrows():
            report += f"""
### {row['Source']}
- **Rows**: {row['Rows']:,}
- **Columns**: {row['Columns']}
- **Symbols**: {row['Symbols']}
- **Date Range**: {row['Date_Range']}
"""
    
    report += f"""
## 🎯 Model Performance

"""
    
    for target_name, result in results.items():
        report += f"""### {target_name.replace('_', ' ').title()}
- **Best Model**: {result['model_type']}
- **Accuracy**: {result['accuracy']:.1%}
- **Random Forest**: {result['rf_accuracy']:.1%}
- **Gradient Boosting**: {result['gb_accuracy']:.1%}

"""
    
    report += f"""
## 🚀 Usage Instructions

### Load Trained Models
```python
import joblib
import pandas as pd

# Load models
direction_1h = joblib.load('models/ensemble/Direction_1h_model.pkl')
strong_move = joblib.load('models/ensemble/Strong_Move_model.pkl')
direction_5h = joblib.load('models/ensemble/Direction_5h_model.pkl')

# Load feature columns
with open('models/ensemble/feature_columns.txt', 'r') as f:
    feature_columns = f.read().strip().split('\\n')
```

### Make Predictions
```python
# Prepare your market data with same features
market_data = prepare_features(current_market_data)
features = market_data[feature_columns].values

# Get predictions
pred_1h = direction_1h.predict_proba(features)[:, 1]  # Probability of up move
pred_strong = strong_move.predict_proba(features)[:, 1]  # Probability of strong move  
pred_5h = direction_5h.predict_proba(features)[:, 1]  # Probability of 5h up move

print(f"1H Direction: {{pred_1h[0]:.1%}}")
print(f"Strong Move: {{pred_strong[0]:.1%}}")
print(f"5H Direction: {{pred_5h[0]:.1%}}")
```

## ⚠️ Important Notes
- **Test thoroughly** before live trading
- **Use proper risk management** 
- **Monitor performance** continuously
- **Retrain periodically** with new data

## 📈 Next Steps
1. Backtest on out-of-sample data
2. Implement in paper trading
3. Add more sophisticated features
4. Consider deep learning models
5. Implement portfolio optimization

---
*Generated by AI Forex Trading System*
"""
    
    with open("models/PERFORMANCE_REPORT.md", "w", encoding='utf-8') as f:
        f.write(report)
    
    print("✅ Report saved: models/PERFORMANCE_REPORT.md")

def main():
    """Main training execution"""
    print("🚀 Advanced Multi-Asset AI Model Training")
    print("=" * 60)
    
    try:
        # Step 1: Load all data
        master_data, data_summary = load_all_available_data()
        if master_data is None:
            print("❌ No data available for training!")
            print("💡 Run one of these first:")
            print("   - python download_extended_data.py")
            print("   - python import_mt5_csvs.py")
            return
        
        # Step 2: Create advanced features
        processed_data = create_advanced_features(master_data)
        
        # Step 3: Prepare multi-target training data
        X, y_tuple, feature_columns = prepare_multi_target_data(processed_data)
        if X is None:
            return
        
        # Step 4: Train ensemble models
        models, results = train_ensemble_models(X, y_tuple, feature_columns)
        
        # Step 5: Create performance report
        create_performance_report(results, data_summary)
        
        print("\n" + "=" * 60)
        print("🎉 ADVANCED TRAINING COMPLETE!")
        print("\nWhat you now have:")
        print("✅ Multi-asset dataset (Forex, Gold, Silver, Oil)")
        print("✅ Advanced feature engineering")
        print("✅ Ensemble models for multiple targets:")
        for target_name, result in results.items():
            print(f"   • {target_name}: {result['accuracy']:.1%} accuracy")
        print("✅ Complete performance report")
        
        print("\n🎯 Ready for:")
        print("• Backtesting")
        print("• Paper trading")
        print("• Live trading (with caution)")
        print("• Further model improvements")
        
        print("\n📖 Check models/PERFORMANCE_REPORT.md for detailed results!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()