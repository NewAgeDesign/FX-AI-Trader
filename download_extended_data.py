"""
Extended Data Downloader
Downloads Gold, Silver, Oil data in multiple timeframes (1M, 5M, 15M)
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from data_handler import DataHandler
from loguru import logger

def download_commodities_data():
    """Download commodities data in multiple timeframes"""
    print("🔄 Downloading Commodities Data (Gold, Silver, Oil)")
    print("=" * 60)
    
    handler = DataHandler()
    
    # Commodity symbols (Yahoo Finance format)
    commodities = {
        'GOLD': 'GC=F',      # Gold Futures
        'SILVER': 'SI=F',    # Silver Futures  
        'OIL_WTI': 'CL=F',   # WTI Crude Oil
        'OIL_BRENT': 'BZ=F'  # Brent Crude Oil
    }
    
    # Timeframes - Yahoo Finance supports these intervals
    timeframes = {
        '1M': '1m',
        '5M': '5m', 
        '15M': '15m'
    }
    
    # Periods (Yahoo Finance limits for shorter timeframes)
    periods = {
        '1M': '7d',    # 1-minute data: max 7 days
        '5M': '60d',   # 5-minute data: max 60 days  
        '15M': '60d'   # 15-minute data: max 60 days
    }
    
    all_data = []
    download_summary = []
    
    for commodity_name, symbol in commodities.items():
        print(f"\n📊 Downloading {commodity_name} ({symbol})...")
        
        for tf_name, tf_code in timeframes.items():
            try:
                print(f"   {tf_name} timeframe...")
                
                data = handler.download_yahoo_data(
                    symbol=symbol,
                    period=periods[tf_name],
                    interval=tf_code
                )
                
                if data is not None and len(data) > 0:
                    # Add metadata
                    data['Commodity'] = commodity_name
                    data['Timeframe'] = tf_name
                    data['Symbol'] = f"{commodity_name}_{tf_name}"
                    
                    all_data.append(data)
                    
                    print(f"   ✅ {tf_name}: {len(data)} rows")
                    download_summary.append({
                        'Commodity': commodity_name,
                        'Timeframe': tf_name,
                        'Rows': len(data),
                        'Date_Range': f"{data.index.min()} to {data.index.max()}"
                    })
                else:
                    print(f"   ❌ {tf_name}: No data received")
                    
            except Exception as e:
                print(f"   ❌ {tf_name}: Error - {e}")
    
    if all_data:
        # Combine all commodity data
        combined_data = pd.concat(all_data, ignore_index=False)
        combined_data = combined_data.sort_index()
        
        # Save raw commodity data
        os.makedirs("data/commodities", exist_ok=True)
        handler.save_data(combined_data, "commodities/raw_commodities_data.csv")
        
        print(f"\n✅ Total commodities data: {len(combined_data)} rows")
        print(f"   Saved to: data/commodities/raw_commodities_data.csv")
        
        # Save summary
        summary_df = pd.DataFrame(download_summary)
        handler.save_data(summary_df, "commodities/download_summary.csv")
        
        return combined_data
    else:
        print("❌ No commodity data downloaded!")
        return None

def setup_csv_import_folder():
    """Set up folder structure for importing CSV files from MT5"""
    print("\n🔄 Setting up CSV import folder structure...")
    
    # Create organized folder structure
    folders = [
        "data/mt5_imports",
        "data/mt5_imports/gold", 
        "data/mt5_imports/silver",
        "data/mt5_imports/oil",
        "data/mt5_imports/forex"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created: {folder}")
    
    # Create instruction file
    instructions = """# MT5 CSV Import Instructions

## How to Export Data from MetaTrader 5

1. **Open MT5** and go to Tools → History Center (F2)

2. **For each symbol you want:**
   - Select the symbol (XAUUSD for Gold, XAGUSD for Silver, etc.)
   - Choose the timeframe (M1, M5, M15)
   - Right-click → Export → CSV file
   
3. **Save files in the appropriate folders:**
   ```
   data/mt5_imports/gold/     -> XAUUSD_M1.csv, XAUUSD_M5.csv, XAUUSD_M15.csv
   data/mt5_imports/silver/   -> XAGUSD_M1.csv, XAGUSD_M5.csv, XAGUSD_M15.csv  
   data/mt5_imports/oil/      -> USOUSD_M1.csv, USOUSD_M5.csv, USOUSD_M15.csv
   data/mt5_imports/forex/    -> EURUSD_M1.csv, GBPUSD_M1.csv, etc.
   ```

4. **Expected CSV format:**
   Date,Time,Open,High,Low,Close,Volume
   2023.01.01,00:00:00,1.2345,1.2350,1.2340,1.2348,100

5. **Run the import script:**
   ```bash
   python import_mt5_csvs.py
   ```

## Recommended Symbols to Export:

### 🥇 Metals:
- XAUUSD (Gold)
- XAGUSD (Silver)
- XPTUSD (Platinum) - optional
- XPDUSD (Palladium) - optional

### 🛢️ Oil:
- USOUSD (US Oil / WTI)
- UKOUSD (UK Oil / Brent) - if available

### 💱 Major Forex:
- EURUSD, GBPUSD, USDJPY, USDCHF
- AUDUSD, USDCAD, NZDUSD

### ⏰ Timeframes:
- M1 (1 minute) - for scalping strategies
- M5 (5 minutes) - for short-term trading
- M15 (15 minutes) - for swing trading

## Tips:
- Export at least 3-6 months of data for good training
- Larger datasets = better AI performance
- Make sure CSV files have headers
- Check for missing data gaps
"""
    
    with open("data/mt5_imports/README_IMPORT_INSTRUCTIONS.txt", "w", encoding='utf-8') as f:
        f.write(instructions)
    
    print("✅ Created import instructions: data/mt5_imports/README_IMPORT_INSTRUCTIONS.txt")
    print("\n📋 You can now:")
    print("   1. Export CSV files from MT5 into the folders above")
    print("   2. Or run this script to download from Yahoo Finance")

def create_csv_importer():
    """Create script to import MT5 CSV files"""
    print("\n🔄 Creating CSV import script...")
    
    importer_script = '''"""
MT5 CSV Importer
Imports CSV files exported from MetaTrader 5
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import sys

sys.path.append('src')
from data_handler import DataHandler

def import_mt5_csvs():
    """Import all MT5 CSV files from the import folders"""
    print("🔄 Importing MT5 CSV Files...")
    print("=" * 50)
    
    handler = DataHandler()
    all_imported_data = []
    
    # Define import folders and their corresponding asset types
    import_config = {
        'data/mt5_imports/gold/': 'GOLD',
        'data/mt5_imports/silver/': 'SILVER', 
        'data/mt5_imports/oil/': 'OIL',
        'data/mt5_imports/forex/': 'FOREX'
    }
    
    for folder_path, asset_type in import_config.items():
        if not os.path.exists(folder_path):
            print(f"⚠️  Folder not found: {folder_path}")
            continue
            
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        if not csv_files:
            print(f"📂 {asset_type}: No CSV files found in {folder_path}")
            continue
            
        print(f"\\n📊 Processing {asset_type} files...")
        
        for csv_file in csv_files:
            try:
                print(f"   Reading: {os.path.basename(csv_file)}")
                
                # Read CSV with flexible parsing
                df = pd.read_csv(csv_file)
                
                # Handle different possible column formats from MT5
                expected_columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                
                if len(df.columns) >= 6:  # At least OHLCV data
                    # Standardize column names
                    if len(df.columns) == 7:
                        df.columns = expected_columns
                    elif len(df.columns) == 6:
                        df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close']
                        df['Volume'] = 0  # Add zero volume if missing
                    
                    # Combine Date and Time columns
                    if 'Date' in df.columns and 'Time' in df.columns:
                        df['DateTime'] = pd.to_datetime(
                            df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                            errors='coerce'
                        )
                    else:
                        # Assume first column is datetime
                        df['DateTime'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                    
                    # Set datetime as index
                    df.set_index('DateTime', inplace=True)
                    
                    # Remove original Date/Time columns
                    df = df.drop(columns=['Date', 'Time'], errors='ignore')
                    
                    # Add metadata
                    filename = os.path.basename(csv_file)
                    symbol_name = filename.replace('.csv', '')
                    
                    df['Symbol'] = symbol_name
                    df['Asset_Type'] = asset_type
                    
                    # Remove rows with invalid data
                    df = df.dropna()
                    
                    if len(df) > 0:
                        all_imported_data.append(df)
                        print(f"   ✅ {symbol_name}: {len(df)} rows imported")
                    else:
                        print(f"   ❌ {symbol_name}: No valid data after cleaning")
                        
                else:
                    print(f"   ❌ {os.path.basename(csv_file)}: Invalid format (need at least 6 columns)")
                    
            except Exception as e:
                print(f"   ❌ Error reading {os.path.basename(csv_file)}: {e}")
    
    if all_imported_data:
        # Combine all imported data
        combined_data = pd.concat(all_imported_data, ignore_index=False)
        combined_data = combined_data.sort_index()
        
        print(f"\\n✅ Successfully imported {len(combined_data)} total rows")
        print(f"   Date range: {combined_data.index.min()} to {combined_data.index.max()}")
        
        # Save imported data
        os.makedirs("data/imported", exist_ok=True)
        handler.save_data(combined_data, "imported/mt5_imported_data.csv")
        print(f"   Saved to: data/imported/mt5_imported_data.csv")
        
        # Show summary by asset type
        print("\\n📊 Import Summary:")
        summary = combined_data.groupby(['Asset_Type', 'Symbol']).size().reset_index(name='Rows')
        for _, row in summary.iterrows():
            print(f"   {row['Asset_Type']:<8} {row['Symbol']:<15} {row['Rows']:>8} rows")
        
        return combined_data
        
    else:
        print("❌ No data was imported!")
        print("\\n💡 Make sure to:")
        print("   1. Export CSV files from MT5")
        print("   2. Place them in the correct folders")
        print("   3. Check the README_IMPORT_INSTRUCTIONS.txt")
        return None

if __name__ == "__main__":
    import_mt5_csvs()
'''
    
    with open("import_mt5_csvs.py", "w", encoding='utf-8') as f:
        f.write(importer_script)
    
    print("✅ Created: import_mt5_csvs.py")

def combine_all_data():
    """Combine existing forex data with new commodities data"""
    print("\n🔄 Combining all data sources...")
    
    handler = DataHandler()
    all_datasets = []
    
    # Load existing forex data
    forex_data_path = "data/processed_forex_data.csv"
    if os.path.exists(forex_data_path):
        print("   Loading existing forex data...")
        forex_data = pd.read_csv(forex_data_path, index_col=0, parse_dates=True)
        forex_data['Asset_Type'] = 'FOREX'
        all_datasets.append(forex_data)
        print(f"   ✅ Forex: {len(forex_data)} rows")
    
    # Load commodities data
    commodities_data_path = "data/commodities/raw_commodities_data.csv"
    if os.path.exists(commodities_data_path):
        print("   Loading commodities data...")
        commodities_data = pd.read_csv(commodities_data_path, index_col=0, parse_dates=True)
        commodities_data['Asset_Type'] = 'COMMODITY'
        all_datasets.append(commodities_data)
        print(f"   ✅ Commodities: {len(commodities_data)} rows")
    
    # Load MT5 imported data
    mt5_data_path = "data/imported/mt5_imported_data.csv"  
    if os.path.exists(mt5_data_path):
        print("   Loading MT5 imported data...")
        mt5_data = pd.read_csv(mt5_data_path, index_col=0, parse_dates=True)
        all_datasets.append(mt5_data)
        print(f"   ✅ MT5 Data: {len(mt5_data)} rows")
    
    if all_datasets:
        # Combine everything
        mega_dataset = pd.concat(all_datasets, ignore_index=False)
        mega_dataset = mega_dataset.sort_index()
        
        # Save mega dataset
        os.makedirs("data/combined", exist_ok=True)
        handler.save_data(mega_dataset, "combined/mega_trading_dataset.csv")
        
        print(f"\n🎉 MEGA DATASET CREATED!")
        print(f"   Total rows: {len(mega_dataset):,}")
        print(f"   Date range: {mega_dataset.index.min()} to {mega_dataset.index.max()}")
        print(f"   Unique symbols: {mega_dataset['Symbol'].nunique()}")
        print(f"   Saved to: data/combined/mega_trading_dataset.csv")
        
        # Show breakdown by asset type
        if 'Asset_Type' in mega_dataset.columns:
            print("\n📊 Asset Type Breakdown:")
            breakdown = mega_dataset['Asset_Type'].value_counts()
            for asset_type, count in breakdown.items():
                print(f"   {asset_type:<12} {count:>8,} rows")
        
        return mega_dataset
    
    else:
        print("❌ No datasets found to combine!")
        return None

def main():
    """Main execution"""
    print("🚀 Extended Data Download & Import Setup")
    print("=" * 60)
    
    # Option 1: Set up CSV import structure
    setup_csv_import_folder()
    create_csv_importer()
    
    print("\n" + "="*60)
    print("Choose your approach:")
    print("1️⃣  Download from Yahoo Finance (automated)")
    print("2️⃣  Import from MT5 CSV files (manual export required)")
    print("3️⃣  Both approaches")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice in ['1', '3']:
        # Download commodities data
        commodities_data = download_commodities_data()
    
    if choice in ['2', '3']:
        print("\n💡 To import MT5 data:")
        print("   1. Export CSV files from MT5 to the folders created")
        print("   2. Run: python import_mt5_csvs.py")
    
    # Try to combine all available data
    print("\n" + "="*60)
    combined = combine_all_data()
    
    print("\n🎯 Next Steps:")
    print("✅ Folder structure created for MT5 imports")
    print("✅ CSV importer script ready")
    if choice in ['1', '3']:
        print("✅ Commodities data downloaded")
    print("\n🔄 Run 'python train_extended_model.py' to train with all data!")

if __name__ == "__main__":
    main()