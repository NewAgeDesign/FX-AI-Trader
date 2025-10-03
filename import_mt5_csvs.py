"""
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
            
        print(f"\n📊 Processing {asset_type} files...")
        
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
        
        print(f"\n✅ Successfully imported {len(combined_data)} total rows")
        print(f"   Date range: {combined_data.index.min()} to {combined_data.index.max()}")
        
        # Save imported data
        os.makedirs("data/imported", exist_ok=True)
        handler.save_data(combined_data, "imported/mt5_imported_data.csv")
        print(f"   Saved to: data/imported/mt5_imported_data.csv")
        
        # Show summary by asset type
        print("\n📊 Import Summary:")
        summary = combined_data.groupby(['Asset_Type', 'Symbol']).size().reset_index(name='Rows')
        for _, row in summary.iterrows():
            print(f"   {row['Asset_Type']:<8} {row['Symbol']:<15} {row['Rows']:>8} rows")
        
        return combined_data
        
    else:
        print("❌ No data was imported!")
        print("\n💡 Make sure to:")
        print("   1. Export CSV files from MT5")
        print("   2. Place them in the correct folders")
        print("   3. Check the README_IMPORT_INSTRUCTIONS.txt")
        return None

if __name__ == "__main__":
    import_mt5_csvs()
