"""
MetaTrader 5 Connector
Handles connection to MT5 platform for data retrieval and trade execution
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import time


class MT5Connector:
    """
    Interface for connecting to MetaTrader 5 platform
    """
    
    def __init__(self):
        self.connected = False
        self.account_info = None
        
    def connect(self, login: Optional[int] = None, 
                password: Optional[str] = None, 
                server: Optional[str] = None) -> bool:
        """
        Connect to MetaTrader 5
        
        Args:
            login: Account number (optional if already logged in)
            password: Account password
            server: Broker server
            
        Returns:
            True if connection successful
        """
        try:
            # Initialize MT5 connection
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login if credentials provided
            if login and password and server:
                if not mt5.login(login, password, server):
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
            
            # Get account information
            self.account_info = mt5.account_info()
            if self.account_info is None:
                logger.error("Failed to get account info")
                return False
            
            self.connected = True
            logger.info(f"Connected to MT5. Account: {self.account_info.login}, "
                       f"Balance: {self.account_info.balance}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    def get_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        if not self.connected:
            logger.warning("Not connected to MT5")
            return []
        
        symbols = mt5.symbols_get()
        if symbols is None:
            logger.error("Failed to get symbols")
            return []
        
        return [symbol.name for symbol in symbols if symbol.visible]
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed information about a symbol"""
        if not self.connected:
            logger.warning("Not connected to MT5")
            return None
        
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Failed to get info for symbol: {symbol}")
            return None
        
        return {
            'name': info.name,
            'digits': info.digits,
            'point': info.point,
            'spread': info.spread,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'contract_size': info.trade_contract_size,
            'margin_initial': info.margin_initial,
            'margin_maintenance': info.margin_maintenance
        }
    
    def get_historical_data(self, symbol: str, timeframe: int, 
                           start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_H1)
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected:
            logger.warning("Not connected to MT5")
            return None
        
        try:
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            if rates is None:
                logger.error(f"Failed to get historical data for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Rename columns to standard format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Spread', 'Real_Volume']
            
            logger.info(f"Retrieved {len(df)} bars of {symbol} data")
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None
    
    def get_latest_tick(self, symbol: str) -> Optional[Dict]:
        """Get latest tick data for a symbol"""
        if not self.connected:
            logger.warning("Not connected to MT5")
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None
        
        return {
            'symbol': symbol,
            'time': datetime.fromtimestamp(tick.time),
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid,
            'volume': tick.volume
        }
    
    def place_order(self, symbol: str, order_type: str, volume: float, 
                    price: Optional[float] = None, 
                    sl: Optional[float] = None, 
                    tp: Optional[float] = None,
                    comment: str = "AI Trading Agent") -> Dict:
        """
        Place a trading order
        
        Args:
            symbol: Trading symbol
            order_type: 'buy' or 'sell'
            volume: Order volume in lots
            price: Limit price (None for market orders)
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
            
        Returns:
            Dictionary with order result
        """
        if not self.connected:
            return {"status": "error", "message": "Not connected to MT5"}
        
        try:
            # Get current price for market orders
            if price is None:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    return {"status": "error", "message": "Failed to get current price"}
                price = tick.ask if order_type == 'buy' else tick.bid
            
            # Prepare order request
            order_type_mt5 = mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL
            action = mt5.TRADE_ACTION_DEAL
            
            request = {
                "action": action,
                "symbol": symbol,
                "volume": volume,
                "type": order_type_mt5,
                "price": price,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,  # Good Till Cancelled
                "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate Or Cancel
            }
            
            # Add stop loss and take profit if provided
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order failed: {result.comment} (Code: {result.retcode})"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            logger.info(f"Order executed successfully: {result.order}")
            return {
                "status": "success",
                "order_id": result.order,
                "volume": result.volume,
                "price": result.price,
                "comment": result.comment
            }
            
        except Exception as e:
            error_msg = f"Error placing order: {e}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def get_positions(self) -> List[Dict]:
        """Get current open positions"""
        if not self.connected:
            logger.warning("Not connected to MT5")
            return []
        
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        position_list = []
        for pos in positions:
            position_list.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'buy' if pos.type == mt5.POSITION_TYPE_BUY else 'sell',
                'volume': pos.volume,
                'open_price': pos.price_open,
                'current_price': pos.price_current,
                'profit': pos.profit,
                'comment': pos.comment,
                'time': datetime.fromtimestamp(pos.time)
            })
        
        return position_list
    
    def close_position(self, ticket: int) -> Dict:
        """Close a specific position"""
        if not self.connected:
            return {"status": "error", "message": "Not connected to MT5"}
        
        try:
            # Get position info
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return {"status": "error", "message": "Position not found"}
            
            position = positions[0]
            
            # Prepare close request
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": close_type,
                "position": ticket,
                "comment": "AI Agent Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Position close failed: {result.comment}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            logger.info(f"Position {ticket} closed successfully")
            return {"status": "success", "ticket": ticket}
            
        except Exception as e:
            error_msg = f"Error closing position: {e}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def get_account_balance(self) -> float:
        """Get current account balance"""
        if not self.connected:
            return 0.0
        
        info = mt5.account_info()
        return info.balance if info else 0.0
    
    def get_account_equity(self) -> float:
        """Get current account equity"""
        if not self.connected:
            return 0.0
        
        info = mt5.account_info()
        return info.equity if info else 0.0


# Example usage and testing
if __name__ == "__main__":
    connector = MT5Connector()
    
    # Test connection (will use existing MT5 login)
    if connector.connect():
        print("MT5 connected successfully!")
        
        # Get available symbols
        symbols = connector.get_symbols()
        print(f"Available symbols: {len(symbols)}")
        if symbols:
            print(f"First 10 symbols: {symbols[:10]}")
        
        # Test getting historical data for EURUSD
        if "EURUSD" in symbols:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Last 30 days
            
            data = connector.get_historical_data(
                "EURUSD", 
                mt5.TIMEFRAME_H1, 
                start_date, 
                end_date
            )
            
            if data is not None:
                print(f"Historical data shape: {data.shape}")
                print(data.head())
            
            # Get latest tick
            tick = connector.get_latest_tick("EURUSD")
            if tick:
                print(f"Latest EURUSD tick: {tick}")
        
        # Get account info
        print(f"Account balance: ${connector.get_account_balance():.2f}")
        print(f"Account equity: ${connector.get_account_equity():.2f}")
        
        # Get current positions
        positions = connector.get_positions()
        print(f"Open positions: {len(positions)}")
        
        connector.disconnect()
    else:
        print("Failed to connect to MT5")