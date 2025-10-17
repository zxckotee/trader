"""
Bybit API parser for cryptocurrency market data collection.
Implements pagination to collect historical OHLCV data for multiple timeframes.
"""

import requests
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import os
from pathlib import Path


class BybitParser:
    """
    Parser for collecting cryptocurrency data from Bybit API.
    Supports multiple timeframes and automatic pagination for historical data.
    """
    
    BASE_URL = "https://api.bybit.com"
    
    # Timeframe mapping for different intervals
    TIMEFRAMES = {
        '1m': '1',      # 1 minute
        '3m': '3',      # 3 minutes
        '5m': '5',      # 5 minutes
        '15m': '15',    # 15 minutes
        '30m': '30',    # 30 minutes  
        '1h': '60',     # 1 hour
        '2h': '120',    # 2 hours
        '4h': '240',    # 4 hours
        '6h': '360',    # 6 hours
        '12h': '720',   # 12 hours
        '1d': 'D',      # 1 day
        '1w': 'W'       # 1 week
    }
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the Bybit parser.
        
        Args:
            data_dir: Directory to save collected data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        
        # Set headers to avoid rate limiting
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_popular_symbols(self, limit: int = 50) -> List[str]:
        """
        Get list of popular trading symbols by volume.
        
        Args:
            limit: Maximum number of symbols to return
            
        Returns:
            List of popular symbol names
        """
        url = f"{self.BASE_URL}/v5/market/tickers"
        
        try:
            response = self.session.get(url, params={'category': 'linear'}, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'retCode' in data and data['retCode'] != 0:
                print(f"API Error getting symbols: {data.get('retMsg', 'Unknown error')}")
                return self._get_default_symbols()
            
            # Sort by 24h volume and filter USDT pairs
            tickers = data.get('result', {}).get('list', [])
            usdt_pairs = [
                ticker for ticker in tickers 
                if ticker['symbol'].endswith('USDT') and 
                float(ticker.get('turnover24h', 0)) > 0
            ]
            
            # Sort by volume (turnover24h)
            usdt_pairs.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            
            symbols = [ticker['symbol'] for ticker in usdt_pairs[:limit]]
            print(f"Found {len(symbols)} popular USDT pairs")
            
            return symbols
            
        except Exception as e:
            print(f"Error getting popular symbols: {e}")
            return self._get_default_symbols()
    
    def _get_default_symbols(self) -> List[str]:
        """Get default list of popular symbols."""
        return [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT',
            'SOLUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'LTCUSDT',
            'AVAXUSDT', 'LINKUSDT', 'ATOMUSDT', 'UNIUSDT', 'FILUSDT',
            'ETCUSDT', 'XLMUSDT', 'VETUSDT', 'ICPUSDT', 'FTMUSDT'
        ]
        
    def get_kline_data(self, 
                      symbol: str, 
                      interval: str, 
                      start_time: Optional[int] = None,
                      end_time: Optional[int] = None,
                      limit: int = 1000) -> Dict:
        """
        Get kline/candlestick data from Bybit API.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval (5, 30, 60, D, W)
            start_time: Start time in milliseconds (Unix timestamp)
            end_time: End time in milliseconds (Unix timestamp)
            limit: Number of data points to return (max 1000)
            
        Returns:
            API response as dictionary
        """
        url = f"{self.BASE_URL}/v5/market/kline"
        
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['start'] = start_time
        if end_time:
            params['end'] = end_time
            
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check API response status
            if 'retCode' in data and data['retCode'] != 0:
                print(f"API Error: {data.get('retMsg', 'Unknown error')} (Code: {data['retCode']})")
                return {}
            
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return {}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {}
    
    def parse_kline_response(self, response: Dict) -> pd.DataFrame:
        """
        Parse Bybit kline API response into pandas DataFrame.
        
        Args:
            response: API response dictionary
            
        Returns:
            DataFrame with OHLCV data
        """
        if not response or 'result' not in response:
            return pd.DataFrame()
            
        data = response['result']['list']
        if not data:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # Convert data types
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # Sort by timestamp (Bybit returns data in reverse order)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def collect_historical_data(self, 
                              symbol: str,
                              timeframe: str,
                              start_date: str,
                              end_date: Optional[str] = None,
                              delay: float = 0.01) -> pd.DataFrame:
        """
        Collect historical data with pagination.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe key from TIMEFRAMES dict (e.g., '5m', '1h')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            delay: Delay between API requests in seconds
            
        Returns:
            Combined DataFrame with all historical data
        """
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Use one of {list(self.TIMEFRAMES.keys())}")
            
        interval = self.TIMEFRAMES[timeframe]
        
        # Convert dates to timestamps
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
        
        start_timestamp = int(start_dt.timestamp() * 1000)
        end_timestamp = int(end_dt.timestamp() * 1000)
        
        # Calculate expected number of records for progress tracking
        timeframe_minutes = {
            '1': 1, '3': 3, '5': 5, '15': 15, '30': 30, '60': 60,
            '120': 120, '240': 240, '360': 360, '720': 720,
            'D': 1440, 'W': 10080
        }
        
        if interval in timeframe_minutes:
            total_minutes = (end_timestamp - start_timestamp) / (1000 * 60)
            expected_records = int(total_minutes / timeframe_minutes[interval])
            print(f"Expected approximately {expected_records:,} records for {timeframe}")
        else:
            expected_records = None
        
        all_data = []
        current_start = start_timestamp
        
        print(f"Collecting {symbol} data for {timeframe} from {start_date} to {end_date or 'now'}...")
        
        iteration_count = 0
        max_iterations = 50000  # Increased limit for more data
        
        while current_start < end_timestamp and iteration_count < max_iterations:
            iteration_count += 1
            
            # Progress tracking
            progress = ((current_start - start_timestamp) / (end_timestamp - start_timestamp)) * 100
            current_records = sum(len(df) for df in all_data)
            
            print(f"Iteration {iteration_count}: {datetime.fromtimestamp(current_start/1000).strftime('%Y-%m-%d %H:%M')} "
                  f"({progress:.1f}% complete, {current_records:,} records collected)")
            
            response = self.get_kline_data(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_timestamp,
                limit=1000
            )
            
            df_batch = self.parse_kline_response(response)
            
            if df_batch.empty:
                print("No more data available from API")
                break
            
            print(f"Received {len(df_batch)} records, date range: {df_batch['timestamp'].min()} to {df_batch['timestamp'].max()}")
            all_data.append(df_batch)
            
            # Update start time for next batch
            last_timestamp = df_batch['timestamp'].iloc[-1]
            new_start = int(last_timestamp.timestamp() * 1000) + 1
            
            # Check if we're making progress
            if new_start <= current_start:
                print("No progress in data collection, stopping...")
                break
                
            current_start = new_start
            
            # Safety check: if we got less than expected, we might be at the end
            if len(df_batch) < 1000:
                print(f"Received {len(df_batch)} records (less than 1000), likely reached end of data")
                break
            
            # Add delay to avoid rate limiting
            time.sleep(delay)
        
        if iteration_count >= max_iterations:
            print(f"Reached maximum iterations ({max_iterations}), stopping collection")
            
        if not all_data:
            return pd.DataFrame()
            
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        print(f"Collected {len(combined_df)} data points")
        return combined_df
    
    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            symbol: Trading pair symbol
            timeframe: Timeframe identifier
            
        Returns:
            Path to saved file
        """
        filename = f"{symbol}_{timeframe}.csv"
        filepath = self.data_dir / filename
        
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        
        return str(filepath)
    
    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load previously saved data from CSV.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe identifier
            
        Returns:
            DataFrame with loaded data
        """
        filename = f"{symbol}_{timeframe}.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"File {filepath} not found")
            return pd.DataFrame()
            
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def collect_multi_timeframe_data(self, 
                                   symbol: str,
                                   timeframes: List[str],
                                   start_date: str,
                                   end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Collect data for multiple timeframes.
        
        Args:
            symbol: Trading pair symbol
            timeframes: List of timeframe keys
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary mapping timeframes to DataFrames
        """
        results = {}
        
        for tf in timeframes:
            print(f"\n=== Collecting {tf} data ===")
            df = self.collect_historical_data(symbol, tf, start_date, end_date)
            
            if not df.empty:
                results[tf] = df
                self.save_data(df, symbol, tf)
            else:
                print(f"No data collected for {tf}")
                
        return results


def main():
    """Example usage of BybitParser."""
    parser = BybitParser(data_dir="./data")
    
    # Define parameters
    symbol = "BTCUSDT"
    timeframes = ['5m', '30m', '1h', '1d', '1w']  # All required timeframes
    start_date = "2019-01-01"
    
    # Collect data for all timeframes
    data = parser.collect_multi_timeframe_data(
        symbol=symbol,
        timeframes=timeframes,
        start_date=start_date
    )
    
    # Display summary
    print("\n=== Data Collection Summary ===")
    for tf, df in data.items():
        if not df.empty:
            print(f"{tf}: {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
