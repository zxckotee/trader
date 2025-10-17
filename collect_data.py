#!/usr/bin/env python3
"""
Data collection script for cryptocurrency market data.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.bybit_parser import BybitParser
from src.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Collect cryptocurrency market data')
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT'],
                       help='Trading symbols to collect')
    parser.add_argument('--timeframes', nargs='+', default=['5m', '30m', '1h', '1d', '1w'],
                       help='Timeframes to collect')
    parser.add_argument('--start-date', type=str, default='2019-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD), default: today')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory to save data')
    parser.add_argument('--delay', type=float, default=0.01,
                       help='Delay between API requests (seconds)')
    
    return parser.parse_args()


def main():
    """Main data collection function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    symbols = args.symbols or config.get('data.symbols', ['BTCUSDT'])
    timeframes = args.timeframes or config.get('data.timeframes', ['5m', '30m', '1h', '1d', '1w'])
    start_date = args.start_date or config.get('data.start_date', '2019-01-01')
    data_dir = args.data_dir or config.get('paths.data_dir', './data')
    
    print("=== Cryptocurrency Data Collection ===")
    print(f"Symbols: {symbols}")
    print(f"Timeframes: {timeframes}")
    print(f"Start date: {start_date}")
    print(f"End date: {args.end_date or 'today'}")
    print(f"Data directory: {data_dir}")
    print(f"API delay: {args.delay}s")
    
    # Initialize parser
    parser = BybitParser(data_dir=data_dir)
    
    # Collect data for each symbol
    for symbol in symbols:
        print(f"\n{'='*20} {symbol} {'='*20}")
        
        try:
            symbol_data = parser.collect_multi_timeframe_data(
                symbol=symbol,
                timeframes=timeframes,
                start_date=start_date,
                end_date=args.end_date
            )
            
            if symbol_data:
                print(f"✓ Successfully collected data for {symbol}")
                
                # Display summary
                print("\nData Summary:")
                for tf, df in symbol_data.items():
                    print(f"  {tf:4}: {len(df):,} records "
                          f"({df['timestamp'].min()} to {df['timestamp'].max()})")
            else:
                print(f"✗ Failed to collect data for {symbol}")
                
        except Exception as e:
            print(f"✗ Error collecting data for {symbol}: {e}")
    
    print(f"\nData collection completed. Files saved to {data_dir}/")


if __name__ == "__main__":
    main()
