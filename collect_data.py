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
    parser.add_argument('--symbols', nargs='+',
                       help='Trading symbols to collect (e.g., BTCUSDT ETHUSDT)')
    parser.add_argument('--auto-symbols', action='store_true',
                       help='Automatically collect data for top symbols by volume')
    parser.add_argument('--limit', type=int, default=20,
                       help='Number of top symbols to collect (used with --auto-symbols)')
    parser.add_argument('--timeframes', nargs='+', default=['5m', '30m', '1h', '1d', '1w'],
                       help='Timeframes to collect')
    parser.add_argument('--start-date', type=str, default='2022-01-01',
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
    timeframes = args.timeframes or config.get('data.timeframes', ['5m', '30m', '1h', '1d', '1w'])
    start_date = args.start_date or config.get('data.start_date', '2022-01-01')
    data_dir = args.data_dir or config.get('paths.data_dir', './data')
    
    # Initialize parser
    parser = BybitParser(data_dir=data_dir)
    
    # Determine symbols to collect
    if args.auto_symbols:
        print(f"\n🔍 Fetching top {args.limit} symbols by volume...")
        symbols = parser.get_popular_symbols(limit=args.limit)
        if not symbols:
            print("❌ Failed to fetch symbols. Using default.")
            symbols = ['BTCUSDT', 'ETHUSDT']
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = config.get('data.symbols', ['BTCUSDT', 'ETHUSDT'])
    
    print("\n" + "=" * 60)
    print("CRYPTOCURRENCY DATA COLLECTION")
    print("=" * 60)
    print(f"Symbols: {symbols}")
    print(f"Total symbols: {len(symbols)}")
    print(f"Timeframes: {timeframes}")
    print(f"Start date: {start_date}")
    print(f"End date: {args.end_date or 'today'}")
    print(f"Data directory: {data_dir}")
    print(f"API delay: {args.delay}s")
    print("=" * 60)
    
    # Collect data for each symbol
    success_count = 0
    fail_count = 0
    total_records = 0
    
    for idx, symbol in enumerate(symbols, 1):
        print(f"\n[{idx}/{len(symbols)}] {'='*20} {symbol} {'='*20}")
        
        try:
            symbol_data = parser.collect_multi_timeframe_data(
                symbol=symbol,
                timeframes=timeframes,
                start_date=start_date,
                end_date=args.end_date
            )
            
            if symbol_data:
                print(f"✅ Successfully collected data for {symbol}")
                success_count += 1
                
                # Display summary
                print("\nData Summary:")
                for tf, df in symbol_data.items():
                    records = len(df)
                    total_records += records
                    print(f"  {tf:4}: {records:,} records "
                          f"({df['timestamp'].min()} to {df['timestamp'].max()})")
            else:
                print(f"❌ Failed to collect data for {symbol}")
                fail_count += 1
                
        except Exception as e:
            print(f"❌ Error collecting data for {symbol}: {e}")
            fail_count += 1
    
    # Final summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"✅ Success: {success_count}/{len(symbols)} symbols")
    print(f"❌ Failed: {fail_count}/{len(symbols)} symbols")
    print(f"📊 Total records: {total_records:,}")
    print(f"💾 Data saved to: {data_dir}/")
    print("=" * 60)
    
    if success_count > 0:
        print("\n🚀 Ready to train! Run:")
        print(f"   python train.py --auto-symbols --epochs 300")
    print()


if __name__ == "__main__":
    main()
