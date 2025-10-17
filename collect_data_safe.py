#!/usr/bin/env python3
"""
Безопасный скрипт сбора данных с улучшенной обработкой ошибок.
"""

import argparse
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.bybit_parser import BybitParser
from src.utils.config import load_config


def safe_collect_timeframe(parser, symbol, timeframe, start_date, end_date=None, max_retries=3):
    """
    Безопасный сбор данных для одного таймфрейма с повторными попытками.
    """
    for attempt in range(max_retries):
        try:
            print(f"\n--- Attempt {attempt + 1}/{max_retries} for {symbol} {timeframe} ---")
            
            df = parser.collect_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                delay=0.1  # Увеличенная задержка для стабильности
            )
            
            if not df.empty:
                # Сохранение данных
                filepath = parser.save_data(df, symbol, timeframe)
                print(f"✓ Successfully collected {len(df)} records for {symbol} {timeframe}")
                return df
            else:
                print(f"⚠ No data collected for {symbol} {timeframe} on attempt {attempt + 1}")
                
        except Exception as e:
            print(f"✗ Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting 5 seconds before retry...")
                time.sleep(5)
    
    print(f"✗ Failed to collect data for {symbol} {timeframe} after {max_retries} attempts")
    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Safe cryptocurrency data collection')
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT'],
                       help='Trading symbols to collect')
    parser.add_argument('--timeframes', nargs='+', default=['5m', '30m', '1h', '1d', '1w'],
                       help='Timeframes to collect')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD), default: today')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory to save data')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry attempts per timeframe')
    parser.add_argument('--delay', type=float, default=0.1,
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
    start_date = args.start_date or config.get('data.start_date', '2023-01-01')
    data_dir = args.data_dir or config.get('paths.data_dir', './data')
    
    print("=== Safe Cryptocurrency Data Collection ===")
    print(f"Symbols: {symbols}")
    print(f"Timeframes: {timeframes}")
    print(f"Start date: {start_date}")
    print(f"End date: {args.end_date or 'today'}")
    print(f"Data directory: {data_dir}")
    print(f"Max retries: {args.max_retries}")
    print(f"API delay: {args.delay}s")
    
    # Initialize parser
    parser = BybitParser(data_dir=data_dir)
    
    # Statistics
    total_collected = 0
    failed_collections = []
    
    # Collect data for each symbol and timeframe
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Processing {symbol}")
        print(f"{'='*50}")
        
        symbol_success = 0
        
        for timeframe in timeframes:
            print(f"\n🔄 Collecting {timeframe} data for {symbol}...")
            
            df = safe_collect_timeframe(
                parser=parser,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=args.end_date,
                max_retries=args.max_retries
            )
            
            if df is not None:
                total_collected += len(df)
                symbol_success += 1
                print(f"✅ {timeframe}: {len(df)} records")
            else:
                failed_collections.append(f"{symbol}_{timeframe}")
                print(f"❌ {timeframe}: Failed")
            
            # Пауза между таймфреймами
            time.sleep(1)
        
        print(f"\n📊 {symbol} Summary: {symbol_success}/{len(timeframes)} timeframes collected")
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎯 COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total records collected: {total_collected:,}")
    print(f"Successful collections: {len(symbols) * len(timeframes) - len(failed_collections)}")
    print(f"Failed collections: {len(failed_collections)}")
    
    if failed_collections:
        print(f"\n❌ Failed collections:")
        for failed in failed_collections:
            print(f"   - {failed}")
        print(f"\nTo retry failed collections, run:")
        for failed in failed_collections:
            symbol, tf = failed.split('_')
            print(f"python collect_data_safe.py --symbols {symbol} --timeframes {tf} --start-date {start_date}")
    else:
        print(f"\n🎉 All collections completed successfully!")
    
    print(f"\nData saved to: {data_dir}/")


if __name__ == "__main__":
    main()
