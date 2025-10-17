#!/usr/bin/env python3
"""
Массовый сбор данных для популярных криптовалют с 2017 года.
Автоматически получает список популярных символов и собирает максимум данных.
"""

import argparse
import sys
from pathlib import Path
import time
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.bybit_parser import BybitParser
from src.utils.config import load_config


class MassiveDataCollector:
    """Класс для массового сбора данных."""
    
    def __init__(self, data_dir: str = "./data", max_workers: int = 3):
        self.parser = BybitParser(data_dir=data_dir)
        self.data_dir = Path(data_dir)
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.stats = {
            'total_symbols': 0,
            'completed_symbols': 0,
            'failed_symbols': 0,
            'total_records': 0,
            'start_time': None
        }
    
    def get_collection_plan(self, symbols: list, timeframes: list, start_date: str) -> list:
        """
        Создает план сбора данных с приоритизацией.
        
        Returns:
            List of (symbol, timeframe, priority) tuples
        """
        plan = []
        
        # Приоритеты таймфреймов (чем больше число, тем выше приоритет)
        timeframe_priority = {
            '1d': 10,   # Самый высокий приоритет - дневные данные
            '1w': 9,    # Недельные
            '4h': 8,    # 4-часовые
            '1h': 7,    # Часовые
            '30m': 6,   # 30-минутные
            '15m': 5,   # 15-минутные
            '5m': 4,    # 5-минутные
            '3m': 3,    # 3-минутные
            '1m': 2     # Самый низкий приоритет - минутные данные
        }
        
        for symbol in symbols:
            for timeframe in timeframes:
                priority = timeframe_priority.get(timeframe, 1)
                # Увеличиваем приоритет для топовых монет
                if symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
                    priority += 5
                
                plan.append((symbol, timeframe, priority))
        
        # Сортируем по приоритету (высокий приоритет первым)
        plan.sort(key=lambda x: x[2], reverse=True)
        
        return plan
    
    def collect_single_task(self, symbol: str, timeframe: str, start_date: str, 
                           end_date: str = None, max_retries: int = 3) -> dict:
        """
        Собирает данные для одной пары символ-таймфрейм.
        
        Returns:
            Dict with collection results
        """
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'success': False,
            'records': 0,
            'error': None,
            'file_path': None
        }
        
        for attempt in range(max_retries):
            try:
                # Проверяем, существует ли уже файл
                expected_file = self.data_dir / f"{symbol}_{timeframe}.csv"
                if expected_file.exists():
                    # Загружаем существующий файл для проверки
                    existing_df = self.parser.load_data(symbol, timeframe)
                    if not existing_df.empty and len(existing_df) > 1000:
                        result['success'] = True
                        result['records'] = len(existing_df)
                        result['file_path'] = str(expected_file)
                        with self.lock:
                            print(f"✓ {symbol} {timeframe}: Using existing file with {len(existing_df):,} records")
                        return result
                
                # Собираем новые данные
                with self.lock:
                    print(f"🔄 {symbol} {timeframe}: Starting collection (attempt {attempt + 1}/{max_retries})")
                
                df = self.parser.collect_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    delay=0.05  # Небольшая задержка для стабильности
                )
                
                if not df.empty:
                    # Сохраняем данные
                    file_path = self.parser.save_data(df, symbol, timeframe)
                    
                    result['success'] = True
                    result['records'] = len(df)
                    result['file_path'] = file_path
                    
                    with self.lock:
                        self.stats['total_records'] += len(df)
                        print(f"✅ {symbol} {timeframe}: Collected {len(df):,} records")
                    
                    return result
                else:
                    with self.lock:
                        print(f"⚠️ {symbol} {timeframe}: No data collected on attempt {attempt + 1}")
                    
            except Exception as e:
                error_msg = str(e)
                with self.lock:
                    print(f"❌ {symbol} {timeframe}: Error on attempt {attempt + 1}: {error_msg}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    result['error'] = error_msg
        
        with self.lock:
            self.stats['failed_symbols'] += 1
        
        return result
    
    def collect_all_data(self, symbols: list, timeframes: list, start_date: str, 
                        end_date: str = None) -> dict:
        """
        Массовый сбор данных для всех символов и таймфреймов.
        
        Returns:
            Dict with collection statistics
        """
        self.stats['start_time'] = datetime.now()
        self.stats['total_symbols'] = len(symbols) * len(timeframes)
        
        print(f"🚀 Starting massive data collection")
        print(f"📊 Symbols: {len(symbols)}")
        print(f"⏱️ Timeframes: {timeframes}")
        print(f"📅 Date range: {start_date} to {end_date or 'now'}")
        print(f"🔧 Max workers: {self.max_workers}")
        print(f"📁 Data directory: {self.data_dir}")
        
        # Создаем план сбора
        collection_plan = self.get_collection_plan(symbols, timeframes, start_date)
        
        print(f"\n📋 Collection plan created: {len(collection_plan)} tasks")
        print("🎯 High priority tasks:")
        for symbol, timeframe, priority in collection_plan[:10]:
            print(f"   {symbol} {timeframe} (priority: {priority})")
        
        # Выполняем сбор данных
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Отправляем задачи
            future_to_task = {}
            for symbol, timeframe, priority in collection_plan:
                future = executor.submit(
                    self.collect_single_task, 
                    symbol, timeframe, start_date, end_date
                )
                future_to_task[future] = (symbol, timeframe)
            
            # Собираем результаты
            for future in as_completed(future_to_task):
                symbol, timeframe = future_to_task[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    with self.lock:
                        self.stats['completed_symbols'] += 1
                        progress = (self.stats['completed_symbols'] / self.stats['total_symbols']) * 100
                        
                        print(f"\n📈 Progress: {self.stats['completed_symbols']}/{self.stats['total_symbols']} "
                              f"({progress:.1f}%) - Total records: {self.stats['total_records']:,}")
                
                except Exception as e:
                    print(f"❌ Unexpected error for {symbol} {timeframe}: {e}")
                    with self.lock:
                        self.stats['failed_symbols'] += 1
        
        # Финальная статистика
        end_time = datetime.now()
        duration = end_time - self.stats['start_time']
        
        final_stats = {
            'total_tasks': len(collection_plan),
            'successful_tasks': len([r for r in results if r['success']]),
            'failed_tasks': len([r for r in results if not r['success']]),
            'total_records': self.stats['total_records'],
            'duration_minutes': duration.total_seconds() / 60,
            'results': results
        }
        
        return final_stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Massive cryptocurrency data collection')
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--symbols', nargs='+', 
                       help='Specific symbols to collect (default: auto-detect popular)')
    parser.add_argument('--num-symbols', type=int, default=30,
                       help='Number of popular symbols to auto-collect')
    parser.add_argument('--timeframes', nargs='+', 
                       default=['1d', '1w', '4h', '1h', '30m', '15m', '5m'],
                       help='Timeframes to collect')
    parser.add_argument('--start-date', type=str, default='2017-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD), default: today')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory to save data')
    parser.add_argument('--max-workers', type=int, default=3,
                       help='Maximum parallel workers')
    parser.add_argument('--save-stats', type=str,
                       help='Save collection statistics to JSON file')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("🎯 MASSIVE CRYPTOCURRENCY DATA COLLECTION")
    print("=" * 60)
    
    # Initialize collector
    collector = MassiveDataCollector(
        data_dir=args.data_dir,
        max_workers=args.max_workers
    )
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
        print(f"📝 Using specified symbols: {symbols}")
    else:
        print(f"🔍 Auto-detecting top {args.num_symbols} popular symbols...")
        symbols = collector.parser.get_popular_symbols(limit=args.num_symbols)
        print(f"📈 Top symbols by volume: {symbols[:10]}...")
    
    # Start collection
    stats = collector.collect_all_data(
        symbols=symbols,
        timeframes=args.timeframes,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Print final results
    print(f"\n🎉 COLLECTION COMPLETED!")
    print("=" * 60)
    print(f"✅ Successful tasks: {stats['successful_tasks']}")
    print(f"❌ Failed tasks: {stats['failed_tasks']}")
    print(f"📊 Total records collected: {stats['total_records']:,}")
    print(f"⏱️ Duration: {stats['duration_minutes']:.1f} minutes")
    print(f"📁 Data saved to: {args.data_dir}")
    
    # Show top collections
    successful_results = [r for r in stats['results'] if r['success']]
    successful_results.sort(key=lambda x: x['records'], reverse=True)
    
    print(f"\n🏆 TOP COLLECTIONS:")
    for result in successful_results[:10]:
        print(f"   {result['symbol']} {result['timeframe']}: {result['records']:,} records")
    
    # Show failures
    failed_results = [r for r in stats['results'] if not r['success']]
    if failed_results:
        print(f"\n⚠️ FAILED COLLECTIONS:")
        for result in failed_results[:10]:
            print(f"   {result['symbol']} {result['timeframe']}: {result['error']}")
    
    # Save statistics
    if args.save_stats:
        with open(args.save_stats, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"\n💾 Statistics saved to: {args.save_stats}")


if __name__ == "__main__":
    main()
