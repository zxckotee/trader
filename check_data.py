#!/usr/bin/env python3
"""
Скрипт для проверки доступных данных в директории.
Показывает статистику по всем CSV файлам.
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.bybit_parser import BybitParser


def analyze_data_directory(data_dir: str = "./data"):
    """
    Анализирует все CSV файлы в директории данных.
    
    Args:
        data_dir: Путь к директории с данными
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Директория {data_path} не существует")
        return
    
    print(f"📁 Анализ данных в директории: {data_path.absolute()}")
    print("=" * 80)
    
    # Найти все CSV файлы
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        print("❌ CSV файлы не найдены")
        return
    
    print(f"📊 Найдено {len(csv_files)} CSV файлов\n")
    
    # Группировка по символам
    symbols_data = {}
    timeframes = set()
    
    for csv_file in csv_files:
        filename = csv_file.stem
        
        # Попытка разделить на символ и таймфрейм
        parts = filename.split('_')
        if len(parts) >= 2:
            timeframe = parts[-1]
            symbol = '_'.join(parts[:-1])
            
            if symbol not in symbols_data:
                symbols_data[symbol] = {}
            
            # Загрузка и анализ файла
            try:
                df = pd.read_csv(csv_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                symbols_data[symbol][timeframe] = {
                    'records': len(df),
                    'start_date': df['timestamp'].min(),
                    'end_date': df['timestamp'].max(),
                    'file_size_mb': csv_file.stat().st_size / (1024 * 1024),
                    'price_range': (df['close'].min(), df['close'].max())
                }
                timeframes.add(timeframe)
                
            except Exception as e:
                print(f"⚠️ Ошибка чтения {csv_file}: {e}")
    
    # Сортировка символов и таймфреймов
    sorted_symbols = sorted(symbols_data.keys())
    sorted_timeframes = sorted(timeframes, key=lambda x: {
        '1m': 1, '3m': 2, '5m': 3, '15m': 4, '30m': 5, '1h': 6,
        '2h': 7, '4h': 8, '6h': 9, '12h': 10, '1d': 11, '1w': 12
    }.get(x, 999))
    
    # Статистика по символам
    print("🪙 СИМВОЛЫ И ДАННЫЕ:")
    print("-" * 80)
    
    total_records = 0
    total_size_mb = 0
    
    for symbol in sorted_symbols:
        symbol_records = 0
        symbol_size = 0
        available_tf = []
        
        print(f"\n📈 {symbol}:")
        
        for tf in sorted_timeframes:
            if tf in symbols_data[symbol]:
                data = symbols_data[symbol][tf]
                symbol_records += data['records']
                symbol_size += data['file_size_mb']
                available_tf.append(tf)
                
                print(f"   {tf:>4}: {data['records']:>8,} записей | "
                      f"{data['start_date'].strftime('%Y-%m-%d')} → {data['end_date'].strftime('%Y-%m-%d')} | "
                      f"{data['file_size_mb']:>6.1f} MB | "
                      f"${data['price_range'][0]:>8.2f} - ${data['price_range'][1]:>8.2f}")
        
        print(f"   {'ИТОГО':>4}: {symbol_records:>8,} записей | "
              f"{len(available_tf)} таймфреймов | {symbol_size:>6.1f} MB")
        
        total_records += symbol_records
        total_size_mb += symbol_size
    
    # Общая статистика
    print("\n" + "=" * 80)
    print("📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Символов: {len(sorted_symbols)}")
    print(f"   Таймфреймов: {len(sorted_timeframes)} ({', '.join(sorted_timeframes)})")
    print(f"   Всего записей: {total_records:,}")
    print(f"   Общий размер: {total_size_mb:.1f} MB")
    print(f"   Средний размер файла: {total_size_mb/len(csv_files):.1f} MB")
    
    # Рекомендации для обучения
    print("\n🎯 РЕКОМЕНДАЦИИ ДЛЯ ОБУЧЕНИЯ:")
    print("-" * 80)
    
    # Символы с наибольшим количеством данных
    symbol_totals = [(symbol, sum(data['records'] for data in symbols_data[symbol].values())) 
                     for symbol in sorted_symbols]
    symbol_totals.sort(key=lambda x: x[1], reverse=True)
    
    print("📈 Топ символов по количеству данных:")
    for i, (symbol, records) in enumerate(symbol_totals[:10], 1):
        timeframes_count = len(symbols_data[symbol])
        print(f"   {i:2d}. {symbol:<12}: {records:>8,} записей ({timeframes_count} таймфреймов)")
    
    # Команды для обучения
    print(f"\n🚀 КОМАНДЫ ДЛЯ ОБУЧЕНИЯ:")
    
    # Все символы
    all_symbols_str = ' '.join(sorted_symbols)
    print(f"\n# Обучение на ВСЕХ символах:")
    print(f"python train.py --auto-symbols --epochs 50 --force-cpu")
    
    # Топ символы
    top_symbols = [symbol for symbol, _ in symbol_totals[:10]]
    top_symbols_str = ' '.join(top_symbols)
    print(f"\n# Обучение на топ-10 символах:")
    print(f"python train.py --symbols {top_symbols_str} --epochs 50")
    
    # Только дневные данные
    daily_symbols = [symbol for symbol in sorted_symbols if '1d' in symbols_data[symbol]]
    if daily_symbols:
        print(f"\n# Быстрое обучение только на дневных данных:")
        print(f"python train.py --auto-symbols --timeframes 1d --epochs 30 --force-cpu")
    
    return symbols_data


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check available cryptocurrency data')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory to analyze')
    
    args = parser.parse_args()
    
    analyze_data_directory(args.data_dir)


if __name__ == "__main__":
    main()
