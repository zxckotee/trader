#!/usr/bin/env python3
"""
Тест получения популярных символов с Bybit.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.bybit_parser import BybitParser


def main():
    print("🔍 Тестирование получения популярных символов...")
    
    parser = BybitParser()
    
    # Получаем топ-20 символов
    symbols = parser.get_popular_symbols(limit=20)
    
    print(f"\n📈 Топ-{len(symbols)} популярных USDT пар по объему торгов:")
    print("-" * 50)
    
    for i, symbol in enumerate(symbols, 1):
        print(f"{i:2d}. {symbol}")
    
    print(f"\n✅ Успешно получено {len(symbols)} символов")
    
    # Тест сбора небольшого количества данных для первого символа
    if symbols:
        test_symbol = symbols[0]
        print(f"\n🧪 Тестовый сбор данных для {test_symbol} (последние 7 дней, дневные свечи)...")
        
        try:
            df = parser.collect_historical_data(
                symbol=test_symbol,
                timeframe='1d',
                start_date='2024-10-10',
                end_date='2024-10-17'
            )
            
            if not df.empty:
                print(f"✅ Тест успешен: собрано {len(df)} записей")
                print(f"📅 Период: {df['timestamp'].min()} - {df['timestamp'].max()}")
                print(f"💰 Цена: {df['close'].iloc[0]:.2f} -> {df['close'].iloc[-1]:.2f}")
            else:
                print("❌ Тест не удался: данные не получены")
                
        except Exception as e:
            print(f"❌ Ошибка теста: {e}")


if __name__ == "__main__":
    main()
