#!/usr/bin/env python3
"""
Пример использования MoE модели для предсказания криптовалютного рынка.
Демонстрирует полный пайплайн: сбор данных -> обучение -> предсказание.
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.bybit_parser import BybitParser
from src.data.preprocessor import CryptoDataPreprocessor
from src.models.moe_model import MoECryptoPredictor
from src.utils.config import load_config


def demo_data_collection():
    """Демонстрация сбора данных."""
    print("=== Демонстрация сбора данных ===")
    
    # Инициализация парсера
    parser = BybitParser(data_dir="./demo_data")
    
    # Сбор небольшого количества данных для демо
    symbol = "BTCUSDT"
    timeframes = ['5m', '1h']  # Только 2 таймфрейма для быстроты
    start_date = "2024-01-01"
    
    print(f"Собираем данные для {symbol} с {start_date}...")
    
    try:
        data = parser.collect_multi_timeframe_data(
            symbol=symbol,
            timeframes=timeframes,
            start_date=start_date
        )
        
        if data:
            print("✓ Данные успешно собраны:")
            for tf, df in data.items():
                print(f"  {tf}: {len(df)} записей")
            return data
        else:
            print("✗ Не удалось собрать данные")
            return None
            
    except Exception as e:
        print(f"Ошибка при сборе данных: {e}")
        return None


def demo_preprocessing(data_dict):
    """Демонстрация предобработки данных."""
    print("\n=== Демонстрация предобработки данных ===")
    
    if not data_dict:
        print("Нет данных для предобработки")
        return None
    
    # Инициализация препроцессора
    preprocessor = CryptoDataPreprocessor(
        sequence_length=50,  # Уменьшенная длина для демо
        scaler_type='standard'
    )
    
    try:
        # Обработка данных
        processed_data = preprocessor.process_multi_timeframe_data(
            data_dict, fit_scalers=True
        )
        
        print("✓ Данные успешно обработаны:")
        for tf, data in processed_data.items():
            print(f"  {tf}: {data['X'].shape[0]} последовательностей, "
                  f"размерность признаков: {data['X'].shape[2]}")
        
        # Сохранение скейлеров
        preprocessor.save_scalers("./demo_data/demo_scalers.pkl")
        
        return processed_data, preprocessor
        
    except Exception as e:
        print(f"Ошибка при предобработке: {e}")
        return None, None


def demo_model_creation(input_dim, timeframes):
    """Демонстрация создания модели."""
    print("\n=== Демонстрация создания MoE модели ===")
    
    try:
        # Создание модели с уменьшенными размерами для демо
        model = MoECryptoPredictor(
            input_dim=input_dim,
            timeframes=timeframes,
            expert_config={
                'hidden_dim': 128,  # Уменьшено для демо
                'num_layers': 2,    # Уменьшено для демо
                'num_heads': 4,     # Уменьшено для демо
                'dropout': 0.1
            },
            use_gating=True
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Модель создана:")
        print(f"  Общее количество параметров: {total_params:,}")
        print(f"  Обучаемых параметров: {trainable_params:,}")
        print(f"  Экспертов: {len(timeframes)}")
        
        return model
        
    except Exception as e:
        print(f"Ошибка при создании модели: {e}")
        return None


def demo_inference(model, processed_data, preprocessor):
    """Демонстрация инференса модели."""
    print("\n=== Демонстрация инференса ===")
    
    if not model or not processed_data:
        print("Нет модели или данных для инференса")
        return
    
    try:
        import torch
        
        model.eval()
        
        # Берем последнюю последовательность из каждого таймфрейма
        inputs = {}
        for tf, data in processed_data.items():
            if data['X'].shape[0] > 0:
                # Берем последнюю последовательность и добавляем batch dimension
                last_sequence = torch.FloatTensor(data['X'][-1:])
                inputs[tf] = last_sequence
        
        if inputs:
            print(f"Входные данные подготовлены для {len(inputs)} таймфреймов")
            
            # Прямой проход через модель
            with torch.no_grad():
                outputs = model(inputs)
            
            print("✓ Предсказание выполнено:")
            
            # Обработка результатов
            if 'price_change' in outputs:
                price_change_pct = float(outputs['price_change'].item() * 100)
                print(f"  Изменение цены: {price_change_pct:+.2f}%")
            
            if 'direction_logits' in outputs:
                direction_probs = torch.softmax(outputs['direction_logits'], dim=-1)
                direction = 'ВВЕРХ' if torch.argmax(direction_probs).item() == 1 else 'ВНИЗ'
                confidence = float(torch.max(direction_probs).item())
                print(f"  Направление: {direction} (уверенность: {confidence:.3f})")
            
            if 'volatility' in outputs:
                volatility = float(outputs['volatility'].item())
                print(f"  Волатильность: {volatility:.4f}")
            
            # Веса экспертов
            if 'expert_weights' in outputs and outputs['expert_weights'] is not None:
                expert_weights = outputs['expert_weights'].cpu().numpy()[0]
                print("  Веса экспертов:")
                for i, (tf, weight) in enumerate(zip(inputs.keys(), expert_weights)):
                    print(f"    {tf}: {weight:.3f}")
        
        else:
            print("Нет входных данных для инференса")
            
    except Exception as e:
        print(f"Ошибка при инференсе: {e}")


def demo_full_pipeline():
    """Демонстрация полного пайплайна."""
    print("🚀 Запуск демонстрации MoE модели для криптовалютного трейдинга")
    print("=" * 70)
    
    start_time = time.time()
    
    # 1. Сбор данных
    data_dict = demo_data_collection()
    
    if not data_dict:
        print("❌ Не удалось собрать данные. Завершение демо.")
        return
    
    # 2. Предобработка
    processed_data, preprocessor = demo_preprocessing(data_dict)
    
    if not processed_data:
        print("❌ Не удалось обработать данные. Завершение демо.")
        return
    
    # 3. Создание модели
    input_dim = next(iter(processed_data.values()))['X'].shape[2]
    timeframes = list(processed_data.keys())
    
    model = demo_model_creation(input_dim, timeframes)
    
    if not model:
        print("❌ Не удалось создать модель. Завершение демо.")
        return
    
    # 4. Инференс (без обучения для демо)
    demo_inference(model, processed_data, preprocessor)
    
    # Итоги
    elapsed_time = time.time() - start_time
    print(f"\n🎉 Демонстрация завершена за {elapsed_time:.1f} секунд")
    print("\nДля полноценного использования:")
    print("1. Соберите больше данных: python collect_data.py")
    print("2. Обучите модель: python train.py")
    print("3. Делайте предсказания: python predict.py")


def demo_quick_test():
    """Быстрый тест без сбора данных."""
    print("⚡ Быстрый тест архитектуры модели")
    print("=" * 40)
    
    try:
        import torch
        
        # Создание синтетических данных
        batch_size = 2
        seq_len = 50
        input_dim = 30
        timeframes = ['5m', '1h']
        
        # Создание модели
        model = MoECryptoPredictor(
            input_dim=input_dim,
            timeframes=timeframes,
            expert_config={'hidden_dim': 64, 'num_layers': 2, 'num_heads': 4},
            use_gating=True
        )
        
        # Синтетические входные данные
        inputs = {}
        for tf in timeframes:
            inputs[tf] = torch.randn(batch_size, seq_len, input_dim)
        
        # Прямой проход
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        
        print("✓ Архитектура модели работает корректно:")
        print(f"  Входные данные: {batch_size} батча, {seq_len} временных шагов, {input_dim} признаков")
        print(f"  Выходные данные:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.shape}")
        
        print(f"  Параметры модели: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Ошибка в быстром тесте: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Демонстрация MoE модели')
    parser.add_argument('--quick', action='store_true', 
                       help='Быстрый тест без сбора данных')
    
    args = parser.parse_args()
    
    if args.quick:
        demo_quick_test()
    else:
        demo_full_pipeline()
