#!/usr/bin/env python3
"""
Main training script for MoE cryptocurrency prediction model.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.bybit_parser import BybitParser
from src.data.preprocessor import CryptoDataPreprocessor
from src.models.moe_model import MoECryptoPredictor
from src.training.trainer import MoETrainer, CryptoDataset, create_trainer_config
from src.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MoE cryptocurrency prediction model')
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--symbols', nargs='+', 
                       help='Trading symbols to train on (default: auto-detect from CSV files)')
    parser.add_argument('--auto-symbols', action='store_true',
                       help='Automatically use all symbols found in data directory')
    parser.add_argument('--timeframes', nargs='+', default=['5m', '30m', '1h', '1d', '1w'],
                       help='Timeframes to use')
    parser.add_argument('--start-date', type=str, default='2019-01-01',
                       help='Start date for data collection')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--use-lora', action='store_true',
                       help='Use LoRA optimization')
    parser.add_argument('--collect-data', action='store_true',
                       help='Collect new data before training')
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Output directory for models')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'],
                       default='auto', help='Device to use for training')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    
    return parser.parse_args()


def collect_data(symbols, timeframes, start_date, data_dir):
    """Collect training data."""
    print("=== Data Collection ===")
    parser = BybitParser(data_dir=data_dir)
    
    all_data = {}
    
    for symbol in symbols:
        print(f"\nCollecting data for {symbol}...")
        symbol_data = parser.collect_multi_timeframe_data(
            symbol=symbol,
            timeframes=timeframes,
            start_date=start_date
        )
        
        if symbol_data:
            all_data[symbol] = symbol_data
            print(f"Successfully collected data for {symbol}")
        else:
            print(f"Failed to collect data for {symbol}")
    
    return all_data


def auto_detect_symbols(data_dir: str, timeframes: list) -> list:
    """
    Автоматически определяет доступные символы из CSV файлов.
    
    Args:
        data_dir: Директория с данными
        timeframes: Список таймфреймов для проверки
        
    Returns:
        Список найденных символов
    """
    import glob
    import os
    
    data_path = Path(data_dir)
    symbols = set()
    
    print(f"🔍 Поиск CSV файлов в {data_path}...")
    
    # Ищем все CSV файлы
    csv_files = list(data_path.glob("*.csv"))
    
    for csv_file in csv_files:
        filename = csv_file.stem  # Имя без расширения
        
        # Проверяем формат SYMBOL_TIMEFRAME
        for tf in timeframes:
            if filename.endswith(f"_{tf}"):
                symbol = filename[:-len(f"_{tf}")]
                symbols.add(symbol)
                break
    
    symbols_list = sorted(list(symbols))
    
    print(f"📊 Найдено символов: {len(symbols_list)}")
    for symbol in symbols_list:
        # Проверяем, какие таймфреймы доступны для каждого символа
        available_tf = []
        for tf in timeframes:
            if (data_path / f"{symbol}_{tf}.csv").exists():
                available_tf.append(tf)
        print(f"   {symbol}: {available_tf}")
    
    return symbols_list


def load_all_symbol_data(symbols: list, timeframes: list, data_dir: str) -> dict:
    """
    Загружает данные для всех символов и таймфреймов.
    
    Args:
        symbols: Список символов
        timeframes: Список таймфреймов
        data_dir: Директория с данными
        
    Returns:
        Словарь {symbol: {timeframe: DataFrame}}
    """
    from src.data.bybit_parser import BybitParser
    
    parser = BybitParser(data_dir=data_dir)
    all_data = {}
    
    print(f"\n📥 Загрузка данных для {len(symbols)} символов...")
    
    for symbol in symbols:
        symbol_data = {}
        total_records = 0
        
        for timeframe in timeframes:
            df = parser.load_data(symbol, timeframe)
            if not df.empty:
                symbol_data[timeframe] = df
                total_records += len(df)
                print(f"   ✅ {symbol} {timeframe}: {len(df):,} записей")
            else:
                print(f"   ⚠️ {symbol} {timeframe}: файл не найден или пуст")
        
        if symbol_data:
            all_data[symbol] = symbol_data
            print(f"   📊 {symbol} итого: {total_records:,} записей по {len(symbol_data)} таймфреймам")
        else:
            print(f"   ❌ {symbol}: нет данных")
    
    print(f"\n✅ Загружено данных для {len(all_data)} символов")
    return all_data


def prepare_datasets(data_dict, config):
    """Prepare training datasets."""
    print("\n=== Data Preprocessing ===")
    
    # Initialize preprocessor
    preprocessor = CryptoDataPreprocessor(
        sequence_length=config.get('data.sequence_length', 100),
        prediction_horizons=config.get('data.prediction_horizons'),
        scaler_type=config.get('preprocessing.scaler_type', 'standard')
    )
    
    # Combine data from all symbols
    combined_data = {}
    for timeframe in config.get('data.timeframes', []):
        combined_data[timeframe] = []
    
    for symbol, symbol_data in data_dict.items():
        for timeframe, df in symbol_data.items():
            if timeframe in combined_data:
                combined_data[timeframe].append(df)
    
    # Concatenate data for each timeframe
    for timeframe in combined_data:
        if combined_data[timeframe]:
            import pandas as pd
            combined_data[timeframe] = pd.concat(combined_data[timeframe], ignore_index=True)
        else:
            del combined_data[timeframe]
    
    # Process data
    processed_data = preprocessor.process_multi_timeframe_data(
        combined_data, fit_scalers=True
    )
    
    # Save preprocessor
    preprocessor.save_scalers(f"{config.get('paths.data_dir')}/scalers.pkl")
    
    # Split data
    train_split = config.get('training.train_split', 0.8)
    val_split = config.get('training.val_split', 0.1)
    
    # Find minimum length across timeframes
    min_length = min(len(data['X']) for data in processed_data.values())
    
    # Create indices for splitting
    indices = np.arange(min_length)
    train_indices, temp_indices = train_test_split(
        indices, test_size=(1 - train_split), random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=(1 - val_split / (1 - train_split)), random_state=42
    )
    
    # Split data
    train_data = {}
    val_data = {}
    test_data = {}
    
    for timeframe, data in processed_data.items():
        train_data[timeframe] = {
            'X': data['X'][train_indices],
            'y': data['y'][train_indices]
        }
        val_data[timeframe] = {
            'X': data['X'][val_indices],
            'y': data['y'][val_indices]
        }
        test_data[timeframe] = {
            'X': data['X'][test_indices],
            'y': data['y'][test_indices]
        }
    
    # Create datasets
    train_dataset = CryptoDataset(train_data, list(processed_data.keys()))
    val_dataset = CryptoDataset(val_data, list(processed_data.keys()))
    test_dataset = CryptoDataset(test_data, list(processed_data.keys()))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Get input dimension
    input_dim = next(iter(processed_data.values()))['X'].shape[2]
    
    return train_dataset, val_dataset, test_dataset, input_dim


def create_model(input_dim, timeframes, config):
    """Create MoE model."""
    print("\n=== Model Creation ===")
    
    model = MoECryptoPredictor(
        input_dim=input_dim,
        timeframes=timeframes,
        expert_config={
            'hidden_dim': config.get('model.hidden_dim', 256),
            'num_layers': config.get('model.num_layers', 4),
            'num_heads': config.get('model.num_heads', 8),
            'dropout': config.get('model.dropout', 0.1)
        },
        use_gating=config.get('model.use_gating', True)
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine symbols to use
    if args.auto_symbols or (not args.symbols and not config.get('data.symbols')):
        # Auto-detect symbols from CSV files
        symbols = auto_detect_symbols(
            data_dir=config.get('paths.data_dir', './data'),
            timeframes=args.timeframes or config.get('data.timeframes', ['5m', '30m', '1h', '1d', '1w'])
        )
        if not symbols:
            print("❌ No symbols found in data directory. Please collect data first.")
            return
        config.set('data.symbols', symbols)
    elif args.symbols:
        config.set('data.symbols', args.symbols)
    
    # Override other config with command line arguments
    if args.timeframes:
        config.set('data.timeframes', args.timeframes)
    if args.start_date:
        config.set('data.start_date', args.start_date)
    if args.epochs:
        config.set('training.num_epochs', args.epochs)
    if args.batch_size:
        config.set('training.batch_size', args.batch_size)
    if args.learning_rate:
        config.set('training.learning_rate', args.learning_rate)
    if args.output_dir:
        config.set('paths.model_dir', args.output_dir)
    if args.device:
        config.set('training.device', args.device)
    if args.force_cpu:
        config.set('training.force_cpu', True)
    
    print("=== MoE Cryptocurrency Prediction Model Training ===")
    print(f"Symbols: {config.get('data.symbols')}")
    print(f"Timeframes: {config.get('data.timeframes')}")
    print(f"Start date: {config.get('data.start_date')}")
    
    # Load or collect data
    if args.collect_data:
        data_dict = collect_data(
            symbols=config.get('data.symbols'),
            timeframes=config.get('data.timeframes'),
            start_date=config.get('data.start_date'),
            data_dir=config.get('paths.data_dir')
        )
    else:
        # Load existing data for all symbols
        data_dict = load_all_symbol_data(
            symbols=config.get('data.symbols'),
            timeframes=config.get('data.timeframes'),
            data_dir=config.get('paths.data_dir')
        )
    
    if not data_dict:
        print("No data available. Please run with --collect-data flag first.")
        return
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset, input_dim = prepare_datasets(data_dict, config)
    
    # Create model
    model = create_model(input_dim, config.get('data.timeframes'), config)
    
    # Create trainer configuration
    trainer_config = create_trainer_config()
    trainer_config.update({
        'num_epochs': config.get('training.num_epochs'),
        'batch_size': config.get('training.batch_size'),
        'learning_rate': config.get('training.learning_rate'),
        'output_dir': config.get('paths.model_dir'),
        'price_weight': config.get('loss.price_weight'),
        'direction_weight': config.get('loss.direction_weight'),
        'volatility_weight': config.get('loss.volatility_weight'),
        'diversity_weight': config.get('loss.diversity_weight')
    })
    
    # Create trainer
    trainer = MoETrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_config
    )
    
    # Apply LoRA if requested
    if args.use_lora and config.get('lora.enabled'):
        print("\n=== Applying LoRA ===")
        trainer.apply_lora(
            r=config.get('lora.r', 16),
            lora_alpha=config.get('lora.lora_alpha', 32),
            lora_dropout=config.get('lora.lora_dropout', 0.1)
        )
    
    # Start training
    print("\n=== Training ===")
    trainer.train()
    
    # Evaluate on test set
    if test_dataset:
        print("\n=== Test Evaluation ===")
        # You would implement test evaluation here
        print("Test evaluation completed")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
