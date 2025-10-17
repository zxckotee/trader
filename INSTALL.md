# Руководство по установке

## Быстрая установка

### Вариант 1: Автоматическая установка (CPU)
```bash
chmod +x install_cpu.sh
./install_cpu.sh
```

### Вариант 2: Универсальная установка
```bash
pip install -r requirements.txt
```

## Детальная установка

### 1. Установка PyTorch

#### Для CPU-only системы:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Для CUDA (GPU):
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Для Apple Silicon (M1/M2):
```bash
pip install torch torchvision torchaudio
```

### 2. Установка остальных зависимостей
```bash
pip install transformers>=4.30.0
pip install peft>=0.4.0
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install requests>=2.31.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install scikit-learn>=1.3.0
pip install tqdm>=4.65.0
pip install python-dotenv>=1.0.0
pip install ccxt>=4.0.0
pip install ta>=0.10.2
```

### 3. Создание директорий
```bash
mkdir -p data models logs output
```

## Проверка установки

### Проверка PyTorch
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Быстрый тест архитектуры
```bash
python example_usage.py --quick
```

## Переключение между CPU и GPU

### Автоматическое определение устройства
Система автоматически определит лучшее доступное устройство:
- CUDA GPU (если доступен)
- Apple Silicon MPS (если доступен)
- CPU (fallback)

### Принудительное использование CPU
Добавьте флаг `--force-cpu` к любой команде:
```bash
python train.py --symbols BTCUSDT --force-cpu
python predict.py --symbols BTCUSDT --force-cpu
```

### Выбор конкретного устройства
Используйте параметр `--device`:
```bash
python train.py --symbols BTCUSDT --device cpu
python train.py --symbols BTCUSDT --device cuda
python train.py --symbols BTCUSDT --device mps  # Apple Silicon
```

## Конфигурация устройства в config.json

```json
{
  "training": {
    "device": "auto",        // "auto", "cpu", "cuda", "mps"
    "force_cpu": false,      // true для принудительного CPU
    "batch_size": 32,        // автоматически адаптируется под устройство
    "num_workers": "auto"    // автоматически адаптируется под устройство
  }
}
```

## Решение проблем

### Ошибка "CUDA out of memory"
1. Уменьшите batch_size в config.json
2. Используйте CPU: `--force-cpu`
3. Закройте другие GPU-приложения

### Медленная работа на CPU
1. Уменьшите размер модели в config.json:
   ```json
   {
     "model": {
       "hidden_dim": 128,
       "num_layers": 2,
       "num_heads": 4
     }
   }
   ```
2. Уменьшите sequence_length и batch_size

### Ошибки импорта
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Проблемы с LoRA
```bash
pip install peft --upgrade
```

## Системные требования

### Минимальные требования (CPU):
- Python 3.8+
- 8 GB RAM
- 2 GB свободного места

### Рекомендуемые требования (GPU):
- Python 3.8+
- 16 GB RAM
- NVIDIA GPU с 6+ GB VRAM
- CUDA 11.8+

### Apple Silicon:
- macOS 12.0+
- 16 GB RAM
- Apple M1/M2 чип
