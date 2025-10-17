# ML проект предсказания криптовалютного рынка с архитектурой MoE

## Описание проекта

Этот проект реализует систему предсказания криптовалютного рынка с использованием архитектуры **Mixture of Experts (MoE)**. Система анализирует рынок на разных временных интервалах:

- **Короткие интервалы** (5 минут, 30 минут, 1 час) - основные для предсказания
- **Долгосрочные интервалы** (1 день, 1 неделя) - для понимания направления и общей ситуации

Модель состоит из 5 экспертов, каждый из которых специализируется на определенном временном интервале. В отличие от обычных трансформеров, которые предсказывают вероятности токенов, наша модель предсказывает **конкретные изменения цены** в процентах.

## Архитектура

- **MoE (Mixture of Experts)** с 5 экспертами для разных временных интервалов
- **Transformer-based** архитектура для каждого эксперта
- **Gating Network** для взвешивания предсказаний экспертов
- **Multi-task Learning** (предсказание цены, направления и волатильности)
- **LoRA оптимизация** для эффективного обучения

## Технологический стек

- **Python 3.8+**
- **PyTorch** - основной фреймворк для ML
- **Transformers** - архитектура модели
- **PEFT (LoRA)** - эффективная адаптация модели
- **pandas, numpy** - обработка данных
- **scikit-learn** - метрики и предобработка
- **requests** - API взаимодействие
- **ta** - технические индикаторы

## Структура проекта

```
trader_ml/
├── src/
│   ├── data/
│   │   ├── bybit_parser.py      # Парсер данных с Bybit API
│   │   └── preprocessor.py      # Предобработка данных
│   ├── models/
│   │   └── moe_model.py         # MoE архитектура
│   ├── training/
│   │   └── trainer.py           # Пайплайн обучения
│   ├── inference/
│   │   └── predictor.py         # Модуль предсказаний
│   └── utils/
│       └── config.py            # Управление конфигурацией
├── train.py                     # Скрипт обучения
├── predict.py                   # Скрипт предсказаний
├── collect_data.py              # Скрипт сбора данных
├── config.json                  # Конфигурация проекта
└── requirements.txt             # Зависимости
```

## Установка и настройка

### Автоматическая установка

1. **Для CPU-only версии:**
```bash
./install_cpu.sh
```

2. **Для GPU версии (CUDA):**
```bash
pip install -r requirements.txt
```

### Ручная установка

1. **Установите PyTorch:**
```bash
# Для CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Для CUDA (замените cu118 на вашу версию CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. **Установите остальные зависимости:**
```bash
pip install -r requirements.txt
```

3. **Создайте необходимые директории:**
```bash
mkdir -p data models logs output
```

4. **Настройте конфигурацию** (опционально):
Отредактируйте `config.json` под ваши нужды.

## Использование

### 1. Сбор данных

Соберите исторические данные с Bybit API:

```bash
# Сбор данных для BTCUSDT с 2019 года
python collect_data.py --symbols BTCUSDT --start-date 2019-01-01

# Сбор данных для нескольких символов
python collect_data.py --symbols BTCUSDT ETHUSDT ADAUSDT --timeframes 5m 30m 1h 1d 1w
```

### 2. Обучение модели

Запустите обучение MoE модели:

```bash
# Базовое обучение (автоматический выбор устройства)
python train.py --symbols BTCUSDT --epochs 100

# Принудительное использование CPU
python train.py --symbols BTCUSDT --force-cpu --epochs 50

# Использование GPU (если доступен)
python train.py --symbols BTCUSDT --device cuda --epochs 100

# Обучение с LoRA оптимизацией
python train.py --symbols BTCUSDT ETHUSDT --use-lora --epochs 50

# Обучение с автоматическим сбором данных
python train.py --collect-data --symbols BTCUSDT --start-date 2019-01-01
```

### 3. Предсказания

Делайте предсказания с помощью обученной модели:

```bash
# Одиночное предсказание (автоматический выбор устройства)
python predict.py --symbols BTCUSDT

# Принудительное использование CPU
python predict.py --symbols BTCUSDT --force-cpu

# Использование GPU для инференса
python predict.py --symbols BTCUSDT --device cuda

# Предсказания для нескольких символов с детальным выводом
python predict.py --symbols BTCUSDT ETHUSDT --expert-outputs --output predictions.json

# Режим реального времени (каждые 5 минут)
python predict.py --symbols BTCUSDT --real-time --interval 300
```

## Конфигурация

Основные параметры в `config.json`:

```json
{
  "data": {
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "timeframes": ["5m", "30m", "1h", "1d", "1w"],
    "sequence_length": 100
  },
  "model": {
    "hidden_dim": 256,
    "num_layers": 4,
    "num_heads": 8
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "num_epochs": 100
  }
}
```

## Особенности реализации

### Архитектура MoE

- **5 экспертов** - по одному для каждого временного интервала
- **Gating Network** - определяет веса экспертов на основе входных данных
- **Multi-head выходы** - предсказание цены, направления и волатильности

### Управление устройствами (CPU/GPU)

Проект поддерживает автоматическое переключение между CPU и GPU:

- **Автоматический выбор** - система сама определяет лучшее устройство
- **Принудительный CPU** - флаг `--force-cpu` для использования только CPU
- **Выбор устройства** - параметр `--device` для явного указания (cpu/cuda/mps)
- **Оптимизация параметров** - автоматическая настройка batch_size и workers
- **Совместимость** - поддержка CUDA, Apple Silicon (MPS) и CPU

Примеры использования:
```bash
# Автоматический выбор устройства
python train.py --symbols BTCUSDT

# Принудительное использование CPU
python train.py --symbols BTCUSDT --force-cpu

# Использование конкретного GPU
python train.py --symbols BTCUSDT --device cuda

# Apple Silicon (M1/M2)
python train.py --symbols BTCUSDT --device mps
```

### Предобработка данных

- **Технические индикаторы** - RSI, MACD, Bollinger Bands, SMA, EMA
- **Нормализация** - StandardScaler или MinMaxScaler
- **Последовательности** - скользящие окна для временных рядов

### LoRA оптимизация

- **Эффективная адаптация** - обучение только части параметров
- **Низкий ранг** - r=16, alpha=32
- **Целевые модули** - attention и feed-forward слои

## API данных Bybit

Проект использует Bybit REST API для получения исторических данных:

- **Endpoint**: `GET /v5/market/kline`
- **Параметры**: category, symbol, interval, start, end, limit
- **Пагинация**: автоматическая обработка лимита в 1000 свечей
- **Rate limiting**: задержка 0.01 секунды между запросами

## Результаты и метрики

Модель предоставляет следующие предсказания:

1. **Изменение цены** (%) - регрессия
2. **Направление движения** - классификация (вверх/вниз)
3. **Волатильность** - предсказание будущей волатильности

Метрики оценки:
- **MSE, MAE, R²** для регрессии цены
- **Accuracy** для классификации направления
- **Confidence scores** для оценки уверенности

## Примеры использования

### Быстрый старт

```bash
# 1. Соберите данные
python collect_data.py --symbols BTCUSDT --start-date 2022-01-01

# 2. Обучите модель
python train.py --symbols BTCUSDT --epochs 50 --use-lora

# 3. Сделайте предсказание
python predict.py --symbols BTCUSDT --expert-outputs
```

### Продвинутое использование

```bash
# Обучение на нескольких символах с кастомной конфигурацией
python train.py \
  --config custom_config.json \
  --symbols BTCUSDT ETHUSDT ADAUSDT \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.0005 \
  --use-lora

# Непрерывные предсказания в реальном времени
python predict.py \
  --symbols BTCUSDT ETHUSDT \
  --real-time \
  --interval 300 \
  --duration 24
```

## Лицензия

MIT License

## Контакты

Для вопросов и предложений создавайте Issues в репозитории.
