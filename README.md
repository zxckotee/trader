# 🚀 ML проект предсказания криптовалютного рынка с архитектурой MoE

## 🎯 Описание проекта

Профессиональная система предсказания криптовалютного рынка с использованием архитектуры **Mixture of Experts (MoE)** и **430M параметров**. Система анализирует **ВСЕ аспекты криптовалюты**: цены, объемы, активность китов, давление покупателей/продавцов на разных временных интервалах.

### 🔥 Ключевые особенности:

- **430M параметров** - мощная модель уровня GPT-2 Medium
- **Multi-Timeframe Analysis** - 5 экспертов для 5 таймфреймов
- **Полный Volume Analysis** - обнаружение китов и манипуляций
- **Multi-Task Learning** - цена, направление, волатильность
- **LoRA оптимизация** - эффективное обучение на GPU
- **RTX 2060 Super friendly** - работает на 8GB VRAM!

---

## 📊 ЧТО АНАЛИЗИРУЕТ МОДЕЛЬ

### 1. 📈 ЦЕНОВЫЕ ПАТТЕРНЫ
```
✅ Тренды: SMA, EMA, MACD
✅ Импульс: RSI, Stochastic
✅ Волатильность: Bollinger Bands, ATR
✅ Экстремумы и диапазоны
```

### 2. 🐋 АНАЛИЗ ОБЪЕМОВ (КРИТИЧНО!)
```
✅ Обнаружение китов: volume_spike, volume_ratio
✅ Накопление/Распределение: OBV, A/D Line
✅ Давление Buy/Sell: CMF, MFI
✅ Сила движений: Force Index, Ease of Movement
✅ Volume-Price тренды: VPT, VWAP
```

### 3. ⏰ MULTI-TIMEFRAME
```
✅ 5m   - краткосрочные паттерны (скальпинг)
✅ 30m  - среднесрочные тренды
✅ 1h   - часовые циклы
✅ 1d   - дневные тренды
✅ 1w   - долгосрочные циклы
```

### 4. 🎯 ПРЕДСКАЗАНИЯ
```
✅ Price Change   - изменение цены в %
✅ Direction      - направление (вверх/вниз)
✅ Volatility     - будущая волатильность (риск)
```

---

## 🏗️ Архитектура Модели

```
📊 MoE (Mixture of Experts) с 430M параметров

├─ 5 Transformer Experts (по 86M каждый):
│  ├─ Expert_5m   : 768 hidden × 12 layers
│  ├─ Expert_30m  : 768 hidden × 12 layers
│  ├─ Expert_1h   : 768 hidden × 12 layers
│  ├─ Expert_1d   : 768 hidden × 12 layers
│  └─ Expert_1w   : 768 hidden × 12 layers
│
├─ Gating Network : Умный выбор экспертов
│
└─ Output Heads:
   ├─ Price Head      : Tanh + scale ([-0.1, 0.1])
   ├─ Direction Head  : Softmax (0/1)
   └─ Volatility Head : Sigmoid + scale ([0, 0.1])

💾 Memory: ~6.4 GB (training), ~1.6 GB (inference)
⚡ Speed: ~3-4 epochs/hour (RTX 2060 Super)
```

---

## 🛠️ Установка

### 1. Клонирование репозитория
```bash
git clone <your-repo>
cd trader
```

### 2. Установка зависимостей

**Для GPU (CUDA):**
```bash
pip install -r requirements.txt
```

**Для CPU:**
```bash
pip install -r requirements_cpu.txt
```

### 3. Структура проекта
```
trader/
├── src/
│   ├── data/
│   │   ├── bybit_parser.py      # Парсер Bybit API
│   │   └── preprocessor.py      # 50+ технических индикаторов
│   ├── models/
│   │   └── moe_model.py         # MoE архитектура (430M params)
│   ├── training/
│   │   └── trainer.py           # Multi-GPU training с LoRA
│   └── utils/
│       └── config.py            # Управление конфигурацией
├── train.py                     # 🚀 Основной скрипт обучения
├── predict.py                   # 🔮 Скрипт предсказаний
├── collect_data.py              # 📊 Сбор данных
├── config.json                  # ⚙️ Конфигурация модели
└── README.md                    # 📖 Этот файл
```

---

## 📊 СБОР ДАННЫХ

### Вариант 1: ТОП-20 монет (автоматически)
```bash
python collect_data.py --auto-symbols --limit 20
```

Это загрузит данные для:
- ТОП-20 монет по объему торгов
- Все 5 таймфреймов (5m, 30m, 1h, 1d, 1w)
- С 2022-01-01 до текущей даты

### Вариант 2: Конкретные монеты
```bash
python collect_data.py --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT
```

### Вариант 3: МАКСИМАЛЬНЫЙ сбор (всё доступное)
```bash
python collect_massive_data.py --limit 100 --start-date 2020-01-01
```

**Что собирается:**
```
✅ OHLCV данные (Open, High, Low, Close, Volume)
✅ Turnover (оборот в USDT)
✅ Timestamp (временная метка)
✅ Для каждого таймфрейма отдельно
```

**Где хранится:** `./data/` (CSV файлы)

---

## 🎓 ОБУЧЕНИЕ МОДЕЛИ

### 🚀 Базовое обучение
```bash
python train.py --epochs 300
```

### 🔥 МАКСИМАЛЬНОЕ обучение с LoRA на GPU

**1. С автоматическим определением монет:**
```bash
python train.py \
    --auto-symbols \
    --epochs 300 \
    --use-lora \
    --device cuda
```

**2. С конкретными монетами:**
```bash
python train.py \
    --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT \
    --epochs 300 \
    --use-lora \
    --batch-size 16 \
    --learning-rate 0.00008
```

**3. Продолжить обучение с чекпоинта:**
```bash
python train.py \
    --resume models/checkpoint_epoch_50.pt \
    --epochs 300 \
    --use-lora
```

### ⚙️ Параметры обучения

```bash
--auto-symbols              # Авто-определение ТОП монет
--symbols [...]             # Конкретные монеты
--epochs N                  # Количество эпох (default: 300)
--batch-size N              # Размер батча (default: 16)
--learning-rate LR          # Learning rate (default: 0.00008)
--use-lora                  # Включить LoRA оптимизацию
--resume PATH               # Продолжить с чекпоинта
--device cuda/cpu           # Устройство (auto, cuda, cpu)
--keep-checkpoints N        # Сохранять последние N чекпоинтов
```

### 📈 Что происходит при обучении

```
Epoch 1/300
Training:   100%|████████| 90/90 [00:26<00:00, 3.45it/s, loss=0.17, lr=1.00e-04]
Validation: 100%|████████| 12/12 [00:22<00:00, 1.88s/it]

Train Loss: 0.1234, Val Loss: 0.2345
Train Dir Acc: 0.6336, Val Dir Acc: 0.6067
Val R2: 0.2341

✅ Saved checkpoint to: models/checkpoint_epoch_1.pt
📊 Total checkpoints: 1, keeping last 10
```

### 🎯 Целевые метрики для прибыльной торговли

| Метрика | Минимум | Хорошо | Отлично |
|---------|---------|--------|---------|
| **Val Dir Acc** | > 57% | > 60% | > 65% |
| **Val R²** | > 0.15 | > 0.30 | > 0.50 |
| **Val Loss** | < 1.0 | < 0.5 | < 0.3 |

---

## 🔮 ИСПОЛЬЗОВАНИЕ МОДЕЛИ

### Предсказание для одной монеты
```bash
python predict.py --symbol BTCUSDT --timeframe 5m
```

### Предсказание для нескольких монет
```bash
python predict.py --symbols BTCUSDT ETHUSDT BNBUSDT
```

### Вывод:
```
=== Predictions for BTCUSDT ===
Current Price: $67,234.50

Predictions (5m timeframe):
  Price Change: +0.32% (±0.15%)
  Direction: UP (confidence: 67.3%)
  Volatility: 0.0042 (risk level: MEDIUM)

Recommendation: BUY
Stop Loss: $66,950 (-0.42%)
Take Profit: $67,800 (+0.84%)
```

---

## ⚙️ КОНФИГУРАЦИЯ

### config.json - Основные параметры

```json
{
  "model": {
    "hidden_dim": 768,        // Размерность модели
    "num_layers": 12,         // Количество слоев
    "num_heads": 12,          // Attention heads
    "feedforward_dim": 3072,  // FFN размерность
    "dropout": 0.15           // Dropout rate
  },
  
  "training": {
    "batch_size": 16,                    // Размер батча
    "learning_rate": 0.00008,            // Learning rate
    "gradient_accumulation_steps": 4,    // Gradient accumulation
    "early_stopping_patience": 50        // Early stopping
  },
  
  "loss": {
    "price_weight": 1.0,         // Вес цены
    "direction_weight": 2.0,     // Вес направления (важнее!)
    "volatility_weight": 0.5,    // Вес волатильности
    "diversity_weight": 0.1      // Разнообразие экспертов
  }
}
```

---

## 🧪 ДИАГНОСТИКА И ОТЛАДКА

### Проверка размера модели
```bash
python check_model_size.py
```

Вывод:
```
Total parameters: 429,860,667 (429.9M)
Memory (training): ~6.4 GB
Memory (inference): ~1.6 GB
Fits RTX 2060 Super 8GB: ✅ YES!
```

### Проверка данных
```bash
python debug_data.py
```

### Анализ признаков
```bash
python check_features.py
```

---

## 📊 ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (50+)

### Ценовые индикаторы:
- SMA (20, 50), EMA (12, 26)
- MACD, MACD Signal, MACD Histogram
- RSI (14), Stochastic (K, D)
- Bollinger Bands (upper, middle, lower, width, position)
- ATR (Average True Range)

### 🐋 Volume индикаторы (обнаружение китов):
- **OBV** (On Balance Volume) - накопление/распределение
- **A/D Line** - линия накопления/распределения
- **CMF** (Chaikin Money Flow) - денежный поток
- **MFI** (Money Flow Index) - индекс денежного потока
- **Force Index** - сила движения
- **Ease of Movement** - легкость движения
- **VPT** (Volume-Price Trend) - volume-price тренд
- **VWAP** - цена взвешенная по объему
- **Volume Spike** - обнаружение аномальных объемов
- **Volume Ratio** - отношение к среднему объему

### Статистические признаки:
- Return std (5, 10, 20 периодов)
- Volume std (5, 10, 20 периодов)
- Price extremes (max/min за 5, 10, 20 периодов)

**ИТОГО: 50+ признаков** для каждого таймфрейма!

---

## 🎯 УЛУЧШЕНИЯ И РАСШИРЕНИЯ

### ✅ Что добавлено недавно:

1. **Модель увеличена до 430M параметров** (было ~100M)
   - Hidden dim: 768 (было 256)
   - Layers: 12 (было 4)
   - Heads: 12 (было 8)

2. **Полный Volume Analysis**
   - Обнаружение китов (volume spikes)
   - Давление buy/sell (CMF, MFI)
   - Накопление/распределение (OBV, A/D)
   - Force Index, Ease of Movement, VPT

3. **Улучшенная архитектура Output Heads**
   - Tanh активация для price (ограничение [-0.1, 0.1])
   - Sigmoid для volatility (ограничение [0, 0.1])
   - Предотвращение огромных потерь

4. **Gradient Accumulation**
   - Эффективный размер батча = 64 (16 × 4)
   - Стабильное обучение на GPU 8GB

5. **Улучшенные веса Loss функции**
   - Direction weight = 2.0 (направление важнее точной цены!)
   - Volatility weight = 0.5 (риск-менеджмент)

---

## 🚀 ПРОИЗВОДИТЕЛЬНОСТЬ

### На RTX 2060 Super 8GB:
- **Обучение:** ~3-4 эпохи/час
- **Inference:** ~100 предсказаний/сек
- **Memory:** 6.4 GB (training), 1.6 GB (inference)

### На CPU (16-core):
- **Обучение:** ~0.5-1 эпоха/час
- **Inference:** ~10-20 предсказаний/сек

### Рекомендации:
- **Минимум:** RTX 2060 Super 8GB или аналог
- **Оптимально:** RTX 3070 Ti / RTX 4070 (12GB+)
- **Профи:** RTX 4090 / A100 (24GB+)

---

## 📚 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ

### 1. Быстрый старт (для начинающих)
```bash
# Собрать данные для BTC и ETH
python collect_data.py --symbols BTCUSDT ETHUSDT

# Обучить модель (50 эпох для теста)
python train.py --epochs 50

# Сделать предсказание
python predict.py --symbol BTCUSDT
```

### 2. Профессиональный сценарий
```bash
# Собрать ТОП-20 монет
python collect_data.py --auto-symbols --limit 20

# Максимальное обучение с LoRA
python train.py \
    --auto-symbols \
    --epochs 300 \
    --use-lora \
    --batch-size 16 \
    --learning-rate 0.00008 \
    --device cuda

# Мониторинг в real-time
python predict.py --symbols BTCUSDT ETHUSDT --loop --interval 300
```

### 3. Продвинутый (multi-GPU)
```bash
# Обучение на нескольких GPU
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --auto-symbols \
    --epochs 300 \
    --use-lora \
    --batch-size 32
```

---

## 🐛 TROUBLESHOOTING

### Проблема: Out of Memory (OOM)
**Решение:**
```bash
# Уменьшить batch_size
python train.py --batch-size 8 --gradient-accumulation-steps 8

# Или использовать CPU
python train.py --force-cpu
```

### Проблема: Огромные потери (loss > 1000)
**Решение:** Уже исправлено! Модель использует Tanh/Sigmoid активации.

### Проблема: Модель не учится (accuracy ~50%)
**Решение:**
1. Собрать больше данных (минимум 1000 строк)
2. Увеличить epochs (300+)
3. Проверить learning rate (попробовать 0.0001)

---

## 📈 ROADMAP

### В разработке:
- [ ] Real-time streaming данных
- [ ] Order Book анализ (глубина рынка)
- [ ] Sentiment Analysis (Twitter, Reddit)
- [ ] Multi-currency portfolio optimization
- [ ] Auto-trading бот с risk management

### Планируется:
- [ ] On-chain метрики (для BTC/ETH)
- [ ] Макроэкономические индикаторы
- [ ] Reinforcement Learning для адаптации
- [ ] Web интерфейс для мониторинга

---

## 📝 LICENSE

MIT License - используйте свободно!

---

## 🤝 CONTRIBUTING

Pull requests приветствуются! Для серьёзных изменений сначала откройте issue.

---

## ⚠️ DISCLAIMER

**Этот проект создан в образовательных целях.**

⚠️ Торговля криптовалютами связана с высокими рисками!
⚠️ Никогда не инвестируйте больше, чем можете позволить себе потерять!
⚠️ Модель НЕ гарантирует прибыль!
⚠️ Всегда используйте stop-loss и risk management!

**Автор не несёт ответственности за ваши торговые решения!**

---

## 📧 КОНТАКТЫ

Вопросы? Проблемы? Предложения?

Открывайте Issue на GitHub!

---

**Удачной торговли! 🚀📈💰**
