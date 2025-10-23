# 🚀 ML проект предсказания криптовалютного рынка с архитектурой MoE

## 🎯 Описание проекта

Профессиональная система предсказания криптовалютного рынка с использованием архитектуры **Enhanced Mixture of Experts (MoE)** и **вероятностных распределений**. Система анализирует **ВСЕ аспекты криптовалюты**: цены, объемы, активность китов, давление покупателей/продавцов на разных временных интервалах.

### 🔥 Ключевые особенности:

- **🎲 Вероятностные распределения** - предсказывает РАСПРЕДЕЛЕНИЕ вероятностей, а не одно значение!
- **🌊 Синусоидальное обнаружение** - находит периодические паттерны и экстремумы
- **📊 52M параметров** - оптимизировано для RTX 2060 Super (масштабируется до 1B+)
- **⏰ Multi-Timeframe Analysis** - 5 экспертов для 5 таймфреймов (5m, 30m, 1h, 1d, 1w)
- **🐋 Полный Volume Analysis** - обнаружение китов и манипуляций (70+ индикаторов)
- **🎯 Multi-Task Learning** - 5 выходов: price, direction, volatility, magnitude, percentile
- **💾 Процентные изменения** - обобщенность между всеми монетами (BTC, ETH, SOL, ...)
- **⚡ RTX 2060 Super friendly** - работает на 8GB VRAM с FP16!

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

### 4. 🎯 ПРЕДСКАЗАНИЯ (ВЕРОЯТНОСТНЫЕ РАСПРЕДЕЛЕНИЯ!)
```
✅ Price Change Distribution  - РАСПРЕДЕЛЕНИЕ вероятностей изменения цены
✅ Direction Logits           - логиты направления (вверх/вниз)
✅ Volatility Logits          - логиты волатильности (риск)
✅ Magnitude Logits           - логиты величины движения
✅ Percentile Logits          - логиты перцентиля (экстремальность)
✅ Sinusoidal Detection       - обнаружение синусоидальных паттернов
✅ Extrema Points             - точки экстремумов в распределении
```

**🔥 НОВАЯ КОНЦЕПЦИЯ**: Модель предсказывает НЕ одно значение, а **РАСПРЕДЕЛЕНИЕ ВЕРОЯТНОСТЕЙ**!

Вместо: "Цена вырастет на +2.5%"
Модель говорит: "С вероятностью 40% рост +2-3%, с вероятностью 30% рост +5-7%, с вероятностью 30% падение -2%"

**Преимущества:**
- 📊 Видите ВСЕ возможные сценарии
- 🎯 Оцениваете риски точнее
- 🔮 Обнаруживаете неопределенность рынка
- 🌊 Находите синусоидальные паттерны и экстремумы

---

## 🏗️ Архитектура Модели

```
📊 Enhanced MoE (Mixture of Experts) с вероятностными распределениями
🎯 Текущая конфигурация: ~52M параметров (оптимизировано для RTX 2060 Super)

├─ 5 Transformer Experts (по ~10M каждый):
│  ├─ Expert_5m   : 384 hidden × 8 layers × 8 heads
│  ├─ Expert_30m  : 384 hidden × 8 layers × 8 heads
│  ├─ Expert_1h   : 384 hidden × 8 layers × 8 heads
│  ├─ Expert_1d   : 384 hidden × 8 layers × 8 heads
│  └─ Expert_1w   : 384 hidden × 8 layers × 8 heads
│
├─ Gating Network : Умный выбор экспертов
│
├─ Output Heads (LOGITS):
│  ├─ Price Change Logits    : Raw logits (без активации)
│  ├─ Direction Logits       : Raw logits → CrossEntropy
│  ├─ Volatility Logits      : Raw logits → MSE
│  ├─ Magnitude Logits       : Raw logits → MSE
│  └─ Percentile Logits      : Raw logits → MSE
│
└─ Probability Distribution Module:
   ├─ Discrete Distribution  : [batch, 75 bins] - массив вероятностей
   ├─ Function Approximation : [batch, 6 params] - математическая функция
   │                           P(x) = w1*N(μ1,σ1²) + w2*N(μ2,σ2²)
   ├─ Sinusoidal Detection  : [batch, 1] - синусоидальность
   └─ Extrema Points        : [batch, max_extrema] - точки экстремумов

💾 Memory: ~1.5 GB (training FP16), ~0.2 GB (inference)
⚡ Speed: ~10-15 epochs/hour (RTX 2060 Super, batch_size=16)
🎲 Bins: 75 (диапазон: -10% до +10%)
```

**📈 Масштабируемость:**
- **52M** (текущая) - быстрое обучение, хорошее качество
- **100M** - увеличить hidden_dim до 512, num_layers до 10
- **430M** - увеличить hidden_dim до 768, num_layers до 12
- **1B+** - увеличить hidden_dim до 1024, num_layers до 18

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

### Вывод (ВЕРОЯТНОСТНОЕ РАСПРЕДЕЛЕНИЕ):
```
=== Predictions for BTCUSDT ===
Current Price: $67,234.50

🎲 Probability Distribution (5m timeframe):
  Most Likely Scenarios:
    1. +2.5% to +3.5%  : 35% probability ⭐⭐⭐
    2. +5.0% to +7.0%  : 25% probability ⭐⭐
    3. -1.0% to -2.0%  : 20% probability ⭐⭐
    4. 0% to +1%       : 15% probability ⭐
    5. -3.0% to -5.0%  : 5% probability

📊 Statistical Summary:
  Expected Value: +2.1% (weighted average)
  Direction: UP (logits: 2.34 → 91.2% confidence)
  Volatility: 0.0042 (risk level: MEDIUM)
  Magnitude: 3.2% (strong movement expected)
  
🌊 Pattern Analysis:
  Sinusoidal: YES (confidence: 78%)
  Extrema Points:
    - Local Max at +6.5% (P=0.28)
    - Local Min at -2.0% (P=0.12)
    - Local Max at +2.8% (P=0.35) ⭐ MAIN PEAK

💡 Recommendation: STRONG BUY
  Entry: $67,234 (current)
  Stop Loss: $65,890 (-2.0%, covers 95% scenarios)
  Take Profit 1: $69,120 (+2.8%, main peak)
  Take Profit 2: $71,650 (+6.5%, secondary peak)
  Risk/Reward: 1:3.5 (excellent!)
```

**🔥 Преимущества вероятностного подхода:**
- Видите ВСЕ возможные сценарии, а не только один
- Можете установить несколько Take Profit на разных пиках
- Точнее оцениваете риски (Stop Loss покрывает 95% сценариев)
- Обнаруживаете синусоидальные паттерны и экстремумы

📖 **Подробнее о вероятностных распределениях:** см. [MODEL_PREDICTIONS.md](MODEL_PREDICTIONS.md)

---

## ⚙️ КОНФИГУРАЦИЯ

### config.json - Основные параметры

```json
{
  "model": {
    "hidden_dim": 384,              // Размерность модели (384 для 52M, 768 для 430M)
    "num_layers": 8,                // Количество слоев (8 для 52M, 12 для 430M)
    "num_heads": 8,                 // Attention heads
    "feedforward_dim": 1536,        // FFN размерность (1536 для 52M, 3072 для 430M)
    "num_bins": 75,                 // Количество бинов для распределения
    "use_probability_distribution": true,  // Включить вероятностные распределения
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

### Проблема: Модель не учится (direction accuracy ~50%)
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
