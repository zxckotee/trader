"""
Улучшенная MoE модель с поддержкой вероятностных распределений.
Интегрирует стандартную MoE архитектуру с предсказанием распределений.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math

from .moe_model import MoECryptoPredictor, TimeframeExpert, GatingNetwork
from .probability_distribution import ProbabilityDistributionPredictor, DistributionLoss


class EnhancedTimeframeExpert(TimeframeExpert):
    """
    Улучшенный эксперт с поддержкой вероятностных распределений.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 feedforward_dim: int = None,
                 use_probability_distribution: bool = True,
                 num_bins: int = 100):
        """
        Инициализация улучшенного эксперта.
        
        Args:
            input_dim: Размерность входных признаков
            hidden_dim: Размерность скрытого слоя
            num_layers: Количество слоев трансформера
            num_heads: Количество голов внимания
            dropout: Коэффициент dropout
            feedforward_dim: Размерность feedforward слоя
            use_probability_distribution: Использовать ли вероятностные распределения
            num_bins: Количество бинов для дискретного распределения
        """
        super().__init__(input_dim, hidden_dim, num_layers, num_heads, dropout, feedforward_dim)
        
        self.use_probability_distribution = use_probability_distribution
        
        if use_probability_distribution:
            # Добавляем предиктор распределений
            self.distribution_predictor = ProbabilityDistributionPredictor(
                input_dim=128,  # Увеличенный размер входа для обработки всех индикаторов
                num_bins=num_bins,
                min_value=-0.1,
                max_value=0.1,
                use_function_approximation=True
            )
            
            # Дополнительные головы для распределений (увеличенные для обработки всех индикаторов)
            self.distribution_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 128)  # Увеличенный размер для предиктора распределений
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Прямой проход через улучшенный эксперт.
        
        Args:
            x: Входной тензор [batch_size, seq_len, input_dim]
            
        Returns:
            Словарь с предсказаниями
        """
        # Базовый прямой проход
        base_outputs = super().forward(x)
        
        if self.use_probability_distribution:
            # Получаем скрытые представления для распределений
            x_projected = self.input_projection(x)  # [batch, seq_len, hidden_dim]
            x_encoded = self.pos_encoding(x_projected)
            x_transformed = self.transformer(x_encoded)
            x_final = x_transformed[:, -1, :]  # [batch, hidden_dim]
            x_final = self.layer_norm(x_final)
            x_final = self.dropout(x_final)
            
            # Подготавливаем вход для предиктора распределений
            distribution_input = self.distribution_head(x_final)
            
            # Получаем распределения
            distribution_outputs = self.distribution_predictor(distribution_input)
            
            # Объединяем результаты
            base_outputs.update(distribution_outputs)
            
            # Добавляем информацию о синусоидальности
            if 'function_params' in distribution_outputs:
                is_sinusoidal = self.distribution_predictor.can_be_sinusoidal(
                    distribution_outputs['function_params']
                )
                base_outputs['is_sinusoidal'] = is_sinusoidal
                
                # Находим экстремумы
                extrema_x, extrema_probs = self.distribution_predictor.get_extrema_points(
                    distribution_outputs['function_params']
                )
                base_outputs['extrema_x'] = extrema_x
                base_outputs['extrema_probs'] = extrema_probs
        
        return base_outputs


class EnhancedMoECryptoPredictor(MoECryptoPredictor):
    """
    Улучшенная MoE модель с поддержкой вероятностных распределений.
    """
    
    def __init__(self, 
                 input_dim: int,
                 timeframes: List[str] = None,
                 expert_config: Dict = None,
                 use_gating: bool = True,
                 use_probability_distribution: bool = True,
                 num_bins: int = 100):
        """
        Инициализация улучшенной MoE модели.
        
        Args:
            input_dim: Размерность входных признаков
            timeframes: Список временных интервалов
            expert_config: Конфигурация экспертов
            use_gating: Использовать ли gating network
            use_probability_distribution: Использовать ли вероятностные распределения
            num_bins: Количество бинов для дискретного распределения
        """
        # Инициализируем базовый класс с input_dim
        super().__init__(input_dim)
        
        self.timeframes = timeframes or ['5m', '30m', '1h', '1d', '1w']
        self.num_experts = len(self.timeframes)
        self.use_gating = use_gating
        self.expert_config = expert_config or {}
        
        # Создаем gating network если нужно
        if self.use_gating:
            from .moe_model import GatingNetwork
            self.gating_network = GatingNetwork(
                input_dim=input_dim,
                num_experts=self.num_experts
            )
        
        # Создаем финальные агрегационные слои
        import torch.nn as nn
        self.final_price_layer = nn.Linear(self.num_experts, 1)
        self.final_direction_layer = nn.Linear(self.num_experts * 2, 2)
        self.final_volatility_layer = nn.Linear(self.num_experts, 1)
        self.final_magnitude_layer = nn.Linear(self.num_experts, 1)
        self.final_percentile_layer = nn.Linear(self.num_experts, 1)
        
        self.use_probability_distribution = use_probability_distribution
        self.num_bins = num_bins
        
        # Создаем улучшенных экспертов
        if use_probability_distribution:
            self.experts = nn.ModuleDict()
            for tf in self.timeframes:
                # Извлекаем параметры для базового эксперта
                base_config = {k: v for k, v in self.expert_config.items() 
                             if k not in ['use_probability_distribution', 'num_bins']}
                
                self.experts[tf] = EnhancedTimeframeExpert(
                    input_dim=input_dim,
                    use_probability_distribution=True,
                    num_bins=num_bins,
                    **base_config
                )
        else:
            # Создаем обычных экспертов
            from .moe_model import TimeframeExpert
            self.experts = nn.ModuleDict()
            for tf in self.timeframes:
                self.experts[tf] = TimeframeExpert(
                    input_dim=input_dim,
                    **self.expert_config
                )
        
        # Дополнительные агрегационные слои для распределений
        if use_probability_distribution:
            self.final_distribution_layer = nn.Linear(self.num_experts, 1)
            self.final_sinusoidal_layer = nn.Linear(self.num_experts, 1)
    
    def forward(self, 
                inputs: Dict[str, torch.Tensor],
                target_timeframe: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Прямой проход через улучшенную MoE модель.
        
        Args:
            inputs: Словарь входных данных по временным интервалам
            target_timeframe: Конкретный временной интервал для предсказания
            
        Returns:
            Словарь с агрегированными предсказаниями
        """
        # Базовый прямой проход
        base_outputs = super().forward(inputs, target_timeframe)
        
        if self.use_probability_distribution:
            # Получаем выходы экспертов
            expert_outputs = base_outputs.get('expert_outputs', {})
            
            if expert_outputs:
                # Агрегируем распределения
                distribution_outputs = self._aggregate_distributions(expert_outputs, inputs)
                base_outputs.update(distribution_outputs)
        
        return base_outputs
    
    def _aggregate_distributions(self, 
                                expert_outputs: Dict[str, Dict[str, torch.Tensor]],
                                inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Агрегирует распределения от всех экспертов.
        
        Args:
            expert_outputs: Выходы экспертов
            inputs: Входные данные
            
        Returns:
            Агрегированные распределения
        """
        batch_size = next(iter(expert_outputs.values()))['price_change_logits'].size(0)
        
        # Собираем распределения от всех экспертов
        discrete_probs_list = []
        function_params_list = []
        is_sinusoidal_list = []
        
        for tf in self.timeframes:
            if tf in expert_outputs:
                expert_output = expert_outputs[tf]
                
                if 'discrete_probabilities' in expert_output:
                    discrete_probs_list.append(expert_output['discrete_probabilities'].unsqueeze(1))
                
                if 'function_params' in expert_output:
                    function_params_list.append(expert_output['function_params'].unsqueeze(1))
                
                if 'is_sinusoidal' in expert_output:
                    is_sinusoidal_list.append(expert_output['is_sinusoidal'].float().unsqueeze(1))
            else:
                # Заполняем нулями, если эксперт недоступен
                device = next(iter(expert_outputs.values()))['price_change_logits'].device
                discrete_probs_list.append(torch.zeros(batch_size, 1, self.num_bins, device=device))
                function_params_list.append(torch.zeros(batch_size, 1, 6, device=device))
                is_sinusoidal_list.append(torch.zeros(batch_size, 1, device=device))
        
        # Агрегируем распределения
        aggregated_outputs = {}
        
        if discrete_probs_list:
            discrete_probs_stack = torch.cat(discrete_probs_list, dim=1)  # [batch, num_experts, num_bins]
            
            if self.use_gating and len(expert_outputs) > 1:
                # Используем gating для взвешивания
                gating_input = next(iter(inputs.values()))
                expert_weights = self.gating_network(gating_input)  # [batch, num_experts]
                
                # Применяем веса
                expert_weights = expert_weights.unsqueeze(-1)  # [batch, num_experts, 1]
                aggregated_discrete = (discrete_probs_stack * expert_weights).sum(dim=1)
            else:
                # Простая агрегация
                aggregated_discrete = discrete_probs_stack.mean(dim=1)
            
            aggregated_outputs['aggregated_discrete_probabilities'] = aggregated_discrete
        
        if function_params_list:
            function_params_stack = torch.cat(function_params_list, dim=1)  # [batch, num_experts, 6]
            
            if self.use_gating and len(expert_outputs) > 1:
                expert_weights = self.gating_network(gating_input)
                expert_weights = expert_weights.unsqueeze(-1)
                aggregated_params = (function_params_stack * expert_weights).sum(dim=1)
            else:
                aggregated_params = function_params_stack.mean(dim=1)
            
            aggregated_outputs['aggregated_function_params'] = aggregated_params
        
        if is_sinusoidal_list:
            is_sinusoidal_stack = torch.cat(is_sinusoidal_list, dim=1)  # [batch, num_experts]
            
            if self.use_gating and len(expert_outputs) > 1:
                expert_weights = self.gating_network(gating_input)
                aggregated_sinusoidal = (is_sinusoidal_stack * expert_weights).sum(dim=1)
            else:
                aggregated_sinusoidal = is_sinusoidal_stack.mean(dim=1)
            
            aggregated_outputs['aggregated_is_sinusoidal'] = aggregated_sinusoidal
        
        return aggregated_outputs
    
    def get_probability_function(self, 
                                inputs: Dict[str, torch.Tensor],
                                x_values: torch.Tensor) -> torch.Tensor:
        """
        Получает функцию вероятности для заданных значений x.
        
        Args:
            inputs: Входные данные
            x_values: Значения x для вычисления вероятности
            
        Returns:
            Вероятности для каждого x
        """
        if not self.use_probability_distribution:
            raise ValueError("Вероятностные распределения не включены")
        
        # Получаем предсказания
        outputs = self.forward(inputs)
        
        if 'aggregated_function_params' not in outputs:
            raise ValueError("Параметры функции не найдены")
        
        # Создаем предиктор для вычисления функции
        predictor = ProbabilityDistributionPredictor(
            input_dim=1,  # Не используется для вычисления функции
            use_function_approximation=True
        )
        
        # Вычисляем функцию вероятности
        probabilities = predictor.get_probability_function(
            outputs['aggregated_function_params'],
            x_values
        )
        
        return probabilities
    
    def can_be_sinusoidal(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Проверяет, может ли функция быть синусоидальной.
        
        Args:
            inputs: Входные данные
            
        Returns:
            Булевы значения для каждого примера
        """
        if not self.use_probability_distribution:
            return torch.zeros(inputs[list(inputs.keys())[0]].size(0), dtype=torch.bool)
        
        outputs = self.forward(inputs)
        
        if 'aggregated_is_sinusoidal' in outputs:
            return outputs['aggregated_is_sinusoidal'] > 0.5
        else:
            return torch.zeros(inputs[list(inputs.keys())[0]].size(0), dtype=torch.bool)
    
    def get_extrema_points(self, 
                          inputs: Dict[str, torch.Tensor],
                          num_points: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Находит точки экстремумов функции распределения.
        
        Args:
            inputs: Входные данные
            num_points: Количество точек для поиска экстремумов
            
        Returns:
            Кортеж (x_coords, probabilities) точек экстремумов
        """
        if not self.use_probability_distribution:
            return torch.empty(0), torch.empty(0)
        
        outputs = self.forward(inputs)
        
        if 'aggregated_function_params' not in outputs:
            return torch.empty(0), torch.empty(0)
        
        # Создаем предиктор для поиска экстремумов
        predictor = ProbabilityDistributionPredictor(
            input_dim=1,
            use_function_approximation=True
        )
        
        # Находим экстремумы
        extrema_x, extrema_probs = predictor.get_extrema_points(
            outputs['aggregated_function_params'],
            num_points
        )
        
        return extrema_x, extrema_probs


def create_enhanced_moe_model(input_dim: int, 
                             timeframes: List[str] = None,
                             use_probability_distribution: bool = True,
                             **kwargs) -> EnhancedMoECryptoPredictor:
    """
    Фабричная функция для создания улучшенной MoE модели.
    
    Args:
        input_dim: Размерность входных признаков
        timeframes: Список временных интервалов
        use_probability_distribution: Использовать ли вероятностные распределения
        **kwargs: Дополнительные параметры
        
    Returns:
        Инициализированная улучшенная MoE модель
    """
    return EnhancedMoECryptoPredictor(
        input_dim=input_dim,
        timeframes=timeframes,
        use_probability_distribution=use_probability_distribution,
        **kwargs
    )


if __name__ == "__main__":
    # Тестирование улучшенной модели
    input_dim = 50
    seq_len = 100
    batch_size = 32
    timeframes = ['5m', '30m', '1h', '1d', '1w']
    
    # Создаем модель
    model = create_enhanced_moe_model(
        input_dim=input_dim,
        timeframes=timeframes,
        use_probability_distribution=True
    )
    
    # Создаем тестовые входные данные
    inputs = {}
    for tf in timeframes:
        inputs[tf] = torch.randn(batch_size, seq_len, input_dim)
    
    # Прямой проход
    outputs = model(inputs)
    
    print("Выходы улучшенной модели:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"{key}: {len(value)} экспертов")
    
    # Проверяем синусоидальность
    is_sinusoidal = model.can_be_sinusoidal(inputs)
    print(f"Синусоидальные функции: {is_sinusoidal.sum().item()}/{batch_size}")
    
    # Находим экстремумы
    extrema_x, extrema_probs = model.get_extrema_points(inputs)
    print(f"Найдено экстремумов: {extrema_x.size(0)}")
    
    print(f"Параметры модели: {sum(p.numel() for p in model.parameters()):,}")
