"""
Модуль для работы с вероятностными распределениями в нейронной сети.
Реализует предсказание распределений вероятностей вместо одиночных значений.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import math


class ProbabilityDistributionPredictor(nn.Module):
    """
    Модуль для предсказания вероятностных распределений.
    Может возвращать как массивы, так и математические функции.
    """
    
    def __init__(self, 
                 input_dim: int,
                 num_bins: int = 100,
                 min_value: float = -0.1,
                 max_value: float = 0.1,
                 use_function_approximation: bool = True):
        """
        Инициализация предиктора распределений.
        
        Args:
            input_dim: Размерность входных признаков
            num_bins: Количество бинов для дискретного распределения
            min_value: Минимальное значение для распределения
            max_value: Максимальное значение для распределения
            use_function_approximation: Использовать ли аппроксимацию функций
        """
        super().__init__()
        
        self.num_bins = num_bins
        self.min_value = min_value
        self.max_value = max_value
        self.use_function_approximation = use_function_approximation
        
        # Создаем бины для дискретного распределения
        self.bins = torch.linspace(min_value, max_value, num_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        
        # Нейронная сеть для предсказания параметров распределения (увеличенная для всех индикаторов)
        self.distribution_network = nn.Sequential(
            nn.Linear(input_dim, min(input_dim * 4, 512)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(min(input_dim * 4, 512), min(input_dim * 2, 256)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(min(input_dim * 2, 256), min(input_dim, 128)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(min(input_dim, 128), num_bins)  # Логиты для каждого бина
        )
        
        # Параметры для аппроксимации функций (если используется)
        if use_function_approximation:
            self.function_params_network = nn.Sequential(
                nn.Linear(input_dim, min(input_dim * 2, 256)),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(min(input_dim * 2, 256), min(input_dim, 128)),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(min(input_dim, 128), 6)  # Параметры для смешанного распределения
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Прямой проход через сеть.
        
        Args:
            x: Входные признаки [batch_size, input_dim]
            
        Returns:
            Словарь с предсказаниями распределений
        """
        batch_size = x.size(0)
        
        # Предсказание дискретного распределения
        logits = self.distribution_network(x)  # [batch_size, num_bins]
        probabilities = F.softmax(logits, dim=-1)  # [batch_size, num_bins]
        
        result = {
            'discrete_probabilities': probabilities,
            'logits': logits,
            'bin_centers': self.bin_centers.unsqueeze(0).expand(batch_size, -1)
        }
        
        # Аппроксимация функции (если используется)
        if self.use_function_approximation:
            function_params = self.function_params_network(x)  # [batch_size, 6]
            
            # Параметры для смешанного нормального распределения
            # [mu1, sigma1, weight1, mu2, sigma2, weight2]
            mu1 = function_params[:, 0]
            sigma1 = torch.abs(function_params[:, 1]) + 1e-6
            weight1 = torch.sigmoid(function_params[:, 2])
            
            mu2 = function_params[:, 3]
            sigma2 = torch.abs(function_params[:, 4]) + 1e-6
            weight2 = torch.sigmoid(function_params[:, 5])
            
            # Нормализация весов
            total_weight = weight1 + weight2 + 1e-8
            weight1 = weight1 / total_weight
            weight2 = weight2 / total_weight
            
            result.update({
                'function_params': function_params,
                'mu1': mu1,
                'sigma1': sigma1,
                'weight1': weight1,
                'mu2': mu2,
                'sigma2': sigma2,
                'weight2': weight2
            })
        
        return result
    
    def get_probability_function(self, 
                                function_params: torch.Tensor,
                                x_values: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет вероятность для заданных значений x.
        
        Args:
            function_params: Параметры функции [batch_size, 6]
            x_values: Значения x для вычисления вероятности [batch_size, num_points]
            
        Returns:
            Вероятности для каждого x [batch_size, num_points]
        """
        if not self.use_function_approximation:
            raise ValueError("Функциональная аппроксимация не включена")
        
        # Извлекаем параметры
        mu1 = function_params[:, 0:1]  # [batch_size, 1]
        sigma1 = torch.abs(function_params[:, 1:2]) + 1e-6  # [batch_size, 1]
        weight1 = torch.sigmoid(function_params[:, 2:3])  # [batch_size, 1]
        
        mu2 = function_params[:, 3:4]  # [batch_size, 1]
        sigma2 = torch.abs(function_params[:, 4:5]) + 1e-6  # [batch_size, 1]
        weight2 = torch.sigmoid(function_params[:, 5:6])  # [batch_size, 1]
        
        # Нормализация весов
        total_weight = weight1 + weight2 + 1e-8
        weight1 = weight1 / total_weight
        weight2 = weight2 / total_weight
        
        # Вычисляем плотность вероятности для каждого компонента
        # Нормальное распределение: 1/(σ√(2π)) * exp(-(x-μ)²/(2σ²))
        prob1 = (1 / (sigma1 * math.sqrt(2 * math.pi))) * \
                torch.exp(-0.5 * ((x_values - mu1) / sigma1) ** 2)
        
        prob2 = (1 / (sigma2 * math.sqrt(2 * math.pi))) * \
                torch.exp(-0.5 * ((x_values - mu2) / sigma2) ** 2)
        
        # Смешанное распределение
        total_prob = weight1 * prob1 + weight2 * prob2
        
        return total_prob
    
    def can_be_sinusoidal(self, function_params: torch.Tensor) -> torch.Tensor:
        """
        Проверяет, может ли функция быть синусоидальной с несколькими экстремумами.
        
        Args:
            function_params: Параметры функции [batch_size, 6]
            
        Returns:
            Булевы значения для каждого примера в батче
        """
        if not self.use_function_approximation:
            return torch.zeros(function_params.size(0), dtype=torch.bool, device=function_params.device)
        
        # Извлекаем параметры
        mu1 = function_params[:, 0]
        sigma1 = torch.abs(function_params[:, 1]) + 1e-6
        weight1 = torch.sigmoid(function_params[:, 2])
        
        mu2 = function_params[:, 3]
        sigma2 = torch.abs(function_params[:, 4]) + 1e-6
        weight2 = torch.sigmoid(function_params[:, 5])
        
        # Нормализация весов
        total_weight = weight1 + weight2 + 1e-8
        weight1 = weight1 / total_weight
        weight2 = weight2 / total_weight
        
        # Критерии для синусоидальности:
        # 1. Два компонента с разными центрами
        # 2. Оба компонента имеют значительный вес
        # 3. Сигмы не слишком большие (чтобы избежать слишком широких распределений)
        
        centers_different = torch.abs(mu1 - mu2) > 0.01  # Центры достаточно далеко
        both_significant = (weight1 > 0.2) & (weight2 > 0.2)  # Оба компонента значимы
        sigmas_reasonable = (sigma1 < 0.05) & (sigma2 < 0.05)  # Сигмы не слишком большие
        
        # Дополнительный критерий: разность между центрами должна быть больше суммы сигм
        centers_separated = torch.abs(mu1 - mu2) > (sigma1 + sigma2)
        
        is_sinusoidal = centers_different & both_significant & sigmas_reasonable & centers_separated
        
        return is_sinusoidal
    
    def get_extrema_points(self, function_params: torch.Tensor, 
                          num_points: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Находит точки экстремумов функции распределения.
        
        Args:
            function_params: Параметры функции [batch_size, 6]
            num_points: Количество точек для поиска экстремумов
            
        Returns:
            Кортеж (x_coords, probabilities) точек экстремумов
        """
        if not self.use_function_approximation:
            return torch.empty(0), torch.empty(0)
        
        # Создаем сетку точек для поиска экстремумов
        x_grid = torch.linspace(self.min_value, self.max_value, num_points, 
                               device=function_params.device)
        x_grid = x_grid.unsqueeze(0).expand(function_params.size(0), -1)
        
        # Вычисляем плотность вероятности
        probs = self.get_probability_function(function_params, x_grid)
        
        # Находим экстремумы (производная близка к нулю)
        # Используем численное дифференцирование
        prob_diff = torch.diff(probs, dim=1)
        sign_changes = torch.diff(torch.sign(prob_diff), dim=1)
        
        # Находим точки, где знак производной меняется
        extrema_mask = torch.abs(sign_changes) > 0
        
        extrema_x = []
        extrema_probs = []
        
        for i in range(function_params.size(0)):
            if extrema_mask[i].any():
                extrema_indices = torch.where(extrema_mask[i])[0]
                extrema_x.append(x_grid[i, extrema_indices])
                extrema_probs.append(probs[i, extrema_indices])
            else:
                # Если экстремумов нет, возвращаем максимум
                max_idx = torch.argmax(probs[i])
                extrema_x.append(x_grid[i, max_idx:max_idx+1])
                extrema_probs.append(probs[i, max_idx:max_idx+1])
        
        return torch.cat(extrema_x, dim=0), torch.cat(extrema_probs, dim=0)


class DistributionLoss(nn.Module):
    """
    Функция потерь для обучения распределений.
    """
    
    def __init__(self, 
                 discrete_weight: float = 1.0,
                 function_weight: float = 0.5,
                 smoothness_weight: float = 0.1):
        """
        Инициализация функции потерь.
        
        Args:
            discrete_weight: Вес для дискретного распределения
            function_weight: Вес для функциональной аппроксимации
            smoothness_weight: Вес для регуляризации гладкости
        """
        super().__init__()
        
        self.discrete_weight = discrete_weight
        self.function_weight = function_weight
        self.smoothness_weight = smoothness_weight
        
        self.kl_divergence = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Вычисляет потери.
        
        Args:
            predictions: Предсказания модели
            targets: Целевые значения
            
        Returns:
            Словарь с компонентами потерь
        """
        losses = {}
        
        # Потеря для дискретного распределения
        if 'discrete_probabilities' in predictions:
            # Создаем целевое распределение (нормальное распределение вокруг целевого значения)
            target_probs = self._create_target_distribution(targets, predictions['bin_centers'])
            
            # KL divergence между предсказанным и целевым распределением
            discrete_loss = self.kl_divergence(
                torch.log(predictions['discrete_probabilities'] + 1e-8),
                target_probs
            )
            losses['discrete_loss'] = discrete_loss
        
        # Потеря для функциональной аппроксимации
        if 'function_params' in predictions:
            # Вычисляем целевую функцию и сравниваем с предсказанной
            x_values = predictions['bin_centers']
            pred_probs = predictions['discrete_probabilities']
            
            # Аппроксимируем целевую функцию
            target_function = self._create_target_function(targets, x_values)
            
            function_loss = self.mse_loss(pred_probs, target_function)
            losses['function_loss'] = function_loss
        
        # Регуляризация гладкости
        if 'function_params' in predictions:
            smoothness_loss = self._compute_smoothness_loss(predictions['function_params'])
            losses['smoothness_loss'] = smoothness_loss
        
        # Общая потеря
        total_loss = 0
        if 'discrete_loss' in losses:
            total_loss += self.discrete_weight * losses['discrete_loss']
        if 'function_loss' in losses:
            total_loss += self.function_weight * losses['function_loss']
        if 'smoothness_loss' in losses:
            total_loss += self.smoothness_weight * losses['smoothness_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _create_target_distribution(self, 
                                  targets: torch.Tensor,
                                  bin_centers: torch.Tensor) -> torch.Tensor:
        """Создает целевое распределение."""
        batch_size = targets.size(0)
        num_bins = bin_centers.size(1)
        
        # Создаем нормальное распределение вокруг целевого значения
        target_probs = torch.zeros_like(bin_centers)
        
        for i in range(batch_size):
            target = targets[i]
            sigma = 0.01  # Стандартное отклонение
            
            # Вычисляем вероятность для каждого бина
            probs = torch.exp(-0.5 * ((bin_centers[i] - target) / sigma) ** 2)
            probs = probs / (probs.sum() + 1e-8)  # Нормализация
            
            target_probs[i] = probs
        
        return target_probs
    
    def _create_target_function(self, 
                               targets: torch.Tensor,
                               x_values: torch.Tensor) -> torch.Tensor:
        """Создает целевую функцию."""
        batch_size = targets.size(0)
        num_points = x_values.size(1)
        
        target_function = torch.zeros_like(x_values)
        
        for i in range(batch_size):
            target = targets[i]
            sigma = 0.01
            
            # Создаем нормальное распределение
            probs = torch.exp(-0.5 * ((x_values[i] - target) / sigma) ** 2)
            probs = probs / (probs.sum() + 1e-8)
            
            target_function[i] = probs
        
        return target_function
    
    def _compute_smoothness_loss(self, function_params: torch.Tensor) -> torch.Tensor:
        """Вычисляет потерю гладкости."""
        # Регуляризация для предотвращения слишком резких изменений
        param_diff = torch.diff(function_params, dim=0)
        smoothness_loss = torch.mean(param_diff ** 2)
        
        return smoothness_loss


def create_probability_predictor(input_dim: int, **kwargs) -> ProbabilityDistributionPredictor:
    """
    Фабричная функция для создания предиктора распределений.
    
    Args:
        input_dim: Размерность входных признаков
        **kwargs: Дополнительные параметры
        
    Returns:
        Инициализированный предиктор
    """
    return ProbabilityDistributionPredictor(input_dim=input_dim, **kwargs)


if __name__ == "__main__":
    # Тестирование модуля
    input_dim = 50
    batch_size = 32
    
    # Создаем предиктор
    predictor = create_probability_predictor(input_dim)
    
    # Тестовые данные
    x = torch.randn(batch_size, input_dim)
    
    # Прямой проход
    outputs = predictor(x)
    
    print("Выходы предиктора:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
    
    # Проверяем синусоидальность
    if 'function_params' in outputs:
        is_sinusoidal = predictor.can_be_sinusoidal(outputs['function_params'])
        print(f"Синусоидальные функции: {is_sinusoidal.sum().item()}/{batch_size}")
        
        # Находим экстремумы
        extrema_x, extrema_probs = predictor.get_extrema_points(outputs['function_params'])
        print(f"Найдено экстремумов: {extrema_x.size(0)}")
