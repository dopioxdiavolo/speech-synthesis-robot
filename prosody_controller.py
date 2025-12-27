"""
Модуль управления просодией речи.

Реализует управление просодическими параметрами:
- F0 (интонация, мелодия)
- Energy (громкость)
- Duration (темп, длительность)
- Паузы

Поддерживает различные режимы речи:
- нейтральный
- дружелюбный
- строгий
- предупреждение

Может использовать rule-based подход или простую ML-модель (PyTorch).
"""

import numpy as np
from typing import Optional, Tuple
import torch
import torch.nn as nn


class RuleBasedProsodyController:
    """
    Rule-based контроллер просодии.
    
    Использует правила для генерации просодических контуров
    на основе стиля речи и текста.
    """
    
    # Базовые параметры для разных стилей речи
    SPEECH_STYLES = {
        'neutral': {
            'f0_mean': 150.0,      # Средняя частота F0 (Гц)
            'f0_range': 30.0,      # Диапазон вариации F0
            'f0_contour': 'flat',  # Тип контура ('flat', 'rising', 'falling', 'question')
            'energy_mean': 0.7,     # Средняя энергия (нормализованная)
            'energy_range': 0.2,   # Диапазон вариации энергии
            'duration_factor': 1.0, # Фактор длительности (1.0 = нормальная скорость)
            'pause_duration': 0.1,  # Длительность пауз (секунды)
        },
        'friendly': {
            'f0_mean': 180.0,
            'f0_range': 50.0,
            'f0_contour': 'rising',
            'energy_mean': 0.8,
            'energy_range': 0.3,
            'duration_factor': 0.9,  # Немного быстрее
            'pause_duration': 0.05,
        },
        'strict': {
            'f0_mean': 120.0,
            'f0_range': 20.0,
            'f0_contour': 'falling',
            'energy_mean': 0.9,
            'energy_range': 0.1,
            'duration_factor': 1.1,  # Медленнее
            'pause_duration': 0.2,
        },
        'warning': {
            'f0_mean': 200.0,
            'f0_range': 80.0,
            'f0_contour': 'rising_falling',
            'energy_mean': 1.0,
            'energy_range': 0.4,
            'duration_factor': 1.2,  # Медленнее для акцента
            'pause_duration': 0.15,
        }
    }
    
    def __init__(self, sample_rate: int = 16000):
        """
        Инициализация контроллера.
        
        Args:
            sample_rate: Частота дискретизации
        """
        self.sample_rate = sample_rate
    
    def generate_f0_contour(self,
                           duration: float,
                           style: str = 'neutral',
                           num_syllables: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация F0 контура для заданного стиля.
        
        Физиологический смысл:
        F0 контур отражает изменения напряжения голосовых связок
        и подглоточного давления во время речи.
        
        Args:
            duration: Длительность в секундах
            style: Стиль речи ('neutral', 'friendly', 'strict', 'warning')
            num_syllables: Количество слогов (для модуляции)
            
        Returns:
            (times, f0_values): Временные метки и значения F0
        """
        if style not in self.SPEECH_STYLES:
            style = 'neutral'
        
        params = self.SPEECH_STYLES[style]
        
        # Временная сетка
        time_step = 0.01  # 10 мс
        times = np.arange(0, duration, time_step)
        if len(times) == 0:
            times = np.array([0.0])
        
        f0_mean = params['f0_mean']
        f0_range = params['f0_range']
        contour_type = params['f0_contour']
        
        # Базовый контур
        if contour_type == 'flat':
            f0_base = np.ones_like(times) * f0_mean
        elif contour_type == 'rising':
            # Восходящая интонация
            f0_base = f0_mean + f0_range * (times / duration)
        elif contour_type == 'falling':
            # Нисходящая интонация
            f0_base = f0_mean + f0_range * (1 - times / duration)
        elif contour_type == 'rising_falling':
            # Восходяще-нисходящая (для предупреждений)
            mid_point = duration / 2
            f0_base = np.where(
                times < mid_point,
                f0_mean + f0_range * (times / mid_point),
                f0_mean + f0_range * (1 - (times - mid_point) / mid_point)
            )
        else:
            f0_base = np.ones_like(times) * f0_mean
        
        # Добавление мелкой модуляции (для естественности)
        # Имитирует микро-вариации в вибрации голосовых связок
        modulation = np.sin(2 * np.pi * 3.5 * times) * (f0_range * 0.1)
        
        # Модуляция по слогам
        if num_syllables > 1:
            syllable_modulation = np.sin(2 * np.pi * num_syllables * times / duration) * (f0_range * 0.15)
            modulation += syllable_modulation
        
        f0_values = f0_base + modulation
        
        # Ограничение диапазона
        f0_values = np.clip(f0_values, f0_mean - f0_range, f0_mean + f0_range)
        
        return times, f0_values
    
    def generate_energy_contour(self,
                               duration: float,
                               style: str = 'neutral') -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация контура энергии.
        
        Физиологический смысл:
        Energy контур отражает изменения амплитуды вибрации
        голосовых связок и потока воздуха.
        
        Args:
            duration: Длительность в секундах
            style: Стиль речи
            
        Returns:
            (times, energy_values): Временные метки и значения энергии
        """
        if style not in self.SPEECH_STYLES:
            style = 'neutral'
        
        params = self.SPEECH_STYLES[style]
        
        # Временная сетка
        time_step = 0.01
        times = np.arange(0, duration, time_step)
        if len(times) == 0:
            times = np.array([0.0])
        
        energy_mean = params['energy_mean']
        energy_range = params['energy_range']
        
        # Базовый контур с небольшими вариациями
        energy_base = np.ones_like(times) * energy_mean
        variation = np.sin(2 * np.pi * 2.0 * times) * (energy_range * 0.3)
        
        energy_values = energy_base + variation
        
        # Ограничение диапазона
        energy_values = np.clip(energy_values, 
                               energy_mean - energy_range, 
                               energy_mean + energy_range)
        
        return times, energy_values
    
    def apply_prosody_to_signal(self,
                                signal: np.ndarray,
                                f0_contour: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                energy_contour: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
        """
        Применение просодических контуров к сигналу.
        
        Args:
            signal: Исходный сигнал
            f0_contour: (times, f0_values) - контур F0
            energy_contour: (times, energy_values) - контур энергии
            
        Returns:
            Модифицированный сигнал
        """
        output = signal.copy()
        
        # Применение энергетического контура
        if energy_contour is not None:
            energy_times, energy_values = energy_contour
            # Интерполяция энергии на временную сетку сигнала
            signal_times = np.linspace(0, len(output) / self.sample_rate, len(output))
            energy_interp = np.interp(signal_times, energy_times, energy_values)
            # Применение энергии (нормализация к диапазону [0.3, 1.0])
            energy_normalized = 0.3 + 0.7 * (energy_interp - np.min(energy_interp)) / (np.max(energy_interp) - np.min(energy_interp) + 1e-10)
            output = output * energy_normalized
        
        # Примечание: F0 контур применяется на этапе синтеза,
        # а не пост-фактум к готовому сигналу
        
        return output
    
    def get_duration_factor(self, style: str = 'neutral') -> float:
        """
        Получение фактора длительности для стиля.
        
        Args:
            style: Стиль речи
            
        Returns:
            Фактор длительности
        """
        if style not in self.SPEECH_STYLES:
            style = 'neutral'
        return self.SPEECH_STYLES[style]['duration_factor']
    
    def get_pause_duration(self, style: str = 'neutral') -> float:
        """
        Получение длительности паузы для стиля.
        
        Args:
            style: Стиль речи
            
        Returns:
            Длительность паузы в секундах
        """
        if style not in self.SPEECH_STYLES:
            style = 'neutral'
        return self.SPEECH_STYLES[style]['pause_duration']


class MLProsodyController(nn.Module):
    """
    Простая ML-модель для управления просодией (опционально).
    
    Использует PyTorch для обучения модели генерации просодических контуров.
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        """
        Инициализация ML-модели.
        
        Args:
            input_dim: Размерность входных признаков (стиль, длительность и т.д.)
            hidden_dim: Размерность скрытого слоя
        """
        super(MLProsodyController, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_f0 = nn.Linear(hidden_dim, 50)  # 50 точек для F0 контура
        self.fc_energy = nn.Linear(hidden_dim, 50)  # 50 точек для energy контура
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход модели.
        
        Args:
            x: Входные признаки [batch_size, input_dim]
            
        Returns:
            (f0_contour, energy_contour): Контуры F0 и энергии
        """
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        
        f0_contour = self.tanh(self.fc_f0(h))  # Нормализованный контур F0
        energy_contour = self.tanh(self.fc_energy(h))  # Нормализованный контур энергии
        
        return f0_contour, energy_contour
    
    def generate_contours(self,
                         style: str,
                         duration: float,
                         device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация контуров для заданного стиля.
        
        Args:
            style: Стиль речи
            duration: Длительность
            device: Устройство для вычислений ('cpu' или 'cuda')
            
        Returns:
            (f0_contour, energy_contour): Контуры F0 и энергии
        """
        # Кодирование стиля
        style_encoding = {
            'neutral': 0.0,
            'friendly': 0.33,
            'strict': 0.66,
            'warning': 1.0
        }
        
        style_val = style_encoding.get(style, 0.0)
        
        # Создание входного вектора
        x = torch.tensor([[style_val, duration, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], 
                        dtype=torch.float32, device=device)
        
        # Генерация контуров
        self.eval()
        with torch.no_grad():
            f0_contour, energy_contour = self.forward(x)
        
        # Преобразование в numpy и денормализация
        f0_contour = f0_contour.cpu().numpy()[0]
        energy_contour = energy_contour.cpu().numpy()[0]
        
        # Денормализация (примерные значения)
        f0_contour = 150.0 + 50.0 * f0_contour  # Диапазон 100-200 Гц
        energy_contour = 0.7 + 0.3 * energy_contour  # Диапазон 0.4-1.0
        
        # Создание временной сетки
        times = np.linspace(0, duration, len(f0_contour))
        
        return (times, f0_contour), (times, energy_contour)

