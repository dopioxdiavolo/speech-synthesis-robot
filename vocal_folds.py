"""
Модель голосовых связок (Vocal Folds Model) для source-filter архитектуры.

Реализует физиологически обоснованную модель фонации на уровне источника (source-level),
позволяя управлять параметрами голосовых связок:

ФИЗИОЛОГИЧЕСКИЕ ПАРАМЕТРЫ:
- F0 (fundamental frequency): основная частота вибрации голосовых связок
  Физиологический смысл: зависит от длины, массы и натяжения голосовых связок,
  а также от подглоточного давления воздуха.

- Jitter: вариабельность F0 между циклами вибрации
  Физиологический смысл: отражает естественные микро-вариации в цикле вибрации
  из-за нелинейной динамики голосовых связок. Нормальные значения: 0.5-1.5%
  Повышенный jitter может указывать на патологию или эмоциональное состояние.

- Shimmer: вариабельность амплитуды между циклами
  Физиологический смысл: отражает изменения амплитуды вибрации из-за вариаций
  в потоке воздуха и натяжении связок. Нормальные значения: 3-5%
  Повышенный shimmer связан с напряжением или патологией.

- Phonation Type (тип фонации):
  • modal: нормальная фонация, полное смыкание связок, богатый спектр
  • breathy: придыхательная фонация, неполное смыкание, больше шума
  • pressed: сжатая фонация, избыточное смыкание, напряжённый звук

МОДЕЛЬ:
Основана на двухмассовой модели голосовых связок (two-mass model),
упрощённой для реального времени. Генерирует периодический источник
с управляемыми параметрами для интеграции в source-filter модель.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Tuple, Literal
from enum import Enum


class PhonationType(Enum):
    """Типы фонации голосовых связок."""
    MODAL = "modal"      # Нормальная фонация
    BREATHY = "breathy"  # Придыхательная фонация
    PRESSED = "pressed"  # Сжатая фонация


class VocalFoldsModel:
    """
    Физиологически обоснованная модель голосовых связок.
    
    Генерирует источник звука (source) для source-filter модели,
    контролируя параметры фонации на физиологическом уровне.
    """
    
    # Параметры по умолчанию для разных типов фонации
    PHONATION_PARAMS = {
        PhonationType.MODAL: {
            'jitter': 0.01,      # 1% вариация F0
            'shimmer': 0.03,      # 3% вариация амплитуды
            'noise_ratio': 0.05,  # 5% шумовой компоненты
            'spectral_tilt': -12, # dB/октава (богатый спектр)
            'open_quotient': 0.5, # Отношение открытой фазы к периоду
        },
        PhonationType.BREATHY: {
            'jitter': 0.015,      # 1.5% вариация F0
            'shimmer': 0.05,      # 5% вариация амплитуды
            'noise_ratio': 0.25,  # 25% шумовой компоненты (больше шума)
            'spectral_tilt': -6,  # dB/октава (меньше высоких частот)
            'open_quotient': 0.7, # Больше открытой фазы
        },
        PhonationType.PRESSED: {
            'jitter': 0.005,      # 0.5% вариация F0 (более стабильно)
            'shimmer': 0.02,      # 2% вариация амплитуды
            'noise_ratio': 0.02,  # 2% шумовой компоненты (меньше шума)
            'spectral_tilt': -18, # dB/октава (богаче высокие частоты)
            'open_quotient': 0.4, # Меньше открытой фазы (более сжато)
        }
    }
    
    def __init__(self, sample_rate: int = 16000):
        """
        Инициализация модели голосовых связок.
        
        Args:
            sample_rate: Частота дискретизации в Гц
        """
        self.sample_rate = sample_rate
    
    def generate_source(self,
                       duration: float,
                       f0: float,
                       jitter: Optional[float] = None,
                       shimmer: Optional[float] = None,
                       phonation_type: PhonationType = PhonationType.MODAL,
                       vocal_effort: float = 1.0) -> np.ndarray:
        """
        Генерация источника звука с управляемыми параметрами голосовых связок.
        
        Физиологический смысл:
        Моделирует вибрацию голосовых связок с учётом:
        - Натяжения связок (F0)
        - Естественных вариаций цикла (jitter, shimmer)
        - Типа фонации (степень смыкания связок)
        - Усилия фонации (vocal effort)
        
        Args:
            duration: Длительность в секундах
            f0: Основная частота вибрации (fundamental frequency) в Гц
            jitter: Вариабельность F0 (относительная, например 0.01 = 1%)
                   Если None, используется значение для phonation_type
            shimmer: Вариабельность амплитуды (относительная, например 0.03 = 3%)
                    Если None, используется значение для phonation_type
            phonation_type: Тип фонации (modal, breathy, pressed)
            vocal_effort: Усилие фонации (0.0-2.0), влияет на амплитуду и спектр
            
        Returns:
            Периодический сигнал источника с управляемыми параметрами
        """
        num_samples = int(self.sample_rate * duration)
        if num_samples == 0:
            return np.array([])
        
        t = np.linspace(0, duration, num_samples)
        
        if f0 <= 0:
            return np.zeros(num_samples)
        
        # Получение параметров для типа фонации
        params = self.PHONATION_PARAMS[phonation_type]
        jitter_val = jitter if jitter is not None else params['jitter']
        shimmer_val = shimmer if shimmer is not None else params['shimmer']
        noise_ratio = params['noise_ratio']
        spectral_tilt = params['spectral_tilt']
        open_quotient = params['open_quotient']
        
        # Генерация jitter (вариации F0)
        # Физиологический смысл: естественные микро-вариации из-за нелинейной динамики
        jitter_amount = f0 * jitter_val
        # Генерируем jitter с медленными и быстрыми компонентами
        jitter_slow = np.random.normal(0, jitter_amount * 0.6, num_samples)
        jitter_fast = np.random.normal(0, jitter_amount * 0.4, num_samples)
        # Сглаживаем медленный компонент
        jitter_slow = gaussian_filter1d(jitter_slow, sigma=num_samples / 500)
        # Быстрый компонент остаётся более резким
        jitter_fast = gaussian_filter1d(jitter_fast, sigma=num_samples / 2000)
        jitter_total = jitter_slow + jitter_fast
        
        # Мгновенная частота с jitter
        f0_instantaneous = f0 + jitter_total
        
        # Генерация формы волны с учётом open quotient
        # Физиологический смысл: open quotient определяет отношение времени
        # открытых связок к периоду вибрации
        period_samples = int(self.sample_rate / f0)
        if period_samples < 2:
            period_samples = 2
        
        # Создаём один период с реалистичной формой
        period_phase = np.linspace(0, 1, period_samples)
        period_wave = np.zeros(period_samples)
        
        # Фаза открытия (быстрый подъём)
        open_samples = int(period_samples * open_quotient)
        if open_samples > 0:
            open_phase = np.linspace(0, 1, open_samples)
            # Быстрый подъём с экспоненциальной формой
            period_wave[:open_samples] = np.power(open_phase, 0.5)
        
        # Фаза закрытия (медленный спад)
        close_samples = period_samples - open_samples
        if close_samples > 0:
            close_phase = np.linspace(0, 1, close_samples)
            # Медленный спад
            period_wave[open_samples:] = np.power(1 - close_phase, 1.5)
        
        # Нормализация периода
        if np.max(np.abs(period_wave)) > 0:
            period_wave = period_wave / np.max(np.abs(period_wave))
        
        # Генерация полного сигнала с вариациями периода (jitter)
        source = np.zeros(num_samples)
        current_phase = 0.0
        
        for i in range(num_samples):
            # Текущий период с учётом jitter
            if i > 0:
                period_samples_current = int(self.sample_rate / max(f0_instantaneous[i], 1.0))
                period_samples_current = max(2, period_samples_current)
            else:
                period_samples_current = period_samples
            
            # Индекс в периоде
            period_idx = int(current_phase * period_samples) % period_samples
            
            # Применяем shimmer (вариации амплитуды)
            shimmer_factor = 1.0 + np.random.normal(0, shimmer_val)
            shimmer_factor = np.clip(shimmer_factor, 1.0 - shimmer_val * 3, 1.0 + shimmer_val * 3)
            
            # Значение из периода
            if period_idx < len(period_wave):
                source[i] = period_wave[period_idx] * shimmer_factor
            else:
                source[i] = 0.0
            
            # Обновление фазы с учётом мгновенной частоты
            phase_increment = f0_instantaneous[i] / self.sample_rate
            current_phase = (current_phase + phase_increment) % 1.0
        
        # Добавление гармоник с учётом spectral tilt
        # Физиологический смысл: spectral tilt отражает затухание высоких частот
        # из-за потерь энергии в голосовых связках
        source_with_harmonics = source.copy()
        num_harmonics = min(15, int(self.sample_rate / 2 / f0))
        
        for harmonic in range(2, num_harmonics + 1):
            # Амплитуда гармоники с учётом spectral tilt
            # Spectral tilt в dB/октава: более отрицательный = больше затухание
            db_per_octave = spectral_tilt
            octaves = np.log2(harmonic)
            amplitude_db = db_per_octave * octaves
            amplitude_linear = 10 ** (amplitude_db / 20)
            
            # Дополнительное затухание для естественности
            amplitude = amplitude_linear / (harmonic ** 1.2)
            
            # Фазовый сдвиг для естественности
            phase_shift = np.random.uniform(0, 2 * np.pi)
            
            # Генерация гармоники с учётом jitter
            harmonic_signal = amplitude * np.sin(
                2 * np.pi * f0 * harmonic * t + phase_shift
            )
            # Применяем jitter к гармонике (пропорционально номеру гармоники)
            for i in range(num_samples):
                if i > 0:
                    harmonic_signal[i] *= (1.0 + jitter_total[i] / f0)
            
            source_with_harmonics += harmonic_signal
        
        # Смешивание основного сигнала с гармониками
        source = 0.7 * source + 0.3 * source_with_harmonics
        
        # Добавление шумовой компоненты (для breathy phonation)
        # Физиологический смысл: турбулентный поток воздуха через частично открытые связки
        if noise_ratio > 0:
            noise = np.random.normal(0, 1, num_samples)
            # Фильтруем шум (высокие частоты для турбулентности)
            b, a = signal.butter(4, 1000 / (self.sample_rate / 2), 'high')
            noise = signal.filtfilt(b, a, noise)
            # Нормализуем и смешиваем
            if np.max(np.abs(noise)) > 0:
                noise = noise / np.max(np.abs(noise))
            source = (1 - noise_ratio) * source + noise_ratio * noise
        
        # Применение vocal effort
        # Физиологический смысл: большее усилие → большее подглоточное давление
        # → большая амплитуда и более богатый спектр
        source = source * vocal_effort
        
        # Применение spectral tilt через фильтрацию (для более точного контроля)
        if spectral_tilt < -10:
            # Богатый спектр: подчёркиваем высокие частоты
            b, a = signal.butter(2, 2000 / (self.sample_rate / 2), 'high')
            high_boost = signal.filtfilt(b, a, source) * 0.3
            source = source + high_boost
        elif spectral_tilt > -8:
            # Бедный спектр: ослабляем высокие частоты
            b, a = signal.butter(4, 3000 / (self.sample_rate / 2), 'low')
            source = signal.filtfilt(b, a, source)
        
        # Плавная огибающая для начала и конца
        envelope = np.ones(num_samples)
        fade_samples = int(0.01 * self.sample_rate)  # 10 мс
        if fade_samples > 0 and num_samples > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            envelope[:fade_samples] = fade_in
            envelope[-fade_samples:] = fade_out
        
        source = source * envelope
        
        # Нормализация
        if np.max(np.abs(source)) > 0:
            source = source / np.max(np.abs(source)) * 0.95
        
        return source
    
    def get_phonation_params(self, phonation_type: PhonationType) -> dict:
        """
        Получение параметров для типа фонации.
        
        Args:
            phonation_type: Тип фонации
            
        Returns:
            Словарь с параметрами
        """
        return self.PHONATION_PARAMS[phonation_type].copy()
    
    def set_custom_phonation_params(self,
                                   phonation_type: PhonationType,
                                   jitter: Optional[float] = None,
                                   shimmer: Optional[float] = None,
                                   noise_ratio: Optional[float] = None,
                                   spectral_tilt: Optional[float] = None,
                                   open_quotient: Optional[float] = None):
        """
        Установка пользовательских параметров для типа фонации.
        
        Args:
            phonation_type: Тип фонации
            jitter: Вариабельность F0
            shimmer: Вариабельность амплитуды
            noise_ratio: Отношение шумовой компоненты
            spectral_tilt: Наклон спектра (dB/октава)
            open_quotient: Отношение открытой фазы к периоду
        """
        if jitter is not None:
            self.PHONATION_PARAMS[phonation_type]['jitter'] = jitter
        if shimmer is not None:
            self.PHONATION_PARAMS[phonation_type]['shimmer'] = shimmer
        if noise_ratio is not None:
            self.PHONATION_PARAMS[phonation_type]['noise_ratio'] = noise_ratio
        if spectral_tilt is not None:
            self.PHONATION_PARAMS[phonation_type]['spectral_tilt'] = spectral_tilt
        if open_quotient is not None:
            self.PHONATION_PARAMS[phonation_type]['open_quotient'] = open_quotient
