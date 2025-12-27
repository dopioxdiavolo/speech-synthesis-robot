"""
Биомиметическая модель речевого тракта (Source-Filter модель)

Эта модель основана на физиологическом принципе производства речи:
- Source (источник): вибрация голосовых связок (периодический сигнал) + турбулентный шум
- Filter (фильтр): резонансные свойства речевого тракта (форманты F1, F2, F3)

Физиологический смысл:
- F0 (fundamental frequency): частота вибрации голосовых связок
- F1, F2, F3: резонансные частоты речевого тракта, зависящие от положения языка, губ, челюсти
- Bandwidth: затухание формант, связанное с потерями энергии в речевом тракте
"""

import numpy as np
from scipy import signal
from typing import Tuple


class ArticulatoryModel:
    """
    Упрощённая артикуляторная модель речевого тракта.
    
    Основана на source-filter парадигме:
    1. Генерация источника (периодический + шумовой)
    2. Фильтрация через формантную модель речевого тракта
    """
    
    # Формантные частоты для основных гласных (F1, F2, F3) в Гц
    # Эти значения соответствуют типичным резонансным частотам речевого тракта
    VOWEL_FORMANTS = {
        'a': (730, 1090, 2440),   # [а] - низкий F1, средний F2
        'e': (530, 1840, 2480),   # [э] - средний F1, высокий F2
        'i': (270, 2290, 3010),   # [и] - очень низкий F1, очень высокий F2
        'o': (570, 840, 2410),     # [о] - средний F1, низкий F2
        'u': (300, 870, 2240),     # [у] - низкий F1, низкий F2
        'y': (310, 1920, 2560),    # [ы] - низкий F1, высокий F2
    }
    
    # Полосы пропускания формант (bandwidth) в Гц
    # Отражают затухание резонансов из-за потерь энергии в речевом тракте
    DEFAULT_BANDWIDTHS = (90, 110, 170)  # B1, B2, B3
    
    def __init__(self, sample_rate: int = 16000):
        """
        Инициализация модели.
        
        Args:
            sample_rate: Частота дискретизации в Гц
        """
        self.sample_rate = sample_rate
        
    def generate_voice_source(self, 
                             duration: float,
                             f0: float,
                             voicing_strength: float = 1.0) -> np.ndarray:
        """
        Генерация источника голоса (voice source) - улучшенная версия.
        
        Физиологический смысл:
        - Периодический сигнал моделирует вибрацию голосовых связок
        - Используется более реалистичная модель с естественными вариациями
        - voicing_strength контролирует степень фонации
        
        Args:
            duration: Длительность в секундах
            f0: Основная частота (pitch) в Гц
            voicing_strength: Сила фонации (0.0 - 1.0)
            
        Returns:
            Периодический сигнал источника
        """
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        if f0 <= 0:
            return np.zeros_like(t)
        
        # Добавляем jitter (естественные вариации F0)
        # В реальной речи F0 немного варьируется от цикла к циклу
        jitter_amount = f0 * 0.01  # 1% вариация (типично для естественной речи)
        jitter = np.random.normal(0, jitter_amount, num_samples)
        # Сглаживаем jitter для более естественного звука
        from scipy.ndimage import gaussian_filter1d
        jitter = gaussian_filter1d(jitter, sigma=num_samples / 1000)
        
        # Генерируем более реалистичную форму волны
        # Используем модель с асимметричной формой (как у реальных голосовых связок)
        source = np.zeros(num_samples)
        period_samples = int(self.sample_rate / f0)
        
        # Создаём один период с более реалистичной формой
        period_phase = np.linspace(0, 1, period_samples)
        # Асимметричная форма волны (быстрый подъём, медленный спад)
        period_wave = np.zeros(period_samples)
        for i, phase in enumerate(period_phase):
            if phase < 0.4:  # Быстрый подъём
                period_wave[i] = np.sin(np.pi * phase / 0.4) ** 0.7
            else:  # Медленный спад
                period_wave[i] = (1 - (phase - 0.4) / 0.6) ** 1.5
        
        # Нормализуем период
        if np.max(np.abs(period_wave)) > 0:
            period_wave = period_wave / np.max(np.abs(period_wave))
        
        # Заполняем весь сигнал периодами с вариациями
        for i in range(num_samples):
            if period_samples > 0:
                period_idx = int((i % period_samples))
                # Добавляем небольшие вариации амплитуды (shimmer)
                shimmer = 1.0 + np.random.normal(0, 0.03)  # 3% вариация
                source[i] = period_wave[period_idx] * shimmer
        
        # Добавляем гармоники для более богатого спектра
        source_with_harmonics = source.copy()
        for harmonic in range(2, 8):  # 2-я до 7-й гармоники
            amplitude = 0.5 / (harmonic ** 1.2)  # Затухание гармоник
            phase_shift = np.random.uniform(0, 2 * np.pi)  # Случайный фазовый сдвиг
            source_with_harmonics += amplitude * np.sin(
                2 * np.pi * f0 * harmonic * t + phase_shift
            )
        
        # Смешиваем основной сигнал с гармониками
        source = 0.6 * source + 0.4 * source_with_harmonics
        
        # Применяем огибающую для плавного начала и конца
        envelope = np.ones(num_samples)
        fade_samples = int(0.01 * self.sample_rate)  # 10 мс затухание
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            envelope[:fade_samples] = fade_in
            if len(envelope) > fade_samples:
                envelope[-fade_samples:] = fade_out
        
        source = source * envelope
        
        # Нормализация и применение силы фонации
        if np.max(np.abs(source)) > 0:
            source = source / np.max(np.abs(source)) * voicing_strength
        
        return source
    
    def generate_noise_source(self, duration: float, energy: float = 1.0) -> np.ndarray:
        """
        Генерация шумового источника (для согласных).
        
        Физиологический смысл:
        - Турбулентный поток воздуха создаёт шум
        - Используется для глухих согласных (п, т, к, с, ш и т.д.)
        
        Args:
            duration: Длительность в секундах
            energy: Энергия шума
            
        Returns:
            Шумовой сигнал
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        # Белый шум с фильтрацией для более реалистичного звука
        noise = np.random.normal(0, 1, len(t))
        # Простой фильтр высоких частот для имитации турбулентности
        b, a = signal.butter(4, 500 / (self.sample_rate / 2), 'high')
        noise = signal.filtfilt(b, a, noise)
        return noise * energy
    
    def create_formant_filter(self, 
                             f1: float, f2: float, f3: float,
                             b1: float = None, b2: float = None, b3: float = None) -> Tuple:
        """
        Создание формантного фильтра (резонатора речевого тракта).
        
        Физиологический смысл:
        - Каждая форманта соответствует резонансу в речевом тракте
        - F1 связан с высотой языка (вертикальное положение)
        - F2 связан с передне-задним положением языка
        - F3 связан с формой глотки и ротовой полости
        - Bandwidth отражает затухание из-за потерь энергии
        
        Args:
            f1, f2, f3: Частоты формант в Гц
            b1, b2, b3: Полосы пропускания формант в Гц
            
        Returns:
            Коэффициенты фильтра (b, a) для scipy.signal
        """
        if b1 is None:
            b1, b2, b3 = self.DEFAULT_BANDWIDTHS
        
        # Создаём резонаторы для каждой форманты
        # Используем биквадратные фильтры (biquad) для каждой форманты
        
        # Резонатор для F1
        w1 = 2 * np.pi * f1 / self.sample_rate
        bw1 = 2 * np.pi * b1 / self.sample_rate
        r1 = np.exp(-bw1 / 2)
        a1 = -2 * r1 * np.cos(w1)
        b1_coeff = 1 - r1
        
        # Резонатор для F2
        w2 = 2 * np.pi * f2 / self.sample_rate
        bw2 = 2 * np.pi * b2 / self.sample_rate
        r2 = np.exp(-bw2 / 2)
        a2 = -2 * r2 * np.cos(w2)
        b2_coeff = 1 - r2
        
        # Резонатор для F3
        w3 = 2 * np.pi * f3 / self.sample_rate
        bw3 = 2 * np.pi * b3 / self.sample_rate
        r3 = np.exp(-bw3 / 2)
        a3 = -2 * r3 * np.cos(w3)
        b3_coeff = 1 - r3
        
        # Каскадное соединение резонаторов (последовательная фильтрация)
        # Это соответствует физической модели: сигнал проходит через все резонансы
        b_total = np.convolve([b1_coeff, 0, 0], 
                             np.convolve([b2_coeff, 0, 0], 
                                        [b3_coeff, 0, 0]))
        a_total = np.convolve([1, a1, r1**2], 
                             np.convolve([1, a2, r2**2], 
                                        [1, a3, r3**2]))
        
        return b_total, a_total
    
    def synthesize_vowel(self,
                        vowel: str,
                        duration: float,
                        f0: float,
                        voicing_strength: float = 1.0) -> np.ndarray:
        """
        Синтез гласного звука с динамическими формантами.
        
        Args:
            vowel: Буква гласного ('a', 'e', 'i', 'o', 'u', 'y')
            duration: Длительность в секундах
            f0: Основная частота в Гц
            voicing_strength: Сила фонации
            
        Returns:
            Синтезированный сигнал гласного
        """
        if vowel.lower() not in self.VOWEL_FORMANTS:
            raise ValueError(f"Неизвестный гласный: {vowel}")
        
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        f1_base, f2_base, f3_base = self.VOWEL_FORMANTS[vowel.lower()]
        
        # Динамические форманты - плавно меняются во времени
        # Добавляем небольшие вариации для естественности
        f1_variation = np.random.normal(0, f1_base * 0.02, num_samples)  # 2% вариация
        f2_variation = np.random.normal(0, f2_base * 0.02, num_samples)
        f3_variation = np.random.normal(0, f3_base * 0.015, num_samples)  # 1.5% вариация
        
        # Сглаживаем вариации
        from scipy.ndimage import gaussian_filter1d
        f1_variation = gaussian_filter1d(f1_variation, sigma=num_samples / 500)
        f2_variation = gaussian_filter1d(f2_variation, sigma=num_samples / 500)
        f3_variation = gaussian_filter1d(f3_variation, sigma=num_samples / 500)
        
        # Добавляем плавное изменение формант (имитация артикуляторной динамики)
        f1_trend = f1_base * (1 + 0.05 * np.sin(2 * np.pi * 2.0 * t))  # Медленная модуляция
        f2_trend = f2_base * (1 + 0.03 * np.sin(2 * np.pi * 1.5 * t))
        f3_trend = f3_base * (1 + 0.02 * np.sin(2 * np.pi * 1.0 * t))
        
        f1_dynamic = f1_trend + f1_variation
        f2_dynamic = f2_trend + f2_variation
        f3_dynamic = f3_trend + f3_variation
        
        # Генерация источника
        source = self.generate_voice_source(duration, f0, voicing_strength)
        
        # Применяем формантный фильтр с динамическими формантами
        # Разбиваем на короткие сегменты для применения разных формант
        segment_length = int(0.01 * self.sample_rate)  # 10 мс сегменты
        output = np.zeros(num_samples)
        
        for i in range(0, num_samples, segment_length):
            end_idx = min(i + segment_length, num_samples)
            segment = source[i:end_idx]
            
            # Средние форманты для сегмента
            f1_seg = np.mean(f1_dynamic[i:end_idx])
            f2_seg = np.mean(f2_dynamic[i:end_idx])
            f3_seg = np.mean(f3_dynamic[i:end_idx])
            
            # Создаём фильтр для сегмента
            b, a = self.create_formant_filter(f1_seg, f2_seg, f3_seg)
            
            # Фильтруем сегмент
            filtered_segment = signal.lfilter(b, a, segment)
            
            # Плавное соединение сегментов
            if i > 0:
                fade_length = min(segment_length // 4, 50)
                fade_out = np.linspace(1, 0, fade_length)
                fade_in = np.linspace(0, 1, fade_length)
                output[i-fade_length:i] = output[i-fade_length:i] * fade_out + filtered_segment[:fade_length] * fade_in
                output[i:i+fade_length] = filtered_segment[:fade_length] * fade_in
                output[i+fade_length:end_idx] = filtered_segment[fade_length:]
            else:
                output[i:end_idx] = filtered_segment
        
        # Применяем огибающую для плавного затухания
        envelope = np.ones(num_samples)
        fade_samples = int(0.02 * self.sample_rate)  # 20 мс затухание
        if fade_samples > 0 and num_samples > fade_samples:
            fade_out = np.linspace(1, 0, fade_samples)
            envelope[-fade_samples:] = fade_out
        
        output = output * envelope
        
        # Нормализация
        if np.max(np.abs(output)) > 0:
            output = output / np.max(np.abs(output)) * 0.95  # Немного тише для естественности
        
        return output
    
    def synthesize_syllable(self,
                           vowel: str,
                           consonant_type: str = 'voiced',
                           duration: float = 0.3,
                           f0: float = 150.0,
                           consonant_duration: float = 0.05) -> np.ndarray:
        """
        Синтез слога (согласный + гласный).
        
        Args:
            vowel: Гласный звук
            consonant_type: Тип согласного ('voiced', 'unvoiced', 'none')
            duration: Общая длительность слога
            f0: Основная частота
            consonant_duration: Длительность согласного
            
        Returns:
            Синтезированный сигнал слога
        """
        vowel_duration = duration - consonant_duration
        
        if consonant_type == 'none':
            return self.synthesize_vowel(vowel, duration, f0)
        
        # Генерация согласного
        if consonant_type == 'unvoiced':
            consonant = self.generate_noise_source(consonant_duration, energy=0.5)
        else:  # voiced
            # Звонкий согласный: комбинация шума и фонации
            noise = self.generate_noise_source(consonant_duration, energy=0.3)
            voice = self.generate_voice_source(consonant_duration, f0, voicing_strength=0.5)
            consonant = noise + voice
            consonant = consonant / np.max(np.abs(consonant)) if np.max(np.abs(consonant)) > 0 else consonant
        
        # Генерация гласного
        vowel_sound = self.synthesize_vowel(vowel, vowel_duration, f0)
        
        # Плавное соединение (crossfade)
        fade_length = int(0.01 * self.sample_rate)  # 10 мс
        if fade_length > 0:
            fade_out = np.linspace(1, 0, fade_length)
            fade_in = np.linspace(0, 1, fade_length)
            consonant[-fade_length:] *= fade_out
            vowel_sound[:fade_length] *= fade_in
        
        # Объединение
        syllable = np.concatenate([consonant, vowel_sound])
        
        return syllable
    
    def synthesize_with_formants(self,
                                duration: float,
                                f0: float,
                                f1: float, f2: float, f3: float,
                                voicing_strength: float = 1.0,
                                noise_energy: float = 0.0) -> np.ndarray:
        """
        Синтез с произвольными формантами (для гибкого управления).
        
        Args:
            duration: Длительность
            f0: Основная частота
            f1, f2, f3: Формантные частоты
            voicing_strength: Сила фонации
            noise_energy: Энергия шумовой компоненты
            
        Returns:
            Синтезированный сигнал
        """
        # Генерация источника
        voice_source = self.generate_voice_source(duration, f0, voicing_strength)
        
        if noise_energy > 0:
            noise_source = self.generate_noise_source(duration, noise_energy)
            source = voice_source + noise_source
        else:
            source = voice_source
        
        # Фильтрация через формантный фильтр
        b, a = self.create_formant_filter(f1, f2, f3)
        output = signal.lfilter(b, a, source)
        
        # Нормализация
        if np.max(np.abs(output)) > 0:
            output = output / np.max(np.abs(output))
        
        return output

