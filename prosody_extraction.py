"""
Модуль извлечения просодических характеристик из референсной речи.

Использует parselmouth (Python API для Praat) для анализа:
- F0 contour (мелодия, интонация)
- Energy contour (громкость)
- Duration (длительность сегментов)

Физиологический смысл:
- F0 отражает напряжение голосовых связок и подглоточное давление
- Energy связана с амплитудой вибрации голосовых связок и потоком воздуха
- Duration зависит от артикуляторной скорости и пауз
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.interpolate import interp1d

# Опциональный импорт parselmouth
try:
    import parselmouth
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    parselmouth = None


class ProsodyExtractor:
    """
    Извлечение просодических параметров из аудиофайла.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Инициализация экстрактора.
        
        Args:
            sample_rate: Частота дискретизации для обработки
        """
        self.sample_rate = sample_rate
    
    def extract_f0_contour(self,
                          audio_path: str,
                          time_step: float = 0.01,
                          f0_min: float = 75.0,
                          f0_max: float = 500.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Извлечение F0 контура (мелодии).
        
        Физиологический смысл:
        F0 (fundamental frequency) - частота вибрации голосовых связок.
        Изменения F0 создают интонацию речи.
        
        Args:
            audio_path: Путь к аудиофайлу
            time_step: Шаг времени для анализа (секунды)
            f0_min: Минимальная частота F0 (Гц)
            f0_max: Максимальная частота F0 (Гц)
            
        Returns:
            (times, f0_values): Массивы времени и значений F0
        """
        if not PARSELMOUTH_AVAILABLE:
            raise ImportError(
                "parselmouth не установлен или несовместим с вашей версией Python. "
                "Используйте альтернативный метод extract_f0_contour_simple."
            )
        
        # Загрузка аудио
        sound = parselmouth.Sound(audio_path)
        
        # Извлечение F0 с помощью алгоритма Praat
        pitch = sound.to_pitch_ac(
            time_step=time_step,
            voicing_threshold=0.45,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max
        )
        
        # Получение значений F0
        times = pitch.xs()
        f0_values = np.array([pitch.get_value_at_time(t) for t in times])
        
        # Удаление невалидных значений (unvoiced segments)
        valid_mask = ~np.isnan(f0_values)
        times = times[valid_mask]
        f0_values = f0_values[valid_mask]
        
        return times, f0_values
    
    def extract_energy_contour(self,
                              audio_path: str,
                              time_step: float = 0.01,
                              window_length: float = 0.025,
                              minimum_pitch: float = 75.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Извлечение контура энергии (громкости).
        
        Физиологический смысл:
        Energy отражает амплитуду вибрации голосовых связок и поток воздуха.
        Связана с воспринимаемой громкостью речи.
        
        Args:
            audio_path: Путь к аудиофайлу
            time_step: Шаг времени для анализа (секунды)
            window_length: Длина окна для анализа (секунды)
            minimum_pitch: Минимальная частота для анализа интенсивности
            
        Returns:
            (times, energy_values): Массивы времени и значений энергии (в дБ)
        """
        if not PARSELMOUTH_AVAILABLE:
            raise ImportError(
                "parselmouth не установлен или несовместим с вашей версией Python. "
                "Используйте альтернативный метод extract_energy_contour_simple."
            )
        
        # Загрузка аудио
        sound = parselmouth.Sound(audio_path)
        
        # Извлечение интенсивности (энергии)
        intensity = sound.to_intensity(
            minimum_pitch=minimum_pitch,
            time_step=time_step
        )
        
        # Получение значений интенсивности
        times = intensity.xs()
        energy_db = np.array([intensity.get_value(t) for t in times])
        
        # Удаление невалидных значений
        valid_mask = ~np.isnan(energy_db)
        times = times[valid_mask]
        energy_db = energy_db[valid_mask]
        
        return times, energy_db
    
    def extract_duration(self, audio_path: str) -> float:
        """
        Извлечение общей длительности аудио.
        
        Args:
            audio_path: Путь к аудиофайлу
            
        Returns:
            Длительность в секундах
        """
        if not PARSELMOUTH_AVAILABLE:
            # Альтернативный способ: используем soundfile
            try:
                import soundfile as sf
                with sf.SoundFile(audio_path) as f:
                    return len(f) / f.samplerate
            except ImportError:
                raise ImportError(
                    "parselmouth не доступен. Установите soundfile для базовой функциональности."
                )
        
        sound = parselmouth.Sound(audio_path)
        return sound.duration
    
    def extract_all_prosody(self,
                           audio_path: str,
                           time_step: float = 0.01,
                           f0_min: float = 75.0,
                           f0_max: float = 500.0) -> Dict:
        """
        Извлечение всех просодических параметров.
        
        Args:
            audio_path: Путь к аудиофайлу
            time_step: Шаг времени для анализа
            f0_min: Минимальная частота F0
            f0_max: Максимальная частота F0
            
        Returns:
            Словарь с просодическими параметрами:
            {
                'f0_times': массив времени для F0,
                'f0_values': массив значений F0,
                'energy_times': массив времени для энергии,
                'energy_values': массив значений энергии,
                'duration': общая длительность
            }
        """
        # Извлечение F0
        f0_times, f0_values = self.extract_f0_contour(
            audio_path, time_step, f0_min, f0_max
        )
        
        # Извлечение энергии
        energy_times, energy_values = self.extract_energy_contour(
            audio_path, time_step
        )
        
        # Извлечение длительности
        duration = self.extract_duration(audio_path)
        
        return {
            'f0_times': f0_times,
            'f0_values': f0_values,
            'energy_times': energy_times,
            'energy_values': energy_values,
            'duration': duration
        }
    
    def interpolate_contour(self,
                           source_times: np.ndarray,
                           source_values: np.ndarray,
                           target_times: np.ndarray,
                           method: str = 'linear') -> np.ndarray:
        """
        Интерполяция контура для нового временного масштаба.
        
        Используется для переноса просодии на синтезированную речь
        с другой длительностью.
        
        Args:
            source_times: Временные метки исходного контура
            source_values: Значения исходного контура
            target_times: Временные метки целевого контура
            method: Метод интерполяции ('linear', 'cubic')
            
        Returns:
            Интерполированные значения для target_times
        """
        if len(source_times) < 2:
            # Если недостаточно точек, возвращаем среднее значение
            if len(source_values) > 0:
                return np.full(len(target_times), np.mean(source_values))
            else:
                return np.zeros(len(target_times))
        
        # Создание интерполяционной функции
        if method == 'linear':
            interp_func = interp1d(
                source_times, source_values,
                kind='linear',
                bounds_error=False,
                fill_value=(source_values[0], source_values[-1])
            )
        else:  # cubic
            interp_func = interp1d(
                source_times, source_values,
                kind='cubic',
                bounds_error=False,
                fill_value=(source_values[0], source_values[-1])
            )
        
        # Интерполяция
        interpolated = interp_func(target_times)
        
        return interpolated
    
    def normalize_contour(self,
                         values: np.ndarray,
                         target_mean: Optional[float] = None,
                         target_std: Optional[float] = None) -> np.ndarray:
        """
        Нормализация контура (z-score или к целевому диапазону).
        
        Args:
            values: Значения контура
            target_mean: Целевое среднее значение (если None, сохраняется исходное)
            target_std: Целевое стандартное отклонение (если None, сохраняется исходное)
            
        Returns:
            Нормализованные значения
        """
        if len(values) == 0:
            return values
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return values
        
        # Z-score нормализация
        normalized = (values - mean_val) / std_val
        
        # Применение целевых параметров
        if target_mean is not None:
            normalized = normalized * (target_std if target_std is not None else std_val) + target_mean
        elif target_std is not None:
            normalized = normalized * target_std + mean_val
        
        return normalized

