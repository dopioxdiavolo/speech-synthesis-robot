"""
Основной модуль синтеза речи.

Объединяет все компоненты системы:
- Артикуляторная модель (source-filter)
- Управление просодией
- Перенос просодии из референсной речи

Поддерживает синтез с управляемыми просодическими параметрами
и переносом просодии из референсных аудиофайлов.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from scipy.interpolate import interp1d

from articulatory_model import ArticulatoryModel
from prosody_controller import RuleBasedProsodyController
from prosody_extraction import ProsodyExtractor

# Опциональный импорт TTS
try:
    from tts_integration import create_tts_engine, TTSEngine
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    TTSEngine = None

# Опциональный импорт детектора эмоций
try:
    from emotion_detector import EmotionDetector
    EMOTION_DETECTOR_AVAILABLE = True
except ImportError:
    EMOTION_DETECTOR_AVAILABLE = False
    EmotionDetector = None


class SpeechSynthesizer:
    """
    Основной синтезатор речи.
    
    Интегрирует артикуляторную модель и управление просодией
    для создания синтезированной речи с контролируемыми характеристиками.
    """
    
    def __init__(self, sample_rate: int = 16000, use_tts: bool = False, tts_engine: str = 'silero', use_ml_emotion: bool = True):
        """
        Инициализация синтезатора.
        
        Args:
            sample_rate: Частота дискретизации
            use_tts: Использовать TTS модель для естественного голоса
            tts_engine: Тип TTS движка ('xtts', 'bark', 'edge', 'gtts', 'silero', 'pyttsx3')
                      'xtts' - Coqui XTTS (лучшая для эмоций, рекомендуется!)
                      'bark' - Bark (Suno AI, очень выразительная)
                      'edge' - Edge TTS (Microsoft, хорошее качество)
                      'gtts' - Google TTS
                      'silero' - Silero TTS (offline)
                      'pyttsx3' - pyttsx3 (offline, низкое качество)
            use_ml_emotion: Использовать ML модель для определения эмоций (True) или rule-based (False)
        """
        self.sample_rate = sample_rate
        self.articulatory_model = ArticulatoryModel(sample_rate)
        self.prosody_controller = RuleBasedProsodyController(sample_rate)
        self.prosody_extractor = ProsodyExtractor(sample_rate)
        
        # Инициализация детектора эмоций
        self.use_ml_emotion = use_ml_emotion and EMOTION_DETECTOR_AVAILABLE
        self.emotion_detector = None
        if EMOTION_DETECTOR_AVAILABLE:
            self.emotion_detector = EmotionDetector(use_ml=use_ml_emotion)
        else:
            print("⚠ Детектор эмоций недоступен, используется встроенный rule-based")
        
        # Инициализация TTS (если нужно)
        self.use_tts = use_tts and TTS_AVAILABLE
        self.tts_engine = None
        if self.use_tts:
            self.tts_engine = create_tts_engine(tts_engine, sample_rate=sample_rate)
            if self.tts_engine is None:
                print("⚠ TTS движок не загружен, используется артикуляторная модель")
                self.use_tts = False
    
    def _detect_emotion_from_text(self, text: str) -> Tuple[str, float]:
        """
        Автоматическое определение эмоции из текста.
        
        Анализирует:
        - Ключевые слова (привет, стоп, внимание и т.д.)
        - Пунктуацию (восклицательные знаки, многоточия)
        - Заглавные буквы
        - Длину предложения
        
        Args:
            text: Текст для анализа
            
        Returns:
            Определённый стиль: 'neutral', 'friendly', 'strict', 'warning'
        """
        text_lower = text.lower()
        
        # Ключевые слова для разных эмоций
        friendly_words = [
            'привет', 'здравствуй', 'добро', 'спасибо', 'пожалуйста',
            'отлично', 'хорошо', 'замечательно', 'рад', 'радость',
            'улыбка', 'счастье', 'любовь', 'дружба', 'помощь'
        ]
        
        strict_words = [
            'прекрати', 'нельзя', 'запрещено', 'запретить',
            'строго', 'обязательно', 'требую', 'приказ', 'дисциплина',
            'нарушение', 'наказание', 'виноват', 'ответственность',
            'не делай', 'не трогай', 'нельзя делать', 'не смей'
        ]
        
        warning_words = [
            'внимание', 'осторожно', 'опасно', 'предупреждение',
            'авария', 'пожар', 'тревога', 'срочно', 'немедленно',
            'опасность', 'угроза', 'берегись', 'предупреждаю'
        ]
        
        # Подсчёт совпадений (инициализация переменных)
        friendly_score = sum(1 for word in friendly_words if word in text_lower)
        strict_score = sum(1 for word in strict_words if word in text_lower)
        warning_score = sum(1 for word in warning_words if word in text_lower)
        
        # Специальные случаи для слова "стоп"
        if 'стоп' in text_lower:
            # Если есть слова опасности - это warning, иначе strict
            if any(word in text_lower for word in ['опасно', 'внимание', 'тревога', 'авария', 'пожар']):
                warning_score += 2
            else:
                strict_score += 2
        
        # Анализ пунктуации
        exclamation_count = text.count('!')
        question_count = text.count('?')
        ellipsis_count = text.count('...') + text.count('…')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Бонусы за пунктуацию
        if exclamation_count >= 2:
            warning_score += 2
        elif exclamation_count == 1:
            if warning_score > 0:
                warning_score += 1
            elif strict_score > 0:
                strict_score += 1
            else:
                friendly_score += 0.5
        
        if ellipsis_count > 0:
            strict_score += 1
        
        if caps_ratio > 0.3:  # Много заглавных букв
            warning_score += 2
        
        # Анализ длины и структуры
        if len(text) < 10:  # Очень короткие фразы часто бывают командами
            if any(word in text_lower for word in ['стоп', 'нет', 'не']):
                strict_score += 1
        
        # Определение стиля по максимальному счёту
        scores = {
            'friendly': friendly_score,
            'strict': strict_score,
            'warning': warning_score,
            'neutral': 0
        }
        
        max_score = max(scores.values())
        
        # Если нет явных признаков, используем нейтральный
        if max_score == 0:
            return 'neutral'
        
        # Определяем стиль
        detected_style = max(scores, key=scores.get)
        
        return detected_style, 0.6  # Rule-based уверенность
    
    def synthesize_phrase(self,
                         phrase: str,
                         style: Optional[str] = None,
                         base_f0: Optional[float] = None,
                         duration_factor: Optional[float] = None,
                         use_tts_override: Optional[bool] = None,
                         auto_detect_emotion: bool = True) -> np.ndarray:
        """
        Синтез фразы с автоматическим определением эмоции.
        
        Args:
            phrase: Текст фразы
            style: Стиль речи ('neutral', 'friendly', 'strict', 'warning').
                   Если None и auto_detect_emotion=True, определяется автоматически
            base_f0: Базовая частота F0 (если None, используется стиль по умолчанию)
            duration_factor: Фактор длительности (если None, используется стиль)
            use_tts_override: Переопределить использование TTS (если None, используется self.use_tts)
            auto_detect_emotion: Автоматически определять эмоцию из текста (по умолчанию True)
            
        Returns:
            Синтезированный аудиосигнал
        """
        # Автоматическое определение эмоции, если не указан стиль
        if style is None and auto_detect_emotion:
            # Используем ML детектор если доступен, иначе встроенный
            if self.emotion_detector is not None:
                style, confidence = self.emotion_detector.detect(phrase)
                print(f"🎭 Определена эмоция: {style} (уверенность: {confidence:.2f})")
            else:
                style, confidence = self._detect_emotion_from_text(phrase)
                print(f"🎭 Определена эмоция: {style} (rule-based)")
        elif style is None:
            style = 'neutral'
        
        # Определяем, использовать ли TTS
        use_tts = use_tts_override if use_tts_override is not None else self.use_tts
        
        # Если TTS доступен, используем его для генерации базового голоса
        if use_tts and self.tts_engine is not None:
            try:
                # Маппинг стилей на эмоции для TTS моделей с поддержкой эмоций
                style_to_emotion = {
                    'neutral': 'neutral',
                    'friendly': 'friendly',  # или 'happy'
                    'strict': 'strict',      # или 'angry'
                    'warning': 'warning'     # или 'surprised'
                }
                emotion = style_to_emotion.get(style, 'neutral')
                
                # Генерируем базовую речь с помощью TTS
                # Проверяем, поддерживает ли TTS эмоции
                try:
                    # Пробуем передать эмоцию
                    import inspect
                    sig = inspect.signature(self.tts_engine.synthesize)
                    if 'emotion' in sig.parameters:
                        base_audio = self.tts_engine.synthesize(phrase, emotion=emotion)
                    else:
                        base_audio = self.tts_engine.synthesize(phrase)
                except:
                    # Fallback
                    base_audio = self.tts_engine.synthesize(phrase)
                
                # Применяем нашу систему управления просодией
                # Изменяем темп, громкость согласно стилю
                modified_audio = self._apply_prosody_to_tts_audio(
                    base_audio, style, base_f0, duration_factor
                )
                
                return modified_audio
            except Exception as e:
                print(f"⚠ Ошибка TTS синтеза: {e}, используем артикуляторную модель")
                # Fallback на артикуляторную модель
        
        # Артикуляторная модель (оригинальный метод)
        # Упрощённая обработка: разбиение на гласные
        # В реальной системе здесь был бы фонетический анализ
        vowels = self._extract_vowels(phrase)
        
        if len(vowels) == 0:
            # Если нет гласных, возвращаем тишину
            return np.array([])
        
        # Получение параметров стиля
        if duration_factor is None:
            duration_factor = self.prosody_controller.get_duration_factor(style)
        
        if base_f0 is None:
            style_params = self.prosody_controller.SPEECH_STYLES.get(style, 
                                                                     self.prosody_controller.SPEECH_STYLES['neutral'])
            base_f0 = style_params['f0_mean']
        
        # Синтез каждого гласного с улучшенными переходами
        segments = []
        pause_duration = self.prosody_controller.get_pause_duration(style)
        
        for i, vowel in enumerate(vowels):
            # Длительность сегмента с небольшими вариациями
            base_duration = 0.2 * duration_factor
            # Добавляем естественные вариации длительности
            duration_variation = np.random.uniform(0.9, 1.1)
            segment_duration = base_duration * duration_variation
            
            # Генерация F0 контура для сегмента
            f0_times, f0_values = self.prosody_controller.generate_f0_contour(
                segment_duration, style, num_syllables=len(vowels)
            )
            
            # Синтез с динамическим F0
            segment = self._synthesize_with_f0_contour(
                vowel, segment_duration, f0_times, f0_values
            )
            
            # Улучшенные переходы между сегментами
            if i > 0 and len(segments) > 0:
                # Плавный переход (crossfade)
                fade_length = int(0.03 * self.sample_rate)  # 30 мс переход
                fade_length = min(fade_length, len(segment) // 4, len(segments[-1]) // 4)
                
                if fade_length > 0:
                    fade_out = np.linspace(1, 0, fade_length)
                    fade_in = np.linspace(0, 1, fade_length)
                    
                    # Применяем fade out к предыдущему сегменту
                    if len(segments[-1]) >= fade_length:
                        segments[-1][-fade_length:] *= fade_out
                    
                    # Применяем fade in к текущему сегменту
                    if len(segment) >= fade_length:
                        segment[:fade_length] *= fade_in
            
            segments.append(segment)
            
            # Добавление паузы между сегментами (кроме последнего)
            if i < len(vowels) - 1:
                # Пауза с плавным затуханием
                pause_samples = int(pause_duration * self.sample_rate)
                pause = np.zeros(pause_samples)
                
                # Добавляем небольшой шум в паузу для естественности
                if pause_samples > 100:
                    noise = np.random.normal(0, 0.01, pause_samples)
                    # Фильтруем шум (низкие частоты)
                    from scipy.ndimage import gaussian_filter1d
                    noise = gaussian_filter1d(noise, sigma=pause_samples / 20)
                    pause = noise
                
                segments.append(pause)
        
        # Объединение сегментов
        if len(segments) == 0:
            return np.array([])
        
        output = np.concatenate(segments)
        
        # Применение энергетического контура
        total_duration = len(output) / self.sample_rate
        energy_times, energy_values = self.prosody_controller.generate_energy_contour(
            total_duration, style
        )
        output = self.prosody_controller.apply_prosody_to_signal(
            output, energy_contour=(energy_times, energy_values)
        )
        
        # Финальная обработка для более естественного звука
        # Сглаживание для устранения артефактов
        from scipy.ndimage import gaussian_filter1d
        output = gaussian_filter1d(output, sigma=2)
        
        # Применяем мягкую огибающую
        envelope = np.ones(len(output))
        fade_samples = int(0.01 * self.sample_rate)  # 10 мс
        if len(envelope) > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            envelope[:fade_samples] = fade_in
            envelope[-fade_samples:] = fade_out
        
        output = output * envelope
        
        # Нормализация
        if np.max(np.abs(output)) > 0:
            output = output / np.max(np.abs(output)) * 0.9  # Немного тише
        
        return output
    
    def synthesize_with_prosody_transfer(self,
                                        phrase: str,
                                        reference_audio: str,
                                        duration_factor: float = 1.0) -> np.ndarray:
        """
        Синтез фразы с переносом просодии из референсной речи.
        
        Физиологический смысл:
        Перенос просодии позволяет сохранить интонационные и энергетические
        характеристики исходной речи при синтезе нового текста.
        
        Args:
            phrase: Текст для синтеза
            reference_audio: Путь к референсному аудиофайлу
            duration_factor: Фактор масштабирования длительности
            
        Returns:
            Синтезированный аудиосигнал с перенесённой просодией
        """
        # Извлечение просодии из референса
        prosody = self.prosody_extractor.extract_all_prosody(reference_audio)
        
        # Извлечение гласных из фразы
        vowels = self._extract_vowels(phrase)
        
        if len(vowels) == 0:
            return np.array([])
        
        # Определение длительности синтеза
        target_duration = prosody['duration'] * duration_factor
        
        # Интерполяция F0 контура на новую длительность
        target_times = np.linspace(0, target_duration, 
                                   int(target_duration * self.sample_rate / 100))  # 10 мс шаг
        f0_interp = self.prosody_extractor.interpolate_contour(
            prosody['f0_times'],
            prosody['f0_values'],
            target_times,
            method='cubic'
        )
        
        # Интерполяция энергетического контура
        energy_interp = self.prosody_extractor.interpolate_contour(
            prosody['energy_times'],
            prosody['energy_values'],
            target_times,
            method='linear'
        )
        
        # Нормализация энергии к диапазону [0.3, 1.0]
        energy_min, energy_max = np.min(energy_interp), np.max(energy_interp)
        if energy_max > energy_min:
            energy_normalized = 0.3 + 0.7 * (energy_interp - energy_min) / (energy_max - energy_min)
        else:
            energy_normalized = np.ones_like(energy_interp) * 0.7
        
        # Распределение длительности между гласными
        segment_durations = self._distribute_duration(target_duration, len(vowels))
        
        # Синтез сегментов
        segments = []
        current_time = 0.0
        
        for i, (vowel, seg_duration) in enumerate(zip(vowels, segment_durations)):
            # Извлечение F0 контура для сегмента
            seg_start_idx = int(current_time / 0.01)  # 10 мс шаг
            seg_end_idx = int((current_time + seg_duration) / 0.01)
            seg_end_idx = min(seg_end_idx, len(f0_interp))
            
            if seg_start_idx < len(f0_interp):
                seg_f0_times = target_times[seg_start_idx:seg_end_idx] - current_time
                seg_f0_values = f0_interp[seg_start_idx:seg_end_idx]
                
                # Синтез сегмента
                segment = self._synthesize_with_f0_contour(
                    vowel, seg_duration, seg_f0_times, seg_f0_values
                )
                
                # Применение энергетического контура
                seg_energy = energy_normalized[seg_start_idx:seg_end_idx]
                if len(seg_energy) > 0:
                    energy_signal = np.interp(
                        np.linspace(0, seg_duration, len(segment)),
                        seg_f0_times,
                        seg_energy
                    )
                    segment = segment * energy_signal
                
                segments.append(segment)
                current_time += seg_duration
        
        # Объединение сегментов
        if len(segments) == 0:
            return np.array([])
        
        output = np.concatenate(segments)
        
        return output
    
    def _extract_vowels(self, text: str) -> List[str]:
        """
        Упрощённое извлечение гласных из текста.
        
        В реальной системе здесь был бы полноценный фонетический анализ.
        
        Args:
            text: Входной текст
            
        Returns:
            Список гласных
        """
        vowels = []
        vowel_chars = 'аеёиоуыэюяАЕЁИОУЫЭЮЯaeiouyAEIOUY'
        
        for char in text.lower():
            if char in vowel_chars:
                # Маппинг русских гласных на упрощённые символы
                mapping = {
                    'а': 'a', 'о': 'o', 'у': 'u', 'ы': 'y',
                    'э': 'e', 'и': 'i', 'е': 'e', 'ё': 'o',
                    'ю': 'u', 'я': 'a'
                }
                vowel = mapping.get(char, char)
                if vowel in 'aeiouy':
                    vowels.append(vowel)
        
        return vowels if vowels else ['a']  # По умолчанию 'a'
    
    def _synthesize_with_f0_contour(self,
                                    vowel: str,
                                    duration: float,
                                    f0_times: np.ndarray,
                                    f0_values: np.ndarray) -> np.ndarray:
        """
        Синтез гласного с динамическим F0 контуром.
        
        Args:
            vowel: Гласный звук
            duration: Длительность
            f0_times: Временные метки F0
            f0_values: Значения F0
            
        Returns:
            Синтезированный сигнал
        """
        # Если контур пустой, используем среднее значение
        if len(f0_values) == 0:
            f0_mean = 150.0
        else:
            f0_mean = np.mean(f0_values)
        
        # Разбиение на короткие сегменты для аппроксимации F0 контура
        segment_length = 0.05  # 50 мс сегменты
        num_segments = int(duration / segment_length)
        
        if num_segments == 0:
            num_segments = 1
        
        segments = []
        
        for i in range(num_segments):
            seg_start = i * segment_length
            seg_end = min((i + 1) * segment_length, duration)
            seg_duration = seg_end - seg_start
            
            # Получение F0 для сегмента
            seg_time = (seg_start + seg_end) / 2
            if len(f0_times) > 0 and len(f0_values) > 0:
                seg_f0 = np.interp(seg_time, f0_times, f0_values)
            else:
                seg_f0 = f0_mean
            
            # Синтез сегмента
            segment = self.articulatory_model.synthesize_vowel(
                vowel, seg_duration, seg_f0
            )
            segments.append(segment)
        
        return np.concatenate(segments)
    
    def _apply_prosody_to_tts_audio(self,
                                   audio: np.ndarray,
                                   style: str,
                                   base_f0: Optional[float] = None,
                                   duration_factor: Optional[float] = None) -> np.ndarray:
        """
        Применение просодии к аудио, сгенерированному TTS.
        
        Изменяет темп и громкость согласно стилю.
        Примечание: изменение F0 (pitch) сложнее и требует более продвинутых методов.
        
        Args:
            audio: Базовое аудио от TTS
            style: Стиль речи
            base_f0: Базовая частота F0 (не используется напрямую, но сохраняется для совместимости)
            duration_factor: Фактор длительности (если None, используется стиль)
            
        Returns:
            Модифицированное аудио
        """
        from scipy import signal as scipy_signal
        
        # Получение параметров стиля
        if duration_factor is None:
            duration_factor = self.prosody_controller.get_duration_factor(style)
        
        # Изменение темпа (time stretching)
        if abs(duration_factor - 1.0) > 0.01:
            original_length = len(audio)
            new_length = int(original_length * duration_factor)
            audio = scipy_signal.resample(audio, new_length)
        
        # Применение энергетического контура
        total_duration = len(audio) / self.sample_rate
        energy_times, energy_values = self.prosody_controller.generate_energy_contour(
            total_duration, style
        )
        audio = self.prosody_controller.apply_prosody_to_signal(
            audio, energy_contour=(energy_times, energy_values)
        )
        
        # Нормализация
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        return audio
    
    def _distribute_duration(self, total_duration: float, num_segments: int) -> List[float]:
        """
        Распределение общей длительности между сегментами.
        
        Args:
            total_duration: Общая длительность
            num_segments: Количество сегментов
            
        Returns:
            Список длительностей сегментов
        """
        if num_segments == 0:
            return []
        
        base_duration = total_duration / num_segments
        # Добавляем небольшие вариации для естественности
        variations = np.random.normal(0, base_duration * 0.1, num_segments)
        durations = [base_duration + v for v in variations]
        
        # Нормализация к общей длительности
        total = sum(durations)
        if total > 0:
            durations = [d * total_duration / total for d in durations]
        
        return durations
    
    def save_audio(self, signal: np.ndarray, filename: str):
        """
        Сохранение аудиосигнала в файл.
        
        Args:
            signal: Аудиосигнал
            filename: Имя файла
        """
        try:
            import soundfile as sf
            sf.write(filename, signal, self.sample_rate)
        except ImportError:
            # Fallback на scipy.io.wavfile
            from scipy.io import wavfile
            # Нормализация к 16-bit диапазону
            signal_int16 = np.int16(signal * 32767)
            wavfile.write(filename, self.sample_rate, signal_int16)

