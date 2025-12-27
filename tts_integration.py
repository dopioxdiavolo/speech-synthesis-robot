"""
Интеграция TTS моделей для естественного голоса с эмоциями.

Поддерживает:
- Coqui XTTS (лучшая для эмоций, открытая, поддерживает русский)
- Bark (Suno AI, очень выразительная, поддерживает эмоции)
- Edge TTS (Microsoft, бесплатная, хорошее качество)
- Silero TTS (русский язык, offline)
- pyttsx3 (простой, работает без интернета)
"""

import numpy as np
from typing import Optional
import io


class TTSEngine:
    """
    Базовый класс для TTS движков.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.engine = None
    
    def synthesize(self, text: str) -> np.ndarray:
        """
        Синтез речи из текста.
        
        Args:
            text: Текст для синтеза
            
        Returns:
            Аудиосигнал
        """
        raise NotImplementedError


class SileroTTS(TTSEngine):
    """
    Silero TTS - качественная русскоязычная модель.
    
    Установка: pip install torch torchaudio
    """
    
    def __init__(self, sample_rate: int = 16000, speaker: str = 'aidar_v2'):
        """
        Инициализация Silero TTS.
        
        Args:
            sample_rate: Частота дискретизации
            speaker: Голос ('aidar_v2', 'baya_v2', 'kseniya_v2', 'natasha_v2', 
                           'ruslan_v2', 'irina_v2', 'v5_ru', 'v5_1_ru')
        """
        super().__init__(sample_rate)
        self.speaker = speaker
        self.model = None
        try:
            import torch
            self.device = torch.device('cpu')
        except:
            self.device = None
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели Silero TTS."""
        try:
            import torch
            import os
            
            # Устанавливаем переменную окружения для обхода проблемы с qengine
            os.environ['TORCH_QUANTIZATION_ENGINE'] = 'fbgemm'
            
            # Пробуем загрузить модель v5 (более новая версия)
            try:
                self.model, example_text = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language='ru',
                    speaker='v5_ru',  # Используем v5 вместо aidar_v2
                    trust_repo=True
                )
                self.model.to(self.device)
                self.model.eval()
                print(f"✓ Silero TTS v5 загружен")
            except:
                # Fallback на старую версию
                self.model, example_text = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language='ru',
                    speaker=self.speaker,
                    trust_repo=True
                )
                self.model.to(self.device)
                self.model.eval()
                print(f"✓ Silero TTS загружен (speaker: {self.speaker})")
        except Exception as e:
            print(f"⚠ Ошибка загрузки Silero TTS: {e}")
            print("  Попробуйте: pip install --upgrade torch torchaudio")
            print("  Или используйте pyttsx3: pip install pyttsx3")
            self.model = None
    
    def synthesize(self, text: str) -> np.ndarray:
        """
        Синтез речи с помощью Silero TTS.
        
        Args:
            text: Текст для синтеза
            
        Returns:
            Аудиосигнал
        """
        if self.model is None:
            raise RuntimeError("Silero TTS модель не загружена")
        
        # Синтез
        # Для v5 используется другой API
        if hasattr(self.model, 'apply_tts'):
            result = self.model.apply_tts(
                text=text,
                speaker=self.speaker if self.speaker != 'v5_ru' else None,
                sample_rate=self.sample_rate
            )
            # apply_tts может возвращать список или тензор
            if isinstance(result, (list, tuple)):
                audio = result[0]
            else:
                audio = result
        else:
            # Альтернативный способ для некоторых версий
            audio = self.model(text, sample_rate=self.sample_rate)
        
        # Конвертация в numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Нормализация
        if len(audio) > 0:
            audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
        
        return audio


class EdgeTTS(TTSEngine):
    """
    Edge TTS (Microsoft) - бесплатная, качественная модель.
    
    Установка: pip install edge-tts
    Требует интернет-соединение.
    Поддерживает эмоции через выбор голоса и SSML.
    """
    
    def __init__(self, sample_rate: int = 16000, voice: str = 'ru-RU-SvetlanaNeural'):
        """
        Инициализация Edge TTS.
        
        Args:
            sample_rate: Частота дискретизации
            voice: Голос (по умолчанию русский женский)
                   Доступные: 'ru-RU-SvetlanaNeural', 'ru-RU-DmitryNeural', 
                   'ru-RU-DariyaNeural' и др.
        """
        super().__init__(sample_rate)
        self.voice = voice
        self._init_engine()
    
    def synthesize(self, text: str, emotion: str = "neutral") -> np.ndarray:
        """
        Синтез речи с эмоцией.
        
        Edge TTS сам по себе не поддерживает эмоции напрямую,
        но мы используем разные голоса для разных эмоций.
        Основная эмоциональная обработка происходит через просодию.
        
        Args:
            text: Текст для синтеза
            emotion: Эмоция (используется для выбора голоса, если доступно)
            
        Returns:
            Аудиосигнал
        """
        if self.edge_tts is None:
            raise RuntimeError("Edge TTS не инициализирован")
        
        import asyncio
        import tempfile
        import os
        
        # Выбираем голос в зависимости от эмоции (если доступно)
        emotion_voices = {
            'friendly': 'ru-RU-SvetlanaNeural',  # Женский, дружелюбный
            'happy': 'ru-RU-SvetlanaNeural',
            'strict': 'ru-RU-DmitryNeural',      # Мужской, строгий
            'angry': 'ru-RU-DmitryNeural',
            'warning': 'ru-RU-DariyaNeural',     # Женский, предупреждающий
            'surprised': 'ru-RU-DariyaNeural',
        }
        
        voice_to_use = emotion_voices.get(emotion, self.voice)
        
        async def _synthesize_async():
            """Асинхронный синтез."""
            communicate = self.edge_tts.Communicate(text, voice_to_use)
            
            # Сохраняем во временный файл
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                await communicate.save(tmp_path)
                
                # Загружаем MP3
                try:
                    import soundfile as sf
                    audio, sr = sf.read(tmp_path)
                    if sr != self.sample_rate:
                        from scipy import signal
                        num_samples = int(len(audio) * self.sample_rate / sr)
                        audio = signal.resample(audio, num_samples)
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                except:
                    # Fallback через pydub
                    try:
                        from pydub import AudioSegment
                        audio_segment = AudioSegment.from_mp3(tmp_path)
                        audio_segment = audio_segment.set_frame_rate(self.sample_rate)
                        audio_segment = audio_segment.set_channels(1)
                        audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                        audio = audio / 32768.0
                    except Exception as e:
                        raise RuntimeError(f"Не удалось загрузить аудио: {e}")
                
                return audio
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        # Запускаем асинхронную функцию
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        audio = loop.run_until_complete(_synthesize_async())
        
        # Нормализация
        if len(audio) > 0:
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
        
        return audio.astype(np.float32)
    
    def _init_engine(self):
        """Инициализация Edge TTS."""
        try:
            import edge_tts
            self.edge_tts = edge_tts
            print(f"✓ Edge TTS инициализирован (voice: {self.voice}, поддерживает эмоции!)")
        except ImportError:
            print("⚠ edge-tts не установлен")
            print("  Установите: pip install edge-tts")
            self.edge_tts = None
        except Exception as e:
            print(f"⚠ Ошибка инициализации Edge TTS: {e}")
            self.edge_tts = None


class CoquiXTTS(TTSEngine):
    """
    Coqui XTTS - лучшая модель для эмоционального синтеза.
    
    Установка: pip install TTS
    Поддерживает эмоции через промпты.
    Примечание: может не работать с Python 3.12+
    """
    
    def __init__(self, sample_rate: int = 16000, language: str = 'ru'):
        """
        Инициализация Coqui XTTS.
        
        Args:
            sample_rate: Частота дискретизации
            language: Язык ('ru' для русского)
        """
        super().__init__(sample_rate)
        self.language = language
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели Coqui XTTS."""
        try:
            from TTS.api import TTS
            # Используем XTTS модель (поддерживает эмоции)
            self.model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            print("✓ Coqui XTTS загружен (поддерживает эмоции!)")
        except ImportError:
            print("⚠ TTS (Coqui) не установлен или несовместим с Python 3.12+")
            print("  Попробуйте другую модель (bark, edge)")
            self.model = None
        except Exception as e:
            print(f"⚠ Ошибка загрузки Coqui XTTS: {e}")
            self.model = None
    
    def synthesize(self, text: str, emotion: str = "neutral") -> np.ndarray:
        """
        Синтез речи с эмоцией.
        
        Args:
            text: Текст для синтеза
            emotion: Эмоция ('neutral', 'happy', 'sad', 'angry', 'surprised')
            
        Returns:
            Аудиосигнал
        """
        if self.model is None:
            raise RuntimeError("Coqui XTTS не загружен")
        
        import tempfile
        import os
        
        # Создаём временный файл
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Синтез
            self.model.tts_to_file(
                text=text,
                file_path=tmp_path,
                language=self.language,
                speaker_wav=None  # Можно использовать референсный голос
            )
            
            # Загрузка аудио
            import soundfile as sf
            audio, sr = sf.read(tmp_path)
            
            # Ресемплинг если нужно
            if sr != self.sample_rate:
                from scipy import signal
                num_samples = int(len(audio) * self.sample_rate / sr)
                audio = signal.resample(audio, num_samples)
            
            # Конвертируем в моно если стерео
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Нормализация
            if len(audio) > 0:
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio))
            
            return audio.astype(np.float32)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class BarkTTS(TTSEngine):
    """
    Bark (Suno AI) - очень выразительная модель с поддержкой эмоций.
    
    Установка: pip install bark
    Отлично передаёт эмоции и интонации.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Инициализация Bark TTS.
        
        Args:
            sample_rate: Частота дискретизации
        """
        super().__init__(sample_rate)
        self.model = None
        self.bark_generate = None
        self.bark_sample_rate = 24000  # Bark использует 24kHz
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели Bark."""
        try:
            import torch
            # Исправляем проблему с weights_only в PyTorch 2.6+
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
            
            from bark import SAMPLE_RATE, generate_audio, preload_models
            import os
            
            # Устанавливаем переменные окружения для Bark
            os.environ["SUNO_USE_SMALL_MODELS"] = "1"  # Используем маленькие модели для скорости
            
            # Предзагрузка моделей
            try:
                preload_models()
            except Exception as e:
                print(f"⚠ Предзагрузка моделей Bark не удалась: {e}")
                print("  Продолжаем без предзагрузки...")
            
            self.bark_generate = generate_audio
            self.bark_sample_rate = SAMPLE_RATE
            print("✓ Bark TTS загружен (очень выразительный!)")
        except ImportError:
            print("⚠ bark не установлен")
            print("  Установите: pip install bark")
            self.bark_generate = None
        except Exception as e:
            print(f"⚠ Ошибка загрузки Bark: {e}")
            print("  Попробуйте обновить: pip install --upgrade bark")
            self.bark_generate = None
    
    def synthesize(self, text: str, emotion: str = "neutral") -> np.ndarray:
        """
        Синтез речи с эмоцией.
        
        Args:
            text: Текст для синтеза
            emotion: Эмоция ('neutral', 'happy', 'sad', 'angry', 'surprised', 
                            'friendly', 'strict', 'warning')
            
        Returns:
            Аудиосигнал
        """
        if self.bark_generate is None:
            raise RuntimeError("Bark не загружен")
        
        # Bark использует специальный синтаксис для эмоций
        emotion_tags = {
            'neutral': '',
            'happy': '[laughter]',
            'sad': '[sighs]',
            'angry': '[clears throat]',
            'surprised': '[gasps]',
            'friendly': '[smiles]',
            'strict': '[sternly]',
            'warning': '[urgently]'
        }
        
        # Формируем текст с эмоциональными тегами
        emotion_tag = emotion_tags.get(emotion, '')
        if emotion_tag:
            text_with_emotion = f"{emotion_tag} {text}"
        else:
            text_with_emotion = text
        
        # Синтез (Bark поддерживает русский через специальные промпты)
        # Используем русский промпт
        try:
            audio_array = self.bark_generate(
                text_with_emotion, 
                history_prompt="v2/ru_speaker_9"  # Русский голос
            )
        except:
            # Fallback на английский промпт
            audio_array = self.bark_generate(text_with_emotion)
        
        # Конвертация в numpy
        if hasattr(audio_array, 'cpu'):
            audio = audio_array.cpu().numpy()
        else:
            audio = np.array(audio_array)
        
        # Ресемплинг если нужно
        if self.bark_sample_rate != self.sample_rate:
            from scipy import signal
            num_samples = int(len(audio) * self.sample_rate / self.bark_sample_rate)
            audio = signal.resample(audio, num_samples)
        
        # Конвертируем в моно если стерео
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Нормализация
        if len(audio) > 0:
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
        
        return audio.astype(np.float32)


class GoogleTTS(TTSEngine):
    """
    Google Text-to-Speech (gTTS) - простой и надёжный.
    
    Установка: pip install gtts
    Требует интернет-соединение.
    """
    
    def __init__(self, sample_rate: int = 16000, lang: str = 'ru'):
        """
        Инициализация gTTS.
        
        Args:
            sample_rate: Частота дискретизации
            lang: Язык ('ru' для русского)
        """
        super().__init__(sample_rate)
        self.lang = lang
        self._init_engine()
    
    def _init_engine(self):
        """Инициализация gTTS."""
        try:
            from gtts import gTTS
            self.gTTS = gTTS
            print(f"✓ gTTS инициализирован (lang: {self.lang})")
        except ImportError:
            print("⚠ gtts не установлен")
            print("  Установите: pip install gtts")
            self.gTTS = None
        except Exception as e:
            print(f"⚠ Ошибка инициализации gTTS: {e}")
            self.gTTS = None
    
    def synthesize(self, text: str) -> np.ndarray:
        """
        Синтез речи с помощью gTTS.
        
        Args:
            text: Текст для синтеза
            
        Returns:
            Аудиосигнал
        """
        if self.gTTS is None:
            raise RuntimeError("gTTS не инициализирован")
        
        import tempfile
        import os
        import io
        
        # Создаём временный файл
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Синтез и сохранение
            tts = self.gTTS(text=text, lang=self.lang, slow=False)
            tts.save(tmp_path)
            
            # Пробуем использовать pydub, если доступен ffmpeg
            try:
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_mp3(tmp_path)
                audio_segment = audio_segment.set_frame_rate(self.sample_rate)
                audio_segment = audio_segment.set_channels(1)  # Моно
                
                # Конвертация в numpy
                audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                audio = audio / 32768.0  # Нормализация
            except Exception:
                # Fallback: используем простой способ (требует ffmpeg в системе)
                raise RuntimeError(
                    "gTTS требует ffmpeg для конвертации MP3. "
                    "Установите: brew install ffmpeg (macOS) или apt-get install ffmpeg (Linux)"
                )
            
            # Нормализация
            if len(audio) > 0:
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio))
            
            return audio.astype(np.float32)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class Pyttsx3TTS(TTSEngine):
    """
    pyttsx3 - простой TTS движок (работает без интернета).
    
    Установка: pip install pyttsx3
    """
    
    def __init__(self, sample_rate: int = 16000):
        super().__init__(sample_rate)
        self._init_engine()
    
    def _init_engine(self):
        """Инициализация pyttsx3."""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            
            # Настройки голоса
            voices = self.engine.getProperty('voices')
            if voices:
                # Пытаемся найти русский голос
                for voice in voices:
                    if 'russian' in voice.name.lower() or 'ru' in voice.id.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            # Скорость речи
            self.engine.setProperty('rate', 150)
            
            print("✓ pyttsx3 инициализирован")
        except Exception as e:
            print(f"⚠ Ошибка инициализации pyttsx3: {e}")
            print("  Установите: pip install pyttsx3")
            self.engine = None
    
    def synthesize(self, text: str) -> np.ndarray:
        """
        Синтез речи с помощью pyttsx3.
        
        Примечание: pyttsx3 работает через системные TTS движки,
        поэтому нужно сохранять во временный файл.
        
        Args:
            text: Текст для синтеза
            
        Returns:
            Аудиосигнал
        """
        if self.engine is None:
            raise RuntimeError("pyttsx3 не инициализирован")
        
        import tempfile
        import os
        import soundfile as sf
        from scipy.io import wavfile
        
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Синтез и сохранение
            self.engine.save_to_file(text, tmp_path)
            self.engine.runAndWait()
            
            # Загрузка аудио
            try:
                audio, sr = sf.read(tmp_path)
            except:
                # Fallback на scipy
                sr, audio = wavfile.read(tmp_path)
                audio = audio.astype(np.float32) / 32768.0
            
            # Ресемплинг если нужно
            if sr != self.sample_rate:
                from scipy import signal
                num_samples = int(len(audio) * self.sample_rate / sr)
                audio = signal.resample(audio, num_samples)
            
            # Нормализация
            if len(audio) > 0:
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio))
            
            return audio.astype(np.float32)
        finally:
            # Удаляем временный файл
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def create_tts_engine(engine_type: str = 'edge', **kwargs) -> Optional[TTSEngine]:
    """
    Создание TTS движка.
    
    Args:
        engine_type: Тип движка ('xtts', 'bark', 'edge', 'gtts', 'silero', 'pyttsx3')
                    'xtts' - Coqui XTTS (лучшая для эмоций, рекомендуется!)
                    'bark' - Bark (Suno AI, очень выразительная)
                    'edge' - Edge TTS (Microsoft, хорошее качество)
                    'gtts' - Google TTS (простой)
                    'silero' - Silero TTS (offline, но может не работать)
                    'pyttsx3' - pyttsx3 (offline, низкое качество)
        **kwargs: Дополнительные параметры
        
    Returns:
        TTS движок или None если не удалось создать
    """
    engine_type_lower = engine_type.lower()
    
    # Пробуем в порядке приоритета
    engines_to_try = []
    
    if engine_type_lower == 'xtts':
        engines_to_try = ['xtts', 'bark', 'edge', 'gtts', 'silero', 'pyttsx3']
    elif engine_type_lower == 'bark':
        engines_to_try = ['bark', 'xtts', 'edge', 'gtts', 'silero', 'pyttsx3']
    elif engine_type_lower == 'edge':
        engines_to_try = ['edge', 'xtts', 'bark', 'gtts', 'silero', 'pyttsx3']
    elif engine_type_lower == 'gtts':
        engines_to_try = ['gtts', 'xtts', 'bark', 'edge', 'silero', 'pyttsx3']
    elif engine_type_lower == 'silero':
        engines_to_try = ['silero', 'xtts', 'bark', 'edge', 'gtts', 'pyttsx3']
    elif engine_type_lower == 'pyttsx3':
        engines_to_try = ['pyttsx3', 'xtts', 'bark', 'edge', 'gtts', 'silero']
    else:
        engines_to_try = [engine_type_lower]
    
    for eng_type in engines_to_try:
        try:
            if eng_type == 'xtts':
                return CoquiXTTS(**kwargs)
            elif eng_type == 'bark':
                return BarkTTS(**kwargs)
            elif eng_type == 'edge':
                return EdgeTTS(**kwargs)
            elif eng_type == 'gtts':
                return GoogleTTS(**kwargs)
            elif eng_type == 'silero':
                return SileroTTS(**kwargs)
            elif eng_type == 'pyttsx3':
                return Pyttsx3TTS(**kwargs)
        except Exception as e:
            if eng_type == engine_type_lower:
                print(f"⚠ Не удалось создать {engine_type}: {e}")
            continue
    
    print(f"⚠ Не удалось создать ни один TTS движок")
    return None

