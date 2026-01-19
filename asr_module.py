"""
Модуль распознавания речи (ASR - Automatic Speech Recognition).

Поддерживает различные ASR движки для распознавания речи в реальном времени:
- whisper (OpenAI Whisper) - точное, но медленное
- vosk - быстрый, offline
- speech_recognition - обёртка для разных API
"""

import numpy as np
from typing import Optional, Callable
import threading
import queue


class SpeechRecognizer:
    """
    Распознаватель речи для работы в реальном времени.
    
    Поддерживает различные ASR движки и режимы работы:
    - Реальное время с микрофона
    - Обработка аудиофайлов
    - Callback при распознавании
    """
    
    def __init__(self, asr_engine: str = 'whisper', language: str = 'ru'):
        """
        Инициализация распознавателя речи.
        
        Args:
            asr_engine: Движок ASR ('whisper', 'vosk', 'speech_recognition')
            language: Язык распознавания ('ru', 'en')
        """
        self.asr_engine = asr_engine
        self.language = language
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.recognition_thread = None
        
        # Инициализация выбранного движка
        if asr_engine == 'whisper':
            self._init_whisper()
        elif asr_engine == 'vosk':
            self._init_vosk()
        elif asr_engine == 'speech_recognition':
            self._init_speech_recognition()
        else:
            raise ValueError(f"Неизвестный ASR движок: {asr_engine}")
    
    def _init_whisper(self):
        """Инициализация Whisper."""
        try:
            import whisper
            self.whisper_model = whisper.load_model("base")
            self.whisper_available = True
            print("✓ Whisper ASR инициализирован")
        except ImportError:
            self.whisper_available = False
            print("⚠ Whisper не установлен. Установите: pip install openai-whisper")
        except Exception as e:
            self.whisper_available = False
            print(f"⚠ Ошибка загрузки Whisper: {e}")
    
    def _init_vosk(self):
        """Инициализация Vosk."""
        try:
            import vosk
            import json
            # Нужно скачать модель с https://alphacephei.com/vosk/models
            # model_path = "path/to/vosk-model-ru"
            # self.vosk_model = vosk.Model(model_path)
            # self.vosk_recognizer = vosk.KaldiRecognizer(self.vosk_model, 16000)
            self.vosk_available = False
            print("⚠ Vosk требует загрузки модели. См. https://alphacephei.com/vosk/models")
        except ImportError:
            self.vosk_available = False
            print("⚠ Vosk не установлен. Установите: pip install vosk")
    
    def _init_speech_recognition(self):
        """Инициализация speech_recognition."""
        try:
            import speech_recognition as sr
            self.sr_recognizer = sr.Recognizer()
            # Микрофон инициализируем только при необходимости (при start_listening)
            # чтобы избежать ошибки если PyAudio не установлен
            self.sr_microphone = None
            self.sr_available = True
            print("✓ Speech Recognition инициализирован")
        except ImportError:
            self.sr_available = False
            print("⚠ speech_recognition не установлен. Установите: pip install SpeechRecognition")
        except Exception as e:
            self.sr_available = False
            print(f"⚠ Ошибка инициализации speech_recognition: {e}")
    
    def recognize_from_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """
        Распознавание речи из аудиомассива.
        
        Args:
            audio: Аудиосигнал (numpy array)
            sample_rate: Частота дискретизации
            
        Returns:
            Распознанный текст или None
        """
        if self.asr_engine == 'whisper' and self.whisper_available:
            return self._recognize_whisper(audio, sample_rate)
        elif self.asr_engine == 'speech_recognition' and self.sr_available:
            return self._recognize_sr(audio, sample_rate)
        else:
            print("⚠ ASR движок недоступен")
            return None
    
    def _recognize_whisper(self, audio: np.ndarray, sample_rate: int) -> Optional[str]:
        """Распознавание через Whisper."""
        try:
            import whisper
            import soundfile as sf
            import tempfile
            import os
            
            # Сохраняем во временный файл
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio, sample_rate)
                result = self.whisper_model.transcribe(
                    tmp_file.name,
                    language=self.language
                )
                os.unlink(tmp_file.name)
                return result['text'].strip()
        except Exception as e:
            print(f"⚠ Ошибка распознавания Whisper: {e}")
            return None
    
    def _recognize_sr(self, audio: np.ndarray, sample_rate: int) -> Optional[str]:
        """Распознавание через speech_recognition."""
        try:
            import speech_recognition as sr
            import io
            
            # Конвертируем numpy array в AudioData
            audio_data = sr.AudioData(
                audio.tobytes(),
                sample_rate,
                2  # 16-bit
            )
            
            # Распознаём (использует Google API по умолчанию)
            text = self.sr_recognizer.recognize_google(
                audio_data,
                language=f"{self.language}-RU" if self.language == 'ru' else "en-US"
            )
            return text
        except Exception as e:
            print(f"⚠ Ошибка распознавания: {e}")
            return None
    
    def start_listening(self, callback: Callable[[str], None]):
        """
        Начало прослушивания микрофона в реальном времени.
        
        Args:
            callback: Функция, вызываемая при распознавании текста
        """
        if self.is_listening:
            print("⚠ Уже слушаем микрофон")
            return
        
        if self.asr_engine == 'speech_recognition' and self.sr_available:
            self._start_listening_sr(callback)
        else:
            print("⚠ Реальное время поддерживается только для speech_recognition")
            print("   Используйте recognize_from_audio() для обработки файлов")
    
    def _start_listening_sr(self, callback: Callable[[str], None]):
        """Запуск прослушивания через speech_recognition."""
        import speech_recognition as sr
        
        # Инициализируем микрофон только сейчас (если ещё не инициализирован)
        if self.sr_microphone is None:
            try:
                self.sr_microphone = sr.Microphone()
            except Exception as e:
                print(f"⚠ Ошибка инициализации микрофона: {e}")
                print("   Установите PyAudio:")
                print("   - На macOS: brew install portaudio && pip install pyaudio")
                print("   - На Linux: sudo apt-get install portaudio19-dev && pip install pyaudio")
                print("   - На Windows: pip install pyaudio")
                print("   Или используйте recognize_from_audio() для обработки файлов")
                self.is_listening = False
                return
        
        def listen_loop():
            self.is_listening = True
            with self.sr_microphone as source:
                self.sr_recognizer.adjust_for_ambient_noise(source)
                print("🎤 Слушаю микрофон... (скажите 'стоп' для остановки)")
                
                while self.is_listening:
                    try:
                        audio = self.sr_recognizer.listen(source, timeout=1, phrase_time_limit=5)
                        text = self.sr_recognizer.recognize_google(
                            audio,
                            language=f"{self.language}-RU" if self.language == 'ru' else "en-US"
                        )
                        if text:
                            print(f"🎤 Распознано: {text}")
                            callback(text)
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        continue
                    except Exception as e:
                        if self.is_listening:
                            print(f"⚠ Ошибка: {e}")
        
        self.recognition_thread = threading.Thread(target=listen_loop, daemon=True)
        self.recognition_thread.start()
    
    def stop_listening(self):
        """Остановка прослушивания микрофона."""
        self.is_listening = False
        if self.recognition_thread:
            self.recognition_thread.join(timeout=2.0)
