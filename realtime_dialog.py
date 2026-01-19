"""
Система диалога в реальном времени.

Полный цикл:
1. Пользователь говорит → ASR распознаёт речь
2. Текст → определение эмоции
3. BehaviorController → координация голоса и мимики
4. Синтез ответа через F5-TTS или другой TTS
5. Воспроизведение ответа
6. Визуализация мимики
"""

import numpy as np
from typing import Optional, Callable
import time
import threading

from asr_module import SpeechRecognizer
from speech_synthesizer import SpeechSynthesizer


class RealtimeDialogSystem:
    """
    Система диалога в реальном времени.
    
    Интегрирует:
    - Распознавание речи (ASR)
    - Определение эмоций
    - BehaviorController
    - Синтез речи (F5-TTS или другой TTS)
    - Визуализацию мимики
    """
    
    def __init__(self,
                 asr_engine: str = 'speech_recognition',
                 tts_engine: str = 'f5-tts',  # или 'edge', 'silero'
                 use_f5_tts: bool = True,
                 facial_render_method: str = 'matplotlib'):
        """
        Инициализация системы диалога.
        
        Args:
            asr_engine: Движок ASR ('whisper', 'speech_recognition')
            tts_engine: Движок TTS ('f5-tts', 'edge', 'silero')
            use_f5_tts: Использовать F5-TTS Russian
            facial_render_method: Метод визуализации мимики
        """
        print("=" * 60)
        print("ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ДИАЛОГА В РЕАЛЬНОМ ВРЕМЕНИ")
        print("=" * 60)
        
        # Инициализация ASR
        print("\n1. Инициализация распознавания речи...")
        self.asr = SpeechRecognizer(asr_engine=asr_engine, language='ru')
        
        # Инициализация синтезатора
        print("\n2. Инициализация синтезатора речи...")
        tts_engine_name = 'f5-tts' if use_f5_tts else 'edge'
        self.synthesizer = SpeechSynthesizer(
            sample_rate=24000 if use_f5_tts else 16000,
            use_tts=True,
            tts_engine=tts_engine_name,
            use_vocal_folds=True,
            facial_render_method=facial_render_method
        )
        
        # Проверка доступности F5-TTS
        self.use_f5_tts = use_f5_tts and (
            self.synthesizer.tts_engine is not None and 
            hasattr(self.synthesizer.tts_engine, 'model') and
            self.synthesizer.tts_engine.model is not None
        )
        
        if use_f5_tts and not self.use_f5_tts:
            print("⚠ F5-TTS недоступен, используем Edge TTS")
        
        print("\n✓ Система готова к работе!")
        print("=" * 60)
    
    def process_user_input(self, text: str) -> tuple[Optional[np.ndarray], Optional]:
        """
        Обработка пользовательского ввода (текст).
        
        Args:
            text: Текст от пользователя
            
        Returns:
            (audio, facial_state) - синтезированный ответ
        """
        print(f"\n👤 Пользователь: {text}")
        
        # Определение эмоции и синтез ответа
        # Здесь можно добавить логику генерации ответа на основе текста
        response_text = self._generate_response(text)
        
        print(f"🤖 Робот: {response_text}")
        
        # Синтез ответа через синтезатор (поддерживает F5-TTS если доступен)
        audio, facial_state = self.synthesizer.synthesize_phrase(
            response_text,
            show_facial_expression=True
        )
        
        return audio, facial_state
    
    def _generate_response(self, user_text: str) -> str:
        """
        Генерация ответа на основе пользовательского ввода.
        
        Простая rule-based логика. Можно заменить на LLM.
        """
        user_text_lower = user_text.lower()
        
        # Простые правила для генерации ответов
        if any(word in user_text_lower for word in ['привет', 'здравствуй', 'добрый']):
            return "Привет! Как дела?"
        elif any(word in user_text_lower for word in ['как дела', 'как поживаешь']):
            return "У меня всё отлично, спасибо!"
        elif any(word in user_text_lower for word in ['спасибо', 'благодарю']):
            return "Пожалуйста, всегда рад помочь!"
        elif any(word in user_text_lower for word in ['пока', 'до свидания', 'прощай']):
            return "До свидания! Хорошего дня!"
        elif any(word in user_text_lower for word in ['помощь', 'помоги', 'подскажи']):
            return "Конечно, чем могу помочь?"
        else:
            # Эхо-ответ с эмоцией
            return f"Понял, вы сказали: {user_text}"
    
    def start_dialog(self):
        """
        Запуск диалога в реальном времени.
        
        Слушает микрофон и отвечает.
        """
        print("\n" + "=" * 60)
        print("НАЧАЛО ДИАЛОГА")
        print("=" * 60)
        print("Говорите в микрофон. Скажите 'стоп' для выхода.")
        print("-" * 60)
        
        def on_speech_recognized(text: str):
            """Обработчик распознанной речи."""
            if text.lower() in ['стоп', 'выход', 'закончить']:
                print("\n👋 Завершение диалога...")
                self.asr.stop_listening()
                return
            
            # Обработка ввода
            audio, facial_state = self.process_user_input(text)
            
            # Воспроизведение ответа
            if audio is not None:
                self._play_audio(audio)
        
        # Запуск прослушивания
        self.asr.start_listening(on_speech_recognized)
        
        # Ожидание завершения
        try:
            while self.asr.is_listening:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n⚠ Прервано пользователем")
            self.asr.stop_listening()
    
    def _play_audio(self, audio: np.ndarray):
        """
        Воспроизведение аудио.
        
        Args:
            audio: Аудиосигнал
        """
        try:
            import sounddevice as sd
            sd.play(audio, self.synthesizer.sample_rate)
            sd.wait()  # Ждём окончания воспроизведения
        except ImportError:
            print("⚠ sounddevice не установлен для воспроизведения")
            print("   Установите: pip install sounddevice")
            print("   Или сохраните аудио в файл")
        except Exception as e:
            print(f"⚠ Ошибка воспроизведения: {e}")


def main():
    """Главная функция для запуска диалога."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  МУЛЬТИМОДАЛЬНАЯ СИСТЕМА ДИАЛОГА В РЕАЛЬНОМ ВРЕМЕНИ      ║
    ╚══════════════════════════════════════════════════════════╝
    
    Система работает следующим образом:
    1. Вы говорите в микрофон
    2. Система распознаёт речь (ASR)
    3. Определяет эмоцию из текста
    4. BehaviorController координирует голос и мимику
    5. Синтезирует ответ через F5-TTS или Edge TTS
    6. Воспроизводит ответ и показывает мимику
    
    Требования:
    - Микрофон подключён
    - Для F5-TTS: GPU рекомендуется
    - Для ASR: pip install SpeechRecognition (или openai-whisper)
    """)
    
    # Создание системы
    # Для реального времени лучше использовать 'emoji' или 'ascii' вместо matplotlib
    # matplotlib может вызывать проблемы в фоновых потоках на macOS
    dialog = RealtimeDialogSystem(
        asr_engine='speech_recognition',  # или 'whisper'
        use_f5_tts=False,  # Установите True если есть F5-TTS
        facial_render_method='emoji'  # Изменено на emoji для избежания проблем с потоками
    )
    
    # Запуск диалога
    dialog.start_dialog()


if __name__ == "__main__":
    main()
