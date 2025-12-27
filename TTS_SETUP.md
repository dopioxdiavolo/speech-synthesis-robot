# Настройка TTS для естественного голоса

## Проблема с Silero TTS

Silero TTS может не загружаться из-за проблемы с quantization engine в PyTorch. Это известная проблема.

## Решения

### Вариант 1: Исправить проблему с qengine

```bash
# Установите правильный quantization engine
pip install fbgemm
```

Или установите переменную окружения:
```bash
export TORCH_QUANTIZATION_ENGINE=fbgemm
python3 tts_example.py
```

### Вариант 2: Использовать pyttsx3 (проще)

```bash
pip install pyttsx3
```

Затем в коде:
```python
from speech_synthesizer import SpeechSynthesizer

synthesizer = SpeechSynthesizer(
    use_tts=True,
    tts_engine='pyttsx3'  # Вместо 'silero'
)
```

### Вариант 3: Использовать артикуляторную модель

Система автоматически использует артикуляторную модель, если TTS не доступен:
```python
synthesizer = SpeechSynthesizer(use_tts=False)
```

## Текущий статус

Система работает с артикуляторной моделью (улучшенной версией). TTS можно подключить позже, когда будет решена проблема с quantization.

## Пример использования

```python
from speech_synthesizer import SpeechSynthesizer

# С TTS (если доступен)
synthesizer = SpeechSynthesizer(use_tts=True, tts_engine='silero')

# Без TTS (артикуляторная модель)
synthesizer = SpeechSynthesizer(use_tts=False)

# Синтез
audio = synthesizer.synthesize_phrase("Привет", style='friendly')
synthesizer.save_audio(audio, 'output.wav')
```

