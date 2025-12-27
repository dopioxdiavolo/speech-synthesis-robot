# Руководство по TTS моделям

## ✅ Рекомендуется: Edge TTS (Microsoft)

**Преимущества:**
- ✅ Качественный естественный голос
- ✅ Бесплатный
- ✅ Простая установка
- ✅ Поддерживает русский язык
- ✅ Работает стабильно

**Установка:**
```bash
pip install edge-tts
```

**Использование:**
```python
from speech_synthesizer import SpeechSynthesizer

synthesizer = SpeechSynthesizer(
    use_tts=True,
    tts_engine='edge'
)

audio = synthesizer.synthesize_phrase("Привет!", style='friendly')
synthesizer.save_audio(audio, 'output.wav')
```

**Доступные голоса:**
- `ru-RU-SvetlanaNeural` (женский, по умолчанию)
- `ru-RU-DmitryNeural` (мужской)
- `ru-RU-DariyaNeural` (женский)

## Альтернативы

### Google TTS (gTTS)

**Установка:**
```bash
pip install gtts pydub
# Также нужен ffmpeg: brew install ffmpeg (macOS)
```

**Использование:**
```python
synthesizer = SpeechSynthesizer(
    use_tts=True,
    tts_engine='gtts'
)
```

### Silero TTS (offline)

**Проблема:** Может не работать из-за проблем с quantization в PyTorch.

**Установка:**
```bash
pip install torch torchaudio
```

**Использование:**
```python
synthesizer = SpeechSynthesizer(
    use_tts=True,
    tts_engine='silero'
)
```

### pyttsx3 (offline, низкое качество)

**Установка:**
```bash
pip install pyttsx3
```

**Использование:**
```python
synthesizer = SpeechSynthesizer(
    use_tts=True,
    tts_engine='pyttsx3'
)
```

## Автоматический выбор

Система автоматически пробует разные движки в порядке приоритета:

```python
synthesizer = SpeechSynthesizer(use_tts=True, tts_engine='edge')
# Автоматически попробует: edge → gtts → silero → pyttsx3
```

## Сравнение

| Модель | Качество | Требует интернет | Установка | Рекомендация |
|--------|----------|------------------|-----------|--------------|
| Edge TTS | ⭐⭐⭐⭐⭐ | Да | Простая | ✅ **Рекомендуется** |
| Google TTS | ⭐⭐⭐⭐ | Да | Средняя | ✅ Хорошая альтернатива |
| Silero TTS | ⭐⭐⭐⭐ | Нет | Сложная | ⚠️ Может не работать |
| pyttsx3 | ⭐⭐ | Нет | Простая | ❌ Низкое качество |

## Итог

**Для лучшего качества используйте Edge TTS:**
```bash
pip install edge-tts
python3 final_tts_example.py
```

