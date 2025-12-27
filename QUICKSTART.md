# Быстрый старт

## ✅ Система готова к использованию!

Все зависимости установлены и протестированы.

## 🚀 Как запустить

### 0. Установка TTS для естественного голоса (рекомендуется)

```bash
pip install edge-tts
```

### 1. Простой пример с естественным голосом

```bash
python3 final_tts_example.py
```

Создаст файлы с **естественным голосом** (Edge TTS).

### 2. Простой пример (артикуляторная модель)

```bash
python3 example_usage.py
```

Создаст 4 файла:
- `example_neutral.wav` - нейтральная речь
- `example_friendly.wav` - дружелюбная речь
- `example_high_f0.wav` - речь с высоким тоном
- `example_vowel_a.wav` - синтез гласного 'а'

### 2. Полная демонстрация (все возможности)

```bash
python3 demo.py
```

Создаст множество примеров в папке `output/`:
- Синтез гласных с разными F0
- Фразы в разных стилях (neutral, friendly, strict, warning)
- Примеры управления просодией

### 3. Использование в своём коде

```python
from speech_synthesizer import SpeechSynthesizer

# Создание синтезатора
synthesizer = SpeechSynthesizer(sample_rate=16000)

# Синтез фразы в определённом стиле
audio = synthesizer.synthesize_phrase(
    "Привет, как дела?",
    style='friendly'  # или 'neutral', 'strict', 'warning'
)

# Сохранение результата
synthesizer.save_audio(audio, 'my_output.wav')
```

## 📁 Где находятся результаты

- Простые примеры: в корневой директории (`example_*.wav`)
- Демонстрация: в папке `output/`

## 🎛️ Доступные стили речи

- `neutral` - нейтральный
- `friendly` - дружелюбный (выше F0, быстрее)
- `strict` - строгий (ниже F0, медленнее)
- `warning` - предупреждение (высокий F0, медленнее)

## 🔧 Дополнительные возможности

### Управление F0 напрямую

```python
audio = synthesizer.synthesize_phrase(
    "Тест",
    style='neutral',
    base_f0=200.0  # Высокий тон
)
```

### Изменение темпа

```python
audio = synthesizer.synthesize_phrase(
    "Тест",
    style='neutral',
    duration_factor=1.5  # Медленнее в 1.5 раза
)
```

### Перенос просодии из референсной речи

```python
# Требуется файл reference.wav
audio = synthesizer.synthesize_with_prosody_transfer(
    "Новая фраза",
    reference_audio='reference.wav',
    duration_factor=1.0
)
```

## 🎵 Прослушивание результатов

Откройте созданные `.wav` файлы в любом аудиоплеере:
- macOS: QuickTime, VLC, или просто двойной клик
- Linux: VLC, Audacity, или `aplay file.wav`
- Windows: Windows Media Player, VLC

## ❓ Проблемы?

Если что-то не работает:
1. Убедитесь, что все зависимости установлены: `pip install -r requirements.txt`
2. Проверьте версию Python: `python3 --version` (должна быть 3.10+)
3. См. `INSTALL.md` для подробной информации

## 📚 Дополнительная информация

- `README.md` - полная документация
- `INSTALL.md` - инструкции по установке
- `demo.py` - исходный код демонстрации
- `example_usage.py` - простые примеры

