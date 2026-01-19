# Настройка системы диалога в реальном времени

## 🎯 Что делает система

Полный цикл работы в реальном времени:

```
1. Пользователь говорит в микрофон
   ↓
2. ASR (распознавание речи) → текст
   ↓
3. Определение эмоции из текста
   ↓
4. BehaviorController → координация голоса и мимики
   ↓
5. Синтез ответа через F5-TTS или Edge TTS
   ↓
6. Воспроизведение ответа + визуализация мимики
```

## 📦 Установка зависимостей

### Базовые зависимости (обязательно):

```bash
# Основные библиотеки
pip install numpy scipy soundfile

# Для воспроизведения аудио
pip install sounddevice
```

### Для распознавания речи (ASR):

**Вариант 1: SpeechRecognition (рекомендуется для реального времени)**
```bash
pip install SpeechRecognition
pip install pyaudio  # Для работы с микрофоном
# Требует интернет для Google API
```

**Важно:** PyAudio может требовать дополнительной установки на macOS:
```bash
# Через Homebrew (рекомендуется)
brew install portaudio
pip install pyaudio

# Или через conda
conda install pyaudio
```

**Вариант 2: Whisper (точнее, но медленнее)**
```bash
pip install openai-whisper
# Требует больше ресурсов, но работает offline
```

**Вариант 3: Vosk (быстрый, offline)**
```bash
pip install vosk
# Нужно скачать модель: https://alphacephei.com/vosk/models
```

### Для синтеза речи (TTS):

**F5-TTS Russian (рекомендуется):**
```bash
pip install f5-tts
# Требует GPU для реального времени
# Модель загрузится автоматически при первом запуске
```

**Edge TTS (fallback, не требует GPU):**
```bash
pip install edge-tts
```

### Для визуализации мимики:

```bash
# Matplotlib (рекомендуется)
pip install matplotlib

# Или Pygame (интерактивная анимация)
pip install pygame
```

## 🚀 Запуск системы

### Базовый запуск (с Edge TTS):

```bash
python realtime_dialog.py
```

### С F5-TTS Russian:

Откройте `realtime_dialog.py` и измените:

```python
dialog = RealtimeDialogSystem(
    asr_engine='speech_recognition',
    use_f5_tts=True,  # Использовать F5-TTS
    facial_render_method='matplotlib'
)
```

Затем запустите:
```bash
python realtime_dialog.py
```

## ⚙️ Настройка

### Выбор ASR движка:

В `realtime_dialog.py`:

```python
dialog = RealtimeDialogSystem(
    asr_engine='speech_recognition',  # или 'whisper'
    ...
)
```

**Сравнение ASR движков:**

| Движок | Скорость | Точность | Требования |
|--------|----------|----------|------------|
| speech_recognition | ⚡⚡⚡ Быстрый | ⭐⭐⭐ Хорошая | Интернет |
| whisper | ⚡ Медленный | ⭐⭐⭐⭐⭐ Отличная | GPU/CPU |
| vosk | ⚡⚡⚡ Очень быстрый | ⭐⭐⭐ Хорошая | Offline |

### Выбор TTS движка:

**F5-TTS Russian:**
- ✅ Высокое качество
- ✅ Работает в реальном времени (RTF < 1.0 на GPU)
- ❌ Требует GPU для оптимальной работы
- ❌ Первая загрузка модели занимает время

**Edge TTS:**
- ✅ Быстрый запуск
- ✅ Не требует GPU
- ✅ Хорошее качество
- ❌ Требует интернет

## 🎤 Использование

1. **Запустите систему:**
   ```bash
   python realtime_dialog.py
   ```

2. **Говорите в микрофон:**
   - Система автоматически распознаёт речь
   - Определяет эмоцию
   - Генерирует ответ

3. **Для выхода:**
   - Скажите "стоп", "выход" или "закончить"
   - Или нажмите Ctrl+C

## 🔧 Примеры использования

### Простой диалог:

```python
from realtime_dialog import RealtimeDialogSystem

# Создание системы
dialog = RealtimeDialogSystem(
    asr_engine='speech_recognition',
    use_f5_tts=False,  # Используем Edge TTS
    facial_render_method='matplotlib'
)

# Запуск диалога
dialog.start_dialog()
```

### С F5-TTS и Whisper:

```python
dialog = RealtimeDialogSystem(
    asr_engine='whisper',  # Точное распознавание
    use_f5_tts=True,       # F5-TTS для синтеза
    facial_render_method='pygame'  # Интерактивная анимация
)

dialog.start_dialog()
```

### Обработка конкретного текста (без микрофона):

```python
dialog = RealtimeDialogSystem()

# Обработка текста
audio, facial_state = dialog.process_user_input("Привет, как дела?")

# Сохранение результата
dialog.synthesizer.save_audio(audio, "response.wav")
```

## 📝 Настройка генерации ответов

По умолчанию используется простая rule-based логика в `_generate_response()`.

Для более умных ответов можно:

1. **Интегрировать LLM (GPT, Claude и т.д.):**
   ```python
   def _generate_response(self, user_text: str) -> str:
       # Использовать LLM API
       response = llm_api.generate(user_text)
       return response
   ```

2. **Использовать предобученную модель диалога:**
   - DialoGPT
   - BlenderBot
   - Или другие модели

## 🐛 Решение проблем

### Микрофон не работает:

1. Проверьте подключение микрофона
2. Убедитесь, что микрофон не занят другим приложением
3. На macOS может потребоваться разрешение на доступ к микрофону

### F5-TTS не загружается:

1. Проверьте наличие GPU (рекомендуется)
2. Убедитесь, что установлен: `pip install f5-tts`
3. Первая загрузка модели может занять время (скачивание с HuggingFace)

### Распознавание не работает:

1. Проверьте интернет (для speech_recognition с Google API)
2. Для Whisper: убедитесь, что модель загружена
3. Проверьте уровень громкости микрофона

### Воспроизведение не работает:

1. Установите: `pip install sounddevice`
2. Проверьте настройки аудио системы
3. Альтернатива: сохраняйте в файл и воспроизводите отдельно

## 📊 Производительность

### Рекомендуемые конфигурации:

**Минимальная (CPU):**
- ASR: speech_recognition
- TTS: Edge TTS
- Задержка: ~1-2 секунды

**Оптимальная (GPU):**
- ASR: whisper (или speech_recognition)
- TTS: F5-TTS Russian
- Задержка: < 1 секунда

**Максимальная (GPU + оптимизация):**
- ASR: vosk (offline, быстрый)
- TTS: F5-TTS с оптимизацией (меньше шагов)
- Задержка: < 0.5 секунды

## 🔗 Полезные ссылки

- F5-TTS Russian: https://huggingface.co/hotstone228/F5-TTS-Russian
- F5-TTS документация: https://f5-tts.github.io/
- Whisper: https://github.com/openai/whisper
- SpeechRecognition: https://github.com/Uberi/speech_recognition
