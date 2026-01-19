# Инструкция по установке F5-TTS и настройке микрофона

## 📦 Установка F5-TTS

### Шаг 1: Установка основной библиотеки F5-TTS

```bash
pip install f5-tts
```

### Шаг 2: Дополнительные зависимости для работы с аудио

```bash
# Базовые библиотеки для обработки аудио
pip install numpy scipy soundfile

# Для работы с микрофоном (обязательно!)
pip install sounddevice
pip install pyaudio
```

### Шаг 3: Установка PyAudio на macOS (если возникают проблемы)

PyAudio может требовать дополнительной установки системных библиотек:

```bash
# Через Homebrew (рекомендуется)
brew install portaudio
pip install pyaudio

# Или через conda
conda install pyaudio
```

### Шаг 4: Для распознавания речи (ASR)

**Вариант 1: SpeechRecognition (рекомендуется)**
```bash
pip install SpeechRecognition
```

**Вариант 2: Whisper (точнее, но медленнее)**
```bash
pip install openai-whisper
```

### Шаг 5: Для визуализации мимики (опционально)

```bash
# Matplotlib (рекомендуется)
pip install matplotlib

# Или Pygame (интерактивная анимация)
pip install pygame
```

## 🎤 Настройка микрофона ноутбука

### Проверка доступности микрофона

Создайте тестовый скрипт `test_microphone.py`:

```python
import sounddevice as sd
import numpy as np

# Проверка доступных устройств
print("Доступные аудио устройства:")
print(sd.query_devices())

# Проверка записи с микрофона
print("\nЗапись 3 секунд... Говорите в микрофон!")
duration = 3  # секунды
sample_rate = 16000

recording = sd.rec(int(duration * sample_rate), 
                   samplerate=sample_rate, 
                   channels=1, 
                   dtype='float32')
sd.wait()  # Ждём окончания записи

print("✓ Запись завершена!")
print(f"Уровень сигнала: {np.max(np.abs(recording)):.3f}")
```

Запустите:
```bash
python test_microphone.py
```

### Настройка микрофона в системе

#### macOS:
1. Откройте **Системные настройки** → **Безопасность и конфиденциальность** → **Микрофон**
2. Убедитесь, что Terminal/Python имеет доступ к микрофону
3. Проверьте уровень громкости микрофона в настройках системы

#### Windows:
1. Откройте **Параметры** → **Конфиденциальность** → **Микрофон**
2. Включите доступ к микрофону для приложений
3. Проверьте уровень громкости в настройках звука

#### Linux:
1. Проверьте права доступа к `/dev/audio` или `/dev/dsp`
2. Убедитесь, что пользователь в группе `audio`:
   ```bash
   sudo usermod -a -G audio $USER
   ```

## 🚀 Запуск системы с F5-TTS и микрофоном

### Вариант 1: Диалог в реальном времени

Откройте файл `realtime_dialog.py` и убедитесь, что настройки правильные:

```python
dialog = RealtimeDialogSystem(
    asr_engine='speech_recognition',  # или 'whisper'
    use_f5_tts=True,  # Использовать F5-TTS
    facial_render_method='matplotlib'  # или 'emoji', 'pygame'
)
```

Затем запустите:
```bash
python realtime_dialog.py
```

### Вариант 2: Программное использование

```python
from speech_synthesizer import SpeechSynthesizer
import sounddevice as sd
import numpy as np

# Создание синтезатора с F5-TTS
synthesizer = SpeechSynthesizer(
    sample_rate=24000,  # F5-TTS использует 24000
    use_tts=True,
    tts_engine='f5-tts'
)

# Запись с микрофона
print("Говорите в микрофон (3 секунды)...")
duration = 3
sample_rate = 16000
recording = sd.rec(int(duration * sample_rate), 
                   samplerate=sample_rate, 
                   channels=1)
sd.wait()

# Здесь можно добавить распознавание речи
# text = recognize_speech(recording)

# Синтез ответа через F5-TTS
text = "Привет, это ответ от F5-TTS!"
audio = synthesizer.synthesize_phrase(text, style='friendly')

# Воспроизведение ответа
sd.play(audio, samplerate=24000)
sd.wait()

# Сохранение
synthesizer.save_audio(audio, 'response.wav')
```

## 🔧 Решение проблем

### Проблема: PyAudio не устанавливается

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**Windows:**
```bash
# Скачайте wheel файл с https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
pip install PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl
```

### Проблема: Микрофон не работает

1. **Проверьте доступ к микрофону в настройках системы**
2. **Проверьте, что микрофон не занят другим приложением**
3. **Попробуйте указать устройство явно:**

```python
import sounddevice as sd

# Список устройств
print(sd.query_devices())

# Использование конкретного устройства (замените индекс)
device_id = 1  # Индекс вашего микрофона из списка
recording = sd.rec(..., device=device_id)
```

### Проблема: F5-TTS не загружается

1. **Проверьте установку:**
   ```bash
   pip show f5-tts
   ```

2. **Проверьте наличие GPU (рекомендуется):**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Первая загрузка модели может занять время** (скачивание с HuggingFace)

4. **Если нет GPU, можно использовать CPU:**
   ```python
   engine = F5TTSEngine(device='cpu')  # Будет медленнее
   ```

### Проблема: Низкое качество записи

Увеличьте частоту дискретизации:
```python
sample_rate = 44100  # Вместо 16000
```

## 📋 Полный список команд для установки

Скопируйте и выполните все команды подряд:

```bash
# 1. F5-TTS и базовые библиотеки
pip install f5-tts numpy scipy soundfile

# 2. Работа с микрофоном
pip install sounddevice

# 3. PyAudio (может потребоваться установка системных библиотек)
# macOS:
brew install portaudio
pip install pyaudio

# Linux:
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio

# 4. Распознавание речи
pip install SpeechRecognition

# 5. Визуализация (опционально)
pip install matplotlib
# или
pip install pygame
```

## ✅ Проверка установки

Создайте файл `check_installation.py`:

```python
print("Проверка установки...")

# Проверка F5-TTS
try:
    from f5_tts import F5TTS
    print("✓ F5-TTS установлен")
except ImportError:
    print("✗ F5-TTS не установлен: pip install f5-tts")

# Проверка sounddevice
try:
    import sounddevice as sd
    print("✓ sounddevice установлен")
    print(f"  Доступные устройства: {len(sd.query_devices())}")
except ImportError:
    print("✗ sounddevice не установлен: pip install sounddevice")

# Проверка pyaudio
try:
    import pyaudio
    print("✓ pyaudio установлен")
except ImportError:
    print("✗ pyaudio не установлен: pip install pyaudio")

# Проверка SpeechRecognition
try:
    import speech_recognition as sr
    print("✓ SpeechRecognition установлен")
except ImportError:
    print("✗ SpeechRecognition не установлен: pip install SpeechRecognition")

print("\nПроверка завершена!")
```

Запустите:
```bash
python check_installation.py
```

## 🎯 Быстрый старт

После установки всех зависимостей:

```bash
# 1. Запуск диалога в реальном времени
python realtime_dialog.py

# 2. Или использование в своём коде
python -c "
from speech_synthesizer import SpeechSynthesizer
s = SpeechSynthesizer(use_tts=True, tts_engine='f5-tts')
audio = s.synthesize_phrase('Привет!')
s.save_audio(audio, 'test.wav')
print('Готово!')
"
```

## 📚 Полезные ссылки

- F5-TTS Russian: https://huggingface.co/hotstone228/F5-TTS-Russian
- F5-TTS документация: https://f5-tts.github.io/
- sounddevice документация: https://python-sounddevice.readthedocs.io/
- SpeechRecognition: https://github.com/Uberi/speech_recognition
