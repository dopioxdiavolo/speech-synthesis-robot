# Инструкция по установке

## Решение проблемы с установкой

Если вы столкнулись с ошибкой при установке зависимостей (например, с `googleads`), это не связано с нашими зависимостями напрямую.

### Способ 1: Установка по одной зависимости

Установите зависимости по одной, чтобы избежать конфликтов:

```bash
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install soundfile>=0.10.0
pip install torch>=1.9.0
pip install parselmouth>=0.4.0
```

### Способ 2: Обновление pip и setuptools

Сначала обновите pip и setuptools:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Способ 3: Установка parselmouth отдельно

Parselmouth может требовать системные зависимости. Установите их сначала:

**macOS:**
```bash
brew install praat
pip install parselmouth
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install praat
pip install parselmouth
```

**Windows:**
1. Скачайте Praat с https://www.fon.hum.uva.nl/praat/
2. Установите его
3. Убедитесь, что Praat доступен в PATH
4. `pip install parselmouth`

### Способ 4: Использование виртуального окружения (рекомендуется)

Создайте чистое виртуальное окружение:

```bash
python3 -m venv venv
source venv/bin/activate  # На macOS/Linux
# или
venv\Scripts\activate  # На Windows

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Минимальная установка (без parselmouth)

Если у вас проблемы с установкой parselmouth (особенно на Python 3.12+), вы можете использовать систему без модуля переноса просодии:

```bash
pip install numpy scipy torch soundfile
```

В этом случае функция `synthesize_with_prosody_transfer` не будет работать, но остальной функционал будет доступен.

**Важно:** Система полностью функциональна без parselmouth! Только перенос просодии из референсной речи требует parselmouth.

### Важно: правильный пакет parselmouth

**Используйте `praat-parselmouth`, а не `parselmouth`!**

- ✅ `praat-parselmouth` - правильный пакет для работы с Praat (совместим с Python 3.12+)
- ❌ `parselmouth` - другой пакет для рекламных платформ (не нужен для этого проекта)

**Установка:**
```bash
pip install praat-parselmouth
```

Этот пакет полностью совместим с Python 3.12+ и не требует дополнительных зависимостей.

### Проверка установки

После установки проверьте:

```python
import numpy
import scipy
import torch
import soundfile
try:
    import parselmouth
    print("✓ Все зависимости установлены")
except ImportError:
    print("⚠ parselmouth не установлен (перенос просодии недоступен)")
```

