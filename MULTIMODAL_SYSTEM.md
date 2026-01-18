# Мультимодальная система синтеза речи для социального робота

## Обзор

Система объединяет физиологически обоснованную модель голосовых связок и визуальную экспрессию (мимику) для создания мультимодального синтеза речи.

## Компоненты системы

### 1. Модель голосовых связок (`vocal_folds.py`)

Физиологически обоснованная модель фонации на уровне источника (source-level) для source-filter архитектуры.

#### Параметры:

- **F0 (fundamental frequency)**: основная частота вибрации голосовых связок
  - Физиологический смысл: зависит от длины, массы и натяжения голосовых связок, а также от подглоточного давления воздуха
  
- **Jitter**: вариабельность F0 между циклами вибрации
  - Физиологический смысл: отражает естественные микро-вариации из-за нелинейной динамики голосовых связок
  - Нормальные значения: 0.5-1.5%
  - Повышенный jitter может указывать на патологию или эмоциональное состояние

- **Shimmer**: вариабельность амплитуды между циклами
  - Физиологический смысл: отражает изменения амплитуды вибрации из-за вариаций в потоке воздуха
  - Нормальные значения: 3-5%
  - Повышенный shimmer связан с напряжением или патологией

- **Phonation Type (тип фонации)**:
  - `modal`: нормальная фонация, полное смыкание связок, богатый спектр
  - `breathy`: придыхательная фонация, неполное смыкание, больше шума
  - `pressed`: сжатая фонация, избыточное смыкание, напряжённый звук

#### Использование:

```python
from vocal_folds import VocalFoldsModel, PhonationType

model = VocalFoldsModel(sample_rate=16000)

# Генерация источника с параметрами
source = model.generate_source(
    duration=1.0,
    f0=150.0,
    jitter=0.01,  # 1%
    shimmer=0.03,  # 3%
    phonation_type=PhonationType.MODAL,
    vocal_effort=1.0
)
```

### 2. Визуализация мимики (`facial_expression.py`)

Модуль для отображения состояния экспрессии робота в реальном времени.

#### Состояния:

- `neutral`: нейтральное выражение
- `friendly`: дружелюбное выражение
- `strict`: строгое выражение
- `warning`: предупреждающее выражение

#### Методы визуализации:

- `emoji`: простая текстовая визуализация через emoji (по умолчанию)
- `matplotlib`: статичная визуализация через matplotlib
- `pygame`: интерактивная анимация через pygame
- `streamlit`: веб-интерфейс через streamlit

#### Использование:

```python
from facial_expression import FacialExpressionRenderer, ExpressionState

renderer = FacialExpressionRenderer(render_method='emoji')
renderer.set_expression(ExpressionState.FRIENDLY)
expression_display = renderer.render()
print(expression_display)  # 😊 FRIENDLY
```

### 3. Контроллер поведения (`behavior_controller.py`)

Координирует состояние голоса и мимики на основе стиля речи и внутреннего состояния робота.

#### Входные параметры:

- `style`: стиль речи ('neutral', 'friendly', 'strict', 'warning')
- `internal_state`: внутреннее состояние робота (опционально)
  - `energy`: уровень энергии (0.0-2.0)
  - `stress`: уровень стресса (0.0-1.0)
  - `mood`: настроение ('positive', 'negative', 'neutral')

#### Выходные параметры:

- `voice_state`: параметры голосовых связок
- `facial_state`: состояние мимики

#### Использование:

```python
from behavior_controller import BehaviorController

controller = BehaviorController()

# Вычисление поведения
voice_params, facial_state = controller.compute_behavior(
    style='friendly',
    internal_state={'energy': 1.2, 'stress': 0.1, 'mood': 'positive'}
)

print(f"F0: {voice_params['f0_base']} Гц")
print(f"Phonation type: {voice_params['phonation_type']}")
print(f"Facial state: {facial_state.value}")
```

### 4. Обновлённый SpeechSynthesizer

Интегрирует все компоненты мультимодальной системы.

#### Инициализация:

```python
from speech_synthesizer import SpeechSynthesizer

synthesizer = SpeechSynthesizer(
    sample_rate=16000,
    use_tts=False,  # Использовать TTS или артикуляторную модель
    use_vocal_folds=True,  # Использовать модель голосовых связок
    facial_render_method='emoji'  # Метод визуализации мимики
)
```

#### Синтез с мимикой:

```python
# Синтез с автоматическим определением эмоции и визуализацией мимики
audio, facial_state = synthesizer.synthesize_phrase(
    "Привет, как дела?",
    show_facial_expression=True
)

# Сохранение аудио
synthesizer.save_audio(audio, "output.wav")
print(f"Состояние мимики: {facial_state.value}")
```

#### Синтез с внутренним состоянием:

```python
# Влияние внутреннего состояния робота на голос и мимику
audio, facial_state = synthesizer.synthesize_phrase(
    "Привет!",
    style='neutral',
    internal_state={
        'energy': 1.5,  # Высокая энергия → выше F0
        'stress': 0.3,  # Низкий стресс → меньше jitter
        'mood': 'positive'  # Положительное настроение → friendly мимика
    }
)
```

## Примеры использования

### Базовый пример

```python
from speech_synthesizer import SpeechSynthesizer

synthesizer = SpeechSynthesizer(
    sample_rate=16000,
    use_vocal_folds=True,
    facial_render_method='emoji'
)

texts = [
    "Привет, как дела?",
    "Стоп! Прекрати!",
    "Внимание! Опасность!",
    "Сегодня обычный день."
]

for text in texts:
    audio, facial_state = synthesizer.synthesize_phrase(text)
    print(f"Текст: {text}")
    print(f"Мимика: {facial_state.value}")
    synthesizer.save_audio(audio, f"output_{texts.index(text)}.wav")
```

### Демонстрация физиологических параметров

См. файл `multimodal_demo.py` для полной демонстрации всех возможностей системы.

## Физиологический смысл параметров

### F0 (Fundamental Frequency)

- **Низкий F0 (100-130 Гц)**: расслабленные связки, низкое подглоточное давление
- **Средний F0 (140-180 Гц)**: нормальная речь
- **Высокий F0 (190-250 Гц)**: напряжённые связки, высокое давление, эмоциональное возбуждение

### Jitter

- **Низкий jitter (<0.5%)**: очень стабильная фонация, может звучать искусственно
- **Нормальный jitter (0.5-1.5%)**: естественная речь
- **Высокий jitter (>2%)**: нестабильность, может указывать на патологию или сильный стресс

### Shimmer

- **Низкий shimmer (<2%)**: стабильная амплитуда
- **Нормальный shimmer (3-5%)**: естественные вариации
- **Высокий shimmer (>6%)**: нестабильность амплитуды, напряжение

### Типы фонации

- **Modal**: нормальная речь, оптимальное смыкание связок
- **Breathy**: больше шума, менее эффективная фонация, может выражать неуверенность или усталость
- **Pressed**: избыточное напряжение, может выражать строгость или агрессию

## Интеграция с ROS (опционально)

Для интеграции с ROS можно публиковать состояние мимики в топик:

```python
import rospy
from std_msgs.msg import String

rospy.init_node('robot_expression')
pub = rospy.Publisher('/robot/expression', String, queue_size=10)

# При синтезе речи
audio, facial_state = synthesizer.synthesize_phrase(text)
pub.publish(facial_state.value)
```

## Требования

Все зависимости уже включены в `requirements.txt`. Дополнительные зависимости для визуализации:

- `matplotlib`: для matplotlib визуализации
- `pygame`: для pygame анимации
- `streamlit`: для веб-интерфейса

## Документация

- Физиологический смысл параметров описан в комментариях к коду
- Примеры использования: `multimodal_demo.py`
- Базовая демонстрация: `my_speech.py`
