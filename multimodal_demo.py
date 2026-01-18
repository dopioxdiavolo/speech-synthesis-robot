"""
Демонстрация мультимодальной системы синтеза речи для социального робота.

Показывает работу:
- Модели голосовых связок с физиологическими параметрами
- Визуализации мимики в реальном времени
- Координации голоса и мимики через BehaviorController
"""

from speech_synthesizer import SpeechSynthesizer
from vocal_folds import PhonationType
from facial_expression import ExpressionState
from behavior_controller import BehaviorController
import numpy as np

def demo_multimodal_synthesis():
    """Демонстрация мультимодального синтеза."""
    
    print("=" * 60)
    print("МУЛЬТИМОДАЛЬНАЯ СИСТЕМА СИНТЕЗА РЕЧИ ДЛЯ СОЦИАЛЬНОГО РОБОТА")
    print("=" * 60)
    print()
    
    # Создаём синтезатор с мультимодальной системой
    print("Инициализация системы...")
    # Выберите метод визуализации:
    # 'matplotlib' - красивая графическая визуализация с деталями
    # 'pygame' - интерактивная анимация (требует pygame)
    # 'ascii' - ASCII-арт в терминале
    # 'emoji' - простые смайлики (по умолчанию)
    
    render_method = 'matplotlib'  # Измените на 'pygame' или 'ascii' для других вариантов
    
    synthesizer = SpeechSynthesizer(
        sample_rate=16000,
        use_tts=True,  # ВАЖНО: TTS нужен для нормальной речи!
        tts_engine='edge',  # Edge TTS для качественной речи
        use_vocal_folds=True,  # Модель голосовых связок применяется к TTS аудио
        facial_render_method=render_method
    )
    print("✓ Система инициализирована\n")
    
    # Демонстрация различных стилей
    test_cases = [
        {
            'text': 'Привет, как дела?',
            'style': 'friendly',
            'description': 'Дружелюбное приветствие'
        },
        {
            'text': 'Стоп! Прекрати немедленно!',
            'style': 'strict',
            'description': 'Строгая команда'
        },
        {
            'text': 'Внимание! Опасность!',
            'style': 'warning',
            'description': 'Предупреждение'
        },
        {
            'text': 'Сегодня обычный день.',
            'style': 'neutral',
            'description': 'Нейтральное высказывание'
        }
    ]
    
    print("Демонстрация синтеза с различными стилями:")
    print("-" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['description']}")
        print(f"   Текст: '{case['text']}'")
        print(f"   Стиль: {case['style']}")
        
        # Получаем параметры поведения
        behavior_controller = synthesizer.behavior_controller
        voice_params, facial_state = behavior_controller.compute_behavior(case['style'])
        
        # Показываем параметры голоса
        print(f"\n   Параметры голоса:")
        print(f"   - F0: {voice_params['f0_base']:.1f} Гц")
        print(f"   - Jitter: {voice_params['jitter']*100:.2f}%")
        print(f"   - Shimmer: {voice_params['shimmer']*100:.2f}%")
        print(f"   - Тип фонации: {voice_params['phonation_type'].value}")
        print(f"   - Усилие фонации: {voice_params['vocal_effort']:.2f}")
        
        # Показываем мимику
        expression_display = synthesizer.facial_renderer.render(facial_state)
        print(f"\n   Мимика: {expression_display}")
        
        # Синтезируем речь
        print("\n   ⏳ Синтез речи...")
        audio, facial_state_result = synthesizer.synthesize_phrase(
            case['text'],
            style=case['style'],
            show_facial_expression=True
        )
        
        # Для matplotlib: окно уже открыто, ждём закрытия пользователем
        if synthesizer.facial_renderer.render_method == 'matplotlib':
            print("   💡 Закройте окно matplotlib, чтобы продолжить...")
        
        # Сохраняем результат
        filename = f"multimodal_output_{i}.wav"
        synthesizer.save_audio(audio, filename)
        print(f"\n   ✓ Аудио сохранено: {filename}")
        print(f"   ✓ Длительность: {len(audio)/synthesizer.sample_rate:.2f} сек")
    
    print("\n" + "=" * 60)
    print("Демонстрация влияния внутреннего состояния робота:")
    print("-" * 60)
    
    # Демонстрация влияния внутреннего состояния
    internal_states = [
        {
            'name': 'Высокая энергия',
            'state': {'energy': 1.5, 'stress': 0.2, 'mood': 'positive'}
        },
        {
            'name': 'Стресс',
            'state': {'energy': 0.8, 'stress': 0.8, 'mood': 'negative'}
        },
        {
            'name': 'Нейтральное состояние',
            'state': {'energy': 1.0, 'stress': 0.3, 'mood': 'neutral'}
        }
    ]
    
    test_text = "Привет, как дела?"
    
    for i, internal_case in enumerate(internal_states, 1):
        print(f"\n{i}. {internal_case['name']}")
        print(f"   Внутреннее состояние: {internal_case['state']}")
        
        # Вычисляем поведение с учётом внутреннего состояния
        voice_params, facial_state = synthesizer.behavior_controller.compute_behavior(
            'neutral',  # Базовый стиль
            internal_case['state']
        )
        
        print(f"   → F0: {voice_params['f0_base']:.1f} Гц")
        print(f"   → Jitter: {voice_params['jitter']*100:.2f}%")
        expression_display = synthesizer.facial_renderer.render(facial_state)
        print(f"   → Мимика: {expression_display}")
    
    print("\n" + "=" * 60)
    print("Демонстрация физиологических параметров голосовых связок:")
    print("-" * 60)
    
    # Демонстрация различных типов фонации
    phonation_types = [
        (PhonationType.MODAL, "Нормальная фонация"),
        (PhonationType.BREATHY, "Придыхательная фонация"),
        (PhonationType.PRESSED, "Сжатая фонация")
    ]
    
    for phonation_type, description in phonation_types:
        print(f"\n{description} ({phonation_type.value}):")
        params = synthesizer.vocal_folds_model.get_phonation_params(phonation_type)
        print(f"  - Jitter: {params['jitter']*100:.2f}%")
        print(f"  - Shimmer: {params['shimmer']*100:.2f}%")
        print(f"  - Шумовая компонента: {params['noise_ratio']*100:.1f}%")
        print(f"  - Spectral tilt: {params['spectral_tilt']:.1f} dB/октава")
        print(f"  - Open quotient: {params['open_quotient']:.2f}")
    
    print("\n" + "=" * 60)
    print("✓ Демонстрация завершена!")
    print("=" * 60)


if __name__ == "__main__":
    demo_multimodal_synthesis()
