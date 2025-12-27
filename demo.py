"""
Демонстрационный сценарий системы синтеза речи.

Демонстрирует:
1. Синтез одной фразы в разных стилях речи
2. Перенос просодии из референсной речи
3. Управление просодическими параметрами
"""

import numpy as np
import os
from speech_synthesizer import SpeechSynthesizer


def demo_speech_styles():
    """
    Демонстрация синтеза одной фразы в разных стилях.
    """
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ: Синтез фразы в разных стилях речи")
    print("=" * 60)
    
    synthesizer = SpeechSynthesizer(sample_rate=16000)
    
    # Тестовая фраза (простая, с несколькими гласными)
    phrase = "Привет, как дела?"
    
    # Стили речи
    styles = ['neutral', 'friendly', 'strict', 'warning']
    
    # Создание директории для результатов
    os.makedirs('output', exist_ok=True)
    
    print(f"\nФраза: '{phrase}'")
    print("\nСинтез в разных стилях...")
    
    for style in styles:
        print(f"\n  Стиль: {style}")
        
        # Синтез
        audio = synthesizer.synthesize_phrase(phrase, style=style)
        
        if len(audio) > 0:
            # Сохранение
            filename = f'output/phrase_{style}.wav'
            synthesizer.save_audio(audio, filename)
            print(f"    Сохранено: {filename}")
            print(f"    Длительность: {len(audio) / synthesizer.sample_rate:.2f} сек")
        else:
            print(f"    Ошибка: пустой сигнал")
    
    print("\n" + "=" * 60)


def demo_prosody_control():
    """
    Демонстрация управления просодическими параметрами.
    """
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ: Управление просодическими параметрами")
    print("=" * 60)
    
    synthesizer = SpeechSynthesizer(sample_rate=16000)
    phrase = "Тест просодии"
    
    # Разные значения F0
    f0_values = [100.0, 150.0, 200.0, 250.0]
    
    print(f"\nФраза: '{phrase}'")
    print("\nСинтез с разными значениями F0...")
    
    for f0 in f0_values:
        print(f"\n  F0: {f0} Гц")
        audio = synthesizer.synthesize_phrase(phrase, style='neutral', base_f0=f0)
        
        if len(audio) > 0:
            filename = f'output/prosody_f0_{int(f0)}.wav'
            synthesizer.save_audio(audio, filename)
            print(f"    Сохранено: {filename}")
    
    # Разные факторы длительности
    duration_factors = [0.7, 1.0, 1.3, 1.6]
    
    print("\nСинтез с разными факторами длительности...")
    
    for df in duration_factors:
        print(f"\n  Фактор длительности: {df}")
        audio = synthesizer.synthesize_phrase(phrase, style='neutral', duration_factor=df)
        
        if len(audio) > 0:
            filename = f'output/prosody_duration_{df:.1f}.wav'
            synthesizer.save_audio(audio, filename)
            print(f"    Сохранено: {filename}")
            print(f"    Длительность: {len(audio) / synthesizer.sample_rate:.2f} сек")
    
    print("\n" + "=" * 60)


def demo_prosody_transfer():
    """
    Демонстрация переноса просодии из референсной речи.
    
    Примечание: требует наличия референсного аудиофайла.
    """
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ: Перенос просодии из референсной речи")
    print("=" * 60)
    
    synthesizer = SpeechSynthesizer(sample_rate=16000)
    
    # Референсный файл (если есть)
    reference_files = [
        'reference_audio.wav',
        'reference.wav',
        'input.wav'
    ]
    
    reference_file = None
    for rf in reference_files:
        if os.path.exists(rf):
            reference_file = rf
            break
    
    if reference_file is None:
        print("\n  Референсный аудиофайл не найден.")
        print("  Для демонстрации переноса просодии поместите файл")
        print("  'reference_audio.wav' в текущую директорию.")
        print("  Пропуск демонстрации переноса просодии...")
        return
    
    phrase = "Новая фраза с перенесённой просодией"
    
    print(f"\nРеференсный файл: {reference_file}")
    print(f"Целевая фраза: '{phrase}'")
    print("\nИзвлечение просодии из референса...")
    
    try:
        # Извлечение просодии
        prosody = synthesizer.prosody_extractor.extract_all_prosody(reference_file)
        print(f"  Длительность референса: {prosody['duration']:.2f} сек")
        print(f"  Средний F0: {np.mean(prosody['f0_values']):.1f} Гц")
        print(f"  Диапазон F0: {np.min(prosody['f0_values']):.1f} - {np.max(prosody['f0_values']):.1f} Гц")
        
        # Синтез с переносом просодии
        print("\nСинтез с перенесённой просодией...")
        audio = synthesizer.synthesize_with_prosody_transfer(
            phrase, reference_file, duration_factor=1.0
        )
        
        if len(audio) > 0:
            filename = 'output/prosody_transfer.wav'
            synthesizer.save_audio(audio, filename)
            print(f"  Сохранено: {filename}")
            print(f"  Длительность: {len(audio) / synthesizer.sample_rate:.2f} сек")
        else:
            print("  Ошибка: пустой сигнал")
    
    except ImportError as e:
        print(f"  ⚠ parselmouth недоступен: {e}")
        print("  Функция переноса просодии требует parselmouth.")
        print("  На Python 3.12+ parselmouth может быть несовместим.")
        print("  Остальной функционал работает без parselmouth.")
    except Exception as e:
        print(f"  Ошибка при обработке: {e}")
        print("  Убедитесь, что файл в формате WAV и parselmouth установлен корректно.")
    
    print("\n" + "=" * 60)


def demo_articulatory_model():
    """
    Демонстрация артикуляторной модели (синтез отдельных гласных).
    """
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ: Артикуляторная модель (синтез гласных)")
    print("=" * 60)
    
    synthesizer = SpeechSynthesizer(sample_rate=16000)
    
    vowels = ['a', 'e', 'i', 'o', 'u']
    f0_values = [120.0, 150.0, 180.0]
    
    print("\nСинтез гласных с разными F0...")
    
    for vowel in vowels:
        for f0 in f0_values:
            audio = synthesizer.articulatory_model.synthesize_vowel(
                vowel, duration=0.3, f0=f0
            )
            
            if len(audio) > 0:
                filename = f'output/vowel_{vowel}_f0_{int(f0)}.wav'
                synthesizer.save_audio(audio, filename)
                print(f"  {vowel.upper()} (F0={f0} Гц): {filename}")
    
    print("\n" + "=" * 60)


def main():
    """
    Главная функция демонстрации.
    """
    print("\n" + "=" * 60)
    print("СИСТЕМА ФИЗИОЛОГИЧЕСКОГО МОДЕЛИРОВАНИЯ РЕЧИ")
    print("Учебно-исследовательский прототип для социального робота")
    print("=" * 60)
    
    # Создание директории для результатов
    os.makedirs('output', exist_ok=True)
    
    # Демонстрации
    demo_articulatory_model()
    demo_speech_styles()
    demo_prosody_control()
    demo_prosody_transfer()
    
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print("\nРезультаты сохранены в директории 'output/'")
    print("\nБиомиметические элементы модели:")
    print("  - Source-Filter архитектура (модель речевого тракта)")
    print("  - Формантные резонаторы (F1, F2, F3)")
    print("  - Физиологически интерпретируемые параметры:")
    print("    * F0: частота вибрации голосовых связок")
    print("    * Форманты: резонансы речевого тракта")
    print("    * Energy: амплитуда вибрации и поток воздуха")
    print("    * Duration: артикуляторная скорость")
    print("\nПрименимость в робототехнике:")
    print("  - Управляемый синтез речи для социальных роботов")
    print("  - Адаптация просодии к контексту взаимодействия")
    print("  - Перенос эмоциональных характеристик речи")
    print("  - Интеграция с ROS/ROS2 для робототехнических систем")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

