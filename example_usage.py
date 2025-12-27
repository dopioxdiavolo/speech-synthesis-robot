"""
Пример использования системы синтеза речи.

Простой пример для быстрого старта.
"""

from speech_synthesizer import SpeechSynthesizer

# Создание синтезатора
synthesizer = SpeechSynthesizer(sample_rate=16000)

# Пример 1: Синтез фразы в нейтральном стиле
print("Синтез фразы в нейтральном стиле...")
audio_neutral = synthesizer.synthesize_phrase(
    "Привет, как дела?",
    style='neutral'
)
synthesizer.save_audio(audio_neutral, 'example_neutral.wav')
print("Сохранено: example_neutral.wav")

# Пример 2: Синтез в дружелюбном стиле
print("\nСинтез фразы в дружелюбном стиле...")
audio_friendly = synthesizer.synthesize_phrase(
    "Привет, как дела?",
    style='friendly'
)
synthesizer.save_audio(audio_friendly, 'example_friendly.wav')
print("Сохранено: example_friendly.wav")

# Пример 3: Синтез с заданным F0
print("\nСинтез с высоким F0...")
audio_high_f0 = synthesizer.synthesize_phrase(
    "Тест",
    style='neutral',
    base_f0=200.0
)
synthesizer.save_audio(audio_high_f0, 'example_high_f0.wav')
print("Сохранено: example_high_f0.wav")

# Пример 4: Синтез гласного напрямую через артикуляторную модель
print("\nСинтез гласного 'а'...")
vowel_audio = synthesizer.articulatory_model.synthesize_vowel(
    'a',
    duration=0.5,
    f0=150.0
)
synthesizer.save_audio(vowel_audio, 'example_vowel_a.wav')
print("Сохранено: example_vowel_a.wav")

print("\nПримеры созданы!")

