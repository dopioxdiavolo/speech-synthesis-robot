"""
Пример использования TTS для естественного голоса.
"""

from speech_synthesizer import SpeechSynthesizer

print("=" * 60)
print("ПРИМЕР: Синтез с TTS (естественный голос)")
print("=" * 60)

# Создаём синтезатор с TTS
print("\n1. Инициализация с Silero TTS...")
try:
    synthesizer = SpeechSynthesizer(
        sample_rate=16000,
        use_tts=True,
        tts_engine='silero'
    )
    print("✓ Silero TTS активирован")
except Exception as e:
    print(f"⚠ Silero TTS не доступен: {e}")
    print("  Попробуем pyttsx3...")
    try:
        synthesizer = SpeechSynthesizer(
            sample_rate=16000,
            use_tts=True,
            tts_engine='pyttsx3'
        )
        print("✓ pyttsx3 активирован")
    except Exception as e2:
        print(f"⚠ pyttsx3 не доступен: {e2}")
        print("  Используем артикуляторную модель")
        synthesizer = SpeechSynthesizer(use_tts=False)

# Синтез фразы
phrase = "Привет, как дела?"
styles = ['neutral', 'friendly', 'strict', 'warning']

print(f"\n2. Синтез фразы: '{phrase}'")
print("   Стили:", ', '.join(styles))

for style in styles:
    print(f"\n   Стиль: {style}")
    try:
        audio = synthesizer.synthesize_phrase(phrase, style=style)
        filename = f'tts_{style}.wav'
        synthesizer.save_audio(audio, filename)
        print(f"   ✓ Сохранено: {filename} ({len(audio)/synthesizer.sample_rate:.2f} сек)")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")

print("\n" + "=" * 60)
print("✓ Готово!")
print("=" * 60)
print("\nПримечание:")
print("  - Если TTS не установлен, система использует артикуляторную модель")
print("  - Для Silero TTS: pip install torch torchaudio")
print("  - Для pyttsx3: pip install pyttsx3")

