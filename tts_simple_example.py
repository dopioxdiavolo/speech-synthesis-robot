"""
Простой пример использования системы с TTS (если доступен)
или без TTS (артикуляторная модель).
"""

from speech_synthesizer import SpeechSynthesizer

print("=" * 60)
print("СИНТЕЗ РЕЧИ С УПРАВЛЕНИЕМ ПРОСОДИЕЙ")
print("=" * 60)

# Пробуем создать синтезатор с TTS
print("\n1. Инициализация...")
try:
    # Пробуем Silero TTS
    synthesizer = SpeechSynthesizer(
        sample_rate=16000,
        use_tts=True,
        tts_engine='silero'
    )
    if synthesizer.use_tts and synthesizer.tts_engine is not None:
        print("✓ TTS активирован (естественный голос)")
    else:
        print("✓ Используется артикуляторная модель (улучшенная)")
except:
    # Fallback на артикуляторную модель
    synthesizer = SpeechSynthesizer(use_tts=False)
    print("✓ Используется артикуляторная модель (улучшенная)")

# Синтез фразы в разных стилях
phrase = "Привет, как дела?"
styles = ['neutral', 'friendly', 'strict', 'warning']

print(f"\n2. Синтез фразы: '{phrase}'")
print("   Стили:", ', '.join(styles))

for style in styles:
    print(f"\n   {style.upper()}:")
    try:
        audio = synthesizer.synthesize_phrase(phrase, style=style)
        filename = f'voice_{style}.wav'
        synthesizer.save_audio(audio, filename)
        duration = len(audio) / synthesizer.sample_rate
        print(f"   ✓ {filename} ({duration:.2f} сек)")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")

print("\n" + "=" * 60)
print("✓ ГОТОВО!")
print("=" * 60)
print("\nФайлы сохранены: voice_*.wav")
print("\nПримечание:")
if synthesizer.use_tts and synthesizer.tts_engine is not None:
    print("  ✓ Использован TTS для естественного голоса")
    print("  ✓ Применена система управления просодией")
else:
    print("  ✓ Использована улучшенная артикуляторная модель")
    print("  ✓ Для TTS: см. TTS_SETUP.md")

