"""
Финальный пример: синтез с естественным голосом (Edge TTS).
"""

from speech_synthesizer import SpeechSynthesizer

print("=" * 60)
print("СИНТЕЗ РЕЧИ С ЕСТЕСТВЕННЫМ ГОЛОСОМ")
print("=" * 60)

# Создаём синтезатор с Edge TTS (Microsoft)
print("\n1. Инициализация с Edge TTS...")
synthesizer = SpeechSynthesizer(
    sample_rate=16000,
    use_tts=True,
    tts_engine='edge'  # Edge TTS (Microsoft) - качественный и бесплатный
)

if synthesizer.use_tts and synthesizer.tts_engine is not None:
    print("✓ Edge TTS активирован - теперь у вас естественный голос!")
else:
    print("⚠ Edge TTS не загружен, используется артикуляторная модель")
    print("  Установите: pip install edge-tts")

# Синтез фразы в разных стилях
phrase = "Привет, как дела?"
styles = ['neutral', 'friendly', 'strict', 'warning']

print(f"\n2. Синтез фразы: '{phrase}'")
print("   Стили:", ', '.join(styles))

for style in styles:
    print(f"\n   {style.upper()}:")
    try:
        audio = synthesizer.synthesize_phrase(phrase, style=style)
        filename = f'natural_voice_{style}.wav'
        synthesizer.save_audio(audio, filename)
        duration = len(audio) / synthesizer.sample_rate
        print(f"   ✓ {filename} ({duration:.2f} сек)")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")

print("\n" + "=" * 60)
print("✓ ГОТОВО!")
print("=" * 60)
print("\nФайлы сохранены: natural_voice_*.wav")
print("\nТеперь у вас:")
print("  ✓ Естественный голос (Edge TTS)")
print("  ✓ Управление просодией (стили речи)")
print("  ✓ Контроль над параметрами (темп, громкость)")

