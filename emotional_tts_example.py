"""
Пример синтеза с эмоциональным голосом (Edge TTS с улучшенной обработкой эмоций).
"""

from speech_synthesizer import SpeechSynthesizer

print("=" * 60)
print("ЭМОЦИОНАЛЬНЫЙ СИНТЕЗ РЕЧИ")
print("=" * 60)

# Создаём синтезатор с Edge TTS (поддерживает эмоции через разные голоса)
print("\n1. Инициализация с Edge TTS (эмоции)...")
synthesizer = SpeechSynthesizer(
    sample_rate=16000,
    use_tts=True,
    tts_engine='edge'  # Edge TTS с поддержкой эмоций
)

if synthesizer.use_tts and synthesizer.tts_engine is not None:
    print("✓ Edge TTS с эмоциями активирован!")
    print("  - Разные голоса для разных эмоций")
    print("  - Управление просодией (темп, громкость)")
else:
    print("⚠ Edge TTS не загружен")
    print("  Установите: pip install edge-tts")

# Синтез фразы в разных стилях
phrase = "Привет, как дела?"
styles = ['neutral', 'friendly', 'strict', 'warning']

print(f"\n2. Синтез фразы: '{phrase}'")
print("   Стили:", ', '.join(styles))
print("\n   Каждый стиль использует:")
print("   - neutral: стандартный голос")
print("   - friendly: дружелюбный женский голос")
print("   - strict: строгий мужской голос")
print("   - warning: предупреждающий голос")

for style in styles:
    print(f"\n   {style.upper()}:")
    try:
        audio = synthesizer.synthesize_phrase(phrase, style=style)
        filename = f'emotional_{style}.wav'
        synthesizer.save_audio(audio, filename)
        duration = len(audio) / synthesizer.sample_rate
        print(f"   ✓ {filename} ({duration:.2f} сек)")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")

print("\n" + "=" * 60)
print("✓ ГОТОВО!")
print("=" * 60)
print("\nФайлы сохранены: emotional_*.wav")
print("\nУлучшения:")
print("  ✓ Разные голоса для разных эмоций")
print("  ✓ Управление просодией (темп, громкость)")
print("  ✓ Естественный голос (Edge TTS)")
print("\nСравните с предыдущими версиями - эмоции должны быть более выраженными!")

