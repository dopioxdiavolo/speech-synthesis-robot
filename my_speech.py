from speech_synthesizer import SpeechSynthesizer

# 1. Создаём синтезатор с мультимодальной системой
synthesizer = SpeechSynthesizer(
    sample_rate=16000,  # Частота дискретизации (16 кГц)
    use_tts=True,  # True = использовать TTS, False = артикуляторная модель
    tts_engine='edge',  # 'edge' для эмоционального голоса
    use_vocal_folds=True,  # Использовать модель голосовых связок
    facial_render_method='emoji'  # Метод визуализации мимики
)

# 2. Синтезируем речь - эмоция определяется автоматически!
# Просто передайте текст, система сама поймёт какую эмоцию использовать
# Теперь система также показывает мимику робота!
texts = [
    "Я ТЕБЯ НЕНАВИЖУ!",
    "Внимание! Опасность!",  # Предупреждение
    "Стоп! Прекрати немедленно!",  # Строго
    "Я так рада тебя видеть!"  # Дружелюбно
]

for text in texts:
    print(f"\nТекст: '{text}'")
    # Не указываем style - система сама определит!
    # Теперь возвращается кортеж (audio, facial_state)
    audio, facial_state = synthesizer.synthesize_phrase(text)

    # Сохраняем с именем файла на основе текста
    filename = f"output_{texts.index(text) + 1}.wav"
    synthesizer.save_audio(audio, filename)
    print(f"✓ Сохранено: {filename}")
    if facial_state is not None:
        print(f"✓ Состояние мимики: {facial_state.value}")

print("\n✓ Готово! Система сама определила эмоции для каждого текста!")
print("✓ Мультимодальная система: голос + мимика работают вместе!")
