"""
Интеграция с F5-TTS Russian для синтеза речи.

F5-TTS - диффузионная модель синтеза речи с высоким качеством.
Русская версия: https://huggingface.co/hotstone228/F5-TTS-Russian
"""

import numpy as np
from typing import Optional
import warnings


class F5TTSEngine:
    """
    Интеграция с F5-TTS Russian для синтеза речи.
    
    Требования:
    - pip install f5-tts
    - GPU рекомендуется для реального времени
    """
    
    def __init__(self, sample_rate: int = 24000, device: str = 'auto'):
        """
        Инициализация F5-TTS.
        
        Args:
            sample_rate: Частота дискретизации (F5-TTS использует 24000)
            device: Устройство ('auto', 'cpu', 'cuda')
        """
        self.sample_rate = sample_rate
        self.device = device
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели F5-TTS."""
        try:
            from f5_tts import F5TTS
            
            print("⏳ Загрузка F5-TTS Russian...")
            # Загрузка русской модели
            self.model = F5TTS.from_pretrained(
                "hotstone228/F5-TTS-Russian",
                device=self.device
            )
            print("✓ F5-TTS Russian загружен")
        except ImportError:
            print("⚠ F5-TTS не установлен")
            print("   Установите: pip install f5-tts")
            print("   Или используйте другой TTS движок")
            self.model = None
        except Exception as e:
            print(f"⚠ Ошибка загрузки F5-TTS: {e}")
            self.model = None
    
    def synthesize(self, text: str, speaker_id: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Синтез речи из текста.
        
        Args:
            text: Текст для синтеза
            speaker_id: ID спикера (опционально)
            
        Returns:
            Аудиосигнал или None
        """
        if self.model is None:
            return None
        
        try:
            # Синтез через F5-TTS
            audio = self.model.tts(
                text=text,
                speaker_id=speaker_id,
                speed=1.0
            )
            
            # Конвертация в numpy array
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            elif hasattr(audio, 'numpy'):
                audio = audio.numpy()
            
            # Нормализация
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            # Ресемплинг если нужно
            if self.sample_rate != 24000:
                from scipy import signal
                num_samples = int(len(audio) * self.sample_rate / 24000)
                audio = signal.resample(audio, num_samples)
            
            # Нормализация амплитуды
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95
            
            return audio.astype(np.float32)
        except Exception as e:
            print(f"⚠ Ошибка синтеза F5-TTS: {e}")
            return None
    
    def is_available(self) -> bool:
        """Проверка доступности модели."""
        return self.model is not None
