"""
Модуль для определения эмоций из текста с использованием ML моделей.

Поддерживает:
- Transformers модели (rubert, ruRoBERTa) для sentiment analysis
- Rule-based fallback если модели недоступны
"""

from typing import Dict, Optional, Tuple
import re


class EmotionDetector:
    """
    Детектор эмоций из текста с использованием ML моделей.
    """
    
    def __init__(self, use_ml: bool = True):
        """
        Инициализация детектора эмоций.
        
        Args:
            use_ml: Использовать ML модель (True) или только rule-based (False)
        """
        self.use_ml = use_ml
        self.ml_model = None
        self.tokenizer = None
        
        if use_ml:
            self._load_ml_model()
    
    def _load_ml_model(self):
        """Загрузка ML модели для определения эмоций."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            # Используем русскую модель для sentiment analysis
            model_name = "cointegrated/rubert-tiny-sentiment"
            
            print("Загрузка ML модели для определения эмоций...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.ml_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.ml_model.eval()
            
            print("✓ ML модель загружена (rubert-tiny-sentiment)")
        except ImportError:
            print("⚠ transformers не установлен")
            print("  Установите: pip install transformers torch")
            print("  Используется rule-based детектор")
            self.ml_model = None
            self.use_ml = False
        except Exception as e:
            print(f"⚠ Ошибка загрузки ML модели: {e}")
            print("  Используется rule-based детектор")
            self.ml_model = None
            self.use_ml = False
    
    def detect_emotion_ml(self, text: str) -> Tuple[str, float]:
        """
        Определение эмоции с помощью ML модели.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Tuple (стиль, уверенность): ('neutral', 'friendly', 'strict', 'warning'), confidence
        """
        if self.ml_model is None or self.tokenizer is None:
            return None, 0.0
        
        try:
            import torch
            
            # Токенизация
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.ml_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Модель возвращает: negative, neutral, positive
            # Маппим на наши стили
            scores = predictions[0].tolist()
            
            # Интерпретация результатов
            # negative -> strict или warning (в зависимости от контекста)
            # neutral -> neutral
            # positive -> friendly
            
            negative_score = scores[0] if len(scores) > 0 else 0.0
            neutral_score = scores[1] if len(scores) > 1 else 0.0
            positive_score = scores[2] if len(scores) > 2 else 0.0
            
            # Определяем базовую эмоцию
            if positive_score > 0.5:
                base_emotion = 'friendly'
                confidence = positive_score
            elif negative_score > 0.5:
                # Отрицательная эмоция - нужно определить strict или warning
                # Используем rule-based для уточнения
                if any(word in text.lower() for word in ['внимание', 'опасно', 'тревога', 'пожар', 'авария']):
                    base_emotion = 'warning'
                else:
                    base_emotion = 'strict'
                confidence = negative_score
            else:
                base_emotion = 'neutral'
                confidence = neutral_score
            
            return base_emotion, confidence
            
        except Exception as e:
            print(f"⚠ Ошибка ML предсказания: {e}")
            return None, 0.0
    
    def detect_emotion_rule_based(self, text: str) -> str:
        """
        Rule-based определение эмоции (fallback).
        
        Args:
            text: Текст для анализа
            
        Returns:
            Определённый стиль: 'neutral', 'friendly', 'strict', 'warning'
        """
        text_lower = text.lower()
        
        # Ключевые слова для разных эмоций
        friendly_words = [
            'привет', 'здравствуй', 'добро', 'спасибо', 'пожалуйста',
            'отлично', 'хорошо', 'замечательно', 'рад', 'радость',
            'улыбка', 'счастье', 'любовь', 'дружба', 'помощь'
        ]
        
        strict_words = [
            'прекрати', 'нельзя', 'запрещено', 'запретить',
            'строго', 'обязательно', 'требую', 'приказ', 'дисциплина',
            'нарушение', 'наказание', 'виноват', 'ответственность',
            'не делай', 'не трогай', 'нельзя делать', 'не смей',
            'ненавижу', 'ненависть', 'злой', 'злость'
        ]
        
        warning_words = [
            'внимание', 'осторожно', 'опасно', 'предупреждение',
            'авария', 'пожар', 'тревога', 'срочно', 'немедленно',
            'опасность', 'угроза', 'берегись', 'предупреждаю'
        ]
        
        # Подсчёт совпадений
        friendly_score = sum(1 for word in friendly_words if word in text_lower)
        strict_score = sum(1 for word in strict_words if word in text_lower)
        warning_score = sum(1 for word in warning_words if word in text_lower)
        
        # Специальные случаи для слова "стоп"
        if 'стоп' in text_lower:
            if any(word in text_lower for word in ['опасно', 'внимание', 'тревога', 'авария', 'пожар']):
                warning_score += 2
            else:
                strict_score += 2
        
        # Анализ пунктуации
        exclamation_count = text.count('!')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Бонусы за пунктуацию
        if exclamation_count >= 2:
            warning_score += 2
        elif exclamation_count == 1:
            if warning_score > 0:
                warning_score += 1
            elif strict_score > 0:
                strict_score += 1
        
        if caps_ratio > 0.3:  # Много заглавных букв
            warning_score += 2
        
        # Определение стиля
        scores = {
            'friendly': friendly_score,
            'strict': strict_score,
            'warning': warning_score,
            'neutral': 0
        }
        
        max_score = max(scores.values())
        if max_score == 0:
            return 'neutral'
        
        return max(scores, key=scores.get)
    
    def detect(self, text: str) -> Tuple[str, float]:
        """
        Определение эмоции из текста.
        
        Использует ML модель если доступна, иначе rule-based.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Tuple (стиль, уверенность): ('neutral', 'friendly', 'strict', 'warning'), confidence
        """
        # Пробуем ML модель
        if self.use_ml and self.ml_model is not None:
            emotion, confidence = self.detect_emotion_ml(text)
            if emotion is not None and confidence > 0.3:
                return emotion, confidence
        
        # Fallback на rule-based
        emotion = self.detect_emotion_rule_based(text)
        # Для rule-based уверенность средняя (0.6)
        confidence = 0.6
        
        return emotion, confidence
