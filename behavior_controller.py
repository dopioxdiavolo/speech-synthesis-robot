"""
Контроллер поведения робота (Behavior Controller).

Координирует состояние голоса и мимики на основе:
- style: стиль речи (neutral, friendly, strict, warning)
- internal_state: внутреннее состояние робота (опционально)

Выходные параметры:
- voice_state: параметры голосовых связок (F0, jitter, shimmer, phonation_type)
- facial_state: состояние мимики (neutral, friendly, strict, warning)

Связь между голосом и мимикой:
expression_state = f(style, internal_state)
voice_state = g(style, internal_state)
"""

from typing import Dict, Optional, Tuple, Literal
from enum import Enum
from vocal_folds import PhonationType
from facial_expression import ExpressionState


class BehaviorController:
    """
    Контроллер поведения робота.
    
    Координирует голос и мимику на основе стиля речи
    и внутреннего состояния робота.
    """
    
    # Маппинг стилей на параметры голоса и мимики
    STYLE_TO_VOICE_PARAMS = {
        'neutral': {
            'f0_base': 150.0,
            'f0_range': 30.0,
            'jitter': 0.01,
            'shimmer': 0.03,
            'phonation_type': PhonationType.MODAL,
            'vocal_effort': 1.0,
        },
        'friendly': {
            'f0_base': 180.0,
            'f0_range': 50.0,
            'jitter': 0.012,
            'shimmer': 0.04,
            'phonation_type': PhonationType.MODAL,
            'vocal_effort': 1.2,
        },
        'strict': {
            'f0_base': 120.0,
            'f0_range': 20.0,
            'jitter': 0.008,
            'shimmer': 0.025,
            'phonation_type': PhonationType.PRESSED,
            'vocal_effort': 1.3,
        },
        'warning': {
            'f0_base': 200.0,
            'f0_range': 80.0,
            'jitter': 0.015,
            'shimmer': 0.05,
            'phonation_type': PhonationType.BREATHY,
            'vocal_effort': 1.5,
        }
    }
    
    STYLE_TO_FACIAL_STATE = {
        'neutral': ExpressionState.NEUTRAL,
        'friendly': ExpressionState.FRIENDLY,
        'strict': ExpressionState.STRICT,
        'warning': ExpressionState.WARNING,
    }
    
    def __init__(self):
        """Инициализация контроллера поведения."""
        pass
    
    def compute_behavior(self,
                        style: str,
                        internal_state: Optional[Dict] = None) -> Tuple[Dict, ExpressionState]:
        """
        Вычисление состояния поведения на основе стиля и внутреннего состояния.
        
        Физиологический смысл:
        Координирует параметры голосовых связок и мимики для создания
        согласованного выражения эмоции/настроения робота.
        
        Args:
            style: Стиль речи ('neutral', 'friendly', 'strict', 'warning')
            internal_state: Внутреннее состояние робота (опционально)
                          Может содержать:
                          - energy: уровень энергии (0.0-2.0)
                          - stress: уровень стресса (0.0-1.0)
                          - mood: настроение ('positive', 'negative', 'neutral')
        
        Returns:
            (voice_state, facial_state): 
            - voice_state: словарь с параметрами голоса
            - facial_state: состояние мимики
        """
        # Нормализация стиля
        style = style.lower()
        if style not in self.STYLE_TO_VOICE_PARAMS:
            style = 'neutral'
        
        # Базовые параметры голоса
        voice_params = self.STYLE_TO_VOICE_PARAMS[style].copy()
        
        # Базовое состояние мимики
        facial_state = self.STYLE_TO_FACIAL_STATE[style]
        
        # Модификация на основе внутреннего состояния
        if internal_state is not None:
            # Влияние уровня энергии
            if 'energy' in internal_state:
                energy = internal_state['energy']
                # Больше энергии → выше F0, больше vocal effort
                voice_params['f0_base'] *= (0.8 + 0.4 * energy)
                voice_params['vocal_effort'] *= energy
            
            # Влияние стресса
            if 'stress' in internal_state:
                stress = internal_state['stress']
                # Стресс → больше jitter и shimmer (нестабильность)
                voice_params['jitter'] *= (1.0 + stress * 0.5)
                voice_params['shimmer'] *= (1.0 + stress * 0.3)
                
                # Высокий стресс → pressed phonation
                if stress > 0.7:
                    voice_params['phonation_type'] = PhonationType.PRESSED
                elif stress > 0.4:
                    voice_params['phonation_type'] = PhonationType.MODAL
            
            # Влияние настроения
            if 'mood' in internal_state:
                mood = internal_state['mood']
                if mood == 'positive' and style == 'neutral':
                    # Положительное настроение → более дружелюбное выражение
                    facial_state = ExpressionState.FRIENDLY
                    voice_params['f0_base'] *= 1.1
                elif mood == 'negative' and style == 'neutral':
                    # Отрицательное настроение → более строгое выражение
                    facial_state = ExpressionState.STRICT
                    voice_params['f0_base'] *= 0.9
        
        return voice_params, facial_state
    
    def get_voice_params(self, style: str, internal_state: Optional[Dict] = None) -> Dict:
        """
        Получение параметров голоса для стиля.
        
        Args:
            style: Стиль речи
            internal_state: Внутреннее состояние робота
            
        Returns:
            Словарь с параметрами голоса
        """
        voice_params, _ = self.compute_behavior(style, internal_state)
        return voice_params
    
    def get_facial_state(self, style: str, internal_state: Optional[Dict] = None) -> ExpressionState:
        """
        Получение состояния мимики для стиля.
        
        Args:
            style: Стиль речи
            internal_state: Внутреннее состояние робота
            
        Returns:
            Состояние мимики
        """
        _, facial_state = self.compute_behavior(style, internal_state)
        return facial_state
    
    def set_custom_mapping(self, style: str, voice_params: Optional[Dict] = None, 
                          facial_state: Optional[ExpressionState] = None):
        """
        Установка пользовательского маппинга для стиля.
        
        Args:
            style: Стиль речи
            voice_params: Параметры голоса (если None, не изменяются)
            facial_state: Состояние мимики (если None, не изменяется)
        """
        if voice_params is not None:
            if style in self.STYLE_TO_VOICE_PARAMS:
                self.STYLE_TO_VOICE_PARAMS[style].update(voice_params)
            else:
                self.STYLE_TO_VOICE_PARAMS[style] = voice_params
        
        if facial_state is not None:
            self.STYLE_TO_FACIAL_STATE[style] = facial_state
    
    def get_available_styles(self) -> list:
        """
        Получение списка доступных стилей.
        
        Returns:
            Список стилей
        """
        return list(self.STYLE_TO_VOICE_PARAMS.keys())
