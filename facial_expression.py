"""
Модуль визуализации мимики робота (Facial Expression Module).

Реализует отображение состояния экспрессии робота в реальном времени
при синтезе речи. Поддерживает различные состояния:
- neutral: нейтральное выражение
- friendly: дружелюбное выражение
- strict: строгое выражение
- warning: предупреждающее выражение

Визуализация может быть реализована через:
- matplotlib (графический интерфейс)
- pygame (интерактивная анимация)
- streamlit (веб-интерфейс)
- emoji-аниматор (простая текстовая визуализация)

Состояние мимики связано с состоянием голоса через BehaviorController.
"""

import numpy as np
from typing import Optional, Literal, Dict, Tuple
from enum import Enum
import time
import threading


class ExpressionState(Enum):
    """Состояния мимики робота."""
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    STRICT = "strict"
    WARNING = "warning"


class FacialExpressionRenderer:
    """
    Рендерер мимики робота.
    
    Отображает состояние экспрессии в реальном времени.
    Поддерживает различные методы визуализации.
    """
    
    # Параметры выражений лица (координаты, цвета, формы)
    EXPRESSION_PARAMS = {
        ExpressionState.NEUTRAL: {
            'eye_shape': 'normal',      # Форма глаз
            'eye_openness': 0.8,        # Открытость глаз (0-1)
            'mouth_shape': 'neutral',    # Форма рта
            'mouth_openness': 0.3,       # Открытость рта (0-1)
            'eyebrow_angle': 0.0,        # Угол бровей (радианы)
            'color': (0.7, 0.7, 0.7),   # RGB цвет (серый)
            'emoji': '😐',              # Emoji представление
        },
        ExpressionState.FRIENDLY: {
            'eye_shape': 'happy',
            'eye_openness': 0.9,
            'mouth_shape': 'smile',
            'mouth_openness': 0.5,
            'eyebrow_angle': -0.2,  # Приподнятые брови
            'color': (0.2, 0.8, 0.2),  # Зелёный
            'emoji': '😊',
        },
        ExpressionState.STRICT: {
            'eye_shape': 'narrow',
            'eye_openness': 0.7,
            'mouth_shape': 'frown',
            'mouth_openness': 0.2,
            'eyebrow_angle': 0.3,  # Нахмуренные брови
            'color': (0.8, 0.2, 0.2),  # Красный
            'emoji': '😠',
        },
        ExpressionState.WARNING: {
            'eye_shape': 'wide',
            'eye_openness': 1.0,
            'mouth_shape': 'open',
            'mouth_openness': 0.8,
            'eyebrow_angle': 0.4,  # Высоко поднятые брови
            'color': (1.0, 0.6, 0.0),  # Оранжевый
            'emoji': '⚠️',
        }
    }
    
    def __init__(self, render_method: Literal['matplotlib', 'pygame', 'streamlit', 'emoji', 'ascii'] = 'emoji'):
        """
        Инициализация рендерера.
        
        Args:
            render_method: Метод визуализации
                - 'matplotlib': статичная визуализация через matplotlib
                - 'pygame': интерактивная анимация через pygame
                - 'streamlit': веб-интерфейс через streamlit
                - 'emoji': простая текстовая визуализация через emoji
        """
        self.render_method = render_method
        self.current_state = ExpressionState.NEUTRAL
        self.is_rendering = False
        self.render_thread = None
        
        # Инициализация в зависимости от метода
        if render_method == 'pygame':
            self._init_pygame()
        elif render_method == 'matplotlib':
            self._init_matplotlib()
        elif render_method == 'streamlit':
            self._init_streamlit()
    
    def _init_pygame(self):
        """Инициализация pygame (если доступен)."""
        try:
            import pygame
            pygame.init()
            self.pygame_available = True
            self.screen = pygame.display.set_mode((400, 400))
            pygame.display.set_caption("Робот - Мимика")
            self.clock = pygame.time.Clock()
        except ImportError:
            self.pygame_available = False
            print("⚠ pygame не установлен, используем emoji режим")
            self.render_method = 'emoji'
    
    def _init_matplotlib(self):
        """Инициализация matplotlib."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            self.matplotlib_available = True
            self.fig = None
            self.ax = None
            self._matplotlib_fig = None
            plt.ion()  # Включаем интерактивный режим
        except ImportError:
            self.matplotlib_available = False
            print("⚠ matplotlib не установлен, используем emoji режим")
            self.render_method = 'emoji'
    
    def _init_streamlit(self):
        """Инициализация streamlit (для веб-интерфейса)."""
        try:
            import streamlit as st
            self.streamlit_available = True
        except ImportError:
            self.streamlit_available = False
            print("⚠ streamlit не установлен, используем emoji режим")
            self.render_method = 'emoji'
    
    def set_expression(self, state: ExpressionState):
        """
        Установка состояния выражения.
        
        Args:
            state: Состояние выражения
        """
        self.current_state = state
    
    def render_emoji(self, state: Optional[ExpressionState] = None) -> str:
        """
        Простая текстовая визуализация через emoji.
        
        Args:
            state: Состояние выражения (если None, используется текущее)
            
        Returns:
            Emoji строка
        """
        if state is None:
            state = self.current_state
        
        params = self.EXPRESSION_PARAMS[state]
        emoji = params['emoji']
        
        # Добавляем анимацию (мигание для warning)
        if state == ExpressionState.WARNING:
            # Чередуем emoji для эффекта мигания
            if int(time.time() * 2) % 2 == 0:
                return f"{emoji} {state.value.upper()}"
            else:
                return f"  {state.value.upper()}"
        
        return f"{emoji} {state.value.upper()}"
    
    def render_matplotlib(self, state: Optional[ExpressionState] = None, ax=None):
        """
        Улучшенная визуализация через matplotlib с детальной отрисовкой.
        
        Args:
            state: Состояние выражения
            ax: Ось matplotlib (если None, создаётся новая)
        """
        if not self.matplotlib_available:
            return
        
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Ellipse, Arc, FancyBboxPatch
        from matplotlib.colors import LinearSegmentedColormap
        
        if state is None:
            state = self.current_state
        
        params = self.EXPRESSION_PARAMS[state]
        
        # Используем неинтерактивный backend для избежания проблем с потоками
        import matplotlib
        matplotlib.use('Agg')  # Неинтерактивный backend
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
        else:
            ax.clear()
        
        # Рисуем лицо с градиентом
        # Голова (круг с градиентом)
        head_center = (0.5, 0.5)
        head_radius = 0.4
        
        # Внешний круг (тень)
        shadow = Circle(head_center, head_radius + 0.02, 
                       color='gray', alpha=0.3, fill=True)
        ax.add_patch(shadow)
        
        # Основной круг головы
        circle = Circle(head_center, head_radius, 
                       color='#FFDBAC', fill=True, 
                       edgecolor='#D4A574', linewidth=3)
        ax.add_patch(circle)
        
        # Добавляем блик (световой эффект)
        highlight = Ellipse((0.45, 0.55), 0.15, 0.2, 
                          color='white', alpha=0.3, fill=True)
        ax.add_patch(highlight)
        
        # Глаза с детальной отрисовкой
        eye_openness = params['eye_openness']
        eye_y = 0.6
        
        for eye_x in [0.35, 0.65]:
            if params['eye_shape'] == 'happy':
                # Счастливые глаза (дуги с бликами)
                arc = Arc((eye_x, eye_y), 0.15, 0.1, 
                         angle=0, theta1=0, theta2=180, 
                         color='black', linewidth=4)
                ax.add_patch(arc)
                # Блик в глазу
                highlight = Circle((eye_x - 0.02, eye_y + 0.02), 0.02, 
                                  color='white', fill=True, alpha=0.8)
                ax.add_patch(highlight)
            elif params['eye_shape'] == 'wide':
                # Широкие глаза (большие с радужкой)
                eye_size = 0.08 * eye_openness
                # Белок глаза
                white = Circle((eye_x, eye_y), eye_size, 
                              color='white', fill=True, 
                              edgecolor='black', linewidth=2)
                ax.add_patch(white)
                # Радужка
                iris = Circle((eye_x, eye_y), eye_size * 0.6, 
                             color='#4A90E2', fill=True)
                ax.add_patch(iris)
                # Зрачок
                pupil = Circle((eye_x, eye_y), eye_size * 0.3, 
                              color='black', fill=True)
                ax.add_patch(pupil)
                # Блик
                highlight = Circle((eye_x - 0.01, eye_y + 0.01), 
                                  eye_size * 0.2, 
                                  color='white', fill=True, alpha=0.9)
                ax.add_patch(highlight)
            elif params['eye_shape'] == 'narrow':
                # Узкие глаза (прищуренные)
                eye_size = 0.05 * eye_openness
                # Глаз как эллипс
                eye = Ellipse((eye_x, eye_y), eye_size * 1.5, eye_size, 
                             color='black', fill=True)
                ax.add_patch(eye)
            else:
                # Нормальные глаза
                eye_size = 0.06 * eye_openness
                white = Circle((eye_x, eye_y), eye_size, 
                              color='white', fill=True, 
                              edgecolor='black', linewidth=2)
                ax.add_patch(white)
                iris = Circle((eye_x, eye_y), eye_size * 0.6, 
                             color='#4A90E2', fill=True)
                ax.add_patch(iris)
                pupil = Circle((eye_x, eye_y), eye_size * 0.3, 
                              color='black', fill=True)
                ax.add_patch(pupil)
                highlight = Circle((eye_x - 0.008, eye_y + 0.008), 
                                  eye_size * 0.15, 
                                  color='white', fill=True, alpha=0.9)
                ax.add_patch(highlight)
        
        # Брови с детальной отрисовкой
        eyebrow_angle = params['eyebrow_angle']
        for brow_x in [0.3, 0.7]:
            brow_y = 0.7 + eyebrow_angle * 0.1
            # Бровь как эллипс (более естественно)
            brow = Ellipse((brow_x, brow_y), 0.1, 0.025, 
                          angle=np.degrees(eyebrow_angle) * 15,
                          color='#8B4513', fill=True, 
                          edgecolor='#654321', linewidth=1)
            ax.add_patch(brow)
        
        # Рот с детальной отрисовкой
        mouth_openness = params['mouth_openness']
        mouth_y = 0.35
        
        if params['mouth_shape'] == 'smile':
            # Улыбка (дуга с зубами)
            arc = Arc((0.5, mouth_y), 0.2, 0.15, 
                     angle=0, theta1=180, theta2=360, 
                     color='black', linewidth=4)
            ax.add_patch(arc)
            # Зубы (опционально)
            for tooth_x in [0.45, 0.5, 0.55]:
                tooth = Ellipse((tooth_x, mouth_y + 0.02), 0.02, 0.03, 
                               color='white', fill=True, 
                               edgecolor='black', linewidth=1, alpha=0.8)
                ax.add_patch(tooth)
        elif params['mouth_shape'] == 'frown':
            # Хмурый рот (перевёрнутая дуга)
            arc = Arc((0.5, mouth_y), 0.2, 0.15, 
                     angle=0, theta1=0, theta2=180, 
                     color='black', linewidth=4)
            ax.add_patch(arc)
        elif params['mouth_shape'] == 'open':
            # Открытый рот (овал с языком)
            mouth = Ellipse((0.5, mouth_y), 0.15, 0.2 * mouth_openness, 
                          color='black', fill=True, 
                          edgecolor='black', linewidth=3)
            ax.add_patch(mouth)
            # Внутренность рта
            inner = Ellipse((0.5, mouth_y), 0.12, 0.15 * mouth_openness, 
                           color='#8B0000', fill=True, alpha=0.7)
            ax.add_patch(inner)
            # Язык
            tongue = Ellipse((0.5, mouth_y - 0.02), 0.08, 0.08, 
                            color='#FF69B4', fill=True, alpha=0.8)
            ax.add_patch(tongue)
        else:
            # Нейтральный рот (линия)
            line = plt.Line2D([0.4, 0.6], [mouth_y, mouth_y], 
                            color='black', linewidth=3)
            ax.add_line(line)
        
        # Цвет фона в зависимости от состояния (градиент)
        bg_color = params['color']
        ax.set_facecolor((bg_color[0] * 0.1, bg_color[1] * 0.1, bg_color[2] * 0.1))
        
        # Добавляем рамку с цветом состояния
        border = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                               boxstyle="round,pad=0.02",
                               edgecolor=bg_color, linewidth=4,
                               facecolor='none')
        ax.add_patch(border)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Заголовок с цветом
        title = ax.text(0.5, 0.95, f"Состояние: {state.value.upper()}", 
                       fontsize=16, fontweight='bold',
                       ha='center', va='top',
                       color=bg_color,
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='white', 
                                edgecolor=bg_color, linewidth=2))
        
        plt.tight_layout()
        
        # Сохраняем фигуру в файл для просмотра
        import os
        output_dir = "facial_expressions"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"expression_{state.value}.png")
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"📊 Визуализация сохранена: {filename}")
        
        # Закрываем фигуру чтобы избежать проблем с памятью и потоками
        # На macOS нельзя создавать GUI окна в фоновых потоках
        # Используем только сохранение в файл (backend 'Agg')
        plt.close(fig)
        
        return ax
    
    def render_pygame(self, state: Optional[ExpressionState] = None):
        """
        Улучшенная визуализация через pygame с плавной анимацией.
        
        Args:
            state: Состояние выражения
        """
        if not self.pygame_available:
            return
        
        import pygame
        import math
        
        if state is None:
            state = self.current_state
        
        params = self.EXPRESSION_PARAMS[state]
        
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_rendering = False
                return
        
        # Градиентный фон
        bg_color = tuple(int(c * 255) for c in params['color'])
        for y in range(400):
            ratio = y / 400
            color = tuple(int(bg_color[i] * (1 - ratio * 0.3)) for i in range(3))
            pygame.draw.line(self.screen, color, (0, y), (400, y))
        
        # Рисуем лицо с улучшенной графикой
        center_x, center_y = 200, 200
        radius = 150
        
        # Тень головы
        shadow_offset = 5
        pygame.draw.circle(self.screen, (50, 50, 50), 
                         (center_x + shadow_offset, center_y + shadow_offset), 
                         radius, 0)
        
        # Голова с градиентом
        color = tuple(int(c * 255) for c in params['color'])
        # Основной круг
        pygame.draw.circle(self.screen, (255, 219, 172), (center_x, center_y), radius)
        pygame.draw.circle(self.screen, (212, 165, 116), (center_x, center_y), radius, 3)
        
        # Блик на голове
        highlight_pos = (center_x - 30, center_y - 30)
        pygame.draw.circle(self.screen, (255, 255, 255), highlight_pos, 40, 0)
        pygame.draw.circle(self.screen, (255, 219, 172), highlight_pos, 35, 0)
        
        # Глаза с деталями
        eye_openness = params['eye_openness']
        eye_size = int(25 * eye_openness)
        eye_y = center_y - 30
        
        for eye_x in [center_x - 50, center_x + 50]:
            # Белок глаза
            pygame.draw.circle(self.screen, (255, 255, 255), (eye_x, eye_y), eye_size)
            pygame.draw.circle(self.screen, (0, 0, 0), (eye_x, eye_y), eye_size, 2)
            
            if params['eye_shape'] != 'narrow':
                # Радужка
                iris_size = int(eye_size * 0.6)
                pygame.draw.circle(self.screen, (74, 144, 226), (eye_x, eye_y), iris_size)
                # Зрачок
                pupil_size = int(eye_size * 0.3)
                pygame.draw.circle(self.screen, (0, 0, 0), (eye_x, eye_y), pupil_size)
                # Блик
                highlight_size = int(eye_size * 0.2)
                pygame.draw.circle(self.screen, (255, 255, 255), 
                                 (eye_x - 3, eye_y - 3), highlight_size)
        
        # Брови
        eyebrow_angle = params['eyebrow_angle']
        for brow_x in [center_x - 60, center_x + 60]:
            brow_y = center_y - 60 + int(eyebrow_angle * 20)
            # Бровь как несколько линий
            for i in range(5):
                offset = (i - 2) * 8
                start_pos = (brow_x + offset - 10, brow_y + int(eyebrow_angle * 5))
                end_pos = (brow_x + offset + 10, brow_y - int(eyebrow_angle * 5))
                pygame.draw.line(self.screen, (139, 69, 19), start_pos, end_pos, 3)
        
        # Рот
        mouth_y = center_y + 50
        if params['mouth_shape'] == 'smile':
            # Улыбка с деталями
            points = []
            for x in range(center_x - 60, center_x + 61):
                y = mouth_y - int(30 * math.sin((x - center_x + 60) * math.pi / 120))
                points.append((x, y))
            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 0, 0), False, points, 4)
        elif params['mouth_shape'] == 'frown':
            # Хмурый рот
            points = []
            for x in range(center_x - 60, center_x + 61):
                y = mouth_y + int(30 * math.sin((x - center_x + 60) * math.pi / 120))
                points.append((x, y))
            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 0, 0), False, points, 4)
        elif params['mouth_shape'] == 'open':
            # Открытый рот
            mouth_width = 40
            mouth_height = int(30 * params['mouth_openness'])
            pygame.draw.ellipse(self.screen, (0, 0, 0), 
                              (center_x - mouth_width, mouth_y - mouth_height // 2,
                               mouth_width * 2, mouth_height), 0)
            # Внутренность
            pygame.draw.ellipse(self.screen, (139, 0, 0), 
                              (center_x - mouth_width + 5, mouth_y - mouth_height // 2 + 5,
                               (mouth_width - 5) * 2, mouth_height - 10), 0)
        else:
            # Нейтральный рот
            pygame.draw.line(self.screen, (0, 0, 0), 
                           (center_x - 40, mouth_y), 
                           (center_x + 40, mouth_y), 4)
        
        # Текст состояния с фоном
        font = pygame.font.Font(None, 36)
        text = font.render(state.value.upper(), True, (255, 255, 255))
        text_rect = text.get_rect(center=(center_x, center_y + 130))
        # Фон для текста
        bg_rect = text_rect.inflate(20, 10)
        pygame.draw.rect(self.screen, color, bg_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 2)
        self.screen.blit(text, text_rect)
        
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
    
    def render_ascii(self, state: Optional[ExpressionState] = None) -> str:
        """
        Визуализация через ASCII-арт с детальной отрисовкой.
        
        Args:
            state: Состояние выражения
            
        Returns:
            ASCII-арт строка
        """
        if state is None:
            state = self.current_state
        
        params = self.EXPRESSION_PARAMS[state]
        
        # Шаблоны для разных состояний
        templates = {
            ExpressionState.NEUTRAL: [
                "    ╭─────────╮",
                "   ╱           ╲",
                "  │   ●     ●   │",
                "  │      │      │",
                "  │   ╰─────╯   │",
                "   ╲           ╱",
                "    ╰─────────╯",
                "   NEUTRAL"
            ],
            ExpressionState.FRIENDLY: [
                "    ╭─────────╮",
                "   ╱           ╲",
                "  │   ◉     ◉   │",
                "  │             │",
                "  │   ╰─────╯   │",
                "   ╲           ╱",
                "    ╰─────────╯",
                "   FRIENDLY"
            ],
            ExpressionState.STRICT: [
                "    ╭─────────╮",
                "   ╱           ╲",
                "  │   ◄     ►   │",
                "  │      │      │",
                "  │   ╰─────╯   │",
                "   ╲           ╱",
                "    ╰─────────╯",
                "   STRICT"
            ],
            ExpressionState.WARNING: [
                "    ╭─────────╮",
                "   ╱           ╲",
                "  │   ○     ○   │",
                "  │      │      │",
                "  │   ╰─────╯   │",
                "   ╲           ╱",
                "    ╰─────────╯",
                "   WARNING"
            ]
        }
        
        art = templates.get(state, templates[ExpressionState.NEUTRAL])
        return "\n".join(art)
    
    def start_realtime_rendering(self, update_callback=None):
        """
        Запуск визуализации в реальном времени.
        
        Args:
            update_callback: Функция обратного вызова для обновления состояния
                            (вызывается каждый кадр)
        """
        if self.is_rendering:
            return
        
        self.is_rendering = True
        
        if self.render_method == 'pygame':
            def render_loop():
                while self.is_rendering:
                    if update_callback:
                        update_callback()
                    self.render_pygame()
                    time.sleep(1/30)  # 30 FPS
            
            self.render_thread = threading.Thread(target=render_loop, daemon=True)
            self.render_thread.start()
        
        elif self.render_method == 'matplotlib':
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            
            fig, ax = plt.subplots(figsize=(6, 6))
            
            def animate(frame):
                if update_callback:
                    update_callback()
                self.render_matplotlib(ax=ax)
                return []
            
            anim = animation.FuncAnimation(fig, animate, interval=100, blit=False)
            plt.show()
    
    def stop_realtime_rendering(self):
        """Остановка визуализации в реальном времени."""
        self.is_rendering = False
        if self.render_thread:
            self.render_thread.join(timeout=1.0)
    
    def render(self, state: Optional[ExpressionState] = None) -> Optional[str]:
        """
        Универсальный метод рендеринга.
        
        Args:
            state: Состояние выражения
            
        Returns:
            Для emoji режима: строка с emoji
            Для других режимов: None (визуализация выполняется напрямую)
        """
        if state is None:
            state = self.current_state
        
        if self.render_method == 'emoji':
            return self.render_emoji(state)
        elif self.render_method == 'matplotlib':
            self.render_matplotlib(state)
            # Для matplotlib возвращаем текстовое описание
            return f"[Графическая визуализация: {state.value.upper()}]"
        elif self.render_method == 'pygame':
            self.render_pygame(state)
            return f"[Pygame: {state.value.upper()}]"
        elif self.render_method == 'ascii':
            return self.render_ascii(state)
        else:
            return self.render_emoji(state)  # Fallback
    
    def get_expression_params(self, state: ExpressionState) -> Dict:
        """
        Получение параметров выражения.
        
        Args:
            state: Состояние выражения
            
        Returns:
            Словарь с параметрами
        """
        return self.EXPRESSION_PARAMS[state].copy()
