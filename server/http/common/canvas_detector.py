from __future__ import annotations

import re
from typing import Final, Literal, Optional, TypedDict


class CanvasDecision(TypedDict):
    action: Literal["promote_to_canvas", "keep_in_chat"]
    lang: str | None


# Рекомендуемые пороги по языкам
CANVAS_THRESHOLDS: Final[dict[str, int]] = {
    "python": 15,
    "javascript": 15,
    "js": 15,
    "typescript": 15,
    "ts": 15,
    "java": 15,
    "go": 15,
    "rust": 15,
    "cpp": 15,
    "c": 15,
    "json": 10,
    "yaml": 12,
    "yml": 12,
    "markdown": 40,
    "md": 40,
    "text": 30,
    "shell": 8,
    "bash": 8,
    "sh": 8,
}

CODE_LANGUAGES: Final[set[str]] = {
    "python", "javascript", "typescript", "java", "go", "rust",
    "cpp", "c", "json", "yaml", "yml", "js", "ts", "cs",
    "ruby", "rb", "php", "swift", "kotlin", "kt", "scala",
    "r", "matlab", "julia", "dart", "elixir", "erlang",
    "haskell", "clojure", "lisp", "scheme", "ocaml", "fsharp",
}


class AutoCanvasDetector:
    """Streaming-aware детектор для автоматического переключения в Canvas.
    
    Анализирует поток чанков от LLM и принимает решение о рендеринге
    на основе языка кода и размера блоков.
    """
    
    def __init__(self, default_threshold: int = 15):
        self.buffer = ""
        self.in_code_block = False
        self.code_lines = 0
        self.language: str | None = None
        self.code_start_idx = 0
        self.threshold = default_threshold
        self.decision_made = False
        self._code_fence_pattern = re.compile(r"```(\w+)?")
        self._close_fence_pattern = re.compile(r"```\s*$")
        
    def feed(self, chunk: str) -> Optional[CanvasDecision]:
        """Вызывается на каждом чанке от LLM.
        
        Returns:
            CanvasDecision если решение принято, None если нужно продолжить.
        """
        if self.decision_made:
            return None
            
        self.buffer += chunk
        
        # Ищем начало ``` или ~~~
        if not self.in_code_block:
            match = self._code_fence_pattern.search(self.buffer)
            if match:
                self.in_code_block = True
                self.language = match.group(1) if match.group(1) else "text"
                self.code_start_idx = match.end()
                # Считаем строки сразу после открытия
                code_so_far = self.buffer[self.code_start_idx:]
                self.code_lines = code_so_far.count("\n")
                
                # Проверяем, не закрылся ли блок сразу
                if self._close_fence_pattern.search(code_so_far):
                    return self._decide_render()
        else:
            # Уже в блоке кода - считаем строки
            code_content = self.buffer[self.code_start_idx:]
            self.code_lines = code_content.count("\n")
            
            # Проверяем конец блока
            if self._close_fence_pattern.search(code_content):
                return self._decide_render()
            
            # Проверяем early threshold для длинных блоков
            lang = self.language or "text"
            threshold = CANVAS_THRESHOLDS.get(lang.lower(), self.threshold)
            
            # Для кода - early decision на пороге
            if lang.lower() in CODE_LANGUAGES and self.code_lines >= threshold:
                self.decision_made = True
                return CanvasDecision(action="promote_to_canvas", lang=self.language)
                
        return None
        
    def _decide_render(self) -> CanvasDecision:
        """Принимает финальное решение: Canvas или Chat."""
        self.decision_made = True
        
        lang = self.language or "text"
        lang_lower = lang.lower()
        
        # Получаем порог для языка
        threshold = CANVAS_THRESHOLDS.get(lang_lower, self.threshold)
        
        # Проверяем, является ли язык кодом
        is_code_lang = lang_lower in CODE_LANGUAGES
        
        # Условия для Canvas:
        if self.code_lines > threshold and is_code_lang:
            return CanvasDecision(action="promote_to_canvas", lang=self.language)
        elif self.code_lines > 30:  # Даже plain text, если очень длинный
            return CanvasDecision(action="promote_to_canvas", lang="text")
        else:
            return CanvasDecision(action="keep_in_chat", lang=self.language)
    
    def is_in_code_block(self) -> bool:
        """Возвращает True, если сейчас внутри блока кода."""
        return self.in_code_block and not self.decision_made
    
    def get_code_lines(self) -> int:
        """Возвращает текущее количество строк в блоке кода."""
        if not self.in_code_block:
            return 0
        code_content = self.buffer[self.code_start_idx:]
        return code_content.count("\n")


class SmartRouter:
    """Умный роутер для принятия решений о Canvas на основе контекста."""
    
    def __init__(self):
        self._code_fence_pattern = re.compile(
            r"```(?P<lang>[a-zA-Z0-9_-]*)\n(?P<code>.*?)```",
            re.DOTALL,
        )
        
    def should_use_canvas(
        self,
        content: str,
        context: dict,
    ) -> bool:
        """Определяет, должен ли контент отображаться в Canvas.
        
        Args:
            content: Полный ответ ассистента
            context: Контекст запроса с ключами:
                - user_msg: оригинальный запрос пользователя
                - file_count: количество файлов в запросе
                - mode: режим (edit, create, etc.)
        """
        code_blocks = self._extract_code_blocks(content)
        user_msg = context.get("user_msg", "").lower()
        
        # Правило 1: Один большой блок (>20 строк для общего случая)
        if any(len(block.split("\n")) > 20 for block in code_blocks):
            return True
            
        # Правило 2: Много файлов (user просит несколько файлов)
        if context.get("file_count", 0) > 1:
            return True
            
        # Правило 3: Редактирование существующего файла
        if context.get("mode") == "edit" and code_blocks:
            return True
            
        # Правило 4: Структурированные данные (JSON/YAML > 10 строк)
        for block in code_blocks:
            if block.startswith(("json", "yaml")) and len(block.split("\n")) > 10:
                return True
                
        # Правило 5: Не триггерить на примеры в обучении
        if "например" in user_msg and len(code_blocks) == 1:
            lines = code_blocks[0].split("\n")
            if len(lines) < 30:
                return False
                
        # Правило 6: Явный запрос показать в Canvas
        if any(kw in user_msg for kw in ["canvas", "покажи в canvas", "в канвас"]):
            return True
            
        return bool(code_blocks) and sum(len(b) for b in code_blocks) > 500
    
    def _extract_code_blocks(self, content: str) -> list[str]:
        """Извлекает блоки кода из markdown."""
        blocks = []
        for match in self._code_fence_pattern.finditer(content):
            blocks.append(match.group("code"))
        return blocks


class CanvasMode:
    """Режимы Canvas для edge cases."""
    DISABLED = "disabled"
    MULTI_FILE = "multi_file"
    SINGLE_FILE = "single_file"


def handle_edge_cases(
    content: str,
    user_intent: str,
    detector: AutoCanvasDetector,
) -> str | None:
    """Обрабатывает edge cases для Canvas.
    
    Returns:
        CanvasMode константа или None если стандартная логика.
    """
    user_lower = user_intent.lower()
    
    # Если пользователь явно сказал "покажи в чате" (override)
    if "в чате" in user_lower or "не открывай canvas" in user_lower:
        return CanvasMode.DISABLED
        
    # Если несколько маленьких блоков подряд (библиотека + пример)
    blocks = content.split("```")
    if len(blocks) > 2 and sum(len(b) for b in blocks) > 500:
        return CanvasMode.MULTI_FILE
        
    return None


def get_threshold_for_lang(lang: str | None) -> int:
    """Возвращает порог для конкретного языка."""
    if not lang:
        return CANVAS_THRESHOLDS["text"]
    return CANVAS_THRESHOLDS.get(lang.lower(), CANVAS_THRESHOLDS["text"])


def is_code_language(lang: str | None) -> bool:
    """Проверяет, является ли язык языком программирования."""
    if not lang:
        return False
    return lang.lower() in CODE_LANGUAGES
