from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Final, Literal

from server.http.common import streaming as _streaming
from server.http.common.canvas_detector import (
    AutoCanvasDetector,
    SmartRouter,
    CANVAS_THRESHOLDS,
    CODE_LANGUAGES,
)
from shared.models import JSONValue

# Увеличенные пороги для fallback
CANVAS_LINE_THRESHOLD: Final[int] = 60
CANVAS_CHAR_THRESHOLD: Final[int] = 3000
CANVAS_CODE_LINE_THRESHOLD: Final[int] = 15  # Было 28, теперь 15 для кода
CANVAS_DOCUMENT_LINE_THRESHOLD: Final[int] = 40  # Было 24, теперь 40 для документов
CANVAS_STATUS_CHARS_STEP: Final[int] = 640
_REQUEST_FILENAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<name>[A-Za-z0-9._/\-]+\.[A-Za-z0-9]{1,12})"
)
_CODE_FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_-]*)(?P<sep>\n|\\n)(?P<code>.*?)```",
    re.DOTALL,
)
_ARTIFACT_FILE_EXTENSIONS: Final[set[str]] = {
    "bash",
    "c",
    "cfg",
    "conf",
    "cpp",
    "css",
    "csv",
    "env",
    "go",
    "h",
    "hpp",
    "html",
    "ini",
    "java",
    "js",
    "json",
    "jsx",
    "kt",
    "md",
    "php",
    "py",
    "rb",
    "rs",
    "sh",
    "sql",
    "swift",
    "toml",
    "ts",
    "tsx",
    "txt",
    "xml",
    "yaml",
    "yml",
}
_EXT_TO_MIME: Final[dict[str, str]] = {
    "bash": "text/x-shellscript",
    "c": "text/plain",
    "cpp": "text/plain",
    "py": "text/x-python",
    "js": "text/javascript",
    "jsx": "text/javascript",
    "ts": "text/typescript",
    "tsx": "text/typescript",
    "json": "application/json",
    "html": "text/html",
    "css": "text/css",
    "md": "text/markdown",
    "txt": "text/plain",
    "sh": "text/x-shellscript",
    "yaml": "text/plain",
    "yml": "text/plain",
    "toml": "text/plain",
    "xml": "application/xml",
    "sql": "text/plain",
}


def _is_document_like_output(normalized: str) -> bool:
    """Проверяет, является ли вывод документом (не кодом)."""
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) < CANVAS_DOCUMENT_LINE_THRESHOLD:
        return False
    heading_like = sum(
        1
        for line in lines[: min(len(lines), 20)]
        if line.startswith("#") or line.lower().startswith(("section ", "chapter ", "## "))
    )
    return heading_like >= 2


def _request_likely_canvas(user_input: str) -> bool:
    """Проверяет, указывает ли запрос на использование Canvas.
    
    Учитывает более специфичные ключевые слова, чем раньше.
    """
    normalized = user_input.strip().lower()
    if not normalized:
        return False
    if _extract_named_file_markers(normalized):
        return True
    if re.search(r"\b\d+\s*(files?|файл(а|ов)?)\b", normalized):
        return True
    
    # Более специфичные ключевые слова (убрали общие "скрипт", "script")
    keywords = (
        # Явные запросы на файлы
        "файл",
        "files",
        "readme",
        "документ",
        "module",
        "класс",
        "config",
        "конфиг",
        "целиком",
        "полностью",
        "mini app",
        "mini ap",
        "мини приложение",
        "напиши файл",
        "создай файл",
        "сгенерируй файл",
        "prilozhen",
        "fail",
        "celikom",
        "polnost",
        # Кодовые слова
        "project",
        "модуль",
        "рефактор",
        "refactor",
    )
    return any(token in normalized for token in keywords)


def _analyze_code_blocks(response_text: str) -> list[dict[str, int | str | bool]]:
    """Анализирует блоки кода в ответе.
    
    Returns:
        Список словарей с info о каждом блоке: lang, lines, is_code
    """
    blocks = []
    for match in _CODE_FENCE_PATTERN.finditer(response_text):
        lang = match.group("lang").strip().lower() if match.group("lang") else "text"
        code = match.group("code")
        lines = code.count("\n")
        is_code = lang in CODE_LANGUAGES
        
        # Получаем порог для этого языка
        threshold = CANVAS_THRESHOLDS.get(lang, CANVAS_THRESHOLDS["text"])
        
        blocks.append({
            "lang": lang,
            "lines": lines,
            "is_code": is_code,
            "threshold": threshold,
            "exceeds_threshold": lines >= threshold if is_code else lines >= 30,
        })
    return blocks


def _should_render_result_in_canvas(
    *,
    response_text: str,
    files_from_tools: list[str],
    named_files_count: int,
    force_canvas: bool,
    user_input: str = "",
) -> bool:
    """Определяет, должен ли результат отображаться в Canvas.
    
    Улучшенная версия с учётом языка кода и контекста.
    """
    if force_canvas:
        return True
    if named_files_count > 0:
        return True
    if len(files_from_tools) >= 2:
        return True
        
    normalized = response_text.strip()
    if not normalized:
        return False
        
    # Анализируем блоки кода
    code_blocks = _analyze_code_blocks(response_text)
    
    # Если есть блоки кода, проверяем их по порогам
    if code_blocks:
        # Любой кодовый блок превышает порог -> Canvas
        for block in code_blocks:
            if block["is_code"] and block["exceeds_threshold"]:
                return True
        # Много блоков кода подряд (библиотека + пример)
        if len(code_blocks) > 1 and sum(b["lines"] for b in code_blocks) > 30:
            return True
            
    # Fallback на размер
    lines = normalized.splitlines()
    line_count = len(lines)
    char_count = len(normalized)
    
    if line_count >= CANVAS_LINE_THRESHOLD:
        return True
    if char_count >= CANVAS_CHAR_THRESHOLD:
        return True
        
    has_code_block = "```" in normalized
    if len(files_from_tools) == 1 and has_code_block and line_count >= 12:
        return True
    if has_code_block and line_count >= CANVAS_CODE_LINE_THRESHOLD:
        return True
    if _is_document_like_output(normalized):
        return True
        
    return False


def _sanitize_download_filename(file_name: str) -> str:
    normalized = file_name.replace("\\", "/").strip().split("/")[-1]
    if not normalized:
        return "artifact.txt"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in normalized)
    if safe.startswith("."):
        safe = f"file{safe}"
    return safe or "artifact.txt"


def _safe_zip_entry_name(file_name: str) -> str:
    normalized = file_name.replace("\\", "/").strip()
    parts = [part for part in normalized.split("/") if part and part not in {".", ".."}]
    if not parts:
        return "artifact.txt"
    safe_parts: list[str] = []
    for part in parts:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in part)
        if safe.startswith("."):
            safe = f"file{safe}"
        safe_parts.append(safe or "file")
    return "/".join(safe_parts)


def _artifact_mime_from_ext(ext: str | None) -> str:
    if not ext:
        return "text/plain"
    return _EXT_TO_MIME.get(ext.lower(), "text/plain")


def _normalize_candidate_file_name(raw_name: str) -> str:
    return raw_name.strip().strip(r"\`\"'()[]{}\u003c\u003e.,;:!?")


def _is_probable_named_file(candidate: str) -> bool:
    normalized = _normalize_candidate_file_name(candidate)
    if not normalized:
        return False
    if normalized.startswith("//") or "://" in normalized:
        return False
    if "/" in normalized:
        head = normalized.split("/", 1)[0]
        if "." in head and not head.startswith("."):
            return False
    if "." not in normalized:
        return False
    ext = normalized.rsplit(".", 1)[-1].lower()
    if ext not in _ARTIFACT_FILE_EXTENSIONS:
        return False
    return True


def _extract_named_file_markers(marker_chunk: str) -> list[str]:
    names: list[str] = []
    for match in _REQUEST_FILENAME_PATTERN.finditer(marker_chunk):
        raw_name = match.group("name")
        if not _is_probable_named_file(raw_name):
            continue
        normalized = _normalize_candidate_file_name(raw_name)
        if normalized:
            names.append(normalized)
    return names


def _normalize_code_fence_content(code_raw: str, sep: str) -> str:
    normalized = code_raw
    if sep == "\\n" and "\n" not in normalized and "\\n" in normalized:
        normalized = normalized.replace("\\n", "\n")
    return normalized.rstrip("\n")


def _extract_named_files_from_output(response_text: str) -> list[dict[str, str]]:
    if not response_text.strip():
        return []
    files: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in _CODE_FENCE_PATTERN.finditer(response_text):
        code = _normalize_code_fence_content(match.group("code"), match.group("sep"))
        if not code.strip():
            continue
        lang = match.group("lang").strip().lower()
        marker_start = max(0, match.start() - 220)
        marker_chunk = response_text[marker_start : match.start()]
        marker_matches = _extract_named_file_markers(marker_chunk)
        if not marker_matches:
            continue
        file_name_raw = marker_matches[-1]
        file_name = _safe_zip_entry_name(file_name_raw)
        if file_name in seen:
            continue
        seen.add(file_name)
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        if ext not in _ARTIFACT_FILE_EXTENSIONS:
            continue
        files.append(
            {
                "file_name": file_name,
                "file_ext": ext,
                "language": lang or ext,
                "file_content": code,
            }
        )
    return files


def _build_output_artifacts(
    *,
    response_text: str,
    display_target: Literal["chat", "canvas"],
    files_from_tools: list[str],
) -> list[dict[str, JSONValue]]:
    normalized = response_text.strip()
    if not normalized:
        return []

    now = datetime.now(UTC).isoformat()
    artifacts: list[dict[str, JSONValue]] = []
    named_files = _extract_named_files_from_output(response_text)
    for named_file in named_files:
        file_name = named_file["file_name"]
        file_ext = named_file["file_ext"] or None
        language = named_file["language"] or None
        file_content = named_file["file_content"]
        artifacts.append(
            {
                "id": uuid.uuid4().hex,
                "kind": "output",
                "artifact_kind": "file",
                "title": file_name,
                "content": file_content,
                "file_name": file_name,
                "file_ext": file_ext,
                "language": language,
                "file_content": file_content,
                "created_at": now,
                "display_target": display_target,
            }
        )

    del files_from_tools
    should_include_text = display_target == "canvas"
    if should_include_text and not named_files:
        first_line = next(
            (line.strip() for line in normalized.splitlines() if line.strip()),
            "Result",
        )
        artifacts.append(
            {
                "id": uuid.uuid4().hex,
                "kind": "output",
                "artifact_kind": "text",
                "title": first_line[:80],
                "content": response_text,
                "file_name": None,
                "file_ext": None,
                "language": None,
                "file_content": None,
                "created_at": now,
                "display_target": display_target,
            }
        )

    return artifacts


def _build_canvas_chat_summary(
    *,
    artifact_title: str | None,
    content_preview: str | None = None,
    user_input: str = "",
) -> str:
    """Возвращает осмысленный summary вместо шаблона.
    
    Args:
        artifact_title: Название артефакта
        content_preview: Первые символы контента для превью
        user_input: Оригинальный запрос пользователя
    """
    if isinstance(artifact_title, str):
        normalized = artifact_title.strip()
        if normalized:
            # Добавляем контекст из запроса пользователя
            context = ""
            if content_preview:
                preview = content_preview[:200].replace("\n", " ").strip()
                if len(content_preview) > 200:
                    preview += "..."
                if preview:
                    context = f" — {preview}"
            
            # Для файлов с расширениями - показываем тип
            if "." in normalized:
                ext = normalized.rsplit(".", 1)[-1].lower()
                if ext in CODE_LANGUAGES:
                    return f"📄 {normalized}{context}"
            
            return f"📄 {normalized}{context}"
            
    # Fallback: более информативное сообщение
    if "например" in user_input.lower():
        return "Пример кода (см. Canvas):"
        
    return "Результат сформирован в Canvas."


def _canvas_summary_title_from_artifact(
    artifact: dict[str, JSONValue] | None,
) -> str | None:
    if artifact is None:
        return None
    artifact_kind = artifact.get("artifact_kind")
    if artifact_kind != "file":
        return None
    file_name_raw = artifact.get("file_name")
    if isinstance(file_name_raw, str) and file_name_raw.strip():
        return _sanitize_download_filename(file_name_raw)
    title_raw = artifact.get("title")
    if not isinstance(title_raw, str):
        return None
    normalized = " ".join(title_raw.replace("`", " ").split())
    if not normalized:
        return None
    return normalized[:80]


def _stream_preview_indicates_canvas(preview_text: str) -> bool:
    """Определяет по превью, должен ли стрим идти в Canvas.
    
    Улучшенная версия с использованием AutoCanvasDetector.
    """
    return _streaming._stream_preview_indicates_canvas(
        preview_text,
        canvas_char_threshold=CANVAS_CHAR_THRESHOLD,
        canvas_code_line_threshold=CANVAS_CODE_LINE_THRESHOLD,
        extract_named_files_from_output_fn=_extract_named_files_from_output,
        extract_named_file_markers_fn=_extract_named_file_markers,
    )


def _get_content_preview(response_text: str, max_chars: int = 200) -> str:
    """Возвращает краткое превью контента для чат-сообщения."""
    normalized = response_text.strip()
    if not normalized:
        return ""
    
    # Ищем первую осмысленную строку
    for line in normalized.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("```"):
            preview = stripped[:max_chars]
            if len(stripped) > max_chars:
                preview += "..."
            return preview
    
    return normalized[:max_chars]
