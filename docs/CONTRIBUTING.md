# Contributing to SlavikAI

## Требования

- Python 3.12.
- Зависимости из `requirements.txt`.

## Установка и запуск

1) Создайте виртуальное окружение и установите зависимости (рекомендуется через `Makefile`):

- `make venv`
- (опционально) `make activate` → покажет команду для активации venv в текущем shell

2) Запуск backend (пример):

- `python -m server`

## Конфигурация (фактические точки интеграции)

- Модели: `config/model_config.json`.
- Переключатели инструментов: `config/tools.json` (создаётся при сохранении, см. `config/tools_config.py`).
- Shell whitelist: по умолчанию `config/shell_config.json` (см. `config/shell_config.py`, `tools/shell_tool.py`).

Переменные окружения, которые используются в коде:
- LLM: `OPENROUTER_API_KEY`, `LOCAL_LLM_URL`, `LOCAL_LLM_API_KEY`.
- Web search: `SERPER_API_KEY` (или `config/web_search_api_key.txt`).
- TTS: `TTS_API_KEY`, `TTS_VOICE_ID`.
- STT: `STT_API_KEY`.

## Как запускать тесты и проверки качества

Проект настроен на `ruff` + `mypy (strict)` + `pytest-cov` (см. `pyproject.toml`).

Рекомендуемый порядок:

- `make check` (one-shot прогон всех проверок)

Или вручную:

- `ruff check .`
- `ruff format --check .`
- `python -m mypy .`
- `pytest`

Альтернатива: `bash scripts/check.sh` (one-shot прогон всех проверок).

## Как добавить новый инструмент (Tool)

1) Реализуйте инструмент в `tools/`:

- интерфейс: `tools/protocols.py` (`handle(ToolRequest) -> ToolResult`);
- возвращайте данные через `ToolResult.success(...)` / `ToolResult.failure(...)`;
- валидируйте типы входных аргументов (`request.args`), не полагайтесь на “магические” значения.

2) Соблюдайте sandbox/safe-mode:

- файловые пути должны оставаться внутри `sandbox/` или `sandbox/project/` (см. проверки пути в `tools/filesystem_tool.py`, `tools/workspace_tools.py`, `tools/project_tool.py`);
- если инструмент опасный (сеть/система/внешние эффекты) — добавьте его в safe-mode блок-лист (`core/agent.py` → `SAFE_MODE_TOOLS_OFF`) и покройте тестом.

3) Зарегистрируйте инструмент:

- `core/agent.py` → `_register_tools()` через `ToolRegistry.register(...)`.

4) Добавьте тесты:

- успех + основные ошибки (валидация args);
- ограничения песочницы (попытка выйти за пределы sandbox);
- safe-mode блокировку (если применимо).

## Как писать новые тесты

- Все тесты — в `tests/`, `pytest` обнаружит `test_*.py`.
- Не делайте реальные сетевые запросы:
  - подменяйте `HttpClient`/HTTP слой (пример: `tests/test_tts_stt_tools.py`);
  - для `VectorIndex` подменяйте модель через monkeypatch (пример: `tests/test_vector_index.py`).
- Для SQLite используйте `tmp_path` (пример: `tests/test_memory.py`).

## Workspace tools: как с ними работать

Workspace ограничен `sandbox/project/`:

- список файлов: инструмент `workspace_list`;
- чтение/запись: `workspace_read`/`workspace_write` (есть whitelist расширений и лимиты размера);
- патчи: `workspace_patch` (строгий `unified diff`, лимиты);
- запуск кода: `workspace_run` (только `.py`, таймаут).


## Commit rules и CI

- Формальных правил коммитов в репозитории нет (NOT IMPLEMENTED).
- CI workflow (GitHub Actions) в отслеживаемом репозитории отсутствует (NOT IMPLEMENTED): нет `./.github/workflows/*` в git.
