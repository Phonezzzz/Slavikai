# SlavikAI — обзор проекта

SlavikAI — Python‑агент с планированием и исполнением действий через инструменты в песочнице.

## Назначение

- Чат‑интерфейс + наблюдаемость (трейсы, логи вызовов инструментов).
- LLM-ответы через абстракцию `Brain` (OpenRouter или локальный HTTP endpoint).
- Пошаговые планы (`Planner`) и исполнение (`Executor`) через слой инструментов (`ToolRegistry`).
- Память/фидбек (SQLite) и векторный поиск контекста по индексированным файлам.
- Workspace-редактор в песочнице: list/read/write/patch/run.
- Голос: TTS/STT (HTTP).

## Архитектура (по коду)

**Core**
- `core/agent.py` — главный роутер: команды, планирование/исполнение, safe-mode, сбор контекста.
- `core/planner.py` — классификация сложности и построение `TaskPlan` (LLM или эвристика).
- `core/executor.py` — пошаговое выполнение плана.
- `core/auto_agent.py` — параллельные мини-агенты (простая декомпозиция цели).
- `core/tracer.py` — запись трейсов в `logs/trace.log`.

**LLM**
- `llm/openrouter_brain.py` — клиент OpenRouter.
- `llm/local_http_brain.py` — клиент для локального OpenAI-compatible endpoint.
- `llm/brain_factory.py`, `llm/brain_manager.py`, `llm/types.py` — сборка и типы.

**Tools**
- `tools/tool_registry.py` — реестр, включение/выключение инструментов, safe-mode блок.
- `tools/tool_logger.py` — лог вызовов инструментов в `logs/tool_calls.log`.
- Реальные инструменты: `tools/filesystem_tool.py`, `tools/shell_tool.py`, `tools/web_search_tool.py`,
  `tools/project_tool.py`, `tools/image_*`, `tools/tts_tool.py`, `tools/stt_tool.py`,
  `tools/workspace_tools.py`.

**Memory / Index / Feedback**
- `memory/memory_manager.py` — SQLite память (`memory/memory.db`).
- `memory/feedback_manager.py` — SQLite фидбек (`memory/feedback.db`).
- `memory/vector_index.py` — SQLite векторный индекс (`memory/vectors.db`) + `sentence-transformers`.

## Pipeline одного запроса (Agent → Planner → Executor → Tools → Memory/Index → Workspace)
1. Клиент отправляет текст в `Agent.respond(...)`.
2. `core/agent.py`:
   - если это команда (`/...`) — маршрутизирует в инструменты/режимы;
   - иначе оценивает сложность (`Planner.classify_complexity`);
   - “сложные” запросы: `Planner.build_plan(...)` → `Executor.run(...)` → `ToolRegistry.call(...)`;
   - “простые” запросы: строит контекст (память/фидбек/векторный поиск/workspace) и вызывает `Brain.generate(...)`.
3. Результаты исполнения шагов и вызовов инструментов пишутся в `logs/trace.log` и `logs/tool_calls.log`.
4. Часть ответов/фидбека сохраняется в SQLite (`memory/*`).

## TTS/STT

- `tools/tts_tool.py` генерирует аудио и пишет файл в `sandbox/audio/`.
- `tools/stt_tool.py` читает файл из `sandbox/audio/` и отправляет на распознавание.

## Workspace (list/read/write/patch/run)

Workspace — это каталог `sandbox/project/`.

- `workspace_list` — показать дерево файлов.
- `workspace_read` — прочитать файл (только разрешённые расширения, лимит размера).
- `workspace_write` — записать файл (только разрешённые расширения).
- `workspace_patch` — применить `unified diff` к одному файлу (строгая проверка, лимиты на размер/строки).
- `workspace_run` — запустить `.py` через `sys.executable` (таймаут, cwd=`sandbox/`).

Инструменты workspace вызываются через публичный API `agent.call_tool(...)` (см. `_call_tool`).

## Ограничения (sandbox, safe-mode)

- Большинство файловых операций инструментов ограничены песочницей `sandbox/` и/или `sandbox/project/`.
- `shell` использует whitelist команд из `config/shell_config.json`, запрещает цепочки `&&/||/;`, абсолютные пути и `..`.
- Safe mode (по умолчанию включён) отключает инструменты `web`, `shell`, `project`, `tts`, `stt` (см. `SAFE_MODE_TOOLS_OFF` в `core/agent.py`).

## Структура репозитория

- `config/` — конфиги моделей/tools/shell.
- `core/` — Agent/Planner/Executor/Tracer.
- `llm/` — клиенты LLM.
- `tools/` — инструменты и реестр.
- `memory/` — SQLite хранилища и векторный индекс.
- `sandbox/` — песочница (workspace, аудио).
- `tests/` — pytest тесты (настройки в `pyproject.toml`).
- `scripts/` — скрипты качества.
