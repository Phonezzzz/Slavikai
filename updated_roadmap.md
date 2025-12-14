# Updated Roadmap (по фактическому состоянию кода)

Roadmap разбит на 3 уровня: Stage A (ядро агента), Stage B (Workspace слой), Stage C (Desktop UI/расширения).

## Stage A — ядро агента

### Сделано

- Agent/Planner/Executor: `core/agent.py`, `core/planner.py`, `core/executor.py`.
- DualBrain и режимы `single|dual|critic-only`: `llm/dual_brain.py` + `config/mode_config.py`.
- Tool-слой: `tools/tool_registry.py` + `tools/tool_logger.py` + `core/tool_gateway.py`.
- Safe-mode на уровне ToolRegistry (блок инструментов из `SAFE_MODE_TOOLS_OFF` в `core/agent.py`).
- Трассировка: `core/tracer.py` → `logs/trace.log`.
- Память/фидбек: `memory/memory_manager.py`, `memory/feedback_manager.py`.
- Векторный индекс: `memory/vector_index.py` (namespaces `code/docs`).
- LLM клиенты: `llm/openrouter_brain.py`, `llm/local_http_brain.py` (HTTP, таймауты, валидация ответа).

### Частично

- Планирование и маппинг шаг→действие: операции назначаются по ключевым словам (`core/planner.py::_map_operation`), а `Executor` по умолчанию использует упрощённые запросы (`fs` всегда `list`, `shell` всегда `echo step`).
- AutoAgent: простая декомпозиция по “и” и параллельный вызов `Brain.generate` без инструментов (`core/auto_agent.py`).
- System prompt: есть `config/system_prompts.py`, но применение зависит от `ModelConfig.system_prompt` (автоматической “глобальной” инъекции промпта в агенте нет).

### НЕ сделано (NOT IMPLEMENTED)

- CLI/headless режим (есть только UI + внутренние “/команды” в `Agent.respond`).
- Политики доступа/ролей/пермиссий поверх инструментов (есть только safe-mode и sandbox).
- Структурированная схема “plan → tool_request” без эвристик (в текущем виде это частично/эвристика).

### Следующий шаг

- Усилить связку `Planner/Executor` с реальными операциями (включая `workspace_*`), сохранив тестируемость.
- Явно зафиксировать политику sandbox для `shell` (см. риск с `sandbox_root` в аудите) и покрыть тестом.

## Stage B — Workspace слой

### Сделано

- Workspace tools `list/read/write/patch/run`: `tools/workspace_tools.py` (корень `sandbox/project/`).
- UI WorkspacePanel: `ui/workspace_panel.py` (дерево файлов, редактор, diff preview, запуск, “спросить AI”).
- ProjectTool индексации/поиска по workspace: `tools/project_tool.py` + `memory/vector_index.py`.

### Частично

- `workspace_patch` применяет `unified diff` только к одному файлу (нет многофайловых патчей).
- `workspace_run` запускает только `.py` (другие языки/команды — NOT IMPLEMENTED).
- Индексация проекта не автоматизирована: вызывается только через инструмент `project index`.

### НЕ сделано (NOT IMPLEMENTED)

- Версионирование/история изменений workspace (git-интеграции нет).
- Фоновая индексация/обновление индекса по изменениям файлов.

### Следующий шаг

- Добавить безопасную поддержку многофайловых патчей или явно зафиксировать “single-file only” как контракт.
- Добавить автоматический (или полуавтоматический) `project index` для `sandbox/project/` в UI.

## Stage C — расширения (Desktop приложение PySide6)

### Сделано

- PySide6 UI: `main.py`, `ui/main_window.py`.
- Панели наблюдаемости: `ui/trace_view.py`, `ui/tool_logs_view.py`, `ui/reasoning_panel.py`, `ui/logs_view.py`.
- Memory/Docs/Feedback панели: `ui/memory_view.py`, `ui/docs_panel.py`, `ui/feedback_panel.py`.
- TTS/STT в UI: `ui/chat_view.py` + `ui/audio_player.py` + инструменты `tools/tts_tool.py`, `tools/stt_tool.py`.

### Частично

- ToolsPanel (`ui/tools_panel.py`) управляет не всеми зарегистрированными инструментами (например, нет переключателя для `project`).
- Настройки shell/моделей есть (`ui/settings_dialog.py`), но корректность путей shell config зависит от соблюдения sandbox-ограничений `tools/shell_tool.py`.

### НЕ сделано (NOT IMPLEMENTED)

- Авто-тесты UI (Qt) — в `tests/` тестов UI нет.
- Инсталлятор/пакетирование приложения (pyinstaller/qtdeploy) — отсутствует.

### Следующий шаг

- Добавить минимальный smoke-test UI (хотя бы импорт/создание основных виджетов) или явно закрепить manual QA чеклист.
- Свести UX: уменьшить количество панелей в одном splitter и/или сделать вкладки (архитектурно это локально в `ui/main_window.py`).

