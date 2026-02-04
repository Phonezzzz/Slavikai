# Architecture — Slavik AI

## Цель системы
Локальный агент с пошаговым планированием и набором инструментов (файлы, shell, web, проектный индекс, workspace, TTS/STT, картинки) с трассировкой и памятью.

## Слои и компоненты
- **Core** (`core/*`): `Agent` маршрутизирует запросы, включает Safe Mode для инструментов, строит контекст (memory, feedback hints, prefs, vector search, workspace). `Planner` генерирует 2–8 шагов (LLM или эвристика). `Executor` выполняет шаги последовательно. `AutoAgent` — примитивный параллельный запуск подагентов без инструментов. `Tracer` пишет reasoning/шаги в `logs/trace.log`.
- **LLM** (`llm/*`): `OpenRouterBrain`, `LocalHttpBrain`, фабрика `create_brain`, менеджер `BrainManager`. Конфиги — `config/model_config.json`.
- **Tools Layer** (`tools/*`): регистрируются через `ToolRegistry`, лог вызовов `logs/tool_calls.log`, Safe Mode блокирует `web` и `shell`. Инструменты: filesystem (sandbox), shell (whitelist), web search/fetch (Serper + HTTP fetch), project index/search (VectorIndex), workspace (list/read/write/patch/run в sandbox/project), TTS/STT (HTTP), image analyze/generate (локально), http client (лимиты).
- **Memory & Index** (`memory/*`): `MemoryManager` (SQLite, строгая валидация записей/prefs/facts), `FeedbackManager` (rating/severity/hint, авто-инжект major/fatal), `VectorIndex` (sentence-transformers, namespaces `code`/`docs`, лимиты per-namespace и total).
- **Sandbox/Safety**: fs/shell/workspace ограничены `sandbox` (shell cwd `sandbox`, workspace `sandbox/project`). Safe Mode в реестре отключает `web` и `shell` (другие сетевые инструменты пока не блокируются). ProjectTool не привязан к sandbox (риск).

## Поток одного запроса
1) Если сообщение — команда `/fs|/web|/sh|/plan|/auto|/project|/img...` → прямой вызов инструмента через ToolRegistry.
2) Иначе: классификация сложности (`simple|complex`). Для complex — построение плана (LLM/эвристика), исполнение шагов через Executor+ToolGateway.
3) Для простых ответов — Brain.generate с добавленным контекстом (memory, feedback hints, prefs, vector index, workspace file/selection). Результат сохраняется в память, трассируется.

## Память и индекс
- `MemoryManager`: таблица memory (id/kind/content/tags/meta/timestamp), методы save/search/get_recent/get_user_prefs/get_project_facts.
- `VectorIndex`: SQLite + embeddings, namespaces `code`/`docs`, prune по max_records и max_total, cosine search.
- `FeedbackManager`: рейтинг/серьёзность/hint, статистика, выборка bad/offtopic; major/fatal hints попадают в контекст.

## Workspace слой
- Инструменты `workspace_list/read/write/patch/run` работают в `sandbox/project`, ограниченные расширения (.py/.md/.txt/.json/.toml/.yaml/.yml), лимит размера (2 MB), run только .py с таймаутом.
- Контракт `workspace_patch`: только **single-file** unified hunk patch для одного `path` (обязателен `@@ ... @@`), multi-file diff и заголовки `diff --git` / `---` / `+++` блокируются.

## Наблюдаемость
- Trace: `logs/trace.log` — reasoning, шаги плана, ошибки.
- Tool calls: `logs/tool_calls.log` — инструмент, ok/error, meta/args.

## Ограничения и пробелы
- Safe Mode покрывает только `web` и `shell`; TTS/STT/web-fetch/project index не блокируются.
- ProjectTool работает по абсолютным путям (нет sandbox).
- Планировщик/исполнитель выбирают инструменты по ключевым словам, без строгой схемы.
- VectorIndex загружает sentence-transformers синхронно (потенциальная задержка).
