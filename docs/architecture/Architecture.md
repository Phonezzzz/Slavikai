# Architecture — SlavikAI current runtime

Этот документ фиксирует **текущее фактическое устройство** системы после PR-0..PR-7.
Целевое поведение runtime определено в `docs/architecture/ARCH_CANON.md`.
Если здесь описан legacy-путь, это не делает его целевой архитектурой.

## Цель

SlavikAI сейчас работает как single-user/server-side agent runtime с контурами chat,
workspace/MWV и auto. После PR-0..PR-9 в коде есть честный tool-calling contract,
read-only chat integration через `AgentToolLoop`, split chat/workspace API paths,
debug-only command lane и единый terminal backend. Часть старого runtime ещё остаётся
legacy-обвязкой.

## Основные слои

- **Core** (`core/*`)
  - Оркестрация: `Agent` + mixins.
  - Tool loop: `core/tool_loop.py` вызывает `Brain.generate(..., tools=...)`, исполняет
    `tool_calls` через `ToolGateway` и добавляет `role="tool"` сообщения.
  - Chat integration: `runtime_mode=ask` может использовать только описанные read-capability
    tools; write/exec tool calls остаются вне chat path.
  - Планирование/исполнение: `Planner`/`Executor` больше не извлекают tool args из prose;
    они принимают только explicit `PlanStep.operation` + `PlanStep.tool_args`.
  - Трассировка: `Tracer` (`logs/trace.log`).
- **MWV runtime** (`core/mwv/*`)
  - Цикл `ManagerRuntime -> WorkerRuntime -> VerifierRuntime`.
  - Успех только при `WorkStatus.SUCCESS` и `VerificationStatus.PASSED`.
  - Ограниченный retry через `RunContext.max_retries`.
- **Auto runtime** (`core/auto_runtime.py`)
  - Контур `planner -> coder pool -> merge -> verifier`.
  - Поддерживает паузу `waiting_approval` и resume.
- **LLM слой** (`llm/*`)
  - Провайдеры: `xai`, `openrouter`, `local`, `inception`.
  - Контракт: `LLMMessage.role` включает `tool`, `LLMResult.tool_calls`,
    `Brain.generate(..., tools=None)`.
  - Provider caveat: не каждый backend реально исполняет native provider tool calling;
    unsupported providers обязаны явно игнорировать/ограничивать tools, а не имитировать их regex-парсингом.
  - `openai` используется только для STT endpoint/ключа в UI-настройках (не chat provider).
- **Tools** (`tools/*`)
  - Реестр: `ToolRegistry`.
  - Descriptor: `name`, `description`, `parameters_schema`, capability/risk classes.
  - Журнал вызовов: `logs/tool_calls.log`.
  - Terminal: `tools/terminal_tool.py`, режимы `oneshot|pty`.
- **Storage/Memory** (`memory/*`)
  - `memory/memory.db`, `memory/memory_companion.db`, `memory/vectors.db`.
  - UI sessions пока физически совместимые: `ui_messages.lane` остаётся legacy
    SQLite/import/export detail, но storage adapter уже отдаёт chat/workspace
    domain records без `lane`; полностью отдельные таблицы ещё не введены.

## Маршрутизация запроса (current legacy runtime)

1. Сообщение, начинающееся с `/`, идёт в command lane (`handle_tool_command`) и не проходит через MWV.
   - Разрешены только debug-команды `/trace` и `/end-session`. Подробнее — `docs/for-humans/COMMANDS.md`.
2. Для обычного текста:
   - `runtime_mode=ask` — сразу chat-ветка (без `classify_request`).
   - `runtime_mode=auto` — выполняется классификация/skill-проверка, затем запуск auto-контура.
   - `runtime_mode=act|plan` — в legacy runtime используется `classify_request(...)` (`chat` или `mwv`).
3. Целевой tool path:
   - LLM получает `ToolSpec[]`.
   - LLM возвращает `tool_calls`.
   - Runtime вызывает `ToolGateway`.
   - Результат возвращается в LLM как `LLMMessage(role="tool", tool_call_id=...)`.

## Инструменты (зарегистрированные имена)

- Базовые: `fs`, `web`, `shell`, `project`.
- Медиа: `image_analyze`, `image_generate`, `tts`, `stt`.
- Workspace: `workspace_list`, `workspace_read`, `workspace_write`, `workspace_create`, `workspace_rename`, `workspace_move`, `workspace_delete`, `workspace_patch`, `workspace_run`, `workspace_terminal_run`.
- `workspace_terminal_run` — restricted one-shot режим общего `TerminalTool`.
- `workspace_patch` контракт: single-file hunk patch для одного `path` (без `diff --git` / `---` / `+++` заголовков).

## Sandbox и безопасность

- `fs` работает в `sandbox/`.
- `workspace_*` и `project` ограничены `sandbox/project/`.
- `shell` использует sandbox root + ограничения конфигурации.
- Safe-mode отключает рискованные инструменты через `SAFE_MODE_TOOLS_OFF`, включая `workspace_run` и `workspace_terminal_run`.

## HTTP/UI слой

- OpenAI-совместимые endpoints: `/v1/models`, `/v1/chat/completions`.
- Служебные endpoints: `/slavik/trace/{trace_id}`, `/slavik/tool-calls/{trace_id}`, `/slavik/feedback`, `/slavik/approve-session`.
- UI API и workflow endpoints регистрируются в `server/http/routes.py`.

### slavik_meta.runtime_mode contract (для /v1/chat/completions)

- `runtime_mode=ask|auto` — поддерживаемый opt-in.
- `runtime_mode=plan|act` — `invalid_request_error` (использовать UI workflow).
- без `runtime_mode` — legacy-поведение текущего runtime.

Это не конфликтует с `ARCH_CANON`: `plan|act` являются целевыми runtime-ролями, но не доступны
как `/v1/chat/completions` opt-in.

### UI API endpoint groups

- Sessions/folders: `/ui/api/folders`, `/ui/api/sessions`, `/ui/api/sessions/{session_id}`.
- Workflow: `/ui/api/mode`, `/ui/api/plan/*`, `/ui/api/runtime/init`.
- Chat: `/ui/api/chat/send`, `/ui/api/chat/events/{session_id}`.
- Workspace: `/ui/api/workspace/send`, `/ui/api/workspace/events/{session_id}`, `/ui/api/workspace/*`.
- Legacy compatibility: `/ui/api/events/stream` ещё существует как общий stream, но новый код
  должен использовать split chat/workspace endpoints.

## Backend PTY Terminal API

Реализован общим `tools/terminal_tool.py`; `server/http/handlers/terminal.py` —
только HTTP-оболочка для PTY режима.

Один PTY-терминал на сессию. Доступен только при `policy.profile = yolo`.

### Endpoints

| Метод | Путь | Доступ |
|---|---|---|
| `POST` | `/ui/api/terminal` | yolo only |
| `GET` | `/ui/api/terminal` | владелец сессии |
| `POST` | `/ui/api/terminal/input` | yolo only |
| `POST` | `/ui/api/terminal/resize` | yolo only |
| `POST` | `/ui/api/terminal/close` | владелец сессии |
| `GET` | `/ui/api/terminal/stream` | владелец сессии (SSE) |

### Правила

- `create` / `input` / `resize` требуют `policy.profile = yolo`. Иначе `403 terminal_yolo_required`.
- `get` / `close` / `stream` доступны владельцу сессии без yolo-gate.
- `stream` поддерживает `Last-Event-ID` для replay событий из ring-буфера (256 событий).
- `TerminalTool` регистрируется в `app["terminal_manager"]` при старте; shutdown — через `app.on_cleanup`.
- При удалении сессии (`DELETE /ui/api/sessions/{id}`) PTY-терминал закрывается автоматически.

### Режимы TerminalTool

- `oneshot` — одна команда, без PTY, через `workspace_terminal_run` и tool gateway approvals.
- `pty` — интерактивная сессия через `/ui/api/terminal/*`, yolo-gate, resize и SSE-стрим.

## Проверки качества

`make check` — canonical gate перед любым merge:

- `scripts/check_no_legacy_ui.sh`
- `ruff check .`
- `ruff format --check .`
- `mypy .`
- `npm run typecheck` (UI)
- `pytest` с покрытием (порог ≥ 80%).

## Наблюдаемость

- Trace: `logs/trace.log`.
- Tool calls: `logs/tool_calls.log`.
- UI storage: `.run/ui_sessions.db`.

## Legacy, который нельзя расширять

- `core/agent*.py` остаются крупной mixin-обвязкой и должны постепенно сжиматься вокруг
  `AgentToolLoop`/`ToolGateway`, а не получать новые ветки.
- `classify_request(...)` всё ещё используется в legacy routing; новые tool-capabilities не
  должны добавляться через keyword router.
- `Planner`/`Executor` больше не делают regex extraction. Любое возвращение к парсингу prose
  как source of truth запрещено.
- Command lane не является способом вызова tools. Только `/trace` и `/end-session`.
- `server/terminal_manager.py` — совместимый alias на `TerminalTool`, не отдельная реализация.
