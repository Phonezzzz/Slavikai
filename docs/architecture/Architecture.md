# Architecture — SlavikAI current runtime

Этот документ фиксирует **текущее фактическое устройство** системы после PR-22.
Целевое поведение runtime определено в `docs/architecture/ARCH_CANON.md`.
Если здесь описан legacy-путь, это не делает его целевой архитектурой.

## Цель

SlavikAI сейчас работает как single-user/server-side agent runtime с контурами chat,
workspace/MWV и auto. После PR-0..PR-14 в коде есть честный tool-calling contract,
read-only chat integration через `AgentToolLoop`, auto v1 через `AgentToolLoop`,
split chat/workspace API paths, debug-only command lane и единый terminal backend.
После PR-15 runtime tools имеют LLM descriptions/JSON Schema. После PR-16 auto mode
больше не проходит через `classify_request(...)`. После PR-17 старый auto pipeline
`planner -> coder pool -> merge -> verifier` удалён из runtime entrypoints. После
PR-18 MWV/CodingTask worker не извлекает target path из prose и не делает fake
append-comment edits напрямую; выполнение идёт через explicit `ToolRequest` и
`ToolGateway`. После PR-20 UI message storage физически разделён на `chat_messages` и
`workspace_messages`; старая локальная DB с `ui_messages` destructive reset/recreate.
После PR-21 HTTP send handlers разделены: chat и workspace endpoints читают payload
сами и передают lane во внутренний runtime явно. Часть старого runtime ещё остаётся
legacy-обвязкой.

## Основные слои

- **Core** (`core/*`)
  - Оркестрация: `Agent` + mixins.
  - Tool loop: `core/tool_loop.py` вызывает `Brain.generate(..., tools=...)`, исполняет
    `tool_calls` через `ToolGateway` и добавляет `role="tool"` сообщения.
  - Chat integration: `runtime_mode=ask` может использовать только описанные read-capability
    tools; write/exec tool calls остаются вне chat path.
  - Старые `Planner`/`Executor` удалены как runtime entrypoints; MWV строит
    `TaskPacket` напрямую, а шаги исполняются explicit gateway tool requests.
  - Трассировка: `Tracer` (`logs/trace.log`).
- **MWV runtime** (`core/mwv/*`)
  - Цикл `ManagerRuntime -> WorkerRuntime -> VerifierRuntime`.
  - Worker execution принимает explicit tool requests и исполняет их через `ToolGateway`;
    prose/regex extraction не является runtime contract.
  - Успех только при `WorkStatus.SUCCESS` и `VerificationStatus.PASSED`.
  - Ограниченный retry через `RunContext.max_retries`.
- **Auto runtime** (`core/auto_runtime.py`)
  - Новый entry из `AutoAgent.run_outcome()` идёт через `AgentToolLoop -> ToolGateway -> verifier`.
  - Legacy-контур `planner -> coder pool -> merge -> verifier` больше не имеет
    runtime entrypoint.
  - Auto v1 поддерживает паузу `waiting_approval` и resume через повторный tool-loop run после approval.
- **LLM слой** (`llm/*`)
  - Провайдеры: `xai`, `openrouter`, `local`, `inception`.
  - Контракт: `LLMMessage.role` включает `tool`, `LLMResult.tool_calls`,
    `Brain.generate(..., tools=None)`.
  - Native provider tool calling реализован в primary `local` OpenAI-compatible path.
    `xai`, `openrouter`, `inception` явно отклоняют generic `tools`; xAI web search
    остаётся отдельным provider-native режимом.
  - `openai` используется только для STT endpoint/ключа в UI-настройках (не chat provider).
- **Tools** (`tools/*`)
  - Реестр: `ToolRegistry`.
  - Descriptor: `name`, `description`, `parameters_schema`, capability/risk classes.
  - Журнал вызовов: `logs/tool_calls.log`.
  - Terminal: `tools/terminal_tool.py`, режимы `oneshot|pty`.
- **Storage/Memory** (`memory/*`)
  - `memory/memory.db`, `memory/memory_companion.db`, `memory/vectors.db`.
  - UI message storage физически разделён: `chat_messages` для chat-сообщений и
    `workspace_messages` для workspace-сообщений.
  - Физическая таблица `ui_messages` больше не является текущей схемой. При обнаружении
    старой local DB с `ui_messages` storage делает destructive reset/recreate schema.
  - Локальная `.run/ui_sessions.db`, старые sessions/chats/history disposable by default.
  - `lane` может ещё встречаться в runtime/API/frontend как временный legacy marker до
    PR-21/PR-23, но не является storage/domain discriminator.

## Маршрутизация запроса (current legacy runtime)

1. Сообщение, начинающееся с `/`, идёт в command lane (`handle_tool_command`) и не проходит через MWV.
   - Разрешены только debug-команды `/trace` и `/end-session`. Подробнее — `docs/for-humans/COMMANDS.md`.
2. Для обычного текста:
   - `runtime_mode=ask` — сразу chat-ветка (без `classify_request`).
   - `runtime_mode=auto` — сразу запуск auto v1
     (`AgentToolLoop -> ToolGateway -> verifier`), без `classify_request` и без chat-fallback.
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
- `/ui/api/chat/send` принимает только chat-запросы; `/ui/api/workspace/send` принимает
  только workspace-запросы. Cross-lane payloads отклоняются.
- Legacy `/ui/api/events/stream` удалён из routes. Новый код использует split
  chat/workspace endpoints.

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
- `classify_request(...)` всё ещё используется в legacy `plan|act` routing; новые
  tool-capabilities не должны добавляться через keyword router.
- `lane` не должен оставаться domain discriminator. После storage split он допускается
  только как временный runtime/API/frontend legacy marker для audit/deletion.
- `Planner`/`Executor` удалены из runtime entrypoints. Любое возвращение к парсингу prose
  как source of truth для tool args запрещено.
- Command lane не является способом вызова tools. Только `/trace` и `/end-session`.
- `server/terminal_manager.py` — совместимый alias на `TerminalTool`, не отдельная реализация.
- Planned cleanup roadmap: `docs/architecture/LEGACY_CLEANUP_ROADMAP.md`.
