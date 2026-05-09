# Architecture — SlavikAI current runtime inventory

Этот документ фиксирует **текущее фактическое устройство** системы. Целевое поведение runtime
определено в `docs/architecture/ARCH_CANON.md`. Если здесь описан legacy-путь, это не делает его
целевой архитектурой.

## Цель

SlavikAI сейчас работает как серверный агент с тремя фактическими контурами выполнения: chat, MWV и auto. Система опирается на debug-only command lane, policy/approval контур, sandbox-ограничения и обязательный trace/tool logging.

## Основные слои

- **Core** (`core/*`)
  - Оркестрация: `Agent` + mixins.
  - Планирование/исполнение: native tool-call contracts + `ToolGateway`; `Planner`/`Executor` больше не извлекают tool args из prose.
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
  - `openai` используется только для STT endpoint/ключа в UI-настройках (не chat provider).
- **Tools** (`tools/*`)
  - Реестр: `ToolRegistry`.
  - Журнал вызовов: `logs/tool_calls.log`.
- **Storage/Memory** (`memory/*`)
  - `memory/memory.db`, `memory/memory_companion.db`, `memory/vectors.db`.

## Маршрутизация запроса (current legacy runtime)

1. Сообщение, начинающееся с `/`, идёт в command lane (`handle_tool_command`) и не проходит через MWV.
   - Разрешены только debug-команды `/trace` и `/end-session`. Подробнее — `docs/for-humans/COMMANDS.md`.
2. Для обычного текста:
   - `runtime_mode=ask` — сразу chat-ветка (без `classify_request`).
   - `runtime_mode=auto` — выполняется классификация/skill-проверка, затем запуск auto-контура.
   - `runtime_mode=act|plan` — в legacy runtime используется `classify_request(...)` (`chat` или `mwv`).

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
- Chat/events: `/ui/api/chat/send`, `/ui/api/events/stream`.
- Workspace: `/ui/api/workspace/*`.

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
