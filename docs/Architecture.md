# Architecture — SlavikAI

## Цель

SlavikAI — серверный агент с тремя рабочими контурами выполнения: chat, MWV и auto. Система опирается на command lane, policy/approval контур, sandbox-ограничения и обязательный trace/tool logging.

## Основные слои

- **Core** (`core/*`)
  - Оркестрация: `Agent` + mixins.
  - Планирование/исполнение: `Planner` + `Executor` через `ToolGateway`.
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
- **Tools** (`tools/*`)
  - Реестр: `ToolRegistry`.
  - Журнал вызовов: `logs/tool_calls.log`.
- **Storage/Memory** (`memory/*`)
  - `memory/memory.db`, `memory/memory_companion.db`, `memory/vectors.db`.

## Маршрутизация запроса (факт runtime)

1. Сообщение, начинающееся с `/`, идёт в command lane (`handle_tool_command`) и не проходит через MWV.
2. Для обычного текста:
   - `runtime_mode=ask` — сразу chat-ветка (без `classify_request`).
   - `runtime_mode=auto` — выполняется классификация/skill-проверка, затем запуск auto-контура.
   - `runtime_mode=act|plan` — используется `classify_request(...)` (`chat` или `mwv`).

## Инструменты (зарегистрированные имена)

- Базовые: `fs`, `web`, `shell`, `project`.
- Медиа: `image_analyze`, `image_generate`, `tts`, `stt`.
- Workspace: `workspace_list`, `workspace_read`, `workspace_write`, `workspace_create`, `workspace_rename`, `workspace_move`, `workspace_delete`, `workspace_patch`, `workspace_run`, `workspace_terminal_run`.

## Sandbox и безопасность

- `fs` работает в `sandbox/`.
- `workspace_*` и `project` ограничены `sandbox/project/`.
- `shell` использует sandbox root + ограничения конфигурации.
- Safe-mode отключает рискованные инструменты через `SAFE_MODE_TOOLS_OFF`, включая `workspace_run` и `workspace_terminal_run`.

## HTTP/UI слой

- OpenAI-совместимые endpoints: `/v1/models`, `/v1/chat/completions`.
- Служебные endpoints: `/slavik/trace/{trace_id}`, `/slavik/tool-calls/{trace_id}`, `/slavik/feedback`, `/slavik/approve-session`.
- UI API и workflow endpoints регистрируются в `server/http/routes.py`.

## Наблюдаемость

- Trace: `logs/trace.log`.
- Tool calls: `logs/tool_calls.log`.
- UI storage: `.run/ui_sessions.db`.
