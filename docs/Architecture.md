# Architecture — SlavikAI

## Цель

SlavikAI — локальный/серверный агент с детерминированной маршрутизацией `chat`/`mwv`, инструментами в песочнице и журналированием всех ключевых действий.

## Основные слои

- **Core** (`core/*`)
  - `core/agent.py` + mixin-модули `core/agent_routing.py`, `core/agent_mwv.py`, `core/agent_tools.py`.
  - `Planner` (`core/planner.py`) строит план (LLM или эвристика).
  - `Executor` (`core/executor.py`) исполняет шаги через `ToolGateway`.
  - `Tracer` (`core/tracer.py`) пишет trace в `logs/trace.log`.
- **MWV runtime** (`core/mwv/*`)
  - `ManagerRuntime` -> `WorkerRuntime` -> `VerifierRuntime`.
  - Успех только при `WorkStatus.SUCCESS` + `VerificationStatus.PASSED`.
  - bounded retry с лимитом попыток (через `RunContext.max_retries`).
- **LLM слой** (`llm/*`)
  - Провайдеры: `xai`, `openrouter`, `local`.
  - Фабрика: `llm/brain_factory.py`.
- **Tools** (`tools/*`)
  - Реестр: `tools/tool_registry.py`.
  - Логи вызовов: `logs/tool_calls.log`.
- **Storage/Memory** (`memory/*`)
  - `memory/memory.db`, `memory/memory_companion.db`, `memory/vectors.db`.
  - Векторный индекс `VectorIndex` с lazy-load.

## Маршрутизация запроса

1. `/...` команды -> **command lane** (без MWV), через `Agent.handle_tool_command`.
2. Обычный текст -> `classify_request(...)` из `core/mwv/routing.py`:
   - `chat`: прямой LLM-ответ с контекстом.
   - `mwv`: Manager -> Worker -> Verifier.

## Инструменты

Зарегистрированы в `core/agent.py`:

- `fs`, `web`, `shell`, `project`
- `image_analyze`, `image_generate`
- `tts`, `stt`
- `workspace_list`, `workspace_read`, `workspace_write`, `workspace_patch`, `workspace_run`

## Sandbox и безопасность

- `filesystem`: внутри `sandbox/`.
- `workspace_*` и `project`: внутри `sandbox/project/`.
- `shell`: `sandbox_root` нормализуется относительно `sandbox/` и проверяется от escape.
- Safe-mode block-list (`SAFE_MODE_TOOLS_OFF`):
  - `web`, `web_search`, `shell`, `project`,
  - `tts`, `stt`, `http_client`,
  - `image_analyze`, `image_generate`,
  - `workspace_run`.

## HTTP/UI слой

`server/http_api.py` поднимает OpenAI-совместимые и UI-endpoints:

- `/v1/models`, `/v1/chat/completions`
- `/slavik/trace/{trace_id}`, `/slavik/tool-calls/{trace_id}`
- `/slavik/feedback`, `/slavik/approve-session`
- `/ui/*` API (status, models, sessions, chat, project command, SSE events)

## Наблюдаемость

- Trace событий: `logs/trace.log`.
- Tool-call журнал: `logs/tool_calls.log`.
- Для UI-сессий: `.run/ui_sessions.db`.
