# PROJECT_OVERVIEW — SlavikAI

SlavikAI — Python-агент с command lane, chat/MWV маршрутизацией и auto-orchestrator контуром.

## Что в проекте есть сейчас

- Модельные провайдеры: `xai`, `openrouter`, `local`, `inception`.
- `openai` используется в настройках только для STT (`OPENAI_API_KEY` / audio transcription), не как chat model provider.
- Command lane (`/fs`, `/web`, `/sh`, `/project`, `/plan`, `/auto`, `/imggen`, `/imganalyze`, `/trace`) — ручной режим без MWV.
- Runtime modes:
  - `ask`: chat-ответ без routing-классификатора;
  - `act|plan`: детерминированная классификация `chat|mwv`;
  - `auto`: auto-runtime с предклассификацией/skill-проверкой.
- Для `/v1/chat/completions`:
  - `slavik_meta.runtime_mode=ask|auto` — поддерживаемый opt-in;
  - `slavik_meta.runtime_mode=plan|act` — `invalid_request_error` (с next_steps на UI workflow);
  - без `runtime_mode` — legacy-поведение текущего runtime.
- Workspace-инструменты: list/read/write/create/rename/move/delete/patch/run/terminal_run.
- `workspace_terminal_run` — restricted one-shot command runner, а не PTY terminal session.
- Контракт `workspace_patch`: single-file hunk patch для одного `path` (без `diff --git`/`---`/`+++`).
- Session-based выбор модели, security/tools state и approvals в UI.

## Ключевые модули

- `core/agent.py` + mixin-модули: orchestration, command lane, routing, MWV/auto integration.
- `core/mwv/*`: Manager/Worker/Verifier runtime, retry policy, stop-коды.
- `core/auto_runtime.py`: planner/coder/verifier оркестрация.
- `server/http/routes.py`: фактический список HTTP/UI endpoints.

## Инструменты (факт)

- Базовые: `fs`, `web`, `shell`, `project`.
- Медиа: `image_analyze`, `image_generate`, `tts`, `stt`.
- Workspace: `workspace_list`, `workspace_read`, `workspace_write`, `workspace_create`, `workspace_rename`, `workspace_move`, `workspace_delete`, `workspace_patch`, `workspace_run`, `workspace_terminal_run`.

## Ограничения безопасности

- Sandbox для файловых/workspace/project операций.
- Shell-ограничения (валидация + sandbox root + policy checks).
- Safe-mode блокирует рискованные инструменты через `SAFE_MODE_TOOLS_OFF`.

## Проверки качества

`make check` включает:

- `scripts/check_no_legacy_ui.sh`
- `ruff check .`
- `ruff format --check .`
- `mypy .`
- `npm run typecheck` (UI)
- `pytest` c покрытием (порог >= 80%).

## UI API (часто используемые endpoints)

- Sessions/folders: `/ui/api/folders`, `/ui/api/sessions`, `/ui/api/sessions/{session_id}`.
- Workflow: `/ui/api/mode`, `/ui/api/plan/*`, `/ui/api/runtime/init`.
- Chat/events: `/ui/api/chat/send`, `/ui/api/events/stream`.
- Workspace: `/ui/api/workspace/*`.
