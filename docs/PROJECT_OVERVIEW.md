# PROJECT_OVERVIEW — SlavikAI

SlavikAI — Python-агент с двумя основными режимами выполнения:

- `chat`: ответ без выполнения инструментов.
- `mwv`: цикл Manager -> Worker -> Verifier для задач с изменениями.

## Что в проекте есть сейчас

- Поддержка моделей: `xai`, `openrouter`, `local`.
- Командный режим (`/fs`, `/web`, `/sh`, `/project`, `/plan`, `/auto`, `/imggen`, `/imganalyze`, `/trace`) без MWV.
- Workspace инструменты: `workspace_list/read/write/patch/run`.
- Контракт `workspace_patch`: только single-file hunk patch для одного `path`.
- Session-based выбор модели в UI.
- Safe-mode и approvals.

## Ключевые модули

- `core/agent.py` + mixin-модули: orchestration, tool commands, MWV integration.
- `core/planner.py`: построение плана.
- `core/executor.py`: выполнение шагов плана.
- `core/mwv/*`: Manager/Worker/Verifier runtime.
- `server/http_api.py`: HTTP + UI API.

## Инструменты (факт)

- Файлы: `fs`, `workspace_*`.
- Команды: `shell`.
- Сеть/поиск: `web`.
- Проектный индекс: `project`.
- Медиа: `image_analyze`, `image_generate`, `tts`, `stt`.

## Ограничения безопасности

- Sandbox для файловых/workspace/project операций.
- Ограничения shell-команд (allowlist + проверка аргументов + sandbox root).
- Safe-mode блокирует рискованные инструменты через `SAFE_MODE_TOOLS_OFF`.

## Проверки качества

Основной one-shot прогон:

- `make check`

Включает:

- `ruff check .`
- `ruff format --check .`
- `mypy .`
- `pytest` c покрытием (порог >= 80%).

## UI API (папки и чаты)

- `GET /ui/api/folders` → `{ folders: [{ folder_id, name, created_at, updated_at }] }`
- `POST /ui/api/folders` (body: `{ name }`) → `{ folder: { folder_id, name, created_at, updated_at } }`
- `PATCH /ui/api/sessions/{session_id}/title` (body: `{ title }`) → `{ session_id, title }`
- `PUT /ui/api/sessions/{session_id}/folder` (body: `{ folder_id: string | null }`) → `{ session_id, folder_id }`
- `GET /ui/api/sessions` → `{ sessions: [{ session_id, title, title_override, folder_id, created_at, updated_at, message_count }] }`
