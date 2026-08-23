# Architecture — SlavikAI current runtime

Этот документ фиксирует **текущее фактическое устройство** системы после PR-12 текущей серии.
Иерархия источников истины определена в `docs/SOURCE_OF_TRUTH.md`, статусы claims — в
`docs/runtime_contract_claims.json`, а обязательные runtime-инварианты и target design — в
`docs/architecture/ARCH_CANON.md`.
Product direction (personal agent computer, роли Chat/Computer, backends) — `ARCH_CANON.md §9`.
Computer Mode product invariant ("not a manual IDE", live agent execution surface, primary vs secondary surfaces) — `ARCH_CANON.md §9`.
Если здесь описан legacy-путь, это не делает его целевой архитектурой.

## Цель

SlavikAI сейчас работает как server-side agent runtime с Cloudflare Access owner/member
browser identity и principal isolation. Legacy token-cookie auth остаётся для
локального запуска. Mutable Agent/runtime state принадлежит `(principal_id, session_id)`,
Memory/vector stores принадлежат principal; owner продолжает использовать существующие legacy
DB paths. После PR-0..PR-14 в коде есть tool-calling contract,
read-only chat integration через `AgentToolLoop`, auto v1 через `AgentToolLoop`,
split chat/workspace API paths, debug-only command lane и единый terminal backend.
После PR-15 runtime tools имеют LLM descriptions/JSON Schema. После PR-16 auto mode
больше не проходит через `classify_request(...)`. После PR-17 старый auto pipeline
`planner -> coder pool -> merge -> verifier` удалён из runtime entrypoints. После
PR-18 MWV/CodingTask worker не извлекает target path из prose и не делает fake
append-comment edits напрямую; выполнение идёт через explicit `ToolRequest` и
`ToolGateway`. После PR-20 UI message storage физически разделён на `chat_messages` и
`workspace_messages`; старая локальная DB с `ui_messages` destructive reset/recreate.
После PR-21 HTTP send handler для chat остался единственным (`/ui/api/chat/send`);
workspace/send endpoint удалён. После PR-23 frontend использует только `handleSendChat`;
workspace history берётся из session snapshot. После PR-24 memory/policy-feedback surface
вынесен из `core/agent.py` в `core/agent_memory.py::AgentMemoryMixin`. После PR-25
repeated tool failures больше не создают отдельный `DecisionPacket(tool_fail)`. После
PR-26 workspace IDE начал декомпозицию: layout/resize state и quick-open index/filter logic
вынесены из `workspace-ide.tsx` в feature-модули.
После PR-27 browser UI получил token login с подписанной HttpOnly-cookie, `.env` загружается
до boot auth validation, Plan строит шаги только через structured `submit_plan` tool call,
а Auto проверяет native-tools capability до запуска и использует verifier profile
`tool_outcomes` вне code repository.
После PR-07..PR-12 (текущей серии) Workspace переименован в **Computer** как UI
inspector/runtime для текущей chat session. Добавлены `AgentComputerRuntime`, `ComputerBackend`
Protocol, `LocalComputerBackend` (default) и `ContainerComputerBackend` (opt-in/inactive).
Текущий runtime также содержит отдельный Desktop execution profile. Он использует общий
conversational endpoint и существующий LLM/tool loop, но публикует host tools только при
`mode=desktop`; Chat (`ask`) и Agent (`auto`/Act) сохраняют прежние sandbox semantics.
Часть старого runtime ещё остаётся legacy-обвязкой.

## Основные слои

- **Core** (`core/*`)
  - Оркестрация: `Agent` + mixins.
  - `core/agent.py` теперь остаётся bootstrap/configuration shell для `Agent`.
  - Memory/policy-feedback surface: `core/agent_memory.py::AgentMemoryMixin`.
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
  - Auto v1 не использует Python intent classifier: модель либо возвращает native tool calls для
    действий, либо завершает loop обычным ответом. Непустой response-only результат проходит
    профиль verifier `response_only`; workspace verifier запускается только после tool actions.
  - Auto доступен только для provider с `supports_native_tools=True` (`deepseek`, `local`).
    Несовместимый provider даёт явный stop `native_tools_required`, а не internal error.
  - В repository с canonical `Makefile` verifier запускает `make check`; в generic workspace
    профиль `tool_outcomes` проверяет фактические результаты native tool calls.
- **LLM слой** (`llm/*`)
  - Провайдеры: `xai`, `openrouter`, `local`, `inception`, `deepseek`.
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
  - Desktop host primitives: search/read/atomic write/copy-move-rename/recoverable delete,
    staged archive extraction, argv-only bounded shell, application launch, URL/file open,
    browser search/fetch/open и explicit result verification.
  - `ToolRegistry` фильтрует descriptors и dispatch по `execution_target=sandbox|desktop`;
    Desktop tools не доступны в Chat/Agent snapshots.
- **Agent Computer** (`core/agent_computer.py`, `core/computer_backend.py`, `core/container_computer_backend.py`)
  - Computer — это **inspector/runtime для текущей chat session**, а не отдельный чат.
  - Chat остаётся единственной conversational entrypoint. Computer не имеет assistant
    composer и не является новым chat lane.
  - `AgentComputerRuntime` — backend-agnostic facade; методы: `list_files`, `read_file`,
    `write_file`, `apply_patch`, `run_command`, `git_diff`, `run_tests`, `check`.
  - `ComputerBackend` — `@runtime_checkable` Protocol, определяет те же 8 методов.
  - `LocalComputerBackend` — default implementation; делегирует операции через `ToolGateway`.
    `Agent.make_computer_runtime()` создаёт его по умолчанию.
  - `ContainerComputerBackend` — **opt-in/inactive** alternative; запускает команды через
    `docker/podman run --rm`; в тестах используется `FakeContainerRunner` вместо реального
    Docker daemon. По умолчанию не включён; `LocalComputerBackend` остаётся default.
  - Computer UI read-only по умолчанию; ручные правки файлов (New/Rename/Move/Del/Save)
    разблокируются только при `sessionYoloActive`.
  - Computer activity log (`ComputerActivityLog`) хранится в `computer_events` в session hub,
    не в `workspace_messages`.
- **Storage/Memory** (`memory/*`)
  - `memory/memory.db`, `memory/memory_companion.db`, `memory/vectors.db`.
  - Эти legacy paths принадлежат owner. Для members используются директории
    `memory/principals/<sha256(principal_id)>/` с отдельными Memory, Memory Companion,
    categorized/canonical и vector DB. Email не включается в filesystem path.
  - Каждая session имеет отдельный mutable `Agent` и lock; сессии одного principal разделяют
    только его persistent DB paths, но не short-term/runtime/tool state.
  - Persistent Agent memory stores use one SQLite connection per calling thread through
    `shared/sqlite.py`; HTTP streaming workers never reuse the server thread connection.
  - UI message storage физически разделён: `chat_messages` для chat-сообщений и
    `workspace_messages` как legacy internal table. `workspace_messages` — не
    "workspace chat lane", а остаток storage split (PR-20); новый Computer activity
    идёт в отдельный `computer_events` в session state hub.
  - Физическая таблица `ui_messages` больше не является текущей схемой. При обнаружении
    старой local DB с `ui_messages` storage делает destructive reset/recreate schema.
  - Локальная `.run/ui_sessions.db`, старые sessions/chats/history disposable by default.
  - `lane` может ещё встречаться в runtime/storage как временный legacy marker, но не
    является domain discriminator. `lane="computer"` не существует и не должен появляться.

## Маршрутизация запроса (current legacy runtime)

1. Сообщение, начинающееся с `/`, идёт в command lane (`handle_tool_command`) и не проходит через MWV.
   - Разрешены только debug-команды `/trace` и `/end-session`. Подробнее — `docs/agent/COMMAND_LANE_POLICY.md`.
2. Для обычного текста:
   - `runtime_mode=ask` — сразу chat-ветка (без `classify_request`).
   - `runtime_mode=auto` — сразу запуск auto v1 без `classify_request` и chat-fallback. Модель
     отвечает без tools для conversation-only запроса или выполняет
     `AgentToolLoop -> ToolGateway -> verifier` для workspace actions.
   - `runtime_mode=desktop` — тот же `AgentToolLoop -> ToolGateway -> VerifierRuntime`, но
     с desktop-only descriptors и execution target реального host.
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
- Desktop host profile: `desktop_file_*`, `desktop_archive_extract`, `desktop_shell`,
  `desktop_clipboard`, `desktop_system_info`, `desktop_process`, `desktop_systemd`,
  `desktop_package`, `desktop_session`, `desktop_open`, `desktop_browser`, `desktop_gui`,
  `desktop_verify`. `desktop_process/systemd/package` являются typed semantic capabilities;
  generic shell отказывает в прямых `kill/systemctl/apt` вызовах.
- `desktop` остаётся инфраструктурным HTTP/runtime profile и не входит в shared domain
  `SessionMode`: domain workflow сохраняет только `ask|plan|act|auto`, а Desktop snapshot
  проецируется в безопасный базовый workflow `ask`.

## Sandbox и безопасность

- `fs` работает в `sandbox/`.
- `workspace_*` и `project` ограничены `sandbox/project/`.
- `shell` использует sandbox root + ограничения конфигурации.
- Safe-mode отключает рискованные инструменты через `SAFE_MODE_TOOLS_OFF`, включая `workspace_run` и `workspace_terminal_run`.
- `desktop_*` — намеренное исключение только для `mode=desktop`: canonical host paths
  проверяются `DesktopPathSecurity`, после чего запрос всё равно проходит `ToolGateway`.
  Protected paths, symlink/traversal и собственные policy/enforcement/config resources
  блокируются до tool execution.

## Desktop policy и lifecycle

- `DesktopPolicyRuntime` объединяет persistent snapshot с once/session rules; explicit DENY
  всегда сильнее matching ALLOW.
- Command scope использует явное exact-match поле `command_exact`; wildcard-команды не
  поддерживаются и не могут неявно расширить approval.
- `DesktopPolicyStore` хранит inspectable persistent rules атомарно с mode `0600`.
- UI decision flow поддерживает approve once, approve for session, narrow always allow и deny;
  выбор и применённое policy решение пишутся в существующий trace/tool logging контур с
  redaction payload/secrets.
- Переход из Desktop отменяет активную генерацию, очищает pending decision и session rules.
- При fail/cancel процессы из непроверенного Desktop run завершаются через скрытый
  runtime-only cleanup tool и `ToolGateway`, а не прямым вызовом из runtime.
- Mutating tool success не завершает задачу: typed tools возвращают проверенное structured
  state, browser/GUI interaction требует correlated observation, а generic filesystem/shell
  action — отдельный `desktop_verify`. `AgentToolLoop` даёт ограниченный correction retry.
- Desktop planner получает единый priority contract: native/application API → typed tool →
  filesystem/system/DBus → argv CLI → browser DOM → AT-SPI → visual GUI fallback.
- Browser capability использует Playwright semantic selectors и first-class verified download
  artifacts. После установки Python dependencies нужен browser runtime:
  `python -m playwright install chromium`.
- Push-to-talk использует существующий STT endpoint; в Desktop transcription передаётся в тот
  же `/ui/api/chat/send` pipeline, что и typed request.

## HTTP/UI слой

- OpenAI-совместимые endpoints: `/v1/models`, `/v1/chat/completions`.
- `/v1/models` сейчас возвращает только публичный proxy id `slavik`; provider/model из UI
  Settings являются внутренней runtime-конфигурацией. Проверка `model="slavik"` в chat
  request ещё неполна и отмечена `partial` в registry claims.
- Служебные endpoints: `/slavik/trace/{trace_id}`, `/slavik/tool-calls/{trace_id}`, `/slavik/feedback`, `/slavik/approve-session`.
- UI API и workflow endpoints регистрируются в `server/http/routes.py`.
- Browser UI проходит login через `/ui/api/auth/login` и работает по подписанной
  `HttpOnly`, `SameSite=Strict` cookie. `/v1/*` и service clients сохраняют Bearer auth.
- Plan draft формируется через structured `submit_plan` tool call. Каждый executable step
  содержит ровно одну `allowed_tool_kinds`, совпадающую `inputs.operation`, и `tool_args`.

### slavik_meta.runtime_mode contract (для /v1/chat/completions)

- `runtime_mode=ask|auto` — поддерживаемый opt-in.
- `runtime_mode=plan|act` — `invalid_request_error` (использовать UI workflow).
- без `runtime_mode` — legacy-поведение текущего runtime.

Это не конфликтует с `ARCH_CANON`: `plan|act` являются целевыми runtime-ролями, но не доступны
как `/v1/chat/completions` opt-in.

### UI API endpoint groups

- Sessions/folders: `/ui/api/folders`, `/ui/api/sessions`, `/ui/api/sessions/{session_id}`.
- Workflow: `/ui/api/mode`, `/ui/api/plan/*`, `/ui/api/runtime/init`.
- Desktop approvals: `/ui/api/desktop/approvals` и
  `/ui/api/desktop/approvals/{rule_id}` для inspect/update/remove persistent rules.
- Chat (единственный conversational entrypoint): `/ui/api/chat/send`, `/ui/api/chat/events/{session_id}`.
- Workspace file operations (Computer inspector): `/ui/api/workspace/root`,
  `/ui/api/workspace/tree`, `/ui/api/workspace/file`, `/ui/api/workspace/file/create`,
  `/ui/api/workspace/file/rename`, `/ui/api/workspace/file/move`,
  `/ui/api/workspace/patch`, `/ui/api/workspace/run`,
  `/ui/api/workspace/terminal/run`, `/ui/api/workspace/git-diff`,
  `/ui/api/workspace/index`.
- `/ui/api/workspace/send` и `/ui/api/workspace/events/{session_id}` **не существуют** —
  удалены; workspace не является conversational endpoint.
- `/ui/api/computer/send` **не существует** и не должен появляться.
- Legacy `/ui/api/events/stream` удалён из routes.
- Frontend send flow использует только `handleSendChat`; `handleSendWorkspace` удалён.
  Runtime controller загружает workspace history из session snapshot.
- Computer (workspace-ide.tsx) всё ещё имеет крупный controller-компонент, но layout/resize
  state живёт в `useWorkspaceLayout`, а quick-open indexing/filtering — в
  `workspace-quick-open-index.ts`.

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
- `npm test` (UI regression tests)
- `pytest` с покрытием (порог ≥ 80%).

## Наблюдаемость

- Trace: `logs/trace.log`.
- Tool calls: `logs/tool_calls.log`.
- UI storage: `.run/ui_sessions.db`.

## Legacy, который нельзя расширять

- `core/agent_mwv.py`, `core/agent_tools.py` и `core/agent_routing.py` остаются крупной
  mixin-обвязкой и должны постепенно сжиматься вокруг `AgentToolLoop`/`ToolGateway`, а не
  получать новые ветки.
- `classify_request(...)` всё ещё используется в legacy `plan|act` routing; новые
  tool-capabilities не должны добавляться через keyword router.
- `lane` не должен оставаться domain discriminator. После storage split он допускается
  только как временный runtime/API/frontend legacy marker для audit/deletion.
  `lane="computer"` запрещён — не добавлять.
- `Planner`/`Executor` удалены из runtime entrypoints. Любое возвращение к парсингу prose
  как source of truth для tool args запрещено.
- Command lane не является способом вызова tools. Только `/trace` и `/end-session`.
  Computer activity не является command lane.
- Tool failure threshold не является decision runtime; failures не должны обходить
  gateway/approval через отдельный `DecisionRequired` path.
- `workspace-ide.tsx` больше не должен снова принимать layout/resize или quick-open
  indexing responsibilities; новые workspace UI изменения должны продолжать выносить
  bounded surfaces в `ui/src/features/workspace/*`.
- `server/terminal_manager.py` — совместимый alias на `TerminalTool`, не отдельная реализация.
- Не создавать `/ui/api/computer/send`, `lane="computer"`, или новый conversational
  entrypoint для Computer. Chat остаётся единственной conversational entrypoint.
- `ContainerComputerBackend` — opt-in/inactive. Не переключать default на container
  без явной config/env wiring и отдельного решения. `FakeContainerRunner` —
  test-only utility, не production backend.
