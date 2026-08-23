# ARCH_CANON — runtime canon Ask/Plan/Act/Auto

Этот документ задаёт обязательные runtime-инварианты и target design в иерархии
`docs/SOURCE_OF_TRUTH.md`. Точный статус каждого claim находится только в
`docs/runtime_contract_claims.json`. `docs/architecture/Architecture.md` описывает текущее
фактическое устройство и обязано называть legacy-пути legacy, а не целевым поведением.

## 0) Статус канона

- **Status registry**: `docs/runtime_contract_claims.json`; `partial` и `target` нельзя
  интерпретировать как shipped guarantee.
- **Target runtime**: строгие Ask / Plan / Act и Auto FSM, описанные ниже.
- **Implemented baseline**: tool-calling типы, `AgentToolLoop`, read-only chat tool-loop
  integration, explicit `PlanStep.tool_args`, debug-only command lane, единый chat send
  endpoint (`/ui/api/chat/send`; `/ui/api/workspace/send` удалён), единый `TerminalTool`,
  auto v1 path через `AgentToolLoop -> ToolGateway -> verifier`, complete runtime tool
  descriptors, auto без `classify_request(...)`, удалённый legacy auto pipeline entrypoint,
  MWV/CodingTask execution через explicit `ToolRequest` + `ToolGateway`, destructive
  physical split UI message storage на `chat_messages` и `workspace_messages`, удалённый
  legacy event stream route, frontend использует только `handleSendChat`,
  memory/policy-feedback surface вынесен из `core/agent.py` в `AgentMemoryMixin`,
  tool-fail decision branch удалён, первый срез декомпозиции `workspace-ide.tsx`
  вынес layout/resize state и quick-open indexing/filtering в feature-модули,
  Workspace переименован в Computer (inspector/runtime для chat session), добавлены
  `AgentComputerRuntime`, `ComputerBackend` Protocol, `LocalComputerBackend` (default),
  `ContainerComputerBackend` (opt-in/inactive). Production browser auth проверяет Cloudflare
  Access JWT и выводит owner/member principal из verified email; token login с подписанной
  HttpOnly-cookie остаётся legacy local mode. Plan draft получает executable steps через structured
  `submit_plan` tool call; Auto отклоняет provider без native tools до запуска и применяет
  `tool_outcomes` verifier для generic workspace. Добавлен пользовательский режим Desktop:
  тот же provider-neutral `AgentToolLoop -> ToolGateway -> VerifierRuntime`, но с явным
  execution target `desktop`, host-only tool profile, детерминированной scoped policy
  `ALLOW|ASK|DENY` и lifecycle once/session/persistent approvals.
- **Known partial contracts**: Ask всё ещё допускает explicit Memory write до mode gate и
  hidden vector runtime init; explicit Memory confirmation и strict public `model="slavik"`
  validation ещё не готовы. Agent runtime state изолирован по `(principal_id, session_id)`;
  Memory/canonical/vector stores изолированы по principal, а owner сохраняет legacy DB paths.
- **Current legacy runtime**: token-cookie browser auth, крупные `core/agent_mwv.py`, `core/agent_tools.py`,
  `core/agent_routing.py`, часть `classify_request(...)`, runtime/API/UI `lane` markers
  и legacy UI endpoints ещё существуют как совместимость и не считаются целевой
  архитектурой.
- **`/v1/chat/completions` rollout**: поддерживает только `ask|auto` как opt-in;
  `plan|act` через `/v1` отклоняются и должны идти через UI workflow.
- **UI workflow**: endpoints `/ui/api/plan/*` и mode transitions допустимы только если
  соблюдают этот канон: Plan использует только read-only inspection через `ToolGateway`,
  Act исполняет только packet.

## 1) Канонические роли режимов

1. Ask = **stateless** (target; current `partial`).
2. Plan = **transactional read-only** (current `implemented`).
3. Act = **isolated** (current `partial`).
4. Auto v1 = **native tool loop** (current `implemented`).
5. Auto FSM = **Ask -> Plan -> Act orchestrator** (target).
6. Desktop = **host execution profile**, а не отдельный AI, planner или execution loop.

В пользовательском UI Chat соответствует безопасному диалогу (`ask`), Agent — существующему
изолированному agent/auto execution, Desktop — выполнению через host capabilities. Desktop
ортогонален внутренним Plan/Act ролям: reasoning остаётся у текущего LLM provider, enforcement
и execution target выбираются локальным runtime.

## 2) Инварианты (обязательные)

### Ask (target: stateless, 0 side effects; current: partial)

- Запрещены любые записи:
  - `save_to_memory`
  - `capture_memory_claims_from_text`
  - inbox/canonical writes
  - `vector_index.upsert/index/delete`
- Запрещены write/exec tool calls из ask-ветки.
- Разрешён только read-only контекст (memory/vector read path).
- Если vector runtime не готов, ask делает soft-degrade (ответ без vector-контекста, без hidden init).

До исправления раннего explicit-memory path и `allow_runtime_init=True` это обязательный
target invariant, а не описание текущей полной защиты.

### Plan (current: transactional read-only)

- Plan только формирует/редактирует/валидирует `TaskPacket`.
- Plan может использовать только read-capability tools для inspection через
  `ToolGateway`; write/exec capabilities блокируются на registry boundary.
- Read tool result не является выполнением будущего Act step и не может менять packet scope.
- Plan фиксирует execution-контракт:
  - `policy`
  - `scope`
  - `budgets`
  - `approvals`
  - `verifier`

### Act (target: isolated; current: partial)

- Act исполняет только `TaskPacket.steps`.
- Act не меняет `scope/policy/budgets/verifier`.
- Любое отклонение от packet-контракта => `STOP_TO_CHAT` с `stop_reason_code=REPLAN_REQUIRED`.
- Retry через новый `packet_revision` в Plan, а не через импровизацию в Act.

### Auto v1 (current)

- Current Auto запускает provider-native `AgentToolLoop -> ToolGateway -> verifier`.
- Он не является отдельным runtime и не использует classifier для выбора tools.
- Approvals, sandbox, safe-mode и verifier остаются обязательными execution boundaries.

### Auto FSM (target)

- Целевой Auto гоняет только детерминированный цикл: `Ask -> Plan -> Act`.
- В `runtime_mode=auto` запрещён chat-fallback.
- Budgets обязательны: time/tool_calls/tokens/files/retries.
- При fail/ambiguity/risk Auto останавливается в STOP и ждёт явного решения.

### Desktop (host execution profile)

- Desktop tools имеют `execution_target=desktop` и не публикуются Chat/Agent tool snapshots.
- Все host actions идут через существующие `ToolRequest -> ToolGateway -> ToolRegistry`.
- Policy локальна и детерминирована; LLM не может создать approval или отключить enforcement.
- Explicit `DENY` имеет precedence над `ALLOW`; scope включает tool/action/target/command/risk.
- Пути канонизируются до policy match; symlink/traversal, credentials, policy store и
  enforcement/config resources защищены на execution boundary.
- Typed state-changing actions обязаны вернуть verified structured state. Generic
  filesystem/shell actions требуют `desktop_verify`; browser/GUI interactions — correlated
  post-action observation. Verifier может вернуть bounded correction cycle `AgentToolLoop`.
- Capability order фиксирован: native/API → typed host tool → filesystem/system/DBus → argv
  CLI → browser DOM → AT-SPI → visual GUI. GUI не является отдельным агентом.
- Browser downloads являются host artifacts с canonical destination, size/type metadata и
  existence verification; дальнейшие filesystem/archive tools используют тот же path.
- Process identity включает PID + create time (или retained launcher handle), поэтому reused
  PID не подтверждает состояние исходного процесса.
- При выходе из Desktop pending execution отменяется, once/session approvals очищаются.
- Наблюдения tools, файлов, terminal и browser считаются untrusted data, не approvals.

## 3) TaskPacket v2 (execution contract)

`TaskPacket` обязан содержать:

- `task_id`
- `packet_revision`
- `packet_hash`
- `session_id`
- `trace_id`
- `goal`
- `messages`
- `steps`
- `constraints`
- `policy`
- `scope`
- `budgets`
- `approvals`
- `verifier`
- `context`

`TaskStepContract`:

- `step_id`
- `title`
- `description`
- `allowed_tool_kinds`
- `inputs`
- `expected_outputs`
- `acceptance_checks`

## 4) STOP_TO_CHAT (единый JSON-блок)

Обязательные поля:

- `route`
- `trace_id`
- `stop_reason_code`
- `plan_summary`
- `execution_summary`
- `next_steps`
- `attempts`
- `verifier`

Формат совместим с `docs/agent/STOP_RESPONSES.md` и `MWV_REPORT_JSON`.

## 5) `/init` = RuntimeReset

API:

- `POST /ui/api/runtime/init`

Сбрасывает только transient:

- short-term/runtime workflow state
- approval/decision runtime fields
- pending auto progress/runtime cache
- workspace diffs

Пересобирает runtime:

- tools/policy configs
- safe-mode application
- execution policy snapshot
- readiness checks

Не трогает long-term:

- `memory/*.db`
- categorized/canonical memory
- session history

## 6) Policy boundary (sandbox/index/yolo)

- Policy/scope/budgets/verifier фиксируются в packet и становятся immutable для Act.
- Enforcement строится по packet-policy snapshot, а не по “текущему состоянию”.
- Работа вне workspace root запрещается policy-check на исполнении шага.
- Index-режим: явный, наблюдаемый, с отдельным контуром read/write.
- Sandbox semantics относятся к Agent/Act. Desktop является отдельным explicit execution
  target реального host и использует `DesktopPathSecurity` + scoped Desktop policy вместо
  workspace sandbox; это исключение не распространяется на Chat или Agent.

## 7) Rollout `/v1` (совместимость)

- Если `slavik_meta.runtime_mode` отсутствует -> legacy поведение без изменений.
- `slavik_meta.runtime_mode=ask|auto` -> opt-in в новый runtime router.
- `slavik_meta.runtime_mode=plan|act` -> `invalid_request_error` + `next_steps` на UI workflow.

## 8) Legacy debt boundary

Новые фичи не должны расширять legacy-пути как целевую архитектуру.

Legacy debt после PR-26:

- `core/agent.py` уже сжат до bootstrap/configuration shell; `core/agent_mwv.py`,
  `core/agent_tools.py`, `core/agent_routing.py` остаются крупными mixin-модулями.
- `classify_request(...)` ещё участвует в legacy `plan|act` маршрутизации; он не выбирает
  tools и не является planner. `runtime_mode=auto` больше не проходит через этот classifier.
- Storage больше не использует физическую таблицу `ui_messages`; сообщения разделены на
  `chat_messages` и `workspace_messages`. Старые локальные DB с `ui_messages`
  destructive reset/recreate, без migrations/backward compatibility/import-export adapters.
- `lane` не является domain discriminator. Он допускается только как временный runtime/API
  legacy marker во время audit/deletion и не должен управлять storage/API/frontend flow
  после соответствующих split PR. Frontend send/history flow больше не выбирает endpoint
  через `lane`.
- Legacy `/ui/api/events/stream` удалён из routes. `/ui/api/workspace/send` и
  `/ui/api/workspace/events/{session_id}` удалены — workspace не является
  conversational endpoint. Единственный conversational send — `/ui/api/chat/send`.
  `/ui/api/workspace/*` остаётся только для file/operation endpoints (tree, file,
  patch, run, git-diff и т.п.).
- Primary `local` OpenAI-compatible provider реализует native provider tool calling.
  `xai`, `openrouter`, `inception` явно отклоняют generic `tools`; xAI web search
  остаётся отдельным provider-native режимом.
- Старый auto pipeline `planner -> coder pool -> merge -> verifier` больше не имеет
  runtime entrypoint; auto-запуски через `AutoAgent.run_outcome()` используют auto v1
  tool loop.
- MWV/CodingTask больше не извлекает target path из prose и не применяет fake
  append-comment change напрямую через Python file writes; worker execution требует
  explicit gateway tool requests.
- Старые `Planner`/`Executor` удалены как runtime entrypoints; MWV execution не
  проходит через отдельный plan/executor wrapper.
- Tool failures после gateway call не создают отдельный `DecisionPacket(tool_fail)`;
  failures остаются observability/candidate сигналом и не должны обходить gateway/approval.
- `workspace-ide.tsx` остаётся крупным controller-компонентом, но layout/resize state
  и quick-open index/filter helpers уже вынесены. Дальнейшая UI-декомпозиция должна
  продолжать выделять реальные bounded surfaces без redesign.

Уже не является допустимым legacy для расширения:

- возвращение `Planner`/`Executor` как runtime entrypoints;
- regex extraction tool args из prose;
- regex target extraction / append-comment fake worker в MWV/CodingTask;
- slash-команды для обычных tools;
- отдельная server-only реализация PTY терминала рядом с one-shot runner.

## 9) Product direction — Personal Agent Computer

**Slavikai — это personal agent computer, а не IDE и не coding agent.**

Coding — первый validated/testable сценарий, а не архитектурный центр.
Архитектура строится вокруг универсального agent-computer контура, в котором Chat
и Computer выполняют строго разные роли.

### Computer Mode is not a manual IDE

**Computer Mode is a live agent execution surface.**

Computer Mode не является:
- IDE для ручного редактирования файлов;
- file manager как primary UX;
- coding assistant как главный сценарий.

Computer Mode — это поверхность, которая отображает, что agent делает прямо сейчас:
sandbox/container status, current task, live command/output stream, activity timeline,
changes/diff, checks/tests, approvals, artifacts.

Explorer/editor/terminal — вторичные inspection tools, доступные по запросу,
а не главный экран.

### Computer Mode — primary surface

Основной экран Computer отображает:

- sandbox/container status (local/container, idle/running/completed/failed)
- current task / auto state / plan
- live activity timeline (agent tool calls, durations, results)
- changes/diff summary
- checks/tests summary
- approvals / pending decisions
- artifacts

### Computer Mode — secondary details (по запросу)

- terminal / PTY
- file explorer
- editor / preview
- raw logs

### Роли (фиксированные)

- **Chat** — единственный conversational entrypoint. Пользователь общается только через Chat.
- **Computer** — live agent execution surface для текущей chat-сессии. Не является вторым чатом,
  не принимает сообщения от пользователя напрямую, не имеет своего lane.
- **ComputerBackend** — граница для executable environments. Весь исполняемый код
  проходит через backend, а не напрямую через Python I/O или subprocess.

### Текущие backends

- `LocalComputerBackend` — реализован, default.
- `ContainerComputerBackend` — реализован, opt-in/inactive (SLAVIK_COMPUTER_* env vars).

### Future backends (направление, не реализованы)

- SSH remote host
- Browser automation / visual browsing
- VM / Desktop backend

### Инварианты (non-negotiable)

- Не вводить новые lanes (нет `lane="computer"`, нет `role="computer"`).
- Не обходить `ToolGateway`, approval boundary или `ComputerActivityLog` hooks.
- Новые backends подключаются только через `ComputerBackend` Protocol.
- Computer-события хранятся отдельно от chat-сообщений; в visible chat не попадают.
- Explorer/editor не должны быть primary product surface.
- Нет `/ui/api/computer/send` — Computer не является conversational endpoint.
