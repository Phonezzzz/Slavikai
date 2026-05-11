# ARCH_CANON — runtime canon Ask/Plan/Act/Auto

Этот документ — **source of truth** для runtime-архитектуры и rollout-границ.
`docs/architecture/Architecture.md` описывает текущее фактическое устройство и обязано
называть legacy-пути legacy, а не целевым поведением.

## 0) Статус канона

- **Target runtime**: Ask / Plan / Act / Auto, описанные ниже.
- **Implemented baseline**: tool-calling типы, `AgentToolLoop`, read-only chat tool-loop
  integration, explicit `PlanStep.tool_args`, debug-only command lane, split chat/workspace
  send+SSE endpoints, единый `TerminalTool`, auto v1 path через
  `AgentToolLoop -> ToolGateway -> verifier`, complete runtime tool descriptors, auto без
  `classify_request(...)`, удалённый legacy auto pipeline entrypoint, MWV/CodingTask
  execution через explicit `ToolRequest` + `ToolGateway`, destructive physical split UI
  message storage на `chat_messages` и `workspace_messages`, отдельные chat/workspace
  send handlers, удалённый legacy event stream route, split frontend send entrypoints,
  memory/policy-feedback surface вынесен из `core/agent.py` в `AgentMemoryMixin`,
  tool-fail decision branch удалён.
- **Current legacy runtime**: крупные `core/agent_mwv.py`, `core/agent_tools.py`,
  `core/agent_routing.py`, часть `classify_request(...)`, runtime/API/UI `lane` markers
  и legacy UI endpoints ещё существуют как совместимость и не считаются целевой
  архитектурой.
- **`/v1/chat/completions` rollout**: поддерживает только `ask|auto` как opt-in;
  `plan|act` через `/v1` отклоняются и должны идти через UI workflow.
- **UI workflow**: endpoints `/ui/api/plan/*` и mode transitions допустимы только если
  соблюдают этот канон: Plan не исполняет, Act исполняет только packet.

## 1) Канонические роли режимов

1. Ask = **stateless**.
2. Plan = **transactional**.
3. Act = **isolated**.
4. Auto = **FSM orchestrator**.

## 2) Инварианты (обязательные)

### Ask (stateless, 0 side effects)

- Запрещены любые записи:
  - `save_to_memory`
  - `capture_memory_claims_from_text`
  - inbox/canonical writes
  - `vector_index.upsert/index/delete`
- Запрещены write/exec tool calls из ask-ветки.
- Разрешён только read-only контекст (memory/vector read path).
- Если vector runtime не готов, ask делает soft-degrade (ответ без vector-контекста, без hidden init).

### Plan (transactional)

- Plan только формирует/редактирует/валидирует `TaskPacket`.
- Plan не исполняет инструменты.
- Plan фиксирует execution-контракт:
  - `policy`
  - `scope`
  - `budgets`
  - `approvals`
  - `verifier`

### Act (isolated)

- Act исполняет только `TaskPacket.steps`.
- Act не меняет `scope/policy/budgets/verifier`.
- Любое отклонение от packet-контракта => `STOP_TO_CHAT` с `stop_reason_code=REPLAN_REQUIRED`.
- Retry через новый `packet_revision` в Plan, а не через импровизацию в Act.

### Auto (FSM only)

- Целевой Auto гоняет только детерминированный цикл: `Ask -> Plan -> Act`.
- В `runtime_mode=auto` запрещён chat-fallback.
- Budgets обязательны: time/tool_calls/tokens/files/retries.
- При fail/ambiguity/risk Auto останавливается в STOP и ждёт явного решения.

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

## 7) Rollout `/v1` (совместимость)

- Если `slavik_meta.runtime_mode` отсутствует -> legacy поведение без изменений.
- `slavik_meta.runtime_mode=ask|auto` -> opt-in в новый runtime router.
- `slavik_meta.runtime_mode=plan|act` -> `invalid_request_error` + `next_steps` на UI workflow.

## 8) Legacy debt boundary

Новые фичи не должны расширять legacy-пути как целевую архитектуру.

Legacy debt после PR-25:

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
- Legacy `/ui/api/events/stream` удалён из routes. Workspace-запросы через
  `/ui/api/chat/send` и chat-запросы через `/ui/api/workspace/send` отклоняются.
  Целевой путь — split chat/workspace endpoints.
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

Уже не является допустимым legacy для расширения:

- возвращение `Planner`/`Executor` как runtime entrypoints;
- regex extraction tool args из prose;
- regex target extraction / append-comment fake worker в MWV/CodingTask;
- slash-команды для обычных tools;
- отдельная server-only реализация PTY терминала рядом с one-shot runner.

Roadmap устранения legacy после PR-25: `docs/architecture/LEGACY_CLEANUP_ROADMAP.md`.
