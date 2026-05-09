# ARCH_CANON — runtime canon Ask/Plan/Act/Auto

Этот документ — **source of truth** для runtime-архитектуры и rollout-границ.
`docs/architecture/Architecture.md` описывает текущее фактическое устройство и обязано
называть legacy-пути legacy, а не целевым поведением.

## 0) Статус канона

- **Target runtime**: Ask / Plan / Act / Auto, описанные ниже.
- **Implemented baseline**: tool-calling типы, `AgentToolLoop`, explicit `PlanStep.tool_args`,
  debug-only command lane, split chat/workspace send+SSE endpoints, единый `TerminalTool`.
- **Current legacy runtime**: `core/agent*.py`, часть `classify_request(...)`, storage `lane`
  и legacy UI endpoints ещё существуют как совместимость и не считаются целевой архитектурой.
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

Legacy debt после PR-0..PR-7:

- `core/agent.py`, `core/agent_mwv.py`, `core/agent_tools.py`, `core/agent_routing.py`
  остаются крупными mixin-модулями.
- `classify_request(...)` ещё участвует в legacy `plan|act|auto` маршрутизации.
- Storage ещё хранит `ui_messages.lane`; новые `ChatThread` / `WorkspaceSession` типы
  существуют как domain views, а не как полностью отдельные таблицы.
- Legacy `/ui/api/events/stream` и lane-multiplexed behavior внутри `/ui/api/chat/send`
  ещё есть для совместимости; целевой путь — split chat/workspace endpoints.
- Провайдеры приняли `tools` в контракте, но native tool calling реализован не во всех
  provider backends.

Уже не является допустимым legacy для расширения:

- regex extraction tool args из prose в `Planner`/`Executor`;
- slash-команды для обычных tools;
- отдельная server-only реализация PTY терминала рядом с one-shot runner.
