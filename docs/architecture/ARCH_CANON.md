# ARCH_CANON — runtime canon Ask/Plan/Act/Auto

Этот документ — **source of truth** для целевой архитектуры runtime и rollout-границ.
`docs/architecture/Architecture.md` описывает текущую legacy-инвентаризацию и не является
источником целевого поведения, если расходится с этим документом.

## 0) Статус канона

- **Target runtime**: Ask / Plan / Act / Auto, описанные ниже.
- **Current legacy runtime**: допускается только как временная реализация до PR-цепочки
  native tool calling и разделения Chat/Workspace.
- **`/v1/chat/completions` rollout**: поддерживает только `ask|auto` как opt-in;
  `plan|act` через `/v1` отклоняются и должны идти через UI workflow.
- **UI workflow**: может иметь endpoints `/ui/api/plan/*` и mode transitions, но их смысл
  должен соответствовать этому канону: Plan не исполняет, Act исполняет только packet.

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
- Запрещены tool calls из ask-ветки.
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

- Auto гоняет только детерминированный цикл: `Ask -> Plan -> Act`.
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

До завершения перехода текущие `Planner`/`Executor`, regex routing, command lane tools и общий
Chat/Workspace session state считаются legacy debt. Новые фичи не должны расширять эти пути как
целевую архитектуру.
