# ARCH_GAP_ANALYSIS — переход на Manager → Worker → Verifier

Документ фиксирует разрыв между текущей архитектурой Slavik и целевой моделью Manager → Worker → Verifier.
Фокус: анализ, без кода и без изменений поведения.

## Источники (прочитано)
- `Architecture.md`
- `PROJECT_OVERVIEW.md`
- `DevRules.md`
- `COMMANDS.md`
- `MEMORY_COMPANION_MINIPROMPTS.md`
- `docs/integration/openwebui/API_CONTRACT.md`
- `docs/audit/PHASE_0_FINDINGS.md`
- `docs/MAP.md`
- `docs/LIFECYCLE.md`
- `docs/PHASE_3_IMPLEMENTATION.md`
- `docs/PHASE_5_SAFE_MODE.md`
- `docs/PHASE_5_DECISION_MATRIX.md`
- Код: `core/*`, `llm/*`, `tools/*`, `memory/*`, `shared/*`, `server/*`, `config/*`, `ui/*`, `tests/*`

---

## A) Что уже совпадает (частично) с Manager → Worker → Verifier

1) **Детерминированные проверки уже существуют как скрипт качества**
   - `scripts/check.sh` — ряды линтеров/форматирования/типизации/тестов.
   - Это можно использовать как основу Verifier, но сейчас скрипт не встроен в runtime-пайплайн.

2) **Есть центральный оркестратор действий**
   - `core/agent.py` сейчас играет роль “центра принятия решений” (планирование/исполнение/инструменты/память).
   - Это близко к Manager, но смешивает роли Manager/Worker/Verifier в одном модуле.

3) **Есть инструментарий и явный контур безопасности**
   - `tools/tool_registry.py`, `core/tool_gateway.py`, `ToolResult` и sandbox — важная часть Worker-инструментов.
   - Safe-mode + approvals уже реализованы (Phase 5): `core/approval_policy.py`, `docs/PHASE_5_*`.

4) **Есть трассировка и логирование**
   - `core/tracer.py`, `tools/tool_logger.py` — пригодны для будущего отчёта Verifier/Worker, но сейчас не формируют итоговую “верификацию зелёного статуса”.

---

## B) Что должно быть удалено (DualBrain/critic)

### Концепт DualBrain и режимы (single/dual/critic-only)
**Нужно полностью убрать/декларировать как deprecated и затем удалить**:
- `llm/dual_brain.py` (DualBrain контейнер)
- `core/critic_policy.py` (A+D критик, decide_critic, classify_critic_status)
- `config/mode_config.py` + `config/mode.json` (режимы single/dual/critic-only)
- `config/system_prompts.py` (CRITIC_PROMPT, THINKING_PROMPT в контексте dual логики)
- `config/model_store.py` (critic_config и связанность с dual)
- `server/http_api.py`:
  - выдача моделей `slavik-dual`, `slavik-critic`
  - meta `critic_status`, `critic_reasons`
  - проверка режима dual/critic-only
- UI‑слой:
  - `ui/mode_panel.py` (переключатель DualBrain)
  - `ui/dual_chat_view.py` (двухколоночный чат)
  - `ui/settings_dialog.py` (поля критика)
  - строки в `ui/main_window.py` про critic и режим
- Документация:
  - `docs/PHASE_4_DUALBRAIN.md`
  - упоминания DualBrain/critic в `docs/LIFECYCLE.md`, `docs/MAP.md`, `Architecture.md`, `PROJECT_OVERVIEW.md`, `API_CONTRACT.md`, `PHASE_3_IMPLEMENTATION.md`
- Тесты:
  - `tests/test_critic_policy.py`
  - `tests/test_agent_modes.py`
  - `tests/test_agent_modes_logic.py`
  - `tests/test_plan_critic.py`
  - `tests/test_http_api.py` (часть с dual/critic моделями и critic_status)

### Критические связки в `core/agent.py`
- Методика `_current_mode`, `_review_plan`/`_critic_plan`/`_critic_step`, `last_critic_*`, вызовы критика и режимов.
- В текущем виде это полностью противоречит требованию «Verifier — не LLM».

---

## C) Что отсутствует для Manager → Worker → Verifier

### 1) Явная архитектура Manager → Worker → Verifier (контракты и модели)
- Нет отдельных модулей/контрактов для Manager/Worker/Verifier.
- Нужны структурированные сущности вроде `TaskPacket`, `WorkResult`, `VerificationResult`.
- Нужен единый протокол передачи результатов между ролями (и хранения trace).

### 2) Детерминированный Verifier
- Нет модуля, который запускает `scripts/check.sh` (или эквивалентные команды) и возвращает строгий результат.
- Нет политики «зелёный verifier = единственный успех». Сейчас успех определяется LLM‑ответом.

### 3) Контур авто‑итераций и rollback
- Нет bounded‑retry цикла (макс N попыток, критерии остановки).
- Нет правил rollback/отката при неуспешной проверке.
- Нет явной эскалации (например: “если не прошло — собрать диагностический отчёт” без бесконечной петли).

### 4) Гейтинг действий и итогов
- Safe‑mode/approvals существуют, но не привязаны к Verifier.
- Нет правила “Worker не считается успешным без зелёного Verifier”.

### 5) Канонический “verifier report” для трассы/UX
- Нужен структурированный отчёт: какие команды запускались, статус, stderr/stdout, итог.
- Сейчас trace сохраняет события, но не отражает результат проверки как единственный критерий успеха.

---

## D) Что НЕ трогать сейчас

- **Инструменты и sandbox**: `tools/*`, `shared/sandbox.py`, `ToolRegistry` и safe‑mode механика.
- **Approvals** (Phase 5): `core/approval_policy.py`, `/slavik/approve-session`, decision matrix.
- **Memory Companion**: `memory/memory_companion_store.py`, `shared/policy_models.py`, BatchReview.
- **HTTP gateway** (Phase 3): `server/http_api.py` и контракт `/v1/*` — менять только после утверждения нового API‑контракта.
- **UI и OpenWebUI**: никакой интеграции/форка сейчас.

---

## Где сейчас есть Verifier (частично)

- **Есть только скрипт** `scripts/check.sh`.
- Он не интегрирован в runtime‑поток Agent/Executor.
- Нет кода, который делает verifier обязательным критерием успеха.

---

## Обязательный DoD для миграции (измеримые критерии)

1) **Архитектура**: менеджер/воркер/верифайер представлены как отдельные сущности или модули с явными контрактами.
2) **DualBrain удалён**: ни одного режима `single/dual/critic-only`, ни критика в коде/конфиге/тестах/доках.
3) **Verifier детерминированный**:
   - Использует `scripts/check.sh` (или единый зафиксированный набор команд).
   - Вердикт «зелёный/красный» — единственный критерий успеха.
4) **Гейтинг**: Worker‑результат не применяется, если Verifier красный.
5) **Bounded‑retry**: задан максимум попыток, условия остановки и rollback‑правила.
6) **Качество**:
   - `make check` (или эквивалент) проходит.
   - `pytest` проходит.
   - `mypy` проходит.
   - coverage ≥ 80%.
7) **E2E сценарий**:
   - Manager формирует задачу → Worker предлагает изменения → Verifier запускает проверки → изменения считаются успешными только при зелёном результате.

---

## Примечания по конфликтам инвариантов

- Сейчас LLM‑критик “решает” судьбу плана/ответа — это противоречит требованию «Verifier не LLM».
- Auto‑save в память (если включён) потенциально конфликтует с «explicit policy», требует отдельного решения при миграции.

---

## Статус M0/M1 (scaffolding без wiring)
- Добавлен каркас `core/mwv/*` (модели + детерминированный VerifierRunner), без подключения к Agent.
- Документы миграции и сценариев обновлены (`MIGRATION_MWV_PLAN.md`, `E2E_SCENARIOS.md`).
- Runtime‑pipeline (agent/planner/executor) не менялся.
