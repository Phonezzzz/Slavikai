# RidgeGen × SlavikAI — reconciliation audit

Дата: 2026-09-25. База нового integration worktree: `origin/main=bcaff719cbb83d9779f3760ef7c16bb22995572a`. Старая RidgeGen-ветка `codex/ridgegen-architecture-integration` (`990c001`, `4dbc79f`) сохранена без переписывания. Этот аудит сопоставляет её с production consolidation после PR #48/#49/#50. Source of truth: `docs/SOURCE_OF_TRUTH.md`, `docs/runtime_contract_claims.json`, `docs/architecture/ARCH_CANON.md`, код и mechanism tests. Статус **accepted Target** означает согласованный контракт, а не работающий runtime.

## Что уже есть в main и что остаётся уникальным

| Контур | Фактическое состояние в `bcaff71` | Решение для этого PR |
| --- | --- | --- |
| Auto v1 | `agent_routing → agent_tools → auto_agent → AutoOrchestrator.run_v1 → AgentToolLoop → ToolGateway → verifier`; native tool calls. `core/auto_runtime.py` проверяет `loop_result.error` и **любой** `ToolResult.ok == False` перед verifier; `tests/test_auto_runtime.py` фиксирует `FAILED_WORKER` даже после более позднего успешного вызова. | Переиспользовать. Не переносить старый recovery guard и не ослаблять failed-tool gate. |
| Cloudflare browser identity, owner/member, principal/session | `server/http/common/auth.py`, `server/http/common/request_identity.py`, `server/agent_provider.py::AgentScope`, `server/ui_hub.py` и principal storage реализуют current scoped paths. `/v1` Bearer отделён. | Переиспользовать без новой identity-модели или migration. |
| Tool authority и approvals | `core/tool_gateway.py` применяет `ApprovalContext` и trusted policy до Registry; `core/tool_loop.py` проверяет список model-visible tools. Политика возвращает `policy_reason`, но текущий loop после deny продолжает batch. | Уникальное исправление: завершать sync/stream batch после policy rejection. Не вводить второй gateway. |
| UI cancellation | `respond_stream` получает token, `AgentToolLoop` умеет его проверять, но Auto branch `handle_auto_command` теряет token. Review #51 выявил блокирующие provider/tool/verifier paths, потерю terminal workflow/stream state, неотменяемый approval-resume и гонку с mutating effects. | **Pending отдельный runtime/auto-cancellation PR** после review узкого #51; здесь token не проводится. |
| Task/Run Lifecycle, Verification, Communication | PR #48 добавил Gate 0, Communication/Verification Target; PR #49 согласовал Task/Run Lifecycle Target и ADR-0012. `runtime_contract_claims.json` оставляет все три target. | Считать ограничением, не создавать параллельный lifecycle, completion authority или новый статус claim. |
| Coordination, Context, Memory target snapshots | `MULTI_AGENT_COORDINATION_CONTRACT.md` отсутствует в main; snapshot `4ee7516` pending. Context/Memory snapshots также pending. Auto legacy `run_parallel` не является production child-run coordination. | Никакой реализации child runs/delegation в этом PR. |
| Sensitive Vault, durable recovery | Vault в main отсутствует. Auto approval resume хранит `_paused_runs` в процессе и повторяет goal, это не durable checkpoint. MWV имеет bounded retry, но не общий persistent execution ledger. | Вне scope; не переносить концепции как фиктивную реализацию. |

Старый RidgeGen diff в `core/auto_runtime.py` добавлял отдельную обработку `loop_result.error`; это уже сделано в `main` и теперь должно сосуществовать с более строгой проверкой **любого** failed tool call. Перенос старого файла целиком потерял бы `tool_failure_unrecovered` и тест, запрещающий ложный SUCCESS после unrelated success. В `core/tool_loop.py` старый batch-stop остаётся уникальным; его можно перенести без изменения `ToolGateway`/approval decision. Старые отчётные цифры и база `888b6c2` не являются current evidence.

## Владение state и оставшиеся blockers

| Компонент | Статус | Граница и дефицит |
| --- | --- | --- |
| `RunContext` (`core/mwv/models.py`) | PARTIALLY IMPLEMENTED | MWV attempt/retry/approval context есть; нет authoritative task/revision/run identity, ownership epoch и durable checkpoint. |
| `AgentScope`, UIHub session | IMPLEMENTED для principal/session scope | Mutable Agent и lock принадлежат `(principal_id, session_id)`; session не замещает task/run identity. Shared Desktop coordinator сериализует host action. |
| `ExecutionPolicy` / `ApprovalContext` | PARTIALLY IMPLEMENTED | Проверки на ToolGateway и Desktop boundary есть; единого frozen policy snapshot для будущей delegation нет. |
| `ToolRegistry` / `ToolGateway` | IMPLEMENTED | Единственный разрешённый tool dispatch path; отказ нельзя обходить replanning. |
| `AutoOrchestrator`, `AgentToolLoop` | IMPLEMENTED Auto v1, PARTIAL для целевой оркестрации | `FAILED_WORKER` после failed tool уже current. Только policy batch-stop — scope этого PR; Auto cancellation pending. |
| MWV/verifier | PARTIALLY IMPLEMENTED | Реальные deterministic checks и bounded retry есть, но authoritative acceptance exact task/result revision — Target. |
| Task/Run lifecycle | ACCEPTED TARGET | Контракт принят; authoritative runtime transition owner/store ещё нет. |
| Multi-Agent coordination | CONTRACT ONLY / pending | Нет принятого cross-contract Multi-Agent contract и production child-run authority. |
| Persistent execution state/recovery | PARTIALLY IMPLEMENTED | UI/Auto snapshots и process-local approval resume не гарантируют replay safety или consistency после restart. |
| Memory и credentials | PARTIALLY IMPLEMENTED / Vault MISSING | Explicit Memory confirmation и principal paths есть; run ledger не Memory. Owner-only application credential проверяется, но SafeBox/Sensitive Vault ещё нет. |
| Provider abstraction, audit | PARTIALLY IMPLEMENTED | Brain/provider route и trace есть; native tools доступны не каждому provider, нет immutable run/evidence digest. |

PR #48 зафиксировал Gate 0 и Target Communication/Verification; PR #49 согласовал терминальные cause/disposition, supersession и ordinary Ask identity **на уровне Target**; PR #50 консолидировал production UI/runtime/security изменения. Последующий `bcaff71` сделал Auto fail-closed для любого failed tool call. Открыты authoritative lifecycle store, exact-revision verification acceptance, Multi-Agent contract/delegation, durable recovery, execution ledger и credential authority. Нельзя считать UI snapshot или verifier `PASSED` самодостаточной completion authority.

## 18 acceptance criteria из исходного запроса

`Частично` означает реальную проверяемую часть без полного целевого свойства. Новые тесты этого PR подтверждают **только batch-stop в строке 8**; строки 13 и 15 опираются на существующие тесты `main`. Эта матрица не заявляет live production proof.

| № | Критерий | После этого increment | Фактическая граница |
| --- | --- | --- | --- |
| 1 | Independent agent execution | Частично | `AgentScope` даёт отдельный Agent per principal/session; child agents отсутствуют. |
| 2 | Concurrent run isolation | Частично | Scope locks защищают Agent; durable concurrent task/run identity нет. |
| 3 | Principal isolation | Реализовано для current lanes | JWT, storage, session и Agent scopes покрыты тестами; delegation не существует. |
| 4 | Session isolation | Реализовано для current lanes | UIHub access check и scoped Agent; session не Task. |
| 5 | Owner-only capabilities | Реализовано для current lanes | В том числе TTS application credential; будущий Vault вне scope. |
| 6 | Member permission boundaries | Реализовано для current lanes | Browser role и route checks; child inheritance не проверяется. |
| 7 | Child-agent delegation authority | Не реализовано | Нет production child runs/accepted Multi-Agent contract. |
| 8 | Approval rejection | Реализовано для Auto tool batch | `ToolGateway` deny/ASK; после deny следующий batch call больше не исполняется. |
| 9 | Cancellation propagation | Частично; Auto pending | Cancellation есть в некоторых stream/tool-loop paths, но Auto теряет UI token; terminal и partial-effect semantics не решены. |
| 10 | Interrupted run recovery | Частично | Process-local approval resume; crash-safe recovery отсутствует. |
| 11 | Persistent state consistency | Частично | UI/Auto snapshots есть, authoritative ledger/checkpoint нет. |
| 12 | Evidence validation | Частично | MWV verifier и Auto profiles есть; exact-revision acceptance Target. |
| 13 | False SUCCESS prevention | Частично | Auto failed tool и loop error не завершаются success; отмена Auto и общий Task completion остаются открытыми. |
| 14 | Secret isolation | Частично | Current owner checks и отдельные storage paths; Sensitive Vault отсутствует. |
| 15 | Tool execution failure handling | Реализовано для Auto v1 | `bcaff71` считает любой failed call `FAILED_WORKER`, даже после later success. |
| 16 | Bounded replanning | Частично | MWV `max_retries`; общего Auto durable replan нет и здесь не вводится. |
| 17 | TASK/RUN lifecycle compatibility | Совместимо с Target, не реализовано | Локальные Auto status не превращены в authoritative task transitions. |
| 18 | Multi-Agent Coordination compatibility | Не доказано | Контракт pending; не создаём несовместимых child paths. |

## Anti-pseudo audit и security boundary

Current Auto идёт через native `AgentToolLoop`/`ToolGateway`, а не classifier. `core/auto_agent.py` сохраняет legacy prose `generate_subtasks/run_parallel/run_subagent`, но они не подключены к Auto v1 и не считаются multi-agent runtime. Legacy Plan/Act router остаётся reachable и не расширяется. Новые проверки происходят на существующем loop result и доверенном `policy_reason` из ToolGateway/model-visible gate; отказ завершает batch, повторный вызов модели не обходит его. В этом PR не добавляются adapters, скрытый CLI path, миграции, новые principal, независимый verifier или completion authority.
