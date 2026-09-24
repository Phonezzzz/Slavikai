# Capability Map

**Статус: принятая системная Target-карта Gate 0 (24.09.2026).** Здесь
зафиксированы крупные capability owners и границы. Это не перечень готовых
runtime-подсистем или принятые подробные контракты каждого owner. Current State и область поиска
находятся в [инвентаре](../research/capability-discovery-inventory.md),
[предложении границ](../research/capability-boundary-proposal.md) и
[сверке всех 20 разделов](../research/plan-coverage-and-owner-review.md), а также
[матрице закрытия Gate 0](../research/gate-0-closure-matrix.md).

## Принятые владельцы и ограничения

«Владелец» означает authority для своего факта или решения, а не отдельный
процесс, сервис или базу данных. Общий принцип принят в
[ADR-0011](../decisions/ADR-0011-system-fact-ownership-and-cross-cutting-protocols.md).

| Capability / authority | Авторитетный факт или решение | Основание и предел |
| --- | --- | --- |
| Work Initiation | Утверждённые пользователем правила запуска, их revision/revocation, firing history и стабильный task-creation intent | [ADR-0003](../decisions/ADR-0003-preapproved-autonomous-work-initiation.md), [ADR-0004](../decisions/ADR-0004-work-initiation-ownership-boundary.md); Lifecycle отдельно принимает задачу, firing protocol открыт |
| Task/Run Lifecycle | Принятая задача/ревизия, состояние run, переход и terminal outcome; supersession lineage без old-revision final | [ARCH_CANON](../ARCH_CANON.md), [ADR-0002](../decisions/ADR-0002-terminal-outcome-and-partial-result.md) и [ADR-0010](../decisions/ADR-0010-supersession-lineage-without-separate-final.md); lifecycle snapshot ещё не согласован, поэтому полный enum/transaction contract здесь не утверждён |
| Verification/Acceptance | Criteria-bound evidence, typed outcome и интерфейс authorized acceptance для точных result revisions | [Verification contract](../VERIFICATION_ARCHITECTURE_CONTRACT.md); Lifecycle принимает completion, а partial result принимает пользователь или заранее утверждённые критерии |
| User Communication | Семантические progress/wait/final/notification records и их delivery identity | [Communication contract](../USER_INTERACTION_COMMUNICATION_CONTRACT.md); сообщение и UI stream не меняют task truth |
| Model Access | Различимые access routes, entitlement/capability qualification и явный выбор/fallback | [ADR-0001](../decisions/ADR-0001-subscription-backed-openai-access.md); compliant subscription transport ещё не выбран и не реализован |
| Local Inference Operations | Lifecycle локального host engine, readiness/health и reconciliation | [ADR-0005](../decisions/ADR-0005-local-inference-operations-boundary.md); model acquisition и tuning ещё не приняты |
| Resource Governance | Shared allocation/reservation для задач, детей, моделей, инструментов, проверки и host resources | [ADR-0006](../decisions/ADR-0006-resource-event-evaluation-authority.md); units/ledger/recovery открыты |
| System Evaluation | Cross-task cases, datasets, scored runs и quality findings | [ADR-0006](../decisions/ADR-0006-resource-event-evaluation-authority.md); не принимает результат отдельной задачи |
| Media Interaction Adapters | Отдельные STT/TTS/image/attachment requests и привязка их media artifacts к общим task/policy/communication границам | [ADR-0008](../decisions/ADR-0008-request-scoped-media-interaction.md); непрерывная realtime media session не входит в Target |
| Skills/Reusable Procedures | Инструкции, версия, provenance и eligibility выбранной процедуры | [ADR-0009](../decisions/ADR-0009-governed-external-tool-providers.md); skill text не выдаёт executable permissions |
| External Extension Governance | Подключение, версия, provenance, capabilities, principal scope и revocation внешнего tool provider | [ADR-0009](../decisions/ADR-0009-governed-external-tool-providers.md); каждый вызов проходит Tool Execution и Policy |

## Системные границы, ожидающие отдельного Target contract

Эти крупные области присутствуют в [плане и owner review](../research/plan-coverage-and-owner-review.md).
Их место в системной карте определено, но detailed authority/transaction
contract нельзя объявлять принятым по одному snapshot или текущему классу кода.

| Область | Предлагаемый владелец факта | Что должно быть принято позднее |
| --- | --- | --- |
| User Intent / Planning | Intent и plan proposals; Lifecycle отдельно принимает goal/criteria и plan revision | Граница clarification, material change, replan и plan acceptance; section 4/5 |
| Agent Coordination | Typed coordination history, membership и dependency exchange | Claim/owner epochs, principal/audience scope, durable publication; snapshot `4ee7516`, section 1 |
| Context | Временная, version-bound model-visible проекция | Source eligibility, scope, invalidation/reconstruction; snapshot `4b9a453`, section 2 |
| Memory | Принятое долговременное знание и его provenance/correction/forget | Consent/promotion, retrieval eligibility, deletion closure; snapshot `f883893`, section 3 |
| Tool/Action Execution | Operation intent, attempt, observed/unknown effect и reconciliation | Typed action/target, retry/idempotency, postcondition и recovery; section 7 |
| Result/Artifact Records | Artifact identity, version/integrity, provenance и retention | Candidate/partial/accepted result links, external artifacts и concurrent modification; section 16 |
| Identity / Policy / Approval / Credentials | Раздельные security authorities для principal, effective permission, scoped decision и credential use | Propagation, revocation, delegation и audit; sections 8/9/19, ADR-0007 |
| Deployment / Runtime Operations | Конфигурация, rollout, health и rollback приложения и execution backends | Не смешивать с Local Inference Operations, которое владеет lifecycle локального model engine; аудит deployment/current paths в section 19/20 |

Recovery, persistence, causal events, observability и documentation/claims
governance — сквозные contracts по ADR-0011, а не ещё один владелец перечисленных task,
effect или artifact facts. Каждый owner называет canonical state, material
history, derived projections и восстановление. Resource Governance и System
Evaluation уже имеют отдельные решения ADR-0006.

Identity/principal, policy/approval и credentials остаются отдельными
security authorities для всех строк. Их scope и текущие пути проверены
[выборочным аудитом](../research/principal-policy-credential-acceptance-boundaries.md),
а [ADR-0007](../decisions/ADR-0007-owner-provider-credential-delegation.md)
фиксирует owner-only default и явное делегирование provider key. Детальный
credential-use contract и runtime enforcement ещё открыты. Human oversight собирает разные
решения пользователя и передаёт их соответствующему владельцу; единый
`approved` state для действий, результатов и trigger rules недопустим.

## Открытые границы и проверки

| Область | Текущий статус | Требуемое решение |
| --- | --- | --- |
| Multi-agent Coordination, Context и Memory | Target snapshots `4ee7516`, `4b9a453`, `f883893` не интегрированы; различие shared history, model-visible projection и принятого знания исследовано | Сверить с финальными task/principal/policy identities и принять либо исправить контракты |
| Result/Artifact Records и Tool/Action Execution | Кандидатные owners; текущие UI artifacts и ToolResult не доказывают accepted result | Утвердить identity, версии, integrity, side effects и recovery boundary |
| Media request contracts | Продуктовая граница решена ADR-0008; текущие отдельные пути ещё не подтверждены как единый policy/provenance contract | Аудит consent, retention, provider egress и результата в разделах 5/7/16/18/19 |
| External extension implementation | Scope и отдельный owner решены ADR-0009; runtime подключение не подтверждено | Аудит installation/version/revocation/credentials/action path в разделах 7–10/16/19 |
| Provider credentials | Owner-only default и явное делегирование приняты в [ADR-0007](../decisions/ADR-0007-owner-provider-credential-delegation.md); application-level resolver пока не получает principal | Аудит всех egress paths, credential handle, rotation/revocation и runtime enforcement в разделах 8/9/18/19 |
| Recovery, persistence, events | Сквозные contracts с owner-local canonical facts; текущие UI snapshots/event streams не доказывают durability | Аудит transaction, lease/epoch, replay/dedup и unknown effects в разделах 10–12 |
| Consent, retention и data egress | Approval/Policy хранит consent grant/revocation и решает допустимость конкретного use; owning Memory/Artifact/Media/Extension хранит и удаляет свой факт | Аудит источника согласия, purpose, provider scope, revocation и cross-owner deletion |
| Deployment / Runtime Operations | Отдельный operational owner предложен для setup, health и rollout; current deployment paths не исследованы целиком | Проверить конфигурацию, rollback, secret injection и границу с engine manager; section 19/20 |

Persistence/recovery, causal events, provenance, security audit и
observability являются сквозными контрактами: каждый owner называет canonical
state и историю своих изменений. [ADR-0006](../decisions/ADR-0006-resource-event-evaluation-authority.md)
не делает общий event bus владельцем фактов. Подробные границы — в
[boundaries.md](boundaries.md), потоки — в [data-flows.md](data-flows.md),
зависимости — в [dependencies.md](dependencies.md). Ни одна строка этой карты
не повышает статус в `docs/runtime_contract_claims.json`.
