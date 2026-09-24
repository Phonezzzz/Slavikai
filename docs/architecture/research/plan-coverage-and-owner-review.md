# Gate 0 — покрытие плана и проверка владельцев

**Статус:** исследовательская сверка, 24.09.2026; системная карта принята на
крупном уровне в [Gate 0 matrix](gate-0-closure-matrix.md). Основа: [рабочий план](../roadmap/master-roadmap.md),
[ARCH_CANON](../ARCH_CANON.md), действующие ADR-0001–0010, Communication и
Verification contracts, [Current State matrix](current-state-flow-ownership-matrix.md)
и четыре **неинтегрированных** snapshot contract. Production checkout при
сверке: `920d5f3`; этот документ не утверждает runtime behavior сверх
указанных точечных проверок.

Здесь «owner» означает единственную authority для конкретного факта или
решения. Номер раздела плана может охватывать несколько owners; один owner
может обслуживать несколько разделов. Это не разбиение на процессы или базы.

## Покрытие разделов 1–20

| Раздел плана | Системная классификация и кандидат authority | Граница, без которой карта неполна |
| --- | --- | --- |
| 1 Multi-Agent Coordination | Capability owner координационной истории, membership и сообщений; Lifecycle отдельно принимает claims и task transitions | Неинтегрированный `4ee7516` ещё требует привязки к task revision, principal и owner epoch |
| 2 Context | Capability owner временной model-visible проекции | Context не хранит canonical task, policy, Memory или accepted result; `4b9a453` ещё требует revision/invalidation reconciliation |
| 3 Memory | Capability owner принятого долговременного знания и его provenance/correction/forget | Memory не является Context, transcript, artifact, approval или credential vault; `f883893` требует consent/deletion handoff и решения OQ-MEM-01 о policy-based promotion |
| 4 Task/Run Lifecycle | Capability owner accepted goal/criteria/task/run/subtask state и transition authority | ADR-0002/0010 приняты; `6040640` пока несовместим по `aborted` и supersession/final mapping |
| 5 User Interaction / Communication | User Intent/Decision ingress предлагает изменения; Communication владеет semantic progress/wait/final records и delivery identity | UI собирает input, но не принимает task или policy transition; request-scoped media — отдельный adapter по ADR-0008 |
| 6 Verification | Verification владеет criteria-bound evidence, profiles и typed outcomes; Lifecycle commits acceptance/terminal state | Verifier pass и artifact existence не равны user acceptance или completion |
| 7 Tool Execution | Capability owner typed operation/attempt/outcome и effect reconciliation | Policy допускает действие, Resource Governance ограничивает расход, Artifact фиксирует result; ToolResult не доказывает postcondition |
| 8 Approval / Policy | Security authority для effective policy и отдельная authority для scoped approval decisions | Action approval, plan acceptance, trigger-rule approval и result acceptance имеют разные subjects/lifetimes |
| 9 Identity / Principal / Isolation | Security authority для authenticated principal, role, delegation и scope binding | Browser Access и Bearer automation — разные auth lanes; session/task ID сам по себе не grant |
| 10 Failure & Recovery | Сквозной protocol каждого state/effect owner; Lifecycle решает судьбу task/run, Tool Execution — unknown effect | Нужен явный recovery controller/handoff, но один глобальный store не становится владельцем всех фактов |
| 11 State Persistence | Сквозной contract canonical state/version/transaction/retention для каждого owner | UIHub snapshot, cache, Context и stream — derived/ephemeral, если owning contract не установил обратное |
| 12 Event Architecture | Сквозной causal publication/delivery contract; material event выпускает owner изменённого факта | UI replay, coordination event, security audit и telemetry имеют разные authority и retention |
| 13 Two-Phase Communication | Подзадача Communication, а не новый owner | Meaningful progress и один revision-bound final с replay/deduplication |
| 14 Background Execution | Исполнитель и recovery существующего task/run под authority Lifecycle | Не совпадает с Work Initiation, которое создаёт новую task по approved trigger |
| 15 Resource / Budget | Resource Governance — allocation/reservation/accounting; Lifecycle принимает exhaustion transition | Child/parallel budgets, model/tool/verification/host расход не превышают grant (ADR-0006) |
| 16 Artifacts | Capability owner artifact identity, version/integrity, provenance, retention и result links | Existing file, response text и UI session download не являются accepted result |
| 17 Observability / Audit | Сквозные projections и security audit; System Evaluation отдельно владеет cross-task cases/scores | Trace не authority task truth, eval не принимает отдельный result, redaction/retention обязательны |
| 18 Model Access / Routing | Capability owner route eligibility, qualification, selection/fallback; Local Inference Operations владеет host engine lifecycle | Credential entitlement, protocol, capability provenance и data egress различимы (ADR-0001/0005/0007) |
| 19 Security / Threat Model | Cross-domain trust model с Identity, Policy/Approval и Credentials authorities | Untrusted model/tool/extension input не может повысить authority; provider-key use по ADR-0007 |
| 20 Source of Truth / Consistency | Governance документации и machine-readable runtime claims, не execution authority | Target/current/claim status и доказательства должны быть прослеживаемы; audit не создаёт runtime behavior |

## Обнаруженные области вне нумерации

| Область | Решённая граница | Что ещё не является Gate 0 доказательством runtime |
| --- | --- | --- |
| Work Initiation | ADR-0003/0004: approved rule/firing owner отдельно от Lifecycle и background continuation | Правила, дедупликация и wakeup остаются OQ-CD-04 |
| Media Interaction Adapters | ADR-0008: отдельные request-scoped STT/TTS/image/attachment operations | Единый consent, retention, egress и artifact path ещё не аудирован |
| Skills/Reusable Procedures | Instructions, version/eligibility/provenance; нет executable authority | Текущий manifest selection не означает provider governance |
| External Extension Governance | ADR-0009: provider identity/version/capability/grant/revocation отдельно от skills | Transport, installation, credentials и call-time enforcement ещё не реализованы как Target |
| Local Inference Operations | ADR-0005: host engine lifecycle отдельно от Model Access route | Текущий локальный HTTP client и узкий Ollama launcher не равны manager |
| System Evaluation | ADR-0006: cross-task quality owner отдельно от per-task Verification | Текущие feedback/batch paths не подтверждают зрелый eval system |
| Data consent, retention and egress | Cross-domain policy decision + records у media, Memory, Artifact и provider owners | Нужен точный audit, чтобы consent не был молча выведен из login или выбора tool |
| Deployment / Runtime Operations | Configuration, rollout, process health и rollback принадлежат operational owner; Local Inference Operations владеет только model-host engine | Наличие `deploy/*` и server boot не доказывает единую lifecycle/rollback модель для всех execution backends |

## Решения по owner handoff на уровне Gate 0

1. **Accepted goal/criteria, task/run state и final cause** принадлежат
   Lifecycle. Planning предлагает стратегию; user/authorized policy принимает
   material goal change. Verification предоставляет evidence, Communication
   доставляет outcome. Детальный lifecycle transaction ещё заблокирован.
2. **Эффект и результат** имеют разные owners: Tool Execution фиксирует
   operation/attempt/observed or unknown effect; Artifact фиксирует объект и
   integrity; Verification оценивает evidence; Lifecycle принимает completion.
   Это системное разделение, не готовый общий runtime path.
3. **Право действовать** происходит от Identity + effective Policy/Approval,
   а расход — от Resource Governance. Skills, external providers, Context,
   Memory, model output и UI events предоставляют данные или предложения.
4. **Recovery и persistence** требуют протокола между владельцами, а не
   второго универсального владельца task/effect/approval. Кто запускает
   recovery pass, какие transaction boundaries и retention применяются —
   отдельный аудит разделов 10–12 после базовых owner contracts.
5. **Data consent/egress** — policy-bound decision для конкретного
   principal/source/purpose/provider. Хранение и удаление остаются у owning
   Memory/Artifact/Media/Extension subsystem; security audit фиксирует решение.

## Предел вывода и следующий gate

Покрытие всех 20 номеров и найденных дополнительных областей подтверждает,
что у каждой крупной темы есть место в системной модели. Это **не** доказывает
полноту всех будущих features и не делает snapshots действующими контрактами.
Проверенные owner boundaries перенесены в четыре `system/` документа;
directed dependencies и доверительные переходы сверены. Lifecycle snapshot
явно ограничен как неинтегрированный источник для раздела 4; принятая норма
зафиксирована ADR-0002/0010. Подробные schema/protocol/PR decisions
принадлежат разделам плана.
