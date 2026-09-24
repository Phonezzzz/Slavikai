# Architecture docs

- `ARCH_CANON.md` — runtime canon и границы legacy.
- `Architecture.md` — текущая runtime inventory.
- `USER_INTERACTION_COMMUNICATION_CONTRACT.md` — target semantic contract пользовательской
  коммуникации; не является текущей runtime implementation.
- `research/task-run-communication-audit.md` — evidence, alternatives и rationale.
- `research/lifecycle-communication-verification-reconciliation.md` — audit расхождения lifecycle, communication и verification; решение зафиксировано в ADR-0002, snapshot ещё не интегрирован.
- `research/capability-discovery-inventory.md` — сохранённый candidate inventory и вопросы, исследованные при Gate 0.
- `research/capability-boundary-proposal.md` — исследовательское предложение owners, cross-cutting contracts и зависимостей; не утверждённая системная карта.
- `research/plan-coverage-and-owner-review.md` — сверка всех 20 пунктов плана с владельцами фактов, сквозными контрактами и найденными областями вне нумерации.
- `research/contract-status-inventory.md` — Gate 0 реестр действующих, неинтегрированных и отсутствующих Target-контрактов и runtime claims.
- `research/current-state-flow-ownership-matrix.md` — проверенные входы, исполнители, state и границы доказанного Current State для Gate 0.
- `research/system-flows-and-trust-boundaries.md` — исследовательская схема текущих потоков, границ доверия и предлагаемых Target handoffs; не принятый `system/` контракт.
- `research/state-authority-inventory.md` — Gate 0 inventory владельцев, lifetime и границ durable/ephemeral Current State; не контракт Persistence/Recovery.
- `research/principal-policy-credential-acceptance-boundaries.md` — Gate 0 audit principal, approval, provider credentials и принятия результата; ADR-0007 решает owner-key default/delegation.
- `research/media-extension-operations-boundary-audit.md` — выборочный Current State audit отдельных STT/TTS/attachment paths, static tool/skill registry и deployment surfaces.
- `research/resource-event-evaluation-boundary-audit.md` — Gate 0 audit ресурсов, событий и evaluation; owner split принят в ADR-0006.
- `research/gate-0-closure-matrix.md` — проверяемый record принятия системной карты и её пределов.
- `research/mature-system-boundary-check.md` — первичные reference patterns для work initiation, внешних инструментов и continuous media; SlavikAI scope решён в ADR-0003/0008/0009.
- `research/gate-0-decision-brief.md` — принятые product decisions и оставшиеся подробные engineering boundaries.
- `system/capability-map.md`, `system/boundaries.md`, `system/data-flows.md`, `system/dependencies.md` — принятая крупная Target-системная карта Gate 0; runtime-реализацию эти документы не подтверждают.
- `VERIFICATION_ARCHITECTURE_CONTRACT.md` — target verification/acceptance contract.
- `VERIFICATION_ARCHITECTURE_RESEARCH.md` — current map, external research и alternatives.

---

# Архитектура SlavikAI

Точка входа в долгосрочную архитектурную документацию SlavikAI.

- **Для чего:** ориентироваться в структуре `docs/architecture/`, понимать статус разделов и правила ведения.
- **Сюда:** навигация по разделам, статус каждого раздела, соглашения о ведении документации.
- **Не сюда:** конкретные архитектурные решения, финальный список capabilities, implementation roadmap, выжимки текущего кода.
- **Обновлять:** при добавлении/удалении разделов или файлов, изменении статуса Capability Discovery, изменении правил ведения.

## Правило о Capability Discovery

Gate 0 Capability Discovery принят на системном уровне 24.09.2026. Его карта
не доказывает полноту всех будущих features: новые находки фиксируются и
сверяются с системной картой. Каталог `capabilities/` всё ещё содержит
placeholder-домены и не заменяет принятые `system/` документы. Детальное
проектирование начинается с зависимых контрактов, без повышения их статуса
до runtime implementation.
