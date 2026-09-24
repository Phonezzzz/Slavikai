# ADR-0011 — Один владелец факта, сквозные протоколы без второй authority

- **Статус:** принятое системное Target-решение для Gate 0
- **Дата:** 24.09.2026
- **Владелец решения:** архитектура SlavikAI
- **Затронутые области:** все capability domains, persistence, recovery,
  events, observability, identity/policy, verification, source-of-truth

## Контекст и проблема

Исходные 20 пунктов [рабочего плана](../roadmap/master-roadmap.md) смешивают
capabilities, сквозные контракты, user surfaces и audits. Текущий UIHub,
agent/tool paths и четыре неинтегрированных snapshot contract имеют разные
state lifetimes. Если принять названия разделов за сервисы либо дать event
stream, UI session или Context право владеть task/result фактом, появятся
конкурирующие источники истины. [Сверка владельцев](../research/plan-coverage-and-owner-review.md)
покрывает все 20 пунктов и дополнительные Work Initiation, media, extension,
local inference, evaluation, data consent/egress и operations границы.

## Решение

1. Каждый durable/domain fact и каждое решение имеет **одну logical
   authority** независимо от числа физических stores или процессов.
   [Capability map](../system/capability-map.md) и
   [boundaries](../system/boundaries.md) задают coarse-grain Target owners;
   подробные contracts будут приняты в соответствующих разделах.
2. Lifecycle владеет accepted task/goal/criteria revision, run transition и
   terminal cause. Planning, worker, model, verifier, UI и event consumer
   предлагают или сообщают; они не принимают task completion. ADR-0002/0010
   определяют terminal result/final и supersession. Work Initiation владеет
   preapproved trigger rule/firing, Lifecycle отдельно принимает новую task.
3. Tool Execution владеет operation/attempt и observed/unknown effect;
   Artifact владеет object identity/version/integrity/retention; Verification
   владеет evidence/outcome; Lifecycle принимает result/terminal transition;
   Communication владеет semantic publication/delivery identity. Переходы
   между ними несут principal, source и revision references.
4. Identity, effective Policy, scoped Approval и Credentials имеют раздельные
   security authorities. Пользовательский интерфейс может собрать typed
   решение, но его владелец зависит от subject: goal, action, trigger rule,
   incomplete result или credential delegation. Approval/Policy хранит
   consent/grant и его scope/revocation; Lifecycle хранит принятое изменение
   goal/result, Work Initiation — утверждённое правило запуска. Model/skill/
   provider output не является authorization.
5. Persistence, crash recovery, causal event publication, telemetry и
   documentation/claims consistency являются **сквозными протоколами**.
   Каждый fact owner определяет canonical state, version, material history,
   transaction boundary, retention, reconciliation и производные projections.
   Общий database, bus, UIHub, trace или Context не становится второй
   authority. Recovery controller координирует проверку owning states, но
   не переписывает их без validated transition.
6. Data consent/egress — policy-bound decision для конкретного
   principal/source/purpose/provider; его grant/revocation записывает
   Approval/Policy. Memory, Artifact, Media и Extension owners хранят и удаляют
   свои объекты. System Evaluation оценивает качество
   между задачами, но не принимает результат отдельной задачи.

## Основания

Такое разделение сохраняет единственное место принятия каждого перехода при
разных каналах доставки, agent roles и вариантах хранения. Оно соответствует
действующим Communication/Verification contracts, ADR-0002–0010 и
обнаруженным Current State разрывам; оно **не** утверждает, что текущий
runtime уже обеспечивает эти handoffs. Подробные identities, транзакции и
recovery механизмы остаются предметом разделов 1–20.

## Рассмотренные варианты

- Сделать один session/agent/UI store владельцем task, approval, artifact и
  communication: отклонено, потому что session/процесс и durable task имеют
  разные lifetime и security scopes.
- Считать каждый из 20 пунктов отдельным сервисом: отклонено; разделы 10–13,
  17, 19 и 20 содержат сквозные contracts и audits, а не по одному fact owner.
- Сделать общий event bus или trace единственным source of truth: отклонено;
  доставка и наблюдение не принимают domain transition.

## Последствия и открытые границы

- Четыре `system/` документа должны согласованно отображать owner,
  trust boundary, major data flow и directed dependency. Это решение не
  задаёт process topology, storage engine или wire schema.
- Snapshot `6040640` остаётся pending до согласования с ADR-0002/0010;
  `4ee7516`, `4b9a453` и `f883893` остаются pending до domain reconciliation.
  OQ-MEM-01 отдельно блокирует автоматическую Memory promotion.
- Target owner classification не повышает ни один runtime claim в
  `docs/runtime_contract_claims.json`. Для каждого implementation PR нужны
  current path audit, ТЗ, mechanism tests и claim evidence.
