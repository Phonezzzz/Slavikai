# Data Flows

**Статус: принятая схема крупных Target-потоков Gate 0 (24.09.2026).** Это
семантические пути и владельцы, не wire format или текущий executable flow.
[Current State audit](../research/system-flows-and-trust-boundaries.md)
отдельно отмечает, где runtime пока хранит только UI snapshot и process-local
worker.

## Основной task/result path

```text
authenticated user / bounded automation origin
  → principal + goal/criteria proposal
  → Lifecycle: accepted task/revision/run + policy/budget references
  → plan/context/model: bounded proposals
  → Policy/Approval + Resource Governance: permitted concrete action
  → Tool Execution: attempt → observed/unknown external effect
  → Result/Artifact Record: version + integrity + provenance
  → Verification: evidence against current criteria/result revision
  → authorized acceptance → Lifecycle terminal transition
  → Communication semantic progress/final → channel delivery
```

Каждая стрелка несёт principal, subject identity, revision, provenance и
чувствительность там, где это нужно для решения. Context является временной
model-visible проекцией; Memory хранит отдельно принятое знание и не получает
автоматическую запись из tool result или final. Точные Context/Memory/Artifact
contracts ещё требуют сверки; их snapshots не интегрированы.

## Другие системные пути

| Поток | Владелец перехода | Правило восстановления и предел |
| --- | --- | --- |
| Schedule/external event → новая task | Work Initiation записывает rule/firing и stable intent; Lifecycle принимает или отвергает task creation | После crash/retry intent deduplicate-ится; отозванное правило повторно проверяется. Exact protocol открыт OQ-CD-04. |
| Существующий run → pause/wait/resume | Lifecycle владеет checkpoint/transition; Policy и Resource Governance перепроверяются при resume | Background continuation отличается от нового trigger firing. Неизвестный external effect сначала reconciled, затем возможен retry. |
| Принятый факт → domain event → subscribers | Owner факта принимает transition; Event/Observability доставляют разрешённую проекцию | Material coordination history принадлежит Coordination; UI replay и trace имеют другую гарантию ([ADR-0006](../decisions/ADR-0006-resource-event-evaluation-authority.md)). |
| Context/Memory/Artifact → model route | Каждый источник отдаёт scoped/versioned projection; Model Access проверяет route, credential и egress | Owner key используется другим principal только по явному действующему delegation ([ADR-0007](../decisions/ADR-0007-owner-provider-credential-delegation.md)). Provider response — untrusted input; local engine readiness не заменяет route qualification. |
| Result evidence/feedback → quality evaluation | Verification/Lifecycle сохраняют task outcome; System Evaluation получает разрешённые версии/cases | Cross-task score или feedback не принимают task result и не меняют policy автоматически. |
| Отдельный media request → transcript/image observation/audio artifact | Approval/Policy решает consent/egress; Media adapter исполняет запрос в этом scope; Artifact/Context/Communication используют результат по своим contracts | ADR-0008 исключает отдельный continuous-session loop; media output не выдаёт approval и не завершает task. |
| Skill instruction → external tool proposal | Skill owner выбирает инструкцию; Extension Governance проверяет provider/version/capability; Policy и Tool Execution допускают конкретный call | ADR-0009 запрещает provider/skill output выдавать себе authority; credentials и egress остаются отдельными проверками. |
| Old revision → new accepted revision | Lifecycle фиксирует `superseded` и lineage, Communication показывает статус/correction | По ADR-0010 нет отдельного final старой ревизии только из-за замены; effects/artifacts сохраняют provenance. |
| User change/approval/acceptance → owning decision | Intent/Communication оформляет typed proposal; Lifecycle, Policy/Approval или Work Initiation принимает только решение своего subject | Action approval не принимает incomplete result; result acceptance не даёт tool permission. |
| Crash/restart → восстановление task/effect | Владельцы загружают canonical versions/history; Lifecycle reconciles task/run, Tool Execution reconciles unknown action, Policy перепроверяет grant | Старые UI/Context/event projections перестраиваются и не возвращают устаревшую authority. |
| Concurrent claim/client update → transition | Lifecycle/owning aggregate проверяет expected revision и fencing epoch, сохраняет material history; stale input отклоняется или явно reconciled | Arrival order UI/event stream не является authority для last-write-wins. |
| Contract evolution → persisted state/projection | Владелец версии мигрирует или читает старый state по явному правилу, затем перестраивает derived views | Миграция не повышает runtime claim и не теряет старые effects, approvals, evidence или deletion tombstones. |
| Deployment/configuration → runtime readiness | Operations проверяет config, secrets, health и rollback evidence; Local Inference Operations управляет model-host engine, Model Access квалифицирует route | Example service, running process или endpoint discovery не становятся доказательством готовности capability. |

**Открытые детали:** media consent/provenance, extension lifecycle и
credential delegation/egress mechanism. Их product scope решён в
ADR-0008/0009/0007, но точные contracts и проверка всех runtime paths остаются
для соответствующих разделов. Gate 0 принимает системный маршрут и authority
handoffs, не wire format или исчерпывающую карту реализации.
