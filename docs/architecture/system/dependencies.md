# Dependencies

**Статус: принятый граф крупных Target-зависимостей Gate 0 (24.09.2026).** Стрелка
`A → B` означает, что A нужен подтверждённый contract/решение B для своего
поведения; это не import или порядок запуска процессов.

| Зависимый owner/путь | Необходимая граница | Почему это blocking для design или runtime |
| --- | --- | --- |
| Work Initiation → Identity/Policy + Lifecycle | Rule principal/revision и task-creation acceptance ([ADR-0004](../decisions/ADR-0004-work-initiation-ownership-boundary.md)) | Firing не может создавать task или расширять authority по одному событию |
| Lifecycle → Identity/Policy + Resource Governance | Task/run/revision, effective grants, budget baseline ([ADR-0002](../decisions/ADR-0002-terminal-outcome-and-partial-result.md), [ADR-0006](../decisions/ADR-0006-resource-event-evaluation-authority.md)) | Resume/retry/terminal state требуют актуальных полномочий и bounded work |
| Coordination → Lifecycle + Identity/Policy | Task/run/ownership epoch, membership и audience | Shared event не принимает task transition и не раскрывает private context другому principal; snapshot contract ещё не интегрирован |
| Tool Execution → Policy/Approval + Resource Governance + Lifecycle | Concrete action/target/revision, available grant и effect identity | Permission и budget не выводятся из model output; unknown effect должен reconciled до retry |
| Verification/Acceptance → Lifecycle + Result/Artifact + criteria | Exact criteria/result revisions и evidence provenance | Старое или неполное evidence не завершает новую task revision |
| Communication → Lifecycle + Verification/Acceptance | Accepted transition и final readiness | Progress, UI token и worker text не могут публиковать ложный success |
| Context → Lifecycle + Memory + Policy + artifacts | Scoped/versioned projection | Cached model input не сохраняет старую authority после policy/revision change |
| Memory → Identity/Policy + source owners + consent | Principal/purpose, accepted source provenance, deletion/correction references | Retrieved или summarised content не становится принятой Memory автоматически; OQ-MEM-01 блокирует snapshot promotion rule |
| Model Access → Policy/Credentials + Local Inference Operations (для local route) | Entitlement, capability qualification, egress, engine readiness; owner key use default deny кроме явного delegation ([ADR-0007](../decisions/ADR-0007-owner-provider-credential-delegation.md)) | Доступность endpoint или key не равна qualified route/разрешённому principal; fallback через account/provider boundary явный |
| System Evaluation → разрешённые task/evidence/feedback records | Provenance, consent, retention | Quality loop не является per-task acceptance или автоматическим изменением policy |
| Media adapters → Identity/Policy + Model Access + Artifact/Communication | Request-scoped consent, egress и result provenance ([ADR-0008](../decisions/ADR-0008-request-scoped-media-interaction.md)) | Отдельный STT/TTS вызов не создаёт собственный task/final authority |
| External Extension Governance → Identity/Policy/Credentials + Tool Execution | Provider/version/capability eligibility, typed action, revocation ([ADR-0009](../decisions/ADR-0009-governed-external-tool-providers.md)) | Skill instruction или provider metadata не может разрешить действие |
| Skills/Reusable Procedures → Policy + External Extension Governance (при provider call) | Instruction provenance/version и доступность уже разрешённой capability | Выбор skill не меняет effective permissions, provider version или credentials |
| Revision lineage → Lifecycle + Communication + Verification/Artifacts | Old/new IDs, stale effect fencing, historical result refs ([ADR-0010](../decisions/ADR-0010-supersession-lineage-without-separate-final.md)) | Новая ревизия не создаёт competing old-revision final |
| Planning/Intent → Lifecycle + Identity/Policy | Accepted goal/criteria/plan revision и typed change/clarification | Стратегия или текст пользователя не меняют task truth без authorized transition |
| Artifact Records → Identity/Policy + Tool Execution + Verification | Owner, version/integrity, effect provenance, acceptance reference | Файл или UI download не является проверенным результатом |
| Communication delivery → semantic Communication + channel authorization | Stable artifact ID, audience, replay/correction cursor | Повторная доставка не создаёт новый final или task transition |
| Recovery protocol → owner-local canonical state + Lifecycle + Tool Execution + Policy | State/history, lease/epoch, pending effect, current grant | Restart не должен повторять неизвестный side effect и оживлять stale approval |
| Concurrent owners/clients → authoritative revision/epoch checks | Expected state version, ownership/cancellation epoch и durable transition history | Поздний result или replay не может перезаписать новый task/approval/artifact факт |
| Schema/contract evolution → owner + Source of Truth/claims | Versioned canonical state, explicit migration/read compatibility и rollback evidence | Legacy projection/adapter не объявляется новым Target; изменение claim требует executable proof |
| Deployment/Operations → Identity/Policy + Configuration + owner health | Auth lane, secret injection, process/engine readiness, rollback plan | Успешный boot не заменяет проверку capability/entitlement/effect safety |

Сквозная dependency для всех строк: **canonical persistence → recovery →
derived UI/Context/event projections**. Каждый owner записывает свои material
facts; общий transport не получает authority ([ADR-0006](../decisions/ADR-0006-resource-event-evaluation-authority.md)).
Системное правило об одном logical owner и сквозных протоколах —
[ADR-0011](../decisions/ADR-0011-system-fact-ownership-and-cross-cutting-protocols.md).

## Gates дальнейшей работы

1. Gate 0 принял крупные owners, потоки и зависимости после сверки с
   действующими и четырьмя неинтегрированными Target snapshots. Product choices
   OQ-CD-01/02/03 решены ADR-0008/0009/0010; расхождение Lifecycle snapshot
   с `aborted` и supersession mapping остаётся [blocker](../roadmap/blockers.md)
   для интеграции Lifecycle, но не меняет принятую системную границу.
2. До section 4 contract/runtime: согласовать terminal mapping, authoritative
   acceptance, action-effect identity и recovery. Нельзя переносить snapshot
   `6040640` как готовый contract.
3. До каждого runtime PR: выбрать один проверяемый срез, сверить claims registry
   и реальный execution path, выполнить проверки соответствующего раздела.

Порядок 20 разделов в [рабочем плане](../roadmap/master-roadmap.md) остаётся
зависимым от этих gates; этот граф не объявляет ни один раздел реализованным.
