# Boundaries

**Статус: принятые системные Target-границы Gate 0 (24.09.2026).** Таблица фиксирует
инварианты из действующих контрактов и ADR, а не готовое runtime enforcement.
Неразрешённые продуктовые границы перечислены в
[capability-map.md](capability-map.md) и
[open-questions.md](../research/open-questions.md).

| Переход | Authority и проверка на границе | Нельзя выводить из |
| --- | --- | --- |
| Browser/API ingress → task | Подтверждённый principal, роль и допустимый scope; Lifecycle принимает goal/criteria/revision. Browser Cloudflare identity и bearer automation остаются разными lanes. | Session ID, текст запроса, событие или token stream сами по себе не дают полномочий |
| User clarification/change/decision → owning authority | Intent/Communication принимает сообщение и оформляет typed proposal; Lifecycle принимает goal/plan revision или result acceptance, Policy/Approval — конкретный grant, Work Initiation — trigger rule. | Один общий `approved` boolean не различает action, plan, trigger и incomplete result |
| Preapproved schedule/event → новая task | Work Initiation проверяет rule revision, source и firing; Lifecycle заново проверяет principal/policy/criteria/budget и принимает stable intent. | Приход внешнего события не является approval на tool effects или доказательством выполненной задачи ([ADR-0003/0004](../decisions/ADR-0004-work-initiation-ownership-boundary.md)) |
| Model/context/skill output → действие | Structured action proposal проходит effective policy/approval для конкретных action, target, principal, task/revision и времени; исполнитель фиксирует attempt и outcome. | Model text, skill instruction, tool output и retrieval не меняют policy ([ARCH_CANON](../ARCH_CANON.md)) |
| Action/effect → result/artifact → completion | Evidence связывается с criteria и result revision; Verification сообщает outcome, authorized acceptance и Lifecycle решают terminal transition. | Tool success, существование файла, verifier prose или budget stop не означают completion ([ADR-0002](../decisions/ADR-0002-terminal-outcome-and-partial-result.md)) |
| Accepted task state → communication/delivery | Communication публикует отдельные semantic progress и final; UI/chat/voice/notification доставляют проекцию, с replay и audience scope. | UI event buffer или отрендеренный assistant text не является task authority ([Communication contract](../USER_INTERACTION_COMMUNICATION_CONTRACT.md)) |
| Local/private data → model/provider/extension | Model route, entitlement, credential use, principal/policy, data scope и egress должны быть проверены раздельно; fallback наблюдаем и контролируем. | Конфиг URL, key в store, локальный engine readiness или название модели не доказывают доступ/способность ([ADR-0001](../decisions/ADR-0001-subscription-backed-openai-access.md), [ADR-0005](../decisions/ADR-0005-local-inference-operations-boundary.md)) |
| Owner provider key → member/automation request | По [ADR-0007](../decisions/ADR-0007-owner-provider-credential-delegation.md) default deny; только явное owner delegation для конкретного principal/route/purpose позволяет use на момент provider call. | Owner-only изменение Settings, наличие key или допуск к SlavikAI не являются разрешением на use |
| Media request → transcript/image result/audio artifact | Approval/Policy проверяет consent grant, principal и egress; adapter несёт source/retention references, результат входит в общий task/Communication/Artifact path ([ADR-0008](../decisions/ADR-0008-request-scoped-media-interaction.md)). | STT/TTS endpoint не означает непрерывную media session, а transcript не меняет task truth сам |
| Skill selection → external provider call | Skill даёт инструкции; Extension Governance разрешает provider/version/capability; Tool Execution и Policy проверяют каждый вызов ([ADR-0009](../decisions/ADR-0009-governed-external-tool-providers.md)). | Skill text и provider metadata не дают permissions, credential или права обхода gateway |
| Old task revision → superseded status/new lineage | Lifecycle фиксирует old/new IDs; Communication проецирует статус или correction без old-revision final ([ADR-0010](../decisions/ADR-0010-supersession-lineage-without-separate-final.md)). | Supersession не является `cancelled`/`aborted` или пятой final cause; прошлый доставленный final не переписывается |
| Canonical fact → event/UI/trace/eval | Authority факта записывает transition; consumers получают версию и provenance с собственными retention/replay правилами. | Replay сообщения или оценка качества не создают новый факт ([ADR-0006](../decisions/ADR-0006-resource-event-evaluation-authority.md)) |
| Durable state / crash → recovery decision | Owner fact/history и effect/approval references проверяются по версиям; Lifecycle решает task/run continuation, Tool Execution — unknown-effect reconciliation. | UI snapshot `running`, log, Context или старое approval не являются resume authority |
| Параллельные agents/clients → один state transition | Owning authority проверяет expected revision/owner epoch и принимает либо отклоняет переход; старый worker/client остаётся историческим источником evidence | Последнее сообщение или timestamp не разрешает гонку, не восстанавливает ownership и не переиспользует approval |
| Изменение contract/schema → новая версия | Владелец явно объявляет version, migration/read compatibility и rollback boundary; claims отражают только доказанный runtime status | Доступность старой projection или legacy path не делает новую семантику совместимой автоматически |
| Deployment/config → runtime capability | Operations валидирует configuration, secret injection, rollout и health; Model Access/Local Inference отдельно подтверждают route/engine readiness | Запущенный process или доступный port не доказывает policy, provider entitlement и безопасный rollback |

Для каждого durable fact нужен ровно один logical owner, даже если storage
физически общий ([ADR-0011](../decisions/ADR-0011-system-fact-ownership-and-cross-cutting-protocols.md)).
Восстановление проверяет revision, permission, оставшийся
budget и неизвестные external effects; UI snapshot `running` и короткий
idempotency cache не являются checkpoint/effect reconciliation. Current State
и ограничения доказательств приведены в
[state-authority-inventory.md](../research/state-authority-inventory.md) и
[system-flows-and-trust-boundaries.md](../research/system-flows-and-trust-boundaries.md).

**После Gate 0 открыто:** согласовать полный Lifecycle contract с ADR-0002 и
ADR-0010 и проверить точные owner handoffs в соответствующих разделах
[плана](../research/plan-coverage-and-owner-review.md).
Подробные transaction и retention contracts, включая механизм credential
delegation и extension governance, относятся к соответствующим разделам плана.
