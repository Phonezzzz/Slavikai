# Дорожная карта SlavikAI

Здесь зафиксированы утверждённая последовательность работ, зависимости и статус находок.
[Обнаруженные работы](discovered-work.md) хранят исходные наблюдения,
[блокеры](blockers.md) — нерешённые противоречия, а
[реестр claims](../../runtime_contract_claims.json) определяет подтверждённый объём
runtime. При расхождении действует [иерархия источников](../../SOURCE_OF_TRUTH.md).

## Ближайшая утверждённая работа

**MA-01 · ACCEPTED_NEXT. Подписочный доступ к ChatGPT/GPT.**
Функциональный этап начинается по отдельному поручению после документационного PR. Цель — в обычном браузерном SlavikAI подключить поддерживаемый аккаунт,
выбрать действительно доступную ему GPT-модель, получить завершённый ответ и провести
полный цикл нативного вызова инструмента через действующие `Brain`, `AgentToolLoop`
и `ToolGateway`. Граница определена в
[ADR-0001](../decisions/ADR-0001-subscription-backed-openai-access.md).

- **MA-02 — допуск.** До живого использования подтвердить поддерживаемый способ
  доступа, entitlement конкретного аккаунта и пригодность для выбранного deployment.
  Сверить требования с [официальной документацией OpenAI](https://developers.openai.com/siwc/token-sharing-open-source).
  Подписка или наличие модели в каталоге сами по себе не подтверждают доступ.
- **MA-03 — выбор.** В пределах подписочного маршрута различать отсутствующий,
  отклонённый, непроверенный и подтверждённый доступ. UI и сервер должны согласованно
  обрабатывать недоступные route/model. Привязку аккаунта, principal и полномочий
  сохранять на всём пути, в том числе при продолжении после approval. Скрытое
  переключение аккаунта, провайдера или плательщика исключено.
- **Исходники.** Архив `chatgpt-plan-preserved.AUeGlF` в каталоге `slavikai-context`
  профиля владельца содержит девять файлов, diff и манифест. Восстановление проверено.
  База — [`e05702d`](https://github.com/Phonezzzz/Slavikai/commit/e05702d7002ae0cfd597b35d6ebe190bb96bb880).
  Части OAuth, каталога и Responses остались в черновике без интеграции в штатное
  приложение. Claim `model_access.openai_subscription` остаётся `target`.
- **Приёмка.** Автоматические проверки проверяют маршрут, выбор модели, tool round
  trip, ошибки доступа, отзыв и продолжение. Фактическую пригодность подтверждает
  владелец вручную в браузере на допущенной связке аккаунта, модели и deployment.

## Следующий продуктовый ориентир

Проверить в реальном использовании текущие Chat, Auto, инструменты и Settings.
Поиск специализации продукта отложен до опыта использования стабильной системы.
При отказе показывать известный исход и доступный безопасный следующий шаг.
Неопределённый эффект требует сверки до повторного действия. Записывать наблюдения
владельца с сохранением смысла. Инженерную диагностику вести отдельно. Позиция
попадает в очередь реализации после разбора наблюдения, проверки и согласования
значимого решения.

После MA-01 и первых пользовательских сценариев владелец пересматривает приоритеты
ниже. Кандидаты могут дополнять действующие Target-контракты только после принятия
соответствующих решений.

## Инвентарь работ

`ACCEPTED_NEXT` — утверждённая ближайшая работа. `REQUIRED_DEPENDENCY` — необходимая
граница указанного результата. `CANDIDATE/RESEARCH` — пробел, наблюдение или проект
решения для разбора. `DEFERRED` — работа отложена до подтверждённого сценария.
Порядок строк внутри этих групп не назначает срок реализации. Оценки исходного кода,
тестов и исследований сохраняют свой объём доказательств.

| ID | Статус | Основание и зависимость |
| --- | --- | --- |
| MA-01 | ACCEPTED_NEXT | Подписочный маршрут, штатный выбор и ручная приёмка. Зависит от MA-02 и MA-03 в указанном выше объёме. [ADR-0001](../decisions/ADR-0001-subscription-backed-openai-access.md) |
| MA-02 | REQUIRED_DEPENDENCY | Подтвердить entitlement аккаунта и применимость deployment до живого допуска MA-01. [OQ-MA-16](../research/open-questions.md#oq-ma-16--как-должен-быть-устроен-subscription-backed-chatgptopenai-access) |
| MA-03 | REQUIRED_DEPENDENCY | Подтвердить выбор доступной модели для MA-01. Общее выравнивание статусов всех providers остаётся предметом разбора. Generic catalog, qualification и смена модели уже имеют отдельные пути. [Onboarding](../GENERIC_PROVIDER_ONBOARDING.md), [выбор модели](../../../server/http/handlers/sessions.py) |
| SEC-01 | REQUIRED_DEPENDENCY | Действующая граница для путей, использующих credentials владельца от имени другого principal. Проверить область применимости к выбранному deployment MA-01. Общий аудит model, media и других egress-путей имеет отдельный scope. [ADR-0007](../decisions/ADR-0007-owner-provider-credential-delegation.md) |
| SEC-02 | CANDIDATE/RESEARCH | Уточнить полномочия на запуск Ollama до использования этого host-пути участниками группы. Текущая находка основана на чтении кода. [DW-2026-09-24-01](discovered-work.md#dw-2026-09-24-01--audit-authorization-of-local-ollama-host-launch) |
| EX-01 | CANDIDATE/RESEARCH | Проверить пользовательский Ask при failed tool, отказе и неизвестном эффекте. Различить неустранённую ошибку и успешное исправление. Сценарии из архива `lifecycle-preserved.o1FnGe` сверять с текущими typed observations. [Tool loop](../../../core/tool_loop.py), [TaskRun storage](../TASK_RUN_STORAGE_CONTRACT.md) |
| EV-01 | CANDIDATE/RESEARCH | Различать независимые вызовы и действия, зависящие от наблюдения. Общий критерий multi-call в #148 сам этого не устанавливает. Сравнить на разных сценариях и квалифицированных моделях после появления доступного маршрута. [PR #148](https://github.com/Phonezzzz/Slavikai/pull/148), [оценщик](../../../scripts/goal_behavior_eval.py) |
| LC-01 | CANDIDATE/RESEARCH | Один браузерный Auto-сценарий с независимой проверкой результата и scoped acceptance. Есть capture, submission и узкий ordinary Ask acceptance. Сценарий и критерий Auto выбрать по реальному использованию. [TaskRun storage](../TASK_RUN_STORAGE_CONTRACT.md), [Verification](../VERIFICATION_ARCHITECTURE_CONTRACT.md) |
| LC-02 | CANDIDATE/RESEARCH | Расширять bounded tool evidence и подтверждение эффекта по потребности конкретного потребителя. Текущий capture покрывает ограниченный `workspace_run`. [PR #122](https://github.com/Phonezzzz/Slavikai/pull/122), [Tool outcomes](../TOOL_EXECUTION_RESULT_CONTRACT.md) |
| LC-03 | CANDIDATE/RESEARCH | Выбрать вид безопасного продолжения после ожидания или сверки. Сохранять exact scope, перепроверять исход и полномочия. Текущее read-only recovery и volatile continuations имеют ограниченный объём. [Lifecycle](../TASK_RUN_LIFECYCLE_CONTRACT.md), [PR #147](https://github.com/Phonezzzz/Slavikai/pull/147) |
| LC-04 | CANDIDATE/RESEARCH | Семантический progress и доставка принятого финала после reconnect. Привязать первый срез к принятому terminal state, например после LC-01 при выборе этого сценария. [Communication](../USER_INTERACTION_COMMUNICATION_CONTRACT.md) |
| RG-01 | CANDIDATE/RESEARCH | Проверять следующий реальный разрыв в учёте и допуске. Уже действуют tool/time/token guards и ограниченные контуры файлового бюджета. Сопоставить остаточные сценарии #123/#125/#126. [ADR-0006](../decisions/ADR-0006-resource-event-evaluation-authority.md), [PR #129](https://github.com/Phonezzzz/Slavikai/pull/129), [PR #130](https://github.com/Phonezzzz/Slavikai/pull/130) |
| INT-01 | CANDIDATE/RESEARCH | Уточнить follow-up, ревизию цели, повторный запуск и обязательность verifier для нового сценария. Решение зависит от выбранных LC-01/03 и пользовательской проверки. [Lifecycle §26](../TASK_RUN_LIFECYCLE_CONTRACT.md#26-open-questions-и-implementation-defined-decisions) |
| CTX-01 | CANDIDATE/RESEARCH | Сверить eligibility и invalidation контекста с principal, политикой и ревизией. Исторический снимок `4b9a453`. [Границы владельцев](../research/plan-coverage-and-owner-review.md), [системные зависимости](../system/dependencies.md) |
| MEM-01 | CANDIDATE/RESEARCH | Снимок `f883893` допускает policy-based promotion, конфликтующий с текущим явным подтверждением. Расширение заблокировано до решения владельца по B-2026-09-24-02. [Блокеры](blockers.md), [DevRules §11](../../agent/DevRules.md#11-memory-companion-инварианты-policies-first-без-auto-changes) |
| ART-01 | CANDIDATE/RESEARCH | Связать preview и apply с одной ревизией при выборе соответствующего сценария. При устаревшей ревизии apply должен сообщить о конфликте до записи. [DW-PATCH-REVISION](discovered-work.md#dw-2026-09-10-patch-revision--version-bound-patch-preview-and-apply) |
| MEDIA-01 | CANDIDATE/RESEARCH | Для реального media-запроса проверить consent, egress, происхождение и хранение артефакта. Зависимости зависят от выбранного пути SEC-01 и LC-02. [ADR-0008](../decisions/ADR-0008-request-scoped-media-interaction.md), [аудит](../research/media-extension-operations-boundary-audit.md) |
| OPS-01 | DEFERRED | Local Inference Manager ждёт конкретной пользы и решения по полномочиям SEC-02. Локальные LLM на нынешнем железе — необязательные эксперименты. Каталог и загрузка моделей Hugging Face остаются идеей. [ADR-0005](../decisions/ADR-0005-local-inference-operations-boundary.md) |
| OPS-02 | CANDIDATE/RESEARCH | Отдельно проверить реальное развёртывание, health и восстановление для выбранных сценариев Web/Desktop. Ручная приёмка MA-01 входит в MA-01. [Эксплуатационные границы](../research/media-extension-operations-boundary-audit.md) |
| BG-01 | DEFERRED | Возвращаться при конкретной потребности продолжить начатый запуск без UI. Сначала определить scope, сверку эффекта и полномочия продолжения. [Lifecycle](../TASK_RUN_LIFECYCLE_CONTRACT.md) |
| TRG-01 | DEFERRED | Target запуска по утверждённым правилам сохраняется. Точный протокол firing и создание task ждут выбранного сценария и безопасного lifecycle. [ADR-0003](../decisions/ADR-0003-preapproved-autonomous-work-initiation.md), [ADR-0004](../decisions/ADR-0004-work-initiation-ownership-boundary.md) |
| MAC-01 | DEFERRED | Координация агентов ждёт подтверждённой задачи и ясных полномочий и бюджетов. Выбор локальных моделей или нескольких платных провайдеров из неё не следует. Параллельность проектировать по реальному сценарию. Исторические снимки `d0d842a` и `4ee7516`. [Системная карта](../system/capability-map.md) |
| EXT-01 | DEFERRED | Target внешних tool providers сохраняется. Первый провайдер ждёт конкретного сценария, principal grant, отзыва и action-time enforcement. [ADR-0009](../decisions/ADR-0009-governed-external-tool-providers.md) |
| EV-02 | DEFERRED | Расширять версии cases, scoring и evidence после проверки подхода EV-01 и потребности в сравнении разных задач. [ADR-0006](../decisions/ADR-0006-resource-event-evaluation-authority.md), [ранняя оценка](../../research/GOAL_BEHAVIOR_EVALUATION.md) |

## Подтверждённые срезы и исторические подходы

| ID | Статус | Подтверждённый объём или замена |
| --- | --- | --- |
| DONE-01 | DONE | Gate 0 принят на системном уровне. Карта, владельцы и зависимости зафиксированы. B-2026-09-24-01 закрыт на уровне согласованного Lifecycle contract. [Матрица Gate 0](../research/gate-0-closure-matrix.md), [PR #49](https://github.com/Phonezzzz/Slavikai/pull/49) |
| DONE-02 | DONE | Generic OpenAI-compatible onboarding и ограниченная qualification сохранены. [Контракт](../GENERIC_PROVIDER_ONBOARDING.md), [PR #52](https://github.com/Phonezzzz/Slavikai/pull/52) |
| DONE-03 | DONE | Отказы и отмена в проверенных путях, I0 и ограниченные qualified execution/capture. [Sandbox contract](../SANDBOX_PROCESS_FENCE.md), [PR #121](https://github.com/Phonezzzz/Slavikai/pull/121), [PR #122](https://github.com/Phonezzzz/Slavikai/pull/122) |
| DONE-04 | DONE | Scoped producer submission, узкий ordinary browser Ask acceptance, restart fencing и read-only recovery. Общая Auto acceptance и durable resume остаются в LC-01/03. [TaskRun storage](../TASK_RUN_STORAGE_CONTRACT.md), [PR #131](https://github.com/Phonezzzz/Slavikai/pull/131), [PR #147](https://github.com/Phonezzzz/Slavikai/pull/147) |
| DONE-05 | DONE | Узкий синтетический evaluator, включая эквивалентные пути. [PR #148](https://github.com/Phonezzzz/Slavikai/pull/148) |
| DONE-06 | DONE | Проверяемая browser identity, изоляция и Memory с явным подтверждением, атомарным хранением и browser management в пределах соответствующих claims. [Реестр claims](../../runtime_contract_claims.json) |
| OLD-01 | SUPERSEDED | Старый Lifecycle controller/store заменён действующим TaskRunStore в покрытом объёме. Сценарии сохранены в EX-01 и LC-03. [TaskRun storage](../TASK_RUN_STORAGE_CONTRACT.md) |
| OLD-02 | SUPERSEDED | Lifecycle snapshot `6040640` заменён согласованным контрактом и ADR-0002/0010/0012. Открытые вопросы runtime ведутся в LC-01–04. [Блокеры](blockers.md) |
| OLD-03 | SUPERSEDED | Старый RidgeGen `990c001` и ранний runtime stack Architecture заменены в проверенном объёме. Исследования сохранены. [PR #50](https://github.com/Phonezzzz/Slavikai/pull/50), [PR #51](https://github.com/Phonezzzz/Slavikai/pull/51), [RidgeGen research](../../research/RIDGEGEN_ARCHITECTURE_RESEARCH.md) |
| OLD-04 | SUPERSEDED | Старые пути файловых эффектов в объёме действующих guards. Статусы старых PR и отсутствующие регрессии сверять в RG-01. [PR #123](https://github.com/Phonezzzz/Slavikai/pull/123), [PR #125](https://github.com/Phonezzzz/Slavikai/pull/125), [PR #126](https://github.com/Phonezzzz/Slavikai/pull/126) |
| OLD-05 | OBSOLETE | Неофициальный web-session как исполнение ADR-0001, привязанный к Mercury patch UI и отдельный legacy Auto pipeline исключены из плана. Сохраняются исследования и независимые от провайдера требования ART-01. [Исторические находки](discovered-work.md), [runtime canon](../ARCH_CANON.md) |
