# Open Questions

- **Для чего:** фиксировать открытые вопросы, требующие ответа до или в процессе проектирования.
- **Сюда:** вопрос, почему он важен, варианты/статус, кто или что может дать ответ.
- **Не сюда:** уже отвеченные или решённые вопросы.
- **Обновлять:** при появлении или закрытии вопроса.

## OQ-CD-04 — Каков точный contract правил и firing автономных triggers?

- **Контекст:** ADR-0003 принял в Target запуск новой работы по расписанию
  или внешнему событию **только по заранее утверждённым пользователем правилам**.
  ADR-0004 закрепил Work Initiation как владельца правил и firing history;
  Lifecycle принимает или отклоняет task-creation intent.
  План покрывал background continuation, но не эту отдельную границу.
  Проверенные HTTP entry points начинают работу по UI/API запросу;
  `core/rule_engine.py` сопоставляет правило с уже полученным user message и
  меняет pre-generation instructions. Это не evidence наличия или отсутствия
  иных trigger механизмов во всём deployment.
- **Недостаёт:** точный contract регистрации/ревизии/отзыва правил, source и
  principal binding, trigger scope/expiry, дедупликация, overlap/catch-up,
  wakeup после restart, budget, task-creation acceptance и notification.
- **Владелец:** Capability Discovery, Lifecycle, Identity/Policy, Event и
  Background Architecture.
- **Закрытие:** принятый подробный contract rule/firing identity, revalidation,
  deduplicated task-creation intent, overlap/catch-up и crash reconciliation
  с Lifecycle; owner boundary уже принят в ADR-0004.

## OQ-MEM-01 — Совместима ли policy-based Memory promotion с запретом auto-updates?

- **Контекст:** неинтегрированный Memory snapshot `f883893`, §5.3–5.4,
  допускает automatic acceptance для узкого класса по explicit policy.
  Действующий `docs/agent/DevRules.md` §11 запрещает auto-updates Memory в
  runtime без явного approve, а `docs/SOURCE_OF_TRUTH.md` описывает текущий
  `confirm`/`edit_and_confirm` для каждого сохранения. Это разные по статусу
  источники; наличие snapshot не меняет current contract.
- **Недостаёт:** установить, считается ли предварительно утверждённая
  class/scope/source policy достаточным «явным approve», либо Target сохраняет
  per-record confirmation. До решения нельзя интегрировать snapshot как
  разрешение на автоматическую запись или реализовывать такой path.
- **Владелец:** Memory Architecture, Policy/Approval и владелец продукта при
  необходимости изменить обязательное правило.
- **Закрытие:** одно непротиворечивое нормативное правило для promotion,
  записанное в canonical docs/ADR; Current State и runtime claims остаются
  отдельными от Target.

## Task / Run Lifecycle Audit — communication boundary

**Статус research:** lifecycle-facing semantic findings зафиксированы; полный
`User Interaction / Communication Architecture Audit` остаётся отдельным будущим
scope. Отдельный lifecycle contract в текущем repository ещё не выделен.

### Установленные findings

- Authoritative lifecycle state принадлежит task/run workflow, а не communication
  layer. Направление зависимости: `authoritative lifecycle state -> communication
  decision`; текст assistant сам по себе не переводит task в `completed`, `failed`
  или `cancelled`.
- `progress_update` и `final_response` — разные semantic output classes. Token
  streaming, tool-call events, Computer activity events и background notifications
  могут быть transport/observability signals, но не являются progress update по
  умолчанию и не являются вторым conversational channel.
- Progress допускается при существенном изменении user-visible execution picture:
  significant intermediate result, переход в long-running/background phase,
  `blocked`/`waiting_user_input`/`waiting_approval`, recovery/replan, partial
  verification, существенный plan/ETA change или degraded execution mode. Нужна
  semantic filtering policy; каждый internal event не публикуется пользователю.
- Long-running определяется совокупностью semantic признаков, а не фиксированным
  числом секунд. Thresholds остаются implementation-defined.
- Two-phase означает два класса коммуникации, а не ровно два сообщения:
  `progress -> progress -> final` допустимо, как и один `final` для короткой задачи.
- Final response разрешён только после принятого authoritative terminal outcome:
  verification/acceptance должны быть завершены настолько, насколько требует
  workflow, и lifecycle должен быть `completed` либо соответствующим terminal outcome.
  Model `done`, artifact creation, конец tool loop или отсутствие tool call сами по
  себе недостаточны.
- Terminal communication различает successful, failed, cancelled и partial/incomplete
  outcomes. `waiting_*` и `blocked` — не completion.
- Final readiness, authoritative completion, response generation, delivery и delivery
  acknowledgement — разные состояния. Crash/reconnect/retry требуют idempotency и
  replay policy, которые остаются отдельным design scope.
- Workers/siblings не получают права напрямую публиковать user-facing progress:
  typed internal coordination state/events агрегируются orchestrator/communication
  decision layer с duplicate suppression и явным отражением unresolved conflicts.
- User-visible communications могут возвращаться в context как history, referenced
  artifact или bounded projection, но progress prose не становится authoritative state.
- User message во время run классифицируется как clarification, requirement change,
  cancel, status request или unrelated question; оно не означает автоматическую отмену.

### Current implementation gap

Текущий runtime публикует `response.ready`, `agent.respond.*`, token deltas,
`chat.*` stream/tool events и auto-progress через UIHub; workflow state отдельно
нормализуется как `running/completed/failed/cancelled`, а approval может остановить
execution в `waiting_approval`. Не найдены отдельные authoritative
`progress_update`/`final_response` artifacts, semantic publication gate, terminal
acceptance gate или recovery/idempotent delivery contract. Текущий streaming поэтому
не считается реализацией discovered requirement.

### Следующий architecture scope

Нужен отдельный `User Interaction / Communication Architecture Audit` для semantic
filtering, communication artifacts, delivery/replay/idempotency, interruption,
multi-agent aggregation и context projection. Он не должен создавать новый
conversational lane или превращать Computer activity events во второй чат.

## OQ-MA-01 — Где хранить общие cross-domain research findings?

- **Контекст:** текущие `research/` files покрывают только sources и open questions;
  `system/*` принимает уже подтверждённые системные representations. Research finding
  RF-2026-09-02 временно сохранён в `roadmap/discovered-work.md`.
- **Недостаёт:** результата Capability Discovery о долговременной структуре research corpus.
- **Владелец:** Capability Discovery.
- **Закрытие:** определён canonical document/section для evidence, observations, hypotheses и
  risks, без смешения с roadmap или accepted Target.

## OQ-MA-02 — Какова минимальная decomposition model-access route?

- **Контекст:** reference projects независимо варьируют Model Family, Access Mode,
  Credentials/Auth, Upstream Transport, Protocol Adapter, capabilities и runtime.
- **Недостаёт:** проверка Current State SlavikAI, ownership/cardinality этих сущностей и
  сценарии, где разделение реально предотвращает coupling.
- **Владелец:** Capability Discovery; будущий provider/model research.
- **Закрытие:** принятое decision record определяет identity, boundaries и lifecycle route
  без преждевременного превращения каждого понятия в отдельный domain.

## OQ-MA-03 — Является ли `web_session` поддерживаемым Access Mode?

- **Контекст:** DeepSeek, Qwen, GLM/Z.ai и Kimi reference proxies используют browser/web
  account credentials и internal Web APIs как дополнительный access path. Local E2E evidence
  подтверждает exact chat routes для DeepSeek, GLM/Z.ai и Kimi; Qwen прошёл
  auth/discovery/session stages, но был заблокирован anti-bot challenge до inference response.
- **Недостаёт:** provider-by-provider ToS/compatibility analysis, auth/recovery design,
  security threat model, operational reliability и проверка требуемых capabilities.
- **Владелец:** Capability Discovery совместно с security/auth/provider research.
- **Закрытие:** для каждой model family путь принят, отклонён или оставлен experimental с
  явными constraints; official API остаётся отдельным mode.

## OQ-MA-04 — Как пользователь выбирает Access Mode и как работает fallback?

- **Контекст:** manual selection полезен, но routing/fallback может менять cost, trust,
  privacy, latency и capability semantics.
- **Недостаёт:** preference model, health/eligibility rules, notification/consent semantics и
  запрет silent перехода через trust boundary.
- **Владелец:** provider/model routing research.
- **Закрытие:** определён observable policy-controlled selection/fallback contract.

## OQ-MA-05 — Как управлять web credentials и session lifecycle?

- **Контекст:** cookies, tokens, browser profiles, upstream chat sessions и multi-account
  mappings являются чувствительными principal-scoped state.
- **Недостаёт:** vault contract, acquisition flow, TTL/refresh/relogin, captcha/manual recovery,
  revocation, redaction, account cooldown, sticky session mapping и concurrency rules.
- **Владелец:** authentication/credentials, sensitive vault и session lifecycle research.
- **Закрытие:** threat model и lifecycle contract покрывают normal, expired, challenged,
  rate-limited, revoked и compromised states.

## OQ-MA-06 — Как описывать capability provenance и qualification?

- **Контекст:** outward capability может быть native, normalized adapter-provided,
  prompt-emulated или unavailable.
- **Недостаёт:** точные определения, partial/degraded states, limits, evidence timestamps и
  qualification matrix.
- **Владелец:** Capability Discovery; model capability registry research.
- **Закрытие:** taxonomy проверена минимум для tools/function calling, reasoning, web search,
  files, vision, image generation и video generation и не сводится к provider-wide boolean.

## OQ-MA-07 — Каков canonical internal model contract?

- **Контекст:** Chat Completions, Responses, Messages и native provider protocols имеют
  разные message, tool, reasoning, streaming and error semantics.
- **Недостаёт:** inventory полей и lifecycle events, loss model, unsupported behavior,
  canonical errors/model metadata и round-trip requirements.
- **Владелец:** protocol/model interface research.
- **Закрытие:** contract явно определяет lossless, lossy и unsupported conversions без
  silent fallback.

## OQ-MA-08 — Где проходит граница protocol adapter и capability emulation?

- **Контекст:** parsing prompt-generated tool calls создаёт OpenAI-compatible output, но не
  делает upstream function calling native; обычная normalization тоже выполняется adapter.
- **Недостаёт:** semantic boundary, validation requirements, trust level, observability и
  verifier obligations для каждого класса transformation.
- **Владелец:** protocol adapter и tool execution research.
- **Закрытие:** native normalization и behavioural emulation различимы в contracts, registry
  и traces.

## OQ-MA-09 — Нужен ли `external_experimental_upstream` как отдельный class?

- **Контекст:** FreeKimiAPI proxy использует сторонний keyless endpoint, который не является
  Web Session, official API или local inference.
- **Недостаёт:** trust taxonomy, privacy gate, acceptable data classes, enablement policy,
  health expectations и removal behavior.
- **Владелец:** security/trust и provider routing research.
- **Закрытие:** class принят или заменён другой taxonomy с эквивалентно явными constraints.

## OQ-MA-10 — Как моделировать stateful upstream sessions и multi-account ownership?

- **Контекст:** reference web proxies используют sticky accounts, provider chat IDs,
  cooldown и recovery; files/media tasks также могут принадлежать account/session.
- **Недостаёт:** mapping principal/session/agent → upstream account/chat/artifacts, durable
  versus transient state, crash recovery and isolation rules.
- **Владелец:** context/session, credentials и configuration persistence research.
- **Закрытие:** lifecycle не допускает cross-principal/session leakage и определяет recovery
  после expiration, restart и account switch.

## OQ-MA-12 — Может ли manager выбирать саму local model, а не только engine?

- **Контекст:** SlavikAI hypothesis шире reference skill, который в основном получает уже
  выбранный model repo/path.
- **Недостаёт:** model catalog metadata, quality/capability evidence, hardware fit, workload
  requirements, licensing, storage/download policy и benchmark methodology.
- **Владелец:** local inference/model routing research.
- **Закрытие:** определён reproducible selection/verification contract либо model selection
  явно исключён из manager scope.

## OQ-MA-13 — Какие lifecycle и safety boundaries нужны local inference?

- **Контекст:** launch/benchmark/monitor/reconfiguration потребляют host resources и могут
  устанавливать runtimes, скачивать weights и компилировать kernels.
- **Недостаёт:** authorization, resource budgets, isolation, supply-chain policy, rollback,
  current-load handling, readiness and behavioural verification.
- **Владелец:** execution, hardware/resource, security и observability research.
- **Закрытие:** lifecycle contract различает Model, Inference Runtime/Engine и Local Inference
  Manager и покрывает failure/recovery paths.

## OQ-MA-14 — Как квалифицировать точную route combination?

- **Контекст:** model alias или HTTP `200` не доказывает tools, media, reasoning либо agent
  suitability; FreeNIMAPI отдельно различает protocol tests, scripted client tests и live
  model evidence. Local DeepSeek/Qwen tests дополнительно показывают, что successful auth,
  model discovery и chat/session creation не доказывают inference readiness.
- **Недостаёт:** test matrix для model × access × protocol × capability × runtime, expiry
  rules для evidence и routing response на stale/failed qualification.
- **Владелец:** verification/observability и provider/model routing research.
- **Закрытие:** routing использует актуальный evidence-backed status, а не marketing/model
  catalog claims.

## OQ-MA-15 — Как представлять readiness Web-session route?

- **Контекст:** DeepSeek, GLM/Z.ai и Kimi прошли exact chat inference E2E, тогда как Qwen имел
  valid credentials, рабочий обычный Web UI, account `OK`, models и созданную session, но
  proxy inference был заблокирован upstream anti-bot challenge.
- **Недостаёт:** canonical distinction между process health, authenticated, discovery-ready,
  session-ready, inference-ready, capability-qualified, degraded, challenge-blocked и
  relogin-required; неизвестно, нужны ли это состояния, независимые facets или оба уровня.
- **Владелец:** provider/model routing, auth/session lifecycle, observability и verification
  research.
- **Закрытие:** readiness contract не объявляет route usable по одному account/model health,
  сохраняет typed upstream failure и задаёт recovery/requalification rules.

## OQ-MA-16 — Как должен быть устроен subscription-backed ChatGPT/OpenAI access?

- **Контекст:** semantic preservation audit 2026-09-10 не нашёл explicit target contract
  для доступа к моделям через официальный ChatGPT/OpenAI account и consumer subscription.
  Этот путь нельзя неявно приравнивать к обычному OpenAI API, оплачиваемому и
  авторизуемому как API service.
- **Статус:** обязательность capability принята как Target в
  [`ADR-0001`](../decisions/ADR-0001-subscription-backed-openai-access.md); open остаётся
  конкретный поддерживаемый и разрешённый implementation contract.
- **Недостаёт:** authoritative research по official account/subscription authentication,
  entitlement/quota, поддерживаемым и разрешённым access paths, отделению от API-billed
  services, quota exhaustion, explicit/non-silent fallback, trust/privacy/data-egress
  transitions, защите credentials/session tokens и product/Terms-of-Service boundaries.
- **Владелец:** Capability Discovery совместно с authentication/credentials,
  security/trust и provider/model access research.
- **Закрытие:** authoritative sources и follow-up design выбирают compliant route class и
  явно определяют identity, entitlement, lifecycle, quota/failure behavior, security boundary
  и fallback policy. Отсутствие подтверждённого route блокирует implementation, но не отменяет
  Target без explicit superseding decision.
