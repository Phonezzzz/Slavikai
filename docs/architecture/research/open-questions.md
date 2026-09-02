# Open architecture questions

- **Назначение:** фиксировать вопросы, которые требуют исследования или решения.
- **Содержит:** конкретный вопрос, контекст, недостающие evidence, владельца и условие
  закрытия.
- **Не содержит:** риторические вопросы, скрытые решения, implementation backlog или уже
  закрытые пункты.
- **Обновлять:** при появлении evidence, изменении формулировки, назначении владельца или
  закрытии вопроса.

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

## OQ-MA-11 — Является ли Local Inference Manager отдельным capability domain?

- **Контекст:** reference skill демонстрирует control layer над hardware discovery, engine
  selection, tuning, launch и verification, но не является model или inference engine.
- **Недостаёт:** полная capability map, ownership относительно routing/resource/runtime
  lifecycle и граница между recommendation, provisioning и continuous management.
- **Владелец:** Capability Discovery.
- **Закрытие:** domain boundary принята либо responsibilities распределены без потери единого
  lifecycle owner.

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
