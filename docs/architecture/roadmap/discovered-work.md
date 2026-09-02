# Discovered work

- **Назначение:** сохранять потенциальную работу, обнаруженную во время исследований и
  архитектурного анализа.
- **Содержит:** evidence-backed findings, их источник, область и необходимое следующее
  решение.
- **Не содержит:** автоматически committed roadmap, готовые implementation plans или
  находки без evidence.
- **Обновлять:** при обнаружении, уточнении, переносе в roadmap или отклонении work item.

## RF-2026-09-02 — Model access, protocol adapters and local inference

### Статус и evidence boundary

- **Статус:** research finding; не Target, не ADR и не implementation plan.
- **Capability Discovery:** остаётся incomplete. Finding не определяет окончательные
  capability domains и не делает приведённую decomposition принятой.
- **Источники:** [S-01](../research/sources.md#s-01) —
  [S-06](../research/sources.md#s-06).
- **Evidence boundary:** подтверждены только заявленные структуры и ограничения reference
  projects на проверенных ревизиях. Их README не доказывают, что соответствующий путь
  разрешён upstream Terms of Service, достаточно надёжен или подходит SlavikAI.

### Structural finding

Текущий workspace не имеет общего документа для cross-domain research findings:
`research/sources.md` предназначен для реестра источников, `research/open-questions.md` — для
незакрытых вопросов, а `system/*` — для уже подтверждённых системных представлений. Поэтому
этот evidence-backed finding временно находится в `discovered-work.md`. Нужен отдельный ответ
Capability Discovery на вопрос, требуется ли общий research-findings document. Структура
workspace в рамках этого finding не изменяется.

### Подтверждённые факты о reference projects

| Reference | Подтверждённый паттерн | Существенные детали |
| --- | --- | --- |
| FreeDeepseekAPI | Локальный proxy использует залогиненный DeepSeek Web account и сохранённую browser session для обращения к internal Web API. | Заявлены session reuse/recovery, sticky multi-account pool, reasoning/search modes, OpenAI Chat, Anthropic Messages и OpenAI Responses shims. Tool calls формируются через structured prompt и parsing нескольких text formats. Проект прямо назван experimental web-chat proxy. |
| FreeQwenApi | Browser-based proxy превращает Qwen Chat account/session в local API; это не официальный Qwen API и не local inference. | Заявлены chat, file upload, image generation, video tasks, multi-account rotation и prompt-adapted tool calls. Session tokens expire; anti-bot/captcha, account limits, session-bound files/tasks и temporary media URLs названы ограничениями. |
| FreeGLMKimiAPI | Local proxy работает с GLM/Z.ai, legacy chatglm.cn и Kimi web credentials; direct request может иметь browser fallback. | Заявлены multiple accounts, round-robin/cooldown, OpenAI/Anthropic-compatible endpoints и conversion протокольного text output в OpenAI `tool_calls`. Tool use прямо назван эмуляцией, а captcha/anti-bot и изменение Web API — ограничениями. |
| FreeKimiAPI | Local adapter направляет запросы в сторонний remote/keyless endpoint `cfbt.ccwu.cc`, который хостит Kimi model. | Это не Web Session, не official Moonshot/Kimi API и не local model. Proxy предоставляет Chat/Messages/Responses dialects, simulated tool continuation и reliability wrappers, но README исключает production guarantees и предупреждает не передавать sensitive data. |
| FreeNIMAPI | Local bridge принимает разные downstream API dialects и нормализует их в hosted NVIDIA Chat Completions upstream, затем преобразует ответ обратно. | Нормализуются system messages, tool definitions/calls/results и streaming lifecycle. Bridge заявляет Chat Completions, Messages и Responses endpoints, но не переводит images, documents, signed thinking blocks, hosted web search, MCP namespaces и computer-use items; downstream SSE частично синтезируется после полного upstream response. |
| local-inference-optimizer-skill | Agent skill обследует hardware/workload, выбирает inference engine, настраивает runtime/quantization, запускает server и проверяет smoke/benchmark. | Это не provider и не engine. Вход включает model repo/path или уже идентифицированную model; выбор самой подходящей model для hardware/workload не является основным scope reference skill. |

### Architectural observations

1. **Один `provider` скрывает несколько независимо меняющихся осей.** В reference projects
   одна model family доступна через web session, официальный/hosted API или сторонний endpoint;
   один upstream может выставляться через несколько downstream dialects; одинаковый внешний
   `tool_calls` может быть native, normalized или prompt-emulated. Это evidence в пользу
   исследования decomposition, но не доказательство конкретного class model для SlavikAI.
2. **Web-session access — отдельный lifecycle, а не ещё один API key field.** Browser login,
   cookies/tokens, expiration, captcha, session recovery, account cooldown, sticky mapping и
   upstream chat state создают собственные auth, state, concurrency и recovery concerns.
3. **Capability требует provenance.** Одинаковая outward capability может иметь разную
   семантику и надёжность: native structured upstream contract, adapter-normalized output или
   prompt-emulated behavior. Boolean `supports_tools` не выражает эту разницу.
4. **Protocol compatibility не равна semantic equivalence.** Messages, system instructions,
   tools, tool results, reasoning, streaming и errors могут быть lossy или вообще не иметь
   эквивалента в другом dialect. FreeNIMAPI прямо перечисляет непереводимые media/thinking/search
   items и синтезированный streaming lifecycle.
5. **External/keyless upstream образует другую trust boundary.** Он не должен сливаться с
   official API или web-session access только потому, что local adapter предоставляет тот же
   OpenAI-compatible endpoint.
6. **Local inference имеет control plane и data plane.** Model weights, исполняющий их
   inference runtime/engine и управляющий selection/configuration/lifecycle layer — разные
   сущности. Reference skill демонстрирует control instructions над engines, но не является
   самим runtime и не выполняет широкую automatic model selection.

### Architectural hypotheses — не принятые решения

#### H-01: web session как дополнительный Access Mode

SlavikAI может рассматривать `web_session` рядом с official API access, не заменяя его:

```text
model family
→ selected access mode
→ credentials/session lifecycle
→ upstream transport
→ canonical SlavikAI model interface
```

Потенциальная конфигурация может позволять пользователю отдельно выбрать `official_api` или
`web_session` для DeepSeek, Qwen, Kimi и GLM/Z.ai, если конкретная комбинация фактически
поддерживается и разрешена. Возможные преимущества: consumer/web access без отдельного API
billing этого режима, функции web-продукта, несколько accounts/sessions и дополнительный
fallback path. Это benefits hypothesis, а не обещание бесплатности, доступности или parity с
официальным API.

| Model family | Candidate official path | Candidate additional path |
| --- | --- | --- |
| DeepSeek | `official_api` | `web_session` |
| Qwen | `official_api` | `web_session` |
| Kimi | `official_api` | `web_session` |
| GLM/Z.ai | `official_api` | `web_session` |

Таблица выражает только candidate taxonomy. Наличие, contract и допустимость каждого
official/web route требуют отдельного provider-specific evidence; reference web proxies не
доказывают поддержку официального пути.

#### H-02: capability provenance

Capability Discovery должен проверить модель состояний минимум:

- `native` — upstream предоставляет явный structured contract;
- `adapter_provided` — adapter сохраняет upstream capability, но нормализует её представление;
- `emulated` — adapter создаёт совместимое поведение через prompt/parsing/orchestration;
- `unavailable` — route не может надёжно предоставить capability.

Рабочие определения требуют уточнения: граница `adapter_provided` и `emulated`, degraded
semantics, частичная поддержка и verification status пока не определены. Проверка нужна как
минимум для tools/function calling, reasoning, web search, files, vision, image generation и
video generation. Registry, если он появится, вероятно должен описывать точную комбинацию
model + access path + protocol/runtime, а не только provider family.

#### H-03: `external_experimental_upstream`

Сторонний remote/keyless endpoint может быть отдельным Access Mode или upstream class
`external_experimental_upstream`. Он допустим только как явно experimental path с отдельным
trust/reliability policy. Название и положение в taxonomy не приняты.

#### H-04: Protocol/API dialect ортогонален provider и access

`Protocol/API dialect` предположительно не должен быть жёстко связан с `Provider Family` или
`Access Mode`. Candidate flow:

```text
downstream dialect
→ normalization
→ canonical internal representation
→ upstream adapter/transport
→ reverse normalization and lifecycle mapping
```

Минимальный исследуемый набор dialects: OpenAI Chat Completions, OpenAI Responses,
Anthropic Messages и native provider protocol.

Canonical representation, если будет принята, должна исследовать не только text/messages,
но system instructions, tool definitions, calls/results, reasoning output, streaming events,
errors и model metadata. Нужны explicit unsupported/lossy outcomes вместо silent conversion.

#### H-05: provider decomposition

Capability Discovery должен проверить, обосновано ли разделение следующих понятий:

- Model Family;
- Access Mode;
- Credentials/Auth;
- Upstream Transport;
- Canonical Internal Model Interface;
- Protocol Adapter;
- Model Capabilities и их provenance/verification;
- Runtime / Execution Environment.

Reference examples показывают, что эти оси могут меняться независимо, но не определяют их
окончательные ownership, cardinality или runtime contracts в SlavikAI. В частности, пока не
решено, является ли local inference ещё одним Access Mode, отдельным runtime class или иной
композицией.

#### H-06: Local Inference Manager

В зрелой системе может существовать управляющий слой с candidate lifecycle:

```text
hardware discovery
→ resource assessment
→ workload classification
→ candidate model selection
→ inference engine selection
→ quantization/runtime configuration
→ launch
→ benchmark
→ verification
→ monitoring/reconfiguration
```

Исследуемые inputs: CPU, GPU/vendor, VRAM, RAM, storage, supported runtimes, model
architecture/size/format, quantization, context, latency, throughput, interactive chat versus
agent workload и текущая system load. Более широкий SlavikAI hypothesis выходит за scope
reference skill: manager может выбирать не только engine для заданной model, но и саму model
под hardware и workload. Capability Discovery должна решить, является ли это отдельным
domain; сейчас это не принято.

### Risks

#### Web-session access

- unofficial/internal Web API может измениться без предупреждения;
- session/token/cookies истекают и требуют observable relogin/recovery;
- anti-bot, captcha, proof-of-work и browser fingerprinting могут блокировать automation;
- действуют web rate limits, account restrictions и возможные cooldown/ban paths;
- ToS, допустимость automation и compatibility требуют отдельной проверки для каждого
  upstream и deployment context;
- cookies/tokens эквивалентны чувствительным credentials и требуют principal-scoped vault,
  redaction, rotation, revocation и запрета попадания в logs/config exports;
- capabilities, quotas, context/session semantics и media lifetime могут отличаться от
  official API и между accounts;
- multi-account rotation может нарушить conversation/session ownership без sticky mapping и
  явного recovery contract.

#### Capability and protocol adaptation

- emulated tool call может выглядеть native, но иметь другую надёжность, validation и retry
  semantics;
- parser false positive/negative способен превратить обычный text в action request или
  потерять настоящий call;
- reasoning может быть extracted/approximated, а не официально структурирован;
- system/tool schemas могут обрезаться из-за upstream context limits;
- dialect conversion может потерять ordering, identifiers, parallel calls, error taxonomy,
  streaming boundaries, usage/model metadata или unsupported media;
- synthesized stream повышает latency и не эквивалентен upstream token streaming;
- capability flags без provenance и qualification могут направить agent workload на route,
  который фактически не выполняет tools.

#### External experimental upstream

- низкая reliability и отсутствие control/SLA;
- endpoint может исчезнуть, сменить model или contract без notice;
- неизвестные logging, retention, jurisdiction и operator security practices;
- privacy/security implications делают отправку private code, credentials и personal data
  недопустимой без отдельной trust decision;
- retries/circuit breaker смягчают сбои, но не создают production guarantees и не исправляют
  compromised/untrusted upstream.

#### Local inference management

- неправильная model/engine/quantization комбинация может дать OOM, нестабильность, плохое
  качество или ложную экономию;
- static hardware discovery недостаточна без учёта текущей load и shared resources;
- benchmark на нерепрезентативном prompt/concurrency ведёт к неверному routing;
- launch/monitor/reconfiguration требуют lifecycle ownership, resource budgets, isolation,
  rollback и verification;
- runtime/model downloads и compiled kernels создают supply-chain, storage и compatibility
  risks;
- manager не должен объявлять route healthy только по process/HTTP health без behavioural
  проверки нужных capabilities.

### Cross-domain implications — candidate dependencies, не capability map

| Связанная область | Возможная зависимость / вопрос ownership |
| --- | --- |
| provider/model routing | Route может зависеть от model family, access mode, capability provenance, trust tier, health, cost/quota и workload; fallback не должен молча менять privacy/capability semantics. |
| authentication/credentials | Official keys, web tokens/cookies, browser profiles и local proxy keys имеют разные acquisition, refresh, revocation и exposure contracts. |
| sensitive vault | Web credentials и provider keys требуют principal scope, redaction, audited access, rotation и запрета сериализации в обычную configuration persistence. |
| model capability registry | Capability предположительно принадлежит конкретному route и должна включать source/provenance, limits, verification timestamp и degraded/unavailable state. |
| task/workload classification | Local/remote selection и model/runtime tuning требуют workload shape, latency/throughput/context/media/tool requirements; classifier не должен обходить основной agent/runtime contract. |
| context handling | Web chat state, canonical message history, session recovery, system/tool prompt size и protocol truncation должны иметь явного owner. |
| tool execution | Native versus emulated tool call влияет на validation, trust, approvals and verifier; adapter output остаётся untrusted model output. |
| media handling | Files, vision, image/video generation имеют route-specific upload ownership, task lifecycle, temporary URLs, retention и unsupported conversion paths. |
| hardware/system information | Local manager требует достоверного discovery CPU/GPU/VRAM/RAM/storage/drivers/runtimes и current load. |
| resource management | Engine/model launch требует budgets, placement, concurrency, memory pressure handling и coexistence с другими workloads. |
| runtime lifecycle | Browser auth sessions, local proxies и inference engines требуют start/stop/readiness/recovery/update ownership. |
| observability | Trace должен показывать фактический model/access/upstream/dialect, adapter transformations, capability provenance, retries/fallbacks и sanitized failure reason. |
| verification | Нужна qualification matrix точной комбинации model × access × protocol × capability × runtime, а не только `200 OK` или наличие model alias. |
| fallback/routing | Fallback между official API, web session, experimental upstream и local inference должен быть policy-controlled и observable; автоматический переход через trust boundary опасен. |
| configuration persistence | Persisted config должна хранить stable identities/policy и ссылки на vault entries, но не raw cookies/tokens; session state и durable preferences нужно различать. |

### Required next decision

Не implementation. В рамках Capability Discovery определить, где должен жить общий
cross-domain research narrative, и затем оценить hypotheses H-01—H-06 вместе с полной
capability map. До этого finding остаётся discovered research, а не provider architecture или
Local Inference architecture SlavikAI.

## RF-2026-09-02-WS — Alternative LLM access local E2E evidence

### Статус и scope

- **Статус:** confirmed local research evidence для точных tested routes; не Target, не ADR,
  не implementation plan и не provider-wide guarantee.
- **Capability Discovery:** остаётся incomplete. Evidence уточняет viability/risk Web-session
  access, но не принимает его как обязательный Access Mode или capability domain SlavikAI.
- **Reference evidence:** [S-01](../research/sources.md#s-01) —
  [S-04](../research/sources.md#s-04).
- **Local E2E evidence:** [S-07](../research/sources.md#s-07--local-deepseek-web-session-e2e),
  [S-08](../research/sources.md#s-08--local-qwen-web-session-partial-e2e),
  [S-09](../research/sources.md#s-09--local-glmzai-web-session-e2e),
  [S-10](../research/sources.md#s-10--local-kimi-web-session-e2e) и
  [S-11](../research/sources.md#s-11--local-freekimiapicfbt-external-upstream-test).
- **Evidence boundary:** tests выполнены владельцем SlavikAI на локальной машине; в этом
  workspace зафиксированы exact stimuli и observed results. Эта архитектурная сессия повторно
  проверила актуальные upstream README/HEAD, но не повторяла browser-auth inference с
  пользовательскими credentials.

### DeepSeek Web Session — PASS

Проверенная цепочка:

```text
DeepSeek Web account
→ browser authentication
→ saved token/cookies/session and Chrome profile
→ local FreeDeepseekAPI proxy
→ OpenAI-compatible endpoint
→ DeepSeek Web backend
→ actual model response
```

| Проверка | Observed result |
| --- | --- |
| Proxy base URL | `http://127.0.0.1:9655/v1` |
| Auth material | auth file создан; token и cookies `OK`; Chrome profile создан |
| Runtime/account | proxy status `ok`; account active |
| Discovery/state | models endpoint работает; session reuse доступен |
| Request | `POST /v1/chat/completions`, model `deepseek-chat`, prompt `Ответь ровно: WORKS` |
| Response | `WORKS` |
| Classification | `PASS` для exact chat inference route |

Подтверждён полный application-level E2E path от consumer Web account до фактического model
inference и ответа через local OpenAI-compatible endpoint. PASS ограничен указанной ревизией,
account/session, model, endpoint и временем проверки. Он не повышает status tools, reasoning,
search, streaming, files, vision или media capabilities.

### Qwen Web Session — partial PASS / inference BLOCKED

Проверенная цепочка остановилась после upstream session creation:

```text
Qwen Web account
→ browser authentication
→ saved token/cookies/session
→ local FreeQwenApi proxy
→ model discovery and upstream chat/session creation
→ browser fetch
→ Qwen anti-bot challenge
→ no model response
```

| Проверка | Observed result |
| --- | --- |
| Proxy base URL | `http://127.0.0.1:3264/api` |
| Auth material | browser login; token, cookies и session сохранены |
| Runtime/account | proxy запущен на `127.0.0.1:3264`; account status `OK` |
| Discovery/state | `/api/models` работает; model list получен; upstream chat/session создаётся |
| Request | `POST /api/chat/completions`, model `qwen3.7-max`, prompt `Ответь ровно: WORKS` |
| Response | HTTP `500`: `Qwen anti-bot challenge returned for browser fetch` |
| Classification | `PARTIAL PASS`: auth/discovery/session; `BLOCKED`: actual inference |

Этот результат не является Qwen inference PASS. Он подтверждает работоспособность browser
login, credential persistence, proxy, discovery и части upstream session-control path, но
фактический model response через проверенный route не получен. README S-02 заранее описывает
anti-bot/captcha как известный operational risk; local test переводит этот risk из общего
предупреждения в observed blocker для точной tested combination.

### Остальные проверенные access paths

| Access path | Точный test | Результат |
| --- | --- | --- |
| GLM/Z.ai Web Session через FreeGLMKimiAPI | Real browser-fallback smoke → `GLM_REAL_OK`; `/v1/chat/completions`, `GLM-5.1` → `WORKS` | **FULL E2E PASS** |
| Kimi Web Session через FreeGLMKimiAPI | Kimi Bearer/access token; `/v1/chat/completions`, `kimi-k2.5` → `WORKS` | **FULL E2E PASS** |
| FreeKimiAPI → CFBT external upstream | Local status OK; `@cf/moonshotai/kimi-k2.6` | **INFERENCE FAIL:** upstream `5035`, model unavailable on Workers Free plan |

CFBT path не является Kimi Web Session. Он подтверждён только как отдельный experimental
external upstream, до которого local proxy дошёл, но который не предоставил inference в
момент теста. Дополнительные требования из этого результата не выводятся.

### Confirmed architectural observations

1. **Model/Provider Family не эквивалентен Access Mode.** Kimi успешно прошёл Web-session
   route, тогда как отдельный CFBT route той же model family был недоступен. Health одного
   access path ничего не доказывает о другом.
2. **Web Session остаётся дополнительным path, а не заменой official API.** Локальные PASS
   подтверждают техническую viability exact Web routes, но не отменяют отдельные contracts,
   reliability и capabilities официального API.
3. **Web-session access технически жизнеспособен, но route-specific.** DeepSeek, GLM/Z.ai и
   Kimi tests подтверждают реальные end-to-end routes. Это не доказывает
   provider-independent reliability Web Session как класса.
4. **Authentication readiness не равна inference readiness.** Qwen имел valid saved session,
   account `OK`, model discovery и upstream chat/session creation, но не получил model output.
5. **Process health, account health и model discovery недостаточны для routing.** Route нельзя
   считать usable до реального inference probe; capability-dependent workloads требуют ещё
   отдельной qualification.
6. **Web-session readiness является time-sensitive.** Внутренний Web API, session и anti-bot
   state могут измениться после успешного теста, поэтому evidence нуждается в timestamp и
   requalification policy.
7. **Upstream challenge — не эквивалент внутренней ошибке SlavikAI.** Хотя reference proxy
   вернул generic HTTP `500`, canonical model layer потенциально должен сохранять observed
   cause вроде `upstream_challenge`/`interaction_required`, чтобы recovery и routing не
   трактовали его как обычный server fault. Точная error taxonomy остаётся hypothesis.

### Readiness distinction — без преждевременного design

Qwen evidence требует различать как минимум успешную authentication/session и подтверждённую
actual inference usability. Точная модель states/facets, error taxonomy и recovery contract
здесь не проектируются: это остаётся вопросом Capability Discovery и provider research.

### Risks refined by local evidence

- DeepSeek PASS остаётся snapshot, а не гарантия стабильности internal Web API или session.
- GLM/Z.ai и Kimi PASS также являются snapshots точных routes, а не гарантиями Web API.
- Qwen anti-bot/captcha — observed inference blocker, а не только hypothetical risk.
- Generic `account OK` способен создать ложный positive readiness signal.
- Автоматический retry без typed challenge handling может повторно атаковать blocked upstream,
  усилить rate limits или привести к account restriction.
- Fallback после challenge не должен молча переключать account, Access Mode или trust boundary.
- Credentials остаются sensitive даже когда inference blocked; failure не является поводом
  выводить token/cookies в diagnostics.

### Cross-domain implications

| Связанная область | Уточнение из E2E evidence |
| --- | --- |
| provider/model routing | Eligibility должна учитывать inference probe и freshness evidence, а не только configured/account/models status. |
| authentication/credentials | Successful browser login и credential persistence — отдельный lifecycle milestone, не финальная readiness. |
| sensitive vault | Auth files, tokens, cookies и Chrome profiles требуют protection/redaction независимо от route health. |
| observability/error model | Trace должен различать proxy/process failure, auth failure, discovery failure и upstream anti-bot challenge без раскрытия credentials. |
| verification | Минимальный qualification должен включать deterministic exact-response inference probe; capability claims требуют отдельных probes. |
| recovery/user interaction | Captcha/challenge может требовать явного browser/manual recovery path; silent bypass не предполагается. |
| fallback/routing | Переход к official API, другому account или provider должен быть observable и policy-controlled. |
| configuration persistence | Durable connection config, sensitive auth material, transient upstream session и latest qualification result имеют разные storage/lifecycle semantics. |

### Finding state

DeepSeek, GLM/Z.ai и Kimi Web-session exact chat routes: **confirmed local FULL E2E PASS**.
Qwen auth/discovery/session route: **confirmed local partial PASS**; Qwen actual inference:
**BLOCKED, not confirmed**. CFBT external path: proxy/status PASS, inference unavailable due
upstream plan restriction. Finding остаётся research evidence до Capability Discovery и
provider-specific security/ToS/reliability research.
