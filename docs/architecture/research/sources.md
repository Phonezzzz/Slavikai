# Research sources

- **Назначение:** быть реестром источников, использованных в архитектурной работе.
- **Содержит:** ссылки, автора или владельца, дату доступа, область применения и краткую
  оценку надёжности источника.
- **Не содержит:** выводы без источников, полный research narrative или принятые decisions.
- **Обновлять:** при добавлении, замене, переоценке или признании источника устаревшим.

## Model access, protocol adapters and local inference — 2026-09-02

Все источники ниже проверены по `README.md` текущей на момент доступа ветки `main` и
зафиксированы commit permalink, чтобы последующие изменения upstream не меняли evidence
задним числом. README является первичным источником для заявленного устройства самого
reference project, но не независимым доказательством его production reliability, соответствия
правилам upstream provider или применимости к SlavikAI.

### S-01

- **Источник:** ForgetMeAI,
  [`FreeDeepseekAPI/README.md`](https://github.com/ForgetMeAI/FreeDeepseekAPI/blob/31386b18cd485e5c6677bc0f46ac0ae3d92fbfd5/README.md).
- **Проверенная ревизия:** `31386b18cd485e5c6677bc0f46ac0ae3d92fbfd5`.
- **Дата доступа:** 2026-09-02.
- **Область:** DeepSeek Web browser authentication, сохранённые session credentials,
  internal Web API proxy, session recovery, multi-account pooling, OpenAI/Anthropic/Responses
  compatibility и prompt-based tool-call parsing.
- **Оценка:** authoritative для заявленного design проекта; экспериментальный unofficial
  web proxy, поэтому не подтверждает стабильность внутреннего DeepSeek Web API.

### S-02

- **Источник:** ForgetMeAI,
  [`FreeQwenApi/README.md`](https://github.com/ForgetMeAI/FreeQwenApi/blob/922fd66a76f4cec8134385f84715ec52c5535fe0/README.md).
- **Проверенная ревизия:** `922fd66a76f4cec8134385f84715ec52c5535fe0`.
- **Дата доступа:** 2026-09-02.
- **Область:** Qwen Chat browser session proxy, multi-account rotation, chat/files,
  image/video generation, session-bound artifacts и prompt-based tool-use adaptation.
- **Оценка:** authoritative для заявленного design ForgetMeAI fork; не является официальной
  документацией Alibaba/Qwen и прямо описывает зависимость от меняющегося Web API.

### S-03

- **Источник:** ForgetMeAI,
  [`FreeGLMKimiAPI/README.md`](https://github.com/ForgetMeAI/FreeGLMKimiAPI/blob/8aa1db707b2ffa2e44393f4e7bc40b71cefc35b9/README.md).
- **Проверенная ревизия:** `8aa1db707b2ffa2e44393f4e7bc40b71cefc35b9`.
- **Дата доступа:** 2026-09-02.
- **Область:** GLM/Z.ai, legacy chatglm.cn и Kimi web tokens, browser fallback,
  multi-account handling, OpenAI/Anthropic shims и emulated tool use через prompt protocol.
- **Оценка:** authoritative для заявленного adapter design; не доказывает официальную
  поддержку, долгосрочную совместимость или отсутствие upstream restrictions.

### S-04

- **Источник:** ForgetMeAI,
  [`FreeKimiAPI/README.md`](https://github.com/ForgetMeAI/FreeKimiAPI/blob/998f2fe62ea1a48c300604edec0c9ae14693df94/README.md).
- **Проверенная ревизия:** `998f2fe62ea1a48c300604edec0c9ae14693df94`.
- **Дата доступа:** 2026-09-02.
- **Область:** local proxy к стороннему remote/keyless Kimi endpoint, protocol shims,
  simulated tool loop, retry, jitter и circuit breaker.
- **Оценка:** authoritative для заявленной proxy topology; сам README исключает трактовку
  endpoint как official Moonshot/Kimi API, local model или production infrastructure.

### S-05

- **Источник:** ForgetMeAI,
  [`FreeNIMAPI/README.md`](https://github.com/ForgetMeAI/FreeNIMAPI/blob/bbe988af7bb6fb6d3a97ff4e4d374c4f47c28068/README.md).
- **Проверенная ревизия:** `bbe988af7bb6fb6d3a97ff4e4d374c4f47c28068`.
- **Дата доступа:** 2026-09-02.
- **Область:** hosted NVIDIA Chat Completions upstream, normalization между OpenAI Chat
  Completions, OpenAI Responses и Anthropic Messages, tools/tool results, streaming lifecycle,
  error handling и capability limitations.
- **Оценка:** authoritative для заявленного bridge design и опубликованной test matrix;
  README отдельно различает scripted, historical live и pending evidence, поэтому не даёт
  общей гарантии для любой модели, клиента или protocol combination.

### S-06

- **Источник:** ForgetMeAI,
  [`local-inference-optimizer-skill/README.md`](https://github.com/ForgetMeAI/local-inference-optimizer-skill/blob/06bc24b92dc715996d3762b520e93184cfdae93a/README.md).
- **Проверенная ревизия:** `06bc24b92dc715996d3762b520e93184cfdae93a`.
- **Дата доступа:** 2026-09-02.
- **Область:** agent-guided hardware inspection, inference engine selection, runtime and
  quantization tuning, launch, smoke testing and benchmarking for an identified model/workload.
- **Оценка:** authoritative для scope самого skill. Это instructions/control layer, а не
  provider, model или inference engine; README предполагает, что model repo/path уже задан.

### S-07 — Local DeepSeek Web-session E2E

- **Источник:** transcript реального локального smoke test, предоставленный владельцем
  SlavikAI 2026-09-02.
- **Проверенная связка:** FreeDeepseekAPI
  `31386b18cd485e5c6677bc0f46ac0ae3d92fbfd5`, DeepSeek Web account,
  `http://127.0.0.1:9655/v1`, model `deepseek-chat`.
- **Preconditions:** создан auth file; `token: OK`; `cookies: OK`; создан Chrome profile;
  proxy status `ok`; account active; model discovery и session reuse доступны.
- **Stimulus:** `POST /v1/chat/completions` с prompt `Ответь ровно: WORKS`.
- **Observed result:** API вернул `WORKS`.
- **Оценка:** direct local application-level E2E evidence для точной цепочки
  `web account → web session → local proxy → OpenAI-compatible request → actual inference →
  response` в момент теста. Не доказывает долговременную reliability, другие models,
  streaming, tools или media capabilities. Raw machine logs не включены в workspace;
  архитектурная сессия фиксирует предоставленный operator transcript и не выдает его за
  независимо повторённый здесь test.

### S-08 — Local Qwen Web-session partial E2E

- **Источник:** transcript реального локального smoke test, предоставленный владельцем
  SlavikAI 2026-09-02.
- **Проверенная связка:** FreeQwenApi
  `922fd66a76f4cec8134385f84715ec52c5535fe0`, Qwen Web account,
  `http://127.0.0.1:3264/api`, model `qwen3.7-max`.
- **Подтверждено:** browser login; извлечение и сохранение token/cookies/session; account
  status `OK`; запуск proxy; `/api/models` и model discovery; создание upstream chat/session.
- **Stimulus:** `POST /api/chat/completions` с prompt `Ответь ровно: WORKS`.
- **Observed result:** HTTP `500` с payload
  `{"error":{"message":"Qwen anti-bot challenge returned for browser fetch","type":"server_error"}}`.
- **Оценка:** partial E2E PASS для auth/discovery/session-control частей route и BLOCKED для
  actual model inference. Это не inference PASS. Ошибка согласуется с README S-02, который
  прямо предупреждает об anti-bot/captcha challenge. Raw machine logs не включены в
  workspace; архитектурная сессия фиксирует предоставленный operator transcript.

### S-09 — Local GLM/Z.ai Web-session E2E

- **Источник:** результаты реального локального теста, предоставленные владельцем SlavikAI
  2026-09-02.
- **Проверенная связка:** FreeGLMKimiAPI
  `8aa1db707b2ffa2e44393f4e7bc40b71cefc35b9`, настоящий Z.ai account, browser fallback,
  `/v1/chat/completions`, model `GLM-5.1`.
- **Observed result:** real smoke test вернул `GLM_REAL_OK`; последующий
  OpenAI-compatible request вернул `WORKS`.
- **Оценка:** FULL E2E PASS для exact GLM/Z.ai Web-session chat route в момент теста; не
  является общей гарантией остальных capabilities или будущей доступности internal Web API.

### S-10 — Local Kimi Web-session E2E

- **Источник:** результаты реального локального теста, предоставленные владельцем SlavikAI
  2026-09-02.
- **Проверенная связка:** FreeGLMKimiAPI
  `8aa1db707b2ffa2e44393f4e7bc40b71cefc35b9`, Kimi account и Bearer/access token,
  `/v1/chat/completions`, model `kimi-k2.5`.
- **Observed result:** OpenAI-compatible request вернул `WORKS`.
- **Оценка:** FULL E2E PASS для exact Kimi Web-session chat route в момент теста; не
  распространяется на другие models или capabilities.

### S-11 — Local FreeKimiAPI/CFBT external-upstream test

- **Источник:** результаты реального локального теста, предоставленные владельцем SlavikAI
  2026-09-02.
- **Проверенная связка:** FreeKimiAPI
  `998f2fe62ea1a48c300604edec0c9ae14693df94`, local proxy, third-party keyless
  `cfbt.ccwu.cc`, model `@cf/moonshotai/kimi-k2.6`.
- **Observed result:** local proxy/status работал; inference вернул
  `5035: Model @cf/moonshotai/kimi-k2.6 is not available on the Workers Free plan`.
- **Оценка:** experimental external access path достиг upstream, но inference был недоступен
  из-за upstream plan restriction. Это не Kimi Web Session и не основание для отдельного
  архитектурного требования.
