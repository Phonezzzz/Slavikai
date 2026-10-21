# Generic OpenAI-compatible provider onboarding

Owner Settings accepts a display name, Base URL and API key. A probe calls the normalized
`/models` endpoint with the supplied key. HTTP 401 rejects the key. Saving creates an endpoint
instance ID without a model, stores its name and Base URL in `.run/ui_settings.json`, and stores
the key in the existing mode-0600 `config/api_keys.json` boundary. Saving does not select a model.
Remote endpoints require HTTPS; HTTP is accepted only for loopback runtimes.

The chat session model picker discovers models through `/models` when the owner opens an instance.
If the catalog succeeds, the selected ID must be in it. If the catalog is unavailable or
unsupported, the picker permits a manual opaque model ID; HTTP 401 never permits this fallback.
The selected ID belongs to session state. The existing OpenAI-compatible `LocalHttpBrain` sends
text/chat completions to the stored Base URL. Dynamic instances start fail-closed for native tools;
Auto reports `native_tools_required` and no runtime tool schema is sent to an unqualified endpoint.
Admission в Ask, Plan, Auto и Desktop читает typed snapshot `NativeToolCapability`.
Декларация адаптера имеет provenance `adapter_declaration`; evidence модели начинается
с `unknown` / `not_observed`. Адаптер без native tools сообщает `unsupported`;
custom route без действующего explicit proof — `unqualified`, хотя его адаптер реализует
протокол. Явная проверка даёт model evidence `supported` или `unsupported` с provenance
`inert_protocol_v1`; это не меняет policy authority.
Существующее разрешение встроенных адаптеров называется `legacy_adapter`, а не qualification
модели. Snapshot вычисляется из текущего Brain при каждом admission, не сохраняется и не
кэшируется между изменениями route/model/credentials; endpoint и секреты в него не входят.
Это только capability admission: ToolGateway, policy, approvals и credential authority
остаются отдельными проверками. Session-scoped qualification описана ниже.
Built-in `local` capability and the public
`/v1` proxy model `slavik` are unchanged.

Dynamic instances use owner credentials and are server-side default-deny for members at model
listing, discovery, selection and completion boundaries until explicit delegation is implemented.

TODO: Add first-class `Sign in with ChatGPT` subscription access through a separately designed
auth/entitlement route as required by ADR-0001. An API key provider instance is never treated
as subscription access. Owner credential delegation remains the separate ADR-0007 target gap.

## Transport OpenAI-compatible адаптера

Native SSE в `LocalHttpBrain` и обычный custom Chat используют один scoped HTTP механизм
из `llm/provider_http.py`. Отмена относится к конкретному запросу и прерывает DNS, connect,
TLS, headers и чтение body; consumer close закрывает response/session/connector/resolver.
SSE читается постепенно, UTF-8 и разделение строк сохраняются между сетевыми chunks;
большая строка с tool delta не ограничивается внутренним readline limit aiohttp.
Redirects запрещены в sync и stream requests; Authorization не пересылается на redirect target.
Remote proxy и owner CA bundle сохраняются, loopback не использует proxy environment.
Native stream не повторяет HTTP request после ошибки или отмены. Read/connect timeout
ограничивает сетевое ожидание, а не всю длительность последовательного SSE ответа.
Эта transport предпосылка не подтверждает модель и не включает custom native tools.

## Явная проверка custom model

Owner может нажать «Проверить tools» в текущем session model drawer. POST
`/ui/api/sessions/{session_id}/model/qualify` требует только `{ "confirm": true }`,
verified owner и owned session с выбранной saved custom model. Четыре коротких
synthetic requests проверяют sync/stream tool call и `role=tool` continuation.
Синтетическая `inert_fixture` не зарегистрирована в runtime tools и не выполняет
host action; история чата, Memory и private instructions в probe не передаются.

GET `.../model/capabilities` не запускает probe и не создаёт Agent. Evidence имеет
provenance `inert_protocol_v1`: `supported`, `unsupported` для проваленного protocol
либо `unknown` при transport/cancel/context error. Qualification хранится только
в transient scoped Agent, не в ModelConfig или persisted JSON/DB. Другая session
и restart начинают с unknown. Recheck сначала отзывает старое proof. Runtime
проверяет exact endpoint/model/effective headers/key binding перед native request;
изменение сохранённых config/credential file revisions отзывает proof, включая
rotation away/back. Model override не наследует его. Custom key не подхватывается
из `LOCAL_LLM_API_KEY`.

Probe держит существующий Agent lock и не начинается во время execution/approval;
отмена/disconnect/общий timeout отменяет worker, а changed context не получает proof.
Native protocol admission не даёт новых grants и не меняет ToolGateway/policy/approvals.
Members по-прежнему не получают owner custom route. Built-in `legacy_adapter`
поведение сохранено. Fixture verification доказывает этот flow, а не квалификацию
реального production endpoint. Persistent/multi-session qualification не реализована.
