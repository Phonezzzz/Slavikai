# Source of Truth — contracts and status

Этот документ задаёт единую иерархию источников истины. Он не подменяет runtime-код
пересказом: текущее поведение подтверждается executable checks, а незавершённые решения
помечаются `partial` или `target` в `docs/runtime_contract_claims.json`.

## Иерархия

При расхождении действует следующий порядок:

1. Security/policy enforcement на execution boundary и проверяющие его tests.
2. Machine-readable schemas, configs, API contracts, Make targets и
   `docs/runtime_contract_claims.json`.
3. `docs/architecture/ARCH_CANON.md` — обязательные runtime-инварианты и target design.
4. `docs/architecture/Architecture.md` — наблюдаемое текущее устройство и legacy paths.
5. Пользовательские и contributor guides, которые ссылаются на владельца контракта, а не
   вводят отдельную семантику.

Код не становится правильным только потому, что он существует. Если implementation
расходится с обязательным security/runtime contract, claim получает `partial`; если пути ещё
нет — `target`. Исправлять такое расхождение нужно отдельным coherent PR с mechanism tests.

## Статусы claims

- `implemented` — путь существует и подтверждён указанной executable проверкой.
- `partial` — часть механизма существует, но обязательный invariant ещё нарушается.
- `target` — принятое направление, которое ещё не является shipped behavior.
- `legacy` — reachable текущее поведение, которое нельзя расширять как целевой путь.

Реестр claims является machine-readable индексом статусов, владельцев и доказательств:
`docs/runtime_contract_claims.json`. Изменение статуса обязательно меняет доказательства в
том же PR. Claim без owner/version/source/verification не допускается.

## Принятый deployment contract

Целевая эксплуатация — небольшая закрытая группа за Cloudflare Access:

- Cloudflare Access email OTP остаётся единственным browser login layer;
- origin обязан проверить JWT signature через team JWKS, а также `iss`, `aud`, `exp`, `nbf`
  и обязательный email claim до доверия email;
- `principal_id` выводится только из подтверждённого нормализованного email;
- один настроенный owner имеет административные полномочия, остальные допущенные Access
  пользователи являются members;
- browser identity и Bearer auth для `/v1` automation — разные auth lanes.

Cloudflare browser identity является production default; token browser mode отклоняется
canonical server boot. Для локальной разработки доступен только явный unauth-local bypass.
Runtime Agent создаётся на scope `(principal_id, session_id)`; owner использует существующие
`memory/*.db`, а members —
изолированные hashed principal directories для Memory, canonical atoms и vectors. Persistent
Desktop approvals также имеют explicit principal subject, при этом один общий host coordinator
по-прежнему допускает только один Desktop run одновременно.

## Binding runtime decisions

Auto v1 применяет отдельный native tool-call cap перед каждым следующим call,
включая batch и approved continuation. Completed ToolResult (success/failure/denial)
расходует слот; ожидание approval не расходует его, approved call учитывается один раз.
Cap и выполненный prefix захвачены continuation, resume не получает свежую квоту.
N+1 не передаётся Gateway; ровно N не является автоматическим budget failure.
Лимит итераций остаётся отдельным guard, обычный verifier сохраняется. Это cap
обработанных native calls, не успешных effects, файлов, токенов или hard wall deadline.
Uncapped Ask/Desktop/Act сохраняют прежний контракт. При dispatch/observation fault
Auto сохраняет доступный typed prefix и останавливается без автоматического replay;
последний call с неизвестным результатом не объявляется выполненным или отменённым.

Auto v1 также проверяет исходный monotonic срок run перед model request, после
возврата модели, перед каждым native call/approved remainder и перед verifier.
На `now >= start + max_runtime_seconds` следующая операция не начинается;
ожидание approval входит в это время. Continuation сохраняет исходный срок,
изменение env или новый loop не выдаёт новый запас времени. Cancellation проверяется
раньше expiry на loop boundaries и перед verifier; доступный выполненный prefix
сохраняется при budget stop. Это admission fence: уже выполняющиеся provider/tool/
verifier не прерываются по часам, и успешный verifier может завершиться после срока.
Hard wall deadline, files/tokens/retry accounting и durable recovery этим не реализованы.

`LLMResult.usage` содержит `LLMUsage` только для трёх явно сообщённых неотрицательных
integer counters (`prompt_tokens`, `completion_tokens`, `total_tokens`); bool, float,
string, missing/null/negative metadata дают `None`, сохраняя валидные text/tool calls.
Явный zero допустим. Существующие xAI input/output aliases и canonical-field precedence
сохраняются. Shared stream decoder не создаёт синтетический zero Usage для unknown;
полные updates остаются отдельными событиями. Это validity reported metadata, не
проверка provider billing, aggregation, token quota или вывод отсутствующего total.

Синхронный Auto-loop отдельно наблюдает внешние вызовы `Brain.generate`:
immutable summary содержит started/reported/unknown calls и сумму complete reported
prompt/completion/total без вычисления отсутствующих полей. Общий `total_tokens`
известен только при хотя бы одном вызове и отсутствии unknown. Полученный ответ
учитывается до post-model cancel/expiry; вызов без результата остаётся unknown.
Approval continuation сохраняет summary, synthetic resume не добавляет model call.
Auto-state публикует только числовую `execution_metrics.tool_loop_model_usage`
projection на доступных normal/paused/cancelled/budget/error exits; normalizer
отбрасывает неверные counters и посторонние metadata. Эта projection может проходить
через существующий workflow state transport/storage, но не является durable accounting
или recovery authority. Внутренние HTTP retries, другие model consumers, streaming и
verifier не покрыты: это не весь token расход run, actual billing либо spending cap.
Сам учёт не останавливает run; token admission задаётся отдельным контрактом ниже.
Non-Auto sync callers по умолчанию не включают учёт, streaming не агрегируется.

Синхронный Auto захватывает существующий `max_llm_tokens` и проверяет его только
перед следующим genuine `Brain.generate`, после cancellation/expiry guards. Первый
request разрешён; при unknown usage следующая generation блокируется с причиной
невозможности проверить limit, без утверждения превышения. При complete reported
sum `>= limit` следующая generation блокируется как достигшая threshold. Continuation
сохраняет исходный limit и summary: новый loop/env не выдаёт свежую квоту, uncapped
checkpoint не наследует policy нового loop. Synthetic approved remainder не model
request: уже полученные tools проходят прежние Gateway/approval/call-cap/expiry.
Final response без tools может пройти verifier даже при unknown или threshold.
Unknown и exhausted различаются в Auto error_code/reason; автоматического replay нет.
Это admission fence следующего model request, не hard spending cap: активный request
может превысить остаток, HTTP retries/другие model consumers/verifier не учитываются.
Providers без complete reported usage не могут продолжить многошаговый Auto после
такого ответа. Default120000 и existing loader не меняются; non-Auto/streaming не
получают новый token policy. Files/retries/hard interruption/recovery остаются отдельно.

- Ask: current `implemented` invariant — никаких Memory/tool writes и hidden vector runtime
  initialization; явный запрос на запоминание создаёт только preview/decision и не сохраняет
  данные.
- Plan: current contract — read-only tools только через `ToolGateway`; write/exec запрещены.
- Auto: Auto v1 через `AgentToolLoop -> ToolGateway -> verifier` является current;
  `Ask -> Plan -> Act` FSM является target.
- Public OpenAI-compatible API: current `implemented` contract — `slavik` является
  единственным публичным proxy model id; `/v1/chat/completions` отклоняет любой другой ID до
  Agent/session resolution.
- Memory: current `implemented` contract — запись возможна только после отдельного явного
  `confirm` или `edit_and_confirm`; `reject` не пишет данные. Устаревший
  `auto_save_dialogue` удалён из config/runtime/UI и отклоняется settings API. Confirm endpoint
  сейчас существует только в browser UI lane; `/v1` может вернуть preview, но не применяет его.
  `/end-session` сначала предлагает Да/Нет/Отмена; Да только подготавливает summary для
  отдельного Memory confirm. Закрытие не удаляет историю чата. Точный flow описан в
  `docs/agent/COMMAND_LANE_POLICY.md`.
- Engineering skills: current `implemented` contract — instructions и supporting dependencies
  внедряются per run в Auto v1; MWV report/UI наблюдают выбранный skill и terminal status. Brain
  не мутируется, tool/approval/sandbox/verifier полномочия не расширяются.

## Личные настройки браузера

`personalization`, `appearance` и `composer` принадлежат verified browser principal.
Settings читает их только из `.run/preferences.db`; старые значения общего UI JSON
не импортируются и не используются. Новый principal получает встроенные defaults.
Общие providers/keys, model default, Memory installation config и grants сохраняют
существующих владельцев. Личные настройки не являются Memory и не расширяют полномочия.

GET `/ui/api/settings` возвращает личный snapshot текущего principal и
`preferences_etag` и opaque `preferences_subject` для сброса UI draft при смене identity.
Личный POST требует `If-Match` с этим ETag; другой subject или
устаревшая revision получает 409 без записи, отсутствие precondition — 428.
Installation и личные поля нельзя изменять одним POST. Installation update остаётся
owner-only. `{ "reset_preferences": true }` с текущим ETag возвращает только личные
поля к defaults, увеличивая revision. Неизвестные поля и неверные типы отклоняются до записи.
UI отправляет только изменённые группы; ошибка второй операции не отменяет уже
сохранённую installation operation и сообщается как частичный результат.
После отправленного installation POST ошибка HTTP/transport либо нечитаемый ответ
не доказывает отсутствие записей: UI предупреждает о возможном частичном сохранении
и предлагает заново проверить сохранённые значения перед явным retry. Draft сохраняется,
автоматического POST retry/rollback нет. Local pre-dispatch ошибки не получают такое
предупреждение; confirmed installation + personal conflict сохраняет отдельное сообщение.
Installation success требует читаемого Settings response, raw malformed body не выводится.
Это truthful unknown-outcome disclosure, не atomic multi-file transaction, exact applied-fields
report или изменение существующего порядка server writes.


Tone и private instructions применяются к следующему новому browser chat request.
Snapshot verified principal читается под существующим agent execution lock. Отдельное
user-role сообщение с preferences входит в Ask/chat (sync/stream), Auto и Desktop перед
исходным user goal; system prompts и policy/approval authority не изменяются.
Balanced + пустые инструкции сохраняют прежний request. Preferences не добавляются
в историю чата, Memory или context logs. Structured Plan/MWV без model request
не использует их для изменения packet/operations. Существующий native tool continuation
сохраняет исходные messages даже после изменения Settings. Bearer `/v1` использует defaults
и не импортирует browser preferences.

## Browser hints

Auth status отдаёт opaque `browser_subject` для подтверждённого browser principal,
с no-store/Vary. AuthGate передаёт этот subject через immutable React context до
App bootstrap. Last session/model, explorer visibility и recent roots читаются и
пишутся только в subject namespace; unscoped legacy values не читаются и не мигрируются.
Без валидного subject persistence выключена, hints получают defaults. Ошибка browser
storage не мешает работе App и не создаёт unscoped fallback.

После login subject читается из нового status, не выводится из token в браузере.
Focus/visibility проверка с epoch/abort отбрасывает старые ответы; подтверждённая смена
subject remounts private App. Прежний async callback сохраняет захваченный subject.
Transient background status failure сохраняет ранее подтверждённый App/draft и сообщает
об ошибке; initial failure блокирует bootstrap, HTTP401/403 скрывает private App.
Это не мгновенный identity switch detector и не замена server-side authorization.

Browser hints не grants и не persisted preferences. Existing pending-send protocol
identity/hash остаётся отдельным session-bound lifecycle state и этим срезом не меняется.
Unimported legacy Sidebar.tsx не является production folders UI; server-owned folders
в HistorySidebar сохраняют существующий principal scope.

## Embeddings defaults и загрузка моделей

Неявный embeddings provider всегда `local`: отсутствие файла, пустой `memory`,
пустой `embeddings` и отсутствующий/неизвестный provider дают один default.
Явно сохранённый `openai` остаётся OpenAI; наличие API key само по себе не включает
remote embeddings. OpenAI получает текст индексирования и поисковые queries только
через существующий выбранный remote provider path.

Local runtime загружает только локальные файлы (`local_files_only=True`); download
выполняется существующим owner-only endpoint после `confirm=true`. Отсутствие пакета
или файлов не включает remote fallback: обычный Chat продолжает работу без semantic
retrieval. Следующий вызов может загрузить модель после появления файлов; первая ошибка
не блокирует index навсегда. После явного Download (включая уже cached файлы)
модель остаётся в общем cache загруженных weights, доступном существующим и новым
scoped indexes. Ask может использовать эти weights без загрузки/инициализации runtime.
Замена vector вычисляет embedding до удаления прежней строки, а delete/insert выполняются
в одной транзакции: сбой embedding, metadata или insert сохраняет прежний vector.
Vectors автоматически не пересоздаются.
Inbox TTL относится только к inbox entries, не к chat history, confirmed Memory или vectors.

Browser показывает неблокирующее уведомление при недоступном local semantic search.
GET embeddings status сообщает `can_download` только verified owner при отсутствующих
файлах или известной ошибке setup и наличии библиотеки. `runtime_ready` отдельно сообщает
о загруженных weights: наличие файлов (`state=ready`) не доказывает активацию. Для owner
`can_activate` разрешает явную cache-only активацию; после ошибки доступен подтверждённый
Retry через тот же endpoint. GET не инициализирует модель. Явная кнопка передаёт `confirm=true`;
POST повторно проверяет owner и сохранённую модель. Отсутствующая библиотека требует
действия администратора на сервере; UI не обещает автоматическую установку или agent repair
без доказанной capability. Chat остаётся доступным при loading/error diagnostics.
После Settings refresh/close уведомление читает новый status; старые async responses
при размонтировании игнорируются. Dismiss не меняет config и не запускает инструменты.

## Canonical Memory storage prerequisite

Canonical atoms имеют opaque server revision. Existing principal-scoped stores используют
attached SQLite transaction для exact-revision vector publication и очистки canonical/atoms
vector при подтверждённой storage operation. Encoding выполняется до короткой транзакции;
поздний результат не публикуется после смены revision или удаления. Ошибка encoding сохраняет
предыдущий vector; canonical save по-прежнему может сообщать index_status=partial.
Deprecated pinned atoms исключены из pinned retrieval.

Atomic mutation требует двух file-backed SQLite DB с rollback journal (DELETE/PERSIST/TRUNCATE)
и synchronous >= FULL на соединении операции. WAL/неподдерживаемый режим отклоняется до
изменения данных; journal mode автоматически не переключается. Delete storage primitive
удаляет только выбранный canonical atom и его vector в namespace atoms, сохраняет bounded
receipt без текста (до 24 часов, максимум1024 операции) и проверяет revision при повторе.
История чатов, inbox, preferences и traces не удаляются; secure disk/backups erase не заявляется.
Browser Memory management API читает principal-owned canonical atoms без выбранной модели
или chat session: GET `/ui/api/memory/items` использует bounded keyset pagination.
POST `/ui/api/memory/delete/preview` принимает точные atom_id/revision и создаёт
principal-local opaque operation binding на 300 секунд (не более1024 previews, без текста).
POST `/ui/api/memory/delete/confirm` требует operation_id + confirm=true и вызывает existing
ToolGateway с server-only confirmed_decision_only tool; модель этот tool не получает.
Preview и receipt переживают restart; foreign/stale/expired bindings не применяются.
Browser identity остаётся verified Access JWT; Bearer `/v1` не получает browser Memory API.
Private ответы имеют no-store; POST требует JSON и отклоняет cross-site fetch metadata.
Action journal содержит только opaque operation ID и outcome.
Settings → Memory and data показывает собственные canonical items с полной пагинацией,
значением/status и отдельным preview/Confirm delete для одной exact revision.
Memory responses включают preferences_subject того же verified principal для связи с
Settings snapshot; чужие/поздние ответы отбрасываются. Закрытие не отменяет отправленное
подтверждение: повторно открытый список ждёт завершения известной операции этого subject.
Удаление не очищает уже переданный модели context или сохранённую историю.
Новая destructive capability модели не предоставляется.

## Настроенные budgets контекста

Owner Settings → Memory and data → Advanced показывает read-only snapshot12 полей
ContextBudgetConfig из существующего server loader. GET settings отдаёт отдельное
`context_budget_snapshot` только owner, вне editable `memory`. POST snapshot либо
`memory.context_budget` отклоняется без записи. Значения загружаются при refresh,
источник — Memory configuration с server defaults; per-field file/default provenance
не заявляется. Running Agent может хранить более ранний snapshot. Это chars/items
контекстной сборки, не model token window и не гарантия Auto execution budgets.
Member diagnostics не получает; React не дублирует значения по умолчанию.

Сохранение существующих inbox/embeddings Memory settings сохраняет уже загруженный
ContextBudgetConfig без сброса к defaults. Это сохранность unrelated fields, не budget
editor, live update существующего Agent или новая concurrency/storage guarantee.

## Сохранённые параметры default model

Owner Settings → Models and connections → Advanced показывает отдельный read-only
`model_config_snapshot` из уже загруженной default ModelConfig. Числа ограничены
конечными JS-safe значениями, enums — известными значениями; неверные типы получают
Unavailable, отсутствие — Not set. Содержимое api_key, base_url, extra_headers и
system_prompt не передаётся: показывается только наличие override. Member snapshot
не получает, POST диагностического поля отклоняется без записи.

Источник — сохранённая default configuration с defaults существующего loader при
Settings refresh, отдельно от несохранённого выбора. Это не effective snapshot
работающего чата, проверка provider capability или credential precedence.

Отдельно принятый loader prerequisite восстанавливает сохранённый
`web_search_enabled` только из JSON bool; отсутствие сохраняет False, другой тип
вызывает явную ошибку без содержимого значения. Owner read-only snapshot показывает
loaded bool как «Web search requested», без нового редактора или доказательства
выполненного поиска. Global runtime model гидратируется при старте; чтение Settings
и изменение файла не активируют настройки в уже работающем runtime. Existing session
model/override precedence сохраняется, наследование default options не добавляется.
True выбирает существующие web-search consumers; их prompts, tools, qualification и
policy/approval boundaries этим prerequisite не переписываются.

## Диагностика лимитов вложений

Owner Settings → Developer показывает read-only `attachment_limits_snapshot` из
констант существующего server attachment validator при Settings refresh. Это максимум
вложений, Unicode code points одного content и суммарного content в одном chat request.
Encoded image data URL считается текстом, не исходными file bytes или model tokens.
Browser preparation может иметь другие ограничения; snapshot не доказывает image support
или provider capability. Member snapshot не получает; POST диагностического поля
отклоняется до записи. UI не содержит копии numeric defaults и не меняет caps/send path.
Отсутствующий или malformed snapshot отображается как unavailable.

## Short-term и summary input limits

Owner Settings → Memory and data → Advanced показывает отдельный read-only
`runtime_context_limits_snapshot` из существующих short-term/summarizer constants
при Settings refresh. Short-term buffer ограничивает transient user/assistant entries,
не сохранённую историю и не каждый model request. Browser session-summary preview
читает сохранённую Hub history отдельно: последние непустые user/assistant messages
и ограниченный content каждого, включая serialized attachment text. Unicode code points
content не включают дополнительные role prefixes. Это input slices, не retention,
model token window, remaining capacity или гарантия полноты summary.
Member snapshot не получает; POST отклоняется до записи. Missing/malformed metadata
остаётся unavailable без numeric defaults в React. Этот read view не вызывает summary,
не создаёт Agent и не меняет summary prompt, close lifecycle или отдельный Memory confirm.

## Search backend defaults

Owner Settings → Developer показывает read-only `search_backend_defaults_snapshot`:
только числовые top_k/timeout/max_bytes из свежего WebSearchConfig при refresh.
Provider, keys и endpoints в этот snapshot не входят. Это class defaults, отдельно
от уже созданных tool instances, active chat options и model-native web search.
Top_k ограничивает рассматриваемые candidates, не гарантирует число valid results.
Timeout — аргумент одной HTTP-попытки, не total deadline с retry. Content read budget
не включает добавленный truncation marker и не является строгим output size cap.
Member snapshot не получает, POST отклоняется до записи; missing/malformed metadata
остаётся unavailable без numeric defaults в React. Чтение не создаёт Agent/search tool/
HTTP client, не проверяет credentials и не запускает search/capability probe.

## Speech tool numeric defaults

Owner Settings → Developer показывает read-only `speech_tool_defaults_snapshot` из
свежих TtsConfig/SttConfig при refresh: TTS timeout/input/request text limits и Agent
STT timeout. Только четыре numeric fields, без key/model/voice/language/endpoint/path.
Это class defaults, не snapshot существующего tool. TTS Unicode text измеряется после
trim; несколько chunks означают несколько HTTP requests, timeout не total deadline.
Browser dictation использует отдельный endpoint и не читает SttConfig; показанный
Agent STT timeout не является его настройкой. Existing env-selected TTS model/voice/
format остаются в Speech backend, request overrides — отдельный scope.
Member snapshot не получает, POST отклоняется до записи, missing/malformed metadata
остаётся unavailable без numeric defaults в React. Новый snapshot не вызывает
resolve_*, не создаёт tools/HTTP clients и не запускает speech/capability probes.
Existing env diagnostics, auth/approval и credential resolution не меняются.

## Shell numeric defaults

Owner Settings → Developer показывает read-only `shell_numeric_defaults_snapshot` из
чистого ShellConfig при refresh: timeout_seconds и max_output_chars. Это class defaults,
не загруженная JSON policy или active-run snapshot; существующий JSON может переопределять
их на каждом subprocess вызове. Timeout — аргумент subprocess, не общий task deadline.
Порог Unicode code points применяется после capture: ShellTool обрезает объединённые
stdout/stderr, TerminalTool oneshot — каждый поток отдельно; marker добавляется сверх
порога. Это не memory cap и не строгий returned-output cap, не PTY/Desktop settings,
effective grants, allowlist или root. Member snapshot не получает, POST отклоняется до
записи; missing/malformed metadata остаётся unavailable без numeric defaults в React.
Новый snapshot не вызывает loader/save/normalizer, не читает shell JSON и не создаёт
tools/subprocess. Runtime enforcement и permissions не меняются.

## Custom native tool protocol admission

Custom route начинается Chat-only. Только явная verified owner/session проверка
synthetic sync/stream protocol может создать transient evidence текущего scoped Agent.
Endpoint/model/effective credential context и config/key revisions проверяются перед
native request; recheck отзывает прежнее proof, другой чат/restart получает unknown.
ModelConfig и persisted JSON не содержат proof. Эта capability admission не меняет
ToolGateway, policy, approvals или запрет members использовать owner custom credentials.
Точный flow и ограничения: `docs/architecture/GENERIC_PROVIDER_ONBOARDING.md`.

## Смена модели сессии

Browser setter проверяет owned session, активные Chat/Plan execution и pending
решения под scoped Agent lock. Выполняющийся запрос или ожидающее подтверждение
возвращают 409 с предложением дождаться завершения либо воспользоваться существующей
отменой. Сам сигнал отмены не разрешает смену до settlement worker. Проверка
revision отклоняет поздний результат model catalog после другого переключения,
включая A → B → A. Request-time синхронизация не может записать устаревший снимок
модели поверх нового выбора. Эти guards не изменяют grants, policy или ToolGateway.
Тот же scoped lock защищает создание model-bound decisions, публикацию draft и
project history/results. Шаг переходит в `waiting_approval` до видимости approval decision.
Frontend привязывает результат выбора к текущему поколению чата: поздний success,
error или cleanup не изменяет другой чат и не снимает busy у нового запроса.

## Change protocol

1. Найти claim и его owner в `docs/runtime_contract_claims.json`.
2. Проверить current execution path и mechanism tests.
3. Изменить один canonical contract, реализацию и проверку в одном PR либо оставить честный
   статус `partial`/`target`.
4. Запустить `make preflight` на чистой новой ветке, targeted tests во время работы,
   `make check` перед commit, синхронизировать PR-ветку с `origin/main`, push и выполнить
   `make git-check` последним gate перед merge.

Guides не могут повышать статус claim. Новые compatibility layers, fallback runtime или
permission bypass не считаются реализацией контракта.

## Привязка согласования и запуска плана

POST `/ui/api/plan/approve` и `/ui/api/plan/execute` требуют `plan_id`,
`plan_hash` и положительный целочисленный `plan_revision` показанного плана.
Отсутствие привязки даёт 428, некорректный JSON/тип — 400, замена плана — 409
без согласования, нового decision или запуска. Bodyless approve и revision-only
execute больше не являются допустимыми запросами.

Согласование проверяет captured draft и режим под существующим lock UIHub;
изменённый, удалённый или уже согласованный snapshot не записывается повторно.
Persistence failure возвращает прежний несогласованный state до публикации события.
После успешной записи ошибка доставки события не отменяет факт согласования.
Execute сохраняет существующие start CAS, policy, exact decision continuation и
idempotency: replay принятого запроса относится к исходной привязке даже после
перехода того же плана в running с новой runtime revision. Новый запуск всегда
проверяет текущую полную привязку. Structured decision сохраняет ту же тройку;
ответ на старое подтверждение не может запустить или изменить заменённый план
с совпадающим номером revision. Неполный captured binding отклоняется.
Typed browser helpers передают явно выбранный
PlanEnvelope, без неявной подстановки текущего плана. Готовность карточки плана
этот API-контракт сам по себе не доказывает.

## Storage prerequisite сохранённой session configuration

UIHub хранит отдельную opaque `security_revision` только для сохранённых
`policy_profile` и raw `tools_state`. Existing trusted setters меняют revision при
изменении этих значений; import создаёт новую binding, restore сохраняет её.
Principal-owned snapshot и CAS выполняются под existing UIHub lock. Новый candidate
публикуется в памяти после успешной persistence. SQLite проверяет expected revision
и principal в той же transaction, что сохраняет session и её сообщения; устаревший
общий snapshot не может вернуть прежний config. Инициализация existing schema добавляет
только revision metadata, сохраняя прежние значения и сообщения. Existing startup
restore отдельно разрешает прежнюю нормализацию NULL/пустого principal к настроенному
owner и обрезку пробелов. Этот trusted restore сохраняет revision и её проверку; обычные
writers не получают права менять principal, именованный чужой principal не переназначается.

Owned GET session/security возвращает `security_revision` и quoted `ETag` одного
captured snapshot policy/raw overrides. POST требует точный `If-Match`: отсутствие —
428, неверная форма — 400, устаревшая/чужая binding — 409 без сохранения. Ownership
проверяется независимо. После полной validation payload/root/YOLO под existing
scoped agent lock вызывается одна UIHub CAS operation. Ответ содержит value/source/revision
результата этой операции, private/no-store. Token покрывает только сохранённую session
configuration, не installation defaults и не effective running permissions.

Session Controls разрешает Save только после чтения валидной revision выбранного чата
и отправляет captured token. Conflict блокирует Save до нового чтения при повторном
открытии панели и отдельного явного сохранения; автоматического POST retry нет.
Existing epoch/session/pending-write fences и два YOLO подтверждения сохранены.
Storage-fault проверки покрывают отказ до commit и rollback transaction; неопределённый
результат после commit/crash не получает новой recovery гарантии. Installation defaults,
grants, running execution snapshot, reset и live revoke этой revision не покрываются.
Один UIHub остаётся владельцем runtime state; multi-worker runtime не вводится.

## Чтение session security в браузере

Session Controls и Computer используют проверенный snapshot существующего owned
session/security endpoint. Отсутствующий, неполный, неизвестный или противоречивый
profile/tools_state не становится Sandbox либо safe-mode ON/OFF. При неизвестном
состоянии Computer показывает Unavailable, а controls не предлагают редактирование
выдуманных defaults. Ответ обязан совпадать с requested session_id.

Смена/закрытие панели отбрасывает поздние reads и completions прежней панели.
Повторное открытие той же сессии ждёт известный отправленный POST и затем читает
состояние заново; POST автоматически не повторяется. Завершение отправленного save
запускает отдельное чтение сводки только если сохранённая сессия всё ещё выбрана,
включая закрытую панель. Существующий explicit save,
YOLO confirmations и server enforcement не меняются. Это UI read-state contract,
не authoritative unified grant/source/revision/revoke или Desktop approval view.

## Сохранённые Desktop rules в Settings

Owner Access → Show saved Desktop rules показывает отдельную read-only projection
существующего Desktop store через GET `/ui/api/desktop/approvals?view=settings`.
Rules ограничены текущим verified owner; member не получает view. Opaque
preferences_subject связывает ответ с Settings principal, ответы private/no-store.
Неверный или повреждённый snapshot означает Unavailable, а не пустой список.

View не переписывает rule file, включая существующую интерпретацию legacy subject.
Текущие runtime matching, list_rules, write routes и persisted schema не меняются.
Projection не передаёт description, command contents, email, raw rule IDs либо raw
load errors. File target path показывается только для известных файловых tool/action ролей
существующего structured request path, включая browser download_destination;
XPath, GUI targets и неизвестные роли остаются скрытыми независимо от `/` prefix.
Non-file target и
command/risk restrictions помечаются как скрытые. Saved effect/source не являются
обещанием effective permission: transient once/session approvals и текущий host
lease здесь не перечисляются, DENY сохраняет precedence. UI не изменяет/revokes
rules и не предлагает reset-invalid. Source и lifetime — saved policy при refresh.


## Источник tool configuration чата

Owned GET session/security возвращает отдельный `tool_sources` для семи tool toggles:
`session_override` означает наличие bool в сохранённой карте чата, даже если значение
совпадает с installation default; `installation_configuration` означает отсутствие
такого override и загрузку из существующего installation loader, включая его defaults.
Policy и raw overrides захватываются одним чтением UIHub под существующим lock;
значения и источники tools строятся из одной captured карты. GET private/no-store,
не создаёт grant и не пишет session state. Runtime consumers не меняются; POST
возвращает source metadata из своего committed snapshot вместе с новой revision.

Session Controls показывает источник сохранённой configuration при refresh.
Редактирование tool toggle или применение preset сбрасывает source labels; POST без
metadata оставляет источник unavailable до нового чтения. Смена/закрытие панели
использует существующие epoch/session guards. Текущий save сохраняет всю карту tools,
поэтому override не доказывает, что пользователь отдельно менял данное поле.
Это не provenance конкретного значения file/default, active-run snapshot, provider/tool
capability, effective action permission, approval lifetime/revision или revoke.


Session/security POST проверяет весь combined payload policy/tools до единой CAS-записи
под существующим scoped agent lock. Неверный tools после корректного policy не
сохраняет policy. Existing YOLO/root/ownership guards и valid update semantics
сохраняются. Это отказ до записей при validation error, не CAS/revision и не новая
transaction guarantee при storage failure.


## Model options paused approval request

Owned GET `/ui/api/session/paused-model-request` читает только существующий scoped
Agent и текущий pending tool decision с source `chat.tool_continue`. Синхронный
accessor проверяет continuation identity, session и execution mode под существующим
scoped lock, без model resolver, Agent factory или brain initialization. Ответ
private/no-store и связан с session/decision: finite numeric/enum/bool/null model options
из captured continuation, без history, prompts, credentials, endpoint, headers и tool args.
Missing/settled/чужой или утраченный checkpoint означает unavailable без fallback к
Settings/main_config. Header обязателен; GET не создаёт session, POST отсутствует.
Session Controls читает это отдельно по явной кнопке, отбрасывает ответы прежней панели
или сессии и не добавляет editable/dirty/save state. Snapshot относится только к paused
request при refresh: не general running config, provider acceptance, search execution,
effective permission или гарантия resumability. Existing capture/resume/cancel, persisted
schemas, prompts и approval policy не меняются.
