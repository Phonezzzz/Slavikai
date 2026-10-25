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

Tone и private instructions применяются к следующему новому browser chat request.
Snapshot verified principal читается под существующим agent execution lock. Отдельное
user-role сообщение с preferences входит в Ask/chat (sync/stream), Auto и Desktop перед
исходным user goal; system prompts и policy/approval authority не изменяются.
Balanced + пустые инструкции сохраняют прежний request. Preferences не добавляются
в историю чата, Memory или context logs. Structured Plan/MWV без model request
не использует их для изменения packet/operations. Существующий native tool continuation
сохраняет исходные messages даже после изменения Settings. Bearer `/v1` использует defaults
и не импортирует browser preferences.

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
Это storage prerequisite: пользовательский paged Memory list/delete API и Settings UI ещё
не реализованы, новая destructive capability модели не предоставляется.

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
