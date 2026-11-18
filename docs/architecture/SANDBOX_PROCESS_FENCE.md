# Sandbox process execution: I0 fence

Статус и evidence: `docs/runtime_contract_claims.json`, claim
`runtime.sandbox.process_execution_fence`. Это containment отказа в запуске,
не реализованный isolated backend и не полная security qualification приложения.

## Текущий обязательный контракт

По умолчанию sandbox process execution недоступен во всех следующих контурах.
Единственное явное исключение для managed `workspace_run` описано ниже:

- native `shell`, `workspace_run`, `workspace_terminal_run`;
- manual PTY create/reuse, input и resize;
- process verifier: explicit packet command, default/fallback runner и Git repo probe — кроме узкого исключения штатного verifier ниже;
- manual workspace Git status/diff/stage/unstage/commit и GitHub clone/import;
- command/check/tests/diff/process patch в `ContainerComputerBackend`, включая direct runner.

Отказ происходит после действующих auth/mode/target/policy/safe-mode/path checks,
но до spawn/PTY allocation/записи command input. Где operation не требует дополнительных
input guards, отказ непосредственный. Pending approval не является состоявшимся effect.
Approval/yolo, owner identity, tool args, env option или выбор container image не снимают
fence. Нет permit branch, host fallback или тестового qualification flag.

`ToolResult` содержит `ok=false`, reason `sandbox_execution_unavailable` в metadata и
`execution_started=false`, без выдуманного process exit code/output. Process verifier
возвращает `VerificationStatus.ERROR`, `exit_code=None`, тот же error code и понятную
причину. Required verification нельзя пропускать, заменять PASS или model assurance.
Существующий MWV error guard запрещает worker retry после infrastructure error.
Manual terminal HTTP сообщает 503 с этим code после существующего session/yolo gate.
Остальные endpoints сохраняют существующие delivery/validation error envelopes;
HTTP success/решённый approval не означает process success.

API `is_repo_workspace` использует консервативный `.git` marker самого root (directory
или worktree file), без вызова Git/config. Это прежний scope самого workspace top-level,
а не Git health check. При GIT_DIR/GIT_WORK_TREE/common-dir/config overrides
контекст неоднозначен и также требует repository verification, без раскрытия env values.
Положительный marker требует repository verification; неуспех
запуска probe не используется как доказательство, что verifier можно пропустить.
Наличие Makefile с canonical check по-прежнему выбирает required process verifier.
Process-free response/tool observation criteria остаются отдельными существующими criteria.

## Сохраняемые границы

Tool calls проходят прежний `ToolGateway -> ToolRegistry`. Principal/session ownership,
approvals и budgets не меняются. Ask, file-only operations и process-free verification
сохраняются; Plan/Computer операции, реально требующие process, получают отказ.
Ни один runtime consumer не может исполнять workspace-controlled code под видом
read-only Git, verifier или admin helper.

Explicit Desktop остаётся отдельным существующим host target/policy/lease; fence не
даёт Desktop дополнительных прав и не превращает Sandbox в Desktop. Server-owned
Ollama startup (`OLLAMA_BIN`, фиксированный serve argv) и developer skill manifest build
(фиксированный script из server skills root) не являются workspace process execution;
новые args/executable roots из запроса в эти пути не добавляются. Developer `make check`
и тестовые temp fixtures не являются runtime Sandbox backend.

Inspection/закрытие старого terminal state сохраняют прежние auth и cleanup semantics;
stdin/resize/reuse unqualified host process не разрешаются. Уже запущенные jobs не
становятся изолированными. I0 не заявляет их settlement, автоматический kill, restart
или production cleanup/activation. Live rollout и завершение старых jobs проверяются отдельно.

## Что не реализовано

I0 сам по себе не реализует isolated execution backend, real environment qualification,
новые grants/config bypass, migrations/schemas/stores, hard run timeout, effects ledger,
recovery/replay, полную file/network isolation или raw canonical capture. I0 не завершает R2b/R2/R1.
Наличие Container factory либо fake runner tests не доказывает isolation.

Разрешение process execution потребует отдельного согласованного scope: реальная среда,
trusted principal/workspace/environment binding, недоступность server-private DB/env,
проверка mounts/FDs/processes/sockets/network и controlled positive/negative acceptance.
Принятое узкое исключение реализуется только на описанной ниже boundary.

## Узкое исключение: managed Python workspace

Отдельный explicit rootless profile возвращает только native `workspace_run` через
`ToolGateway -> RunCodeTool -> ComputerBackend.execute_python`. Все остальные process
paths из списка выше остаются закрыты. Ручной Run через `Agent.call_tool`/HTTP и
его category-only resume также остаются unavailable: они не захватывают immutable
workspace/environment approval. Native loop/typed continuation идут через Gateway. Local backend не исполняет Python на host;
выбор image/approval/yolo сам по себе не даёт admission.

Для подключения нужны `SLAVIK_COMPUTER_BACKEND=container`, engine `podman`, workspace
`/workspace`, immutable image ID и canonical absolute `SLAVIK_COMPUTER_ROOTLESS_STATE`.
Все пять `SLAVIK_COMPUTER_CAPTURE_FILE_BYTES|TOTAL_BYTES|ENTRIES|DEPTH|OUTPUT_BYTES`
задаются явно положительными integers. Отдельно обязательны
`SLAVIK_COMPUTER_STAGING_SLOT_BYTES|STAGING_SLOT_INODES|STAGING_SLOTS`:
размер каждого tmpfs slot (кратный странице), число inodes и конечное число slots.
Capture limits не подменяют execution quota; числовых defaults и наследования нет. Runtime ничего
не устанавливает, не скачивает и не запускает daemon. Без этих настроек действует I0.
Точные поддержанные engine/image/config bindings определены в `core/rootless_podman.py`;
иная версия или изменённая binding даёт unavailable, без автоматического ремонта.

Server создаёт отдельный workspace principal под private rootless state. Существующие
host roots не импортируются и остаются file-only. Доверенный pool разделяется между
sessions; один principal имеет общий workspace owner/revision и exclusive OS lease.
Restart сохраняет bytes, но выдаёт новую identity/revision, без восстановления approvals.
Весь control state защищён от выбора как workspace и от Desktop file capabilities.
Шесть native writers проходят тот же owner guard после registry policy checks.

Ingress — ограниченный nofollow capture с отказом для links/special files и race;
в контейнер передаются независимые private staged bytes внутри kernel-bounded tmpfs.
Один principal занимает не более одного active или retained slot; разные principals
могут исполняться параллельно. Совокупный предел staging — число slots × размер
slot (отдельно inodes), включая retained outputs. Краткий allocation lock не
удерживается во время execution. Worker не получает writable host-backed stage.
Доверенный mount owner живёт в существующих rootless namespaces; сервер работает
через отдельный capability-free view. PID/start time, namespaces, helper files и
реальные slot identities входят в environment binding. Потеря owner или неизвестные
старые mounts запрещают admission/reuse; автоматического overmount/recovery нет.
Реальная stopped-container
конфигурация проверяется до start: pinned image, private namespaces, network none,
readonly rootfs, capdropall, no-new-privileges, единственный workspace mount с
nosuid/nodev, ограниченные memory/pids/tmpfs и allowlisted environment. Container
logging отключён и проверяется в actual spec: stdout/stderr не дублируются в
неограниченный host log. Host home,
DB, credentials, control sockets и inherited FDs не передаются. Это qualification
конкретного профиля, не обещание защиты от kernel/engine vulnerabilities.

Collector разделяет raw stdout/stderr bytes, actual child exit и transport outcome.
Публикация происходит только после подтверждённого завершения и удаления собственной
container environment; nofollow capture outputs ограничен до переноса в новый
доверенный host stage, проверяются исходные owner/revision/content, затем
Linux atomic directory exchange. Unsupported exchange не заменяется copy-back.
Concurrent writer даёт conflict, unknown/timeout/output limit — retained stage без
публикации или replay. Такой результат останавливает batch и следующие model/tool calls;
проверка и новый запуск требуют явного решения. Retained stages не имеют GC/recovery
UI и могут сохранять чувствительные bytes; автоматического cleanup unknown outputs нет.
Retained outputs продолжают занимать bounded slot; exhaustion даёт unavailable.
Slot освобождается после подтверждённого cleanup опубликованного stage либо
после явного отказа по файловому бюджету до публикации, если контейнер остановлен
и удалён. При ошибке очистки slot остаётся занятым и требует review. Shutdown с живыми allocations сохраняет их mounts;
чистое завершение требует доставленной команды и подтверждённого unmount.
При ошибке удаления контейнер может остаться, settlement не объявляется успешным.

Для admitted синхронного Auto UI/API и captured approval R2b сохраняет original
run/attempt intent и bounded raw stdout/stderr observation в existing TaskRunStore
до lossy ToolResult view; точный scope — TASK_RUN_STORAGE_CONTRACT.md. Остальные
native callers не получают hidden capture. Полный canonical capture, recovery и
terminal acceptance остаются target. Required process verifier остаётся ERROR/unavailable
без квалифицированного command backend (см. узкое исключение ниже);
этот backend не завершает R2/R1 и не включает production activation.

## Узкое исключение: штатный process verifier (make check)

Второе явное исключение — только штатный `VerifierRuntime` с canonical check
`["make", "-B", "-r", "-f", "Makefile", "check"]` в managed workspace. Все остальные process paths из списка
выше остаются закрыты; shell tools модели не публикуются (это E04-T04),
полная семантика UNVERIFIED/BLOCKED — E05-T03.

- Отдельный pinned image `python:3.12.16-bookworm` с make 4.3
  (Id `28d30c15b9e9c3b1d557441aad9592226e852f7716451d6c8df5b4f0c3ce3d1b`,
  digest `sha256:3c3ba8ad4c85eb63b7568f806434f14e33370056985da7fdf2673621cb1d7e93`),
  отдельный профиль `i1-ubuntu-podman-5.7-command-bounded-v1`. Python-профиль
  (`…-python-bounded-v3`) не трогается; command-ревизия квалифицируется
  независимо — её отсутствие не ломает Python-путь.
- Allowlist ровно `["make", "-B", "-r", "-f", "Makefile", "check"]`: любой другой argv отклоняется до
  любого обращения к engine. Явный `-f Makefile` фиксирует проверяемый
  файл: без него make сам выбрал бы `GNUmakefile`/`makefile`, и хэш
  `Makefile` ничего бы не гарантировал. `-B` заставляет make всегда
  выполнять recipe цели: файл или каталог `check` в workspace больше
  не даёт `up to date` без проверки. `-r` отключает встроенные правила
  make: `Makefile.sh` и любой другой файл больше не могут пересоздать
  доверенный `Makefile` внутри stage через неявное правило `%: %.sh`.
  Точная гарантия `-r`: отключены только встроенные правила. Для
  произвольных Makefile с собственными явными правилами пересборки
  самого себя или с `include` файлов, которые может писать агент,
  этой гарантии недостаточно — такой критерий требует отдельной
  проверки. Срез покрывает только E2E-fixture Makefile без собственных
  правил пересборки и без include. Confinement — тот же, что у Python-пути
  (network none, private namespaces, cap-drop ALL, no-new-privileges,
  readonly rootfs, 256 МиБ / 64 PID, ровно один bind mount staged копии,
  ровно те же 7 env-записей), реальный exit code — через `finished_exit`.
- Verifier запускается на staged snapshot и только наблюдает: stage
  освобождается через `release_unpublished`, артефакты проверки никогда
  не пишутся обратно в workspace.
- Условие 1 (явное разрешение на run): запуск `make check` привязан к
  доверенному `WorkspaceBinding`; само наличие Makefile разрешения не
  даёт. Источник разрешения — вызов auto run владельцем
  (`handle_auto_command` после routing decision gate); запись о нём —
  `_RunFrame` этого run. `_verifier_command_executor` сверяет binding
  с frame (principal/session) и с текущей revision квалифицированного
  executor (live; в самом frame environment revision не хранится),
  а также требует совпадения run_root и квалифицированного профиля/
  staging pool. Без binding или без доступного backend
  `VerifierRuntime.run()` возвращает ERROR
  `sandbox_execution_unavailable`, процесс не запускается.
- Условие 2 (критерий задаёт доверенная сторона): содержимое `Makefile`
  (цель `check`) задаётся до запуска агента; его sha256 фиксируется
  доверенным кодом раннера до начала loop и хранится вне agent-writable
  workspace. Хэш сверяется дважды: ранний fail по живому workspace и
  авторитетный — на байтах того же snapshot, который пойдёт в контейнер
  (executor вычисляет sha256 snapshot'ного `Makefile` до `pool.prepare`).
  Несовпадение или удаление файла даёт ERROR `verifier_criterion_changed`,
  никогда PASSED. Решение о том, что проверка обязательна, живёт вместе
  с критерием до конца run: если критерий был зафиксирован, а `Makefile`
  к концу run исчез, `_verify_after_loop` возвращает ERROR
  `verifier_criterion_changed`, а не `None`. Без зафиксированного критерия
  `Makefile`, появившийся во время loop, никогда не исполняется.
- Валидация до выделения stage: проверки скрипта (`workspace_run`) и
  критерия (verifier) выполняются до `pool.prepare()`, поэтому
  предзапусковой отказ не занимает staging slot и не блокирует следующие
  запуски principal.
- Свежесть после исполнения: успешный результат verifier обязан
  относиться к актуальному workspace, а не к устаревшему snapshot.
  После освобождения stage executor повторно захватывает workspace и
  сравнивает его с проверенным snapshot: изменение (в т.ч. запись
  другой сессии того же principal во время проверки) даёт conflict
  (`workspace_conflict` → ERROR), а не PASSED.
- Ошибка settlement не теряется: если освобождение stage провалилось
  (`cleanup_warning`, retained stage), маппинг возвращает ERROR
  (`workspace_cleanup_failed` / `settlement_failed`) с сохранённым exit
  code проверки, а не тихий PASSED. Retained stage блокирует следующие
  запуски principal и требует вмешательства оператора.
- Маппинг исходов: finished+0 → PASSED; finished+≠0 → FAILED
  (`check_failed`); unavailable → ERROR зеркалом
  `sandbox_verification_unavailable`; unknown/conflict → ERROR, никогда
  фейковый PASSED. `_verify_after_loop`: canonical verifier есть, но backend
  недоступен → ERROR, а не `None` (fail-closed в FAILED_VERIFIER вместо
  COMPLETED).
- Scope: проверка E2E-fixture Makefile в managed workspace. Полный
  `make check` самого репозитория внутри sandbox — вне scope (репозиторий
  нельзя заснять в managed workspace: symlink/размеры; в образе нет
  node/venv). Зелёный fixture E2E нельзя читать как «verifier теперь
  работает для настоящих репозиториев».
