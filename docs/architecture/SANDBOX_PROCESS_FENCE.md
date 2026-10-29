# Sandbox process execution: I0 fence

Статус и evidence: `docs/runtime_contract_claims.json`, claim
`runtime.sandbox.process_execution_fence`. Это containment отказа в запуске,
не реализованный isolated backend и не полная security qualification приложения.

## Текущий обязательный контракт

По умолчанию sandbox process execution недоступен во всех следующих контурах.
Единственное явное исключение для managed `workspace_run` описано ниже:

- native `shell`, `workspace_run`, `workspace_terminal_run`;
- manual PTY create/reuse, input и resize;
- process verifier: explicit packet command, default/fallback runner и Git repo probe;
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
задаются явно положительными integers; новых числовых defaults нет. Runtime ничего
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
в контейнер передаются независимые private staged bytes. Реальная stopped-container
конфигурация проверяется до start: pinned image, private namespaces, network none,
readonly rootfs, capdropall, no-new-privileges, единственный workspace mount с
nosuid/nodev, ограниченные memory/pids/tmpfs и allowlisted environment. Host home,
DB, credentials, control sockets и inherited FDs не передаются. Это qualification
конкретного профиля, не обещание защиты от kernel/engine vulnerabilities.

Collector разделяет raw stdout/stderr bytes, actual child exit и transport outcome.
Публикация происходит только после подтверждённого завершения и удаления собственной
container environment; проверяются outputs, исходные owner/revision/content, затем
Linux atomic directory exchange. Unsupported exchange не заменяется copy-back.
Concurrent writer даёт conflict, unknown/timeout/output limit — retained stage без
публикации или replay. Такой результат останавливает batch и следующие model/tool calls;
проверка и новый запуск требуют явного решения. Retained stages не имеют GC/recovery
UI и могут сохранять чувствительные bytes; автоматического cleanup unknown outputs нет.
При ошибке удаления контейнер может остаться, settlement не объявляется успешным.

Raw bytes пока доступны внутренней boundary только на время вызова; native ToolResult
содержит decoded view. Durable canonical capture, original run/attempt intent и recovery
относятся к незавершённому R2b. Required process verifier остаётся ERROR/unavailable;
этот backend не завершает R2/R1 и не включает production activation.
