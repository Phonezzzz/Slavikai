# Sandbox process execution: I0 fence

Статус и evidence: `docs/runtime_contract_claims.json`, claim
`runtime.sandbox.process_execution_fence`. Это containment отказа в запуске,
не реализованный isolated backend и не полная security qualification приложения.

## Текущий обязательный контракт

Квалифицированной sandbox process environment в current runtime нет. До её отдельной
реализации и проверки process execution недоступен во всех следующих контурах:

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

Нет isolated execution backend, real environment qualification, новых grants/config
bypass, migrations/schemas/stores, hard run timeout, effects ledger, recovery/replay,
полного file/network isolation или raw canonical capture. I0 не завершает R2b/R2/R1.
Наличие Container factory либо fake runner tests не доказывает isolation.

Разрешение process execution потребует отдельного согласованного scope: реальная среда,
trusted principal/workspace/environment binding, недоступность server-private DB/env,
проверка mounts/FDs/processes/sockets/network и controlled positive/negative acceptance.
Это направление остаётся proposal, а не разрешение подключить backend или ослабить fence.
