# Roadmap перестройки SlavikAI для Muse как worker

## Executive architecture decisions

Этот roadmap — **инженерная программа перестройки SlavikAI для Muse как worker**. Канонические документы: [рабочее соглашение Muse](slavikai-how-to-work-with-muse.md), [STATE.md](STATE.md) и данный roadmap. Правило: один тикет → одна ветка → один PR → `make check` → CI → независимое ревью Codex → **ручной merge Muse** → следующий тикет. Параллельные тикеты запрещены. Все содержательные развилки реализации идут владельцу. **GitHub auto-merge не включать.**

Этот roadmap **не исходит из предположения, что существующая архитектура `main` правильная**. Существующий код рассматривается только как набор потенциально полезных строительных блоков. Механизм сохраняется лишь тогда, когда у него есть самостоятельная пользовательская capability, реальный production call path и более дешёвый путь к надёжной реализации, чем удаление/замена. Старые документы, предыдущий аудит и target PDF являются источниками гипотез, но не доказательством фактического состояния кода. Именно так должен работать каждый будущий тикет Muse.

На момент исследования публичный `main` действительно содержит `AgentToolLoop`, `ToolGateway`, `ToolRegistry`, `DesktopRuntime` и generic Playwright-based `DesktopBrowserTool`. `AgentToolLoop` уже является реальным model→tool→observation loop; `ToolRegistry` всё ещё смешивает `ExecutionMode = ask|plan|act|auto|desktop` и `ExecutionTarget = sandbox|desktop`; `AgentRoutingMixin` всё ещё импортирует legacy `classify_request`; `DesktopRuntime` использует тот же `AgentToolLoop`; browser backend умеет generic navigation/read/snapshot/actions, но жизненный цикл браузера сейчас построен вокруг backend/context, а не вокруг отдельного durable Browser Session Resource.

PR #151 по GPT subscription **успешно merged** 2026-11-16 в `main` (`470a9a63280f9dac392f559a6fe99115db2e50d0`), включая исправления approvals и отмену auto-send. Эта интеграция работает и больше не блокирует архитектурный roadmap. Владелец согласовал **два linked Git worktree**: исходный `/home/ki/Project/Slavikai` (`main`), где Muse не работает, и выделенный `/home/ki/Project/Slavikai-rebuild` (`muse/slavikai-rebuild`) для всех её тикетов. Третье дерево нельзя создавать без разрешения. Muse выполняет только последовательные PR из второго дерева, не затрагивая исходное.

**Read-only baseline обновлён 2026-11-16:** `origin/main` = `470a9a63280f9dac392f559a6fe99115db2e50d0` после merge PR #151. `git worktree list` подтвердил исходный `/home/ki/Project/Slavikai` (`main`, сохранён post-merge untracked-файл владельца) и новый `/home/ki/Project/Slavikai-rebuild` (`muse/slavikai-rebuild`, HEAD = `470a9a63280f9dac392f559a6fe99115db2e50d0`). Второй worktree создан, но **реализация там не начиналась**. Каждый новый тикет требует актуальной перепроверки HEAD/status/`origin/main` и [STATE.md](STATE.md); SHA приведён как исторический baseline, не вечная гарантия.

Главное архитектурное решение: **SlavikAI становится одним агентом с независимыми параметрами Run, а не набором конкурирующих режимов**.

Минимальный production contract:

`User/Task → Run → AgentToolLoop → ToolGateway → qualified Executor → Observation → Postcondition → Outcome`

Где `Run` имеет четыре ортогональные оси:

`intent = answer | plan | execute`

`target = workspace | browser:<profile> | owner_host | connector:<id>`

`policy = immutable snapshot разрешений/effects`

`budget = time | tool calls | model tokens | files | bytes | processes`

`Ask`, `Plan`, `Act`, `Auto`, `Desktop` больше не должны означать пять разных execution engines. `Plan` остаётся полезным workflow и approval artifact. `Auto` превращается в policy/budget preset. `Desktop` становится execution target/trust boundary. `Act` исчезает как самостоятельный runtime и означает execute-run с approved plan. MWV не получает права существовать в production только потому, что у него есть Manager/Worker/Verifier классы.

Это направление соответствует уже существующей архитектурной реальности: `AgentToolLoop` принимает `Brain`, `ToolGateway`, tool specs и budgets; `DesktopRuntime` уже использует тот же loop и добавляет host-specific coordination/verification; `ToolGateway` уже является центральной точкой approval/workspace enforcement. Значит, миграция может быть **consolidation**, а не big-bang rewrite.

Первый новый product slice после унификации контракта — **persistent authenticated Browser Session Resource**. Это не browser mode и не site-specific connector. Playwright официально поддерживает persistent context через `userDataDir`; такой context хранит browser session state на диске. Документация также прямо показывает использование отдельного user-data directory для сохранённого auth state. Значит, целевой пользовательский сценарий технически естественен: владелец открывает отдельный профиль SlavikAI, вручную проходит login/MFA, закрывает приложение, после рестарта агент снова открывает тот же профиль и продолжает работать в той же web-session.

Вторая обязательная новая capability — **Owner Host Bridge уровня Desktop Commander**, но архитектурно это не новый агент и не второй execution engine. Это restricted executor для `target=owner_host`.

Для одного Linux-хоста выбран **Unix domain socket**, а не localhost HTTP и не SSH. Причина: bridge и SlavikAI находятся на одной машине; AF_UNIX позволяет опереться одновременно на filesystem permissions и Linux peer credentials, не открывает TCP listener и хорошо интегрируется с systemd socket/service lifecycle. Linux предоставляет `SO_PEERCRED` для получения credentials peer-а Unix socket; systemd socket units позволяют контролировать ownership/mode filesystem socket и socket activation.

HTTP поверх loopback остаётся возможным поздним transport adapter только если появится реальная потребность. SSH используется только если agent runtime и owner host физически разнесены, как уже зафиксировано владельцем.

Для текущего Ubuntu host я **не рекомендую начинать с полной VM**. Начальная архитектура: отдельный OS-level workspace/runtime principal + bounded workspace + service boundaries; owner host доступен только через bridge; browser profile находится за Browser Service. Полная VM или rootless container остаются deployment evolution, если threat model покажет необходимость более сильной isolation. Это уменьшает количество одновременно вводимых технологий и позволяет сначала доказать capabilities. Target PDF сам признаёт VM vs container deployment choice, а не обязательный semantic contract.

Explicit non-goals до завершения foundation:

- scheduler/cron/proactive monitoring;
- subagents;
- distributed workers;
- vector-first memory architecture;
- PTY terminal;
- site-specific browser connectors;
- automatic social posting/trading;
- Kubernetes;
- multi-node orchestration;
- generic plugin framework;
- новая provider abstraction ради красоты;
- миграционные adapters только ради сохранения legacy;
- полная VM только потому, что она была нарисована в target PDF.

Rollback policy для всей серии: каждый тикет должен быть mergeable независимо; нельзя делать промежуточный PR, после которого `main` заведомо неработоспособен. Старый production path удаляется **только в том же PR или следующем строго связанном PR после доказательства replacement path**, причём пока replacement не принят, compatibility допускается лишь как краткоживущий migration bridge с записанным ticket ID на удаление. Постоянные dual paths запрещены.

**Owner decision gates не ограничены пятью случаями.** Muse **обязана спросить владельца при любой содержательной развилке реализации «так или иначе»** с влиянием на архитектуру, поведение, контракты, границы доверия, зависимости или будущую поддержку. Нужны варианты, trade-offs, рекомендация Muse и последствия. Сюда также входят security boundary, новый класс side effects, destructive migration, VM/container/OS-user, root/sudo, расширение host roots и PR #151. Самостоятельно решаются только мелкие детали внутри уже утверждённого способа.

## Architecture blueprint

### Canonical Run contract

Новая архитектура не должна начинаться с создания десятка новых классов. Сначала нужен минимальный immutable контракт запуска.

```text
RunSpec
  run_id
  principal_id
  session_id
  intent
  target
  goal
  approved_plan?
  policy_snapshot
  budgets
  workspace_binding?
  browser_profile_id?
  parent_run_id?
```

```text
RunState
  status
  started_at
  updated_at
  continuation
  observations[]
  effects[]
  pending_approval?
  process_refs[]
  postconditions[]
  outcome?
```

Outcome — **структура, а не плоский enum**, согласованная с [ADR-0002](docs/architecture/decisions/ADR-0002-terminal-outcome-and-partial-result.md):

```text
RunPhase: running | blocked | waiting_user_input | waiting_approval | terminal
TerminalReason (только для terminal): completed | failed | cancelled | aborted
Disposition: full | accepted_partial | none
Verification: passed | failed | unknown | not_applicable
Evidence: postconditions + observations + effect ledger
```

Текстовые метки UI — только **проекции этой структуры**, не источник истины и не подмена `TerminalReason`. Проверенный набор меток (`core/run_outcome.py::RunOutcome.ui_label`, E01-T05): `RUNNING`, `WAITING`, `BLOCKED`, `DONE`, `PARTIAL`, `FAILED`, `CANCELLED`, `ABORTED`, `UNVERIFIED`. `DONE` — тогда и только тогда, когда `completed + full + verification=passed` (для задачи без изменения состояния допустимо `verification=not_applicable`). `completed + accepted_partial` — `PARTIAL`, его нельзя представить полным `DONE`; `final_partial` — только для `completed + accepted_partial`. `aborted` — отдельная метка `ABORTED`, не `FAILED`. `blocked` — `BLOCKED`, `waiting_user_input` / `waiting_approval` — `WAITING`: не-терминальные фазы никогда не выдаются за завершение. При `Verification=unknown` state-changing задача — `UNVERIFIED` **с сохранением причины завершения** (не `DONE`, не автоматический `FAILED`); при `Verification=failed` проверка опровергла заявленный результат — run считается `failed`. Никакой fake PASS из `tool.ok`. `final_success` публикуется только при подтверждённой postcondition (gate принятия по ADR-0002 §4, не изменение таблицы §3): для неподтверждённого full claim финал удерживается, а не публикуется.

`AgentToolLoop` уже хранит iterations, executed tool calls, error/cancelled state, usage и continuation после approval; эти реальные свойства надо использовать, а не переписывать loop с нуля.

### Tool contract

Канонический ToolSpec должен описывать не только JSON schema, но и effect semantics:

```text
name
description
input_schema
target
effect
idempotency
qualification
timeout_class
postcondition_kind?
```

Effects минимально:

```text
read
write
exec
external_send
destructive
privileged
```

Текущий `ToolRegistry` уже хранит capability, risk classes, model exposure и execution targets; значит, migration должна нормализовать эти поля, а не создавать параллельный registry. Сейчас registry mutates execution mode/target internally; конечная цель — передавать immutable per-run policy/target при listing/call, чтобы глобально изменяемое состояние реестра не определяло безопасность одного run.

`ToolGateway` остаётся единственной production точкой:

```text
schema validation
→ qualification check
→ target binding
→ policy decision
→ approval
→ execute
→ record effect
→ return observation
```

Текущий gateway уже связывает approvals с session/request JSON и workspace binding; это хорошая база для дальнейшей унификации.

### Persistent Browser Session Resource

Browser profile — **resource**, а не hidden state внутри tool instance.

```text
BrowserProfile
  profile_id
  principal_id
  root_path
  state = stopped | starting | ready | needs_user | broken
  lease_owner_run_id?
  last_started_at?
```

BrowserService:

```text
open_profile(profile_id)
close_profile(profile_id)
status(profile_id)
list_tabs(profile_id)
navigate(...)
observe(...)
click(...)
type(...)
select(...)
upload(...)
download(...)
back/forward(...)
new_tab/close_tab(...)
screenshot(...)
```

Один profile одновременно имеет максимум один writer lease. Playwright persistent context запускается на отдельном dedicated `user_data_dir`, который не совпадает с основным Chrome владельца. Playwright документирует `launchPersistentContext(userDataDir)` именно как browser context с persistent storage.

Manual-auth handoff:

```text
Agent requests profile
→ BrowserService launches visible Chromium
→ UI marks NEEDS_USER
→ owner enters password/MFA directly in browser
→ owner presses "Готово"
→ BrowserService verifies expected logged-in observation
→ returns control to agent
```

Пароль не проходит через LLM context, tool result или SlavikAI logs.

Критическая trust-boundary: workspace shell **не должен иметь raw access к profile directory**, иначе обещание «агент не видит credentials/session material» становится ложным. BrowserService должен владеть profile directory под отдельным principal или permissions boundary.

### Owner Host Bridge / Desktop Commander architecture

Host Bridge — отдельный local daemon:

```text
Slavik Runtime
   |
   | AF_UNIX
   v
/ run/slavik-host-bridge/bridge.sock
   |
Host Bridge daemon
   |
   + filesystem capability
   + process capability
   + service capability
   + diagnostics capability
   + file-transfer capability
   + optional desktop-control capability
```

Не допускается:

```text
arbitrary unauthenticated shell
root daemon exposed to model
TCP 0.0.0.0 listener
shared unrestricted owner home
implicit sudo
raw secret store access
```

Protocol transport: length-prefixed JSON messages или newline-delimited JSON с жёстким maximum frame size. Для первого варианта достаточно request/response + polling; полноценный bidirectional streaming не нужен до появления доказанной необходимости.

Request:

```json
{
  "protocol_version": 1,
  "request_id": "...",
  "principal": "...",
  "run_id": "...",
  "capability": "process.run",
  "args": {},
  "timeout_ms": 30000
}
```

Response:

```json
{
  "request_id": "...",
  "ok": true,
  "observation": {},
  "effect": {},
  "error": null
}
```

Long process:

```text
process.run → process_id
process.poll → state/stdout/stderr/exit_code
process.kill → signal/result
```

Bridge authorizes client twice: filesystem mode/ownership socket + `SO_PEERCRED` UID/PID check. `SO_PEERCRED` на Linux предоставляет peer credentials для connected Unix socket.

Минимальная allowlisted capability set:

```text
fs.stat
fs.list
fs.read
fs.write
fs.patch
fs.move
fs.delete

process.run
process.poll
process.kill

service.status

transfer.put
transfer.get

diagnostics.host
diagnostics.disk
diagnostics.processes
```

`desktop.observe` / `desktop.control` добавляются отдельным тикетом и отдельной policy capability, потому что GUI control существенно расширяет threat surface.

`service.start/stop/restart`, package installation и privileged operations **не входят в v1**.

Path security:

1. request path нормализуется;
2. absolute path обязан лежать внутри конкретного capability root;
3. `..` запрещён до filesystem access;
4. проверяется resolved real path;
5. symlink traversal наружу запрещается;
6. write destination parent также проверяется;
7. rename/move проверяет source и destination;
8. hard-link edge cases покрываются негативными тестами;
9. protected paths denylisted независимо от root.

Process security:

- только argv array, без implicit `/bin/sh -c`;
- working directory bound к approved root;
- timeout обязателен;
- max stdout/stderr bytes;
- process count quota;
- environment allowlist;
- no inherited sensitive env;
- no privilege escalation;
- kill process group на STOP.

Systemd lifecycle для bridge должен использовать sandboxing primitives вроде `NoNewPrivileges`, filesystem restrictions и `RestrictAddressFamilies=AF_UNIX` там, где они совместимы с фактическими capabilities. systemd документирует `ProtectSystem`, `ProtectHome`, `PrivateDevices`, `RestrictAddressFamilies` и прочие isolation controls; socket unit может отдельно контролировать Unix socket ownership/mode.

Bridge **не принимает решения об approval**. Решение находится в SlavikAI policy layer. Bridge повторно enforce-ит structural boundaries, потому что executor нельзя заставлять доверять вызывающему агенту. Это defence in depth без второго permission framework.

Prompt injection не решается фильтром текста. Любой файл, web page, terminal output или GUI observation считается untrusted data. Только RunSpec + policy snapshot могут дать authorization. Интересно, что текущий `DesktopRuntime` уже содержит именно эту идею в system prompt: tool output и web/file content не могут менять user goal, policy или authorization. Это полезный принцип, который надо поднять из prompt-level convention в execution contract.

STOP semantics:

```text
STOP received
→ mark run cancelling
→ refuse new tool calls
→ signal active managed processes
→ BrowserService cancels current operation
→ HostBridge kills process group for run-owned process refs
→ persist partial effects
→ terminal_reason=cancelled; disposition=accepted_partial|none (по результату); verification=passed|unknown
```

Никакого обещания «rollback всего». Уже совершённый external side effect может быть необратим. UI должен показывать partial effects.

### Durable state, memory and context

Durable task state нужен раньше scheduler.

Минимальные stores:

```text
runs
effects
approvals
process_refs
browser_profiles metadata
conversation summaries
curated memory
```

Первый этап не требует vector DB.

Memory split:

- **run state** — operational;
- **conversation summary** — context compaction;
- **curated memory** — долговременные факты/решения;
- **search corpus** — проектные материалы.

Vector index допускается только как rebuildable derived index.

Target PDF предлагает значительно более богатую layered memory; это разумный возможный end state, но не основание вводить все слои до доказательства чтения/записи каждого из них.

### UI and observability

UI должен отображать:

```text
Run goal
Intent
Execution target
Current action
Pending approval
Effects
Verification
Processes
Browser profile/session state
Owner-host bridge health
STOP
Final outcome
```

Пользователь не должен видеть архитектурные названия `MWV`, `AutoAgent`, `ActRuntime`.

Его язык:

```text
Ответить
Сначала показать план
Выполнить
Где: среда Славика / браузер Славика / этот компьютер
Права: только чтение / спрашивать / разрешённые действия
```

## Critical path and ordered epics

**Программа: 63 тикета, строго последовательно**, один тикет = одна ветка = один PR. Список пересматривается по подтверждённым фактам после эпиков: sunk cost не является основанием сохранять механизм.

| Epic | Tickets | Статус |
|---|---:|---|
| E00 — baseline/docs | 3 | E00-T01 DONE; E00-T02 DONE; E00-T03 DONE |
| E01 — canonical execution | 9 | E01-T08 DONE; E01-T09 waits for E04-T03 slice |
| E02 — persistent browser | 8 | READY после E01 |
| E03 — Host Bridge / Desktop Commander | 11 | READY после E02 |
| E04 — workspace/executor | 6 | T03 slice IN_PROGRESS (prerequisite E01-T09); остальное NEEDS_REFINEMENT |
| E05 — verification/recovery | 7 | NEEDS_REFINEMENT |
| E06 — modes/policy/UI | 6 | NEEDS_REFINEMENT |
| E07 — memory/context | 5 | NEEDS_REFINEMENT |
| E08 — cleanup/operations | 8 | NEEDS_REFINEMENT |
| **Всего** | **63** | **31 READY, 32 предварительный** |

**E04–E08 — не исполнимые ticket specs до refinement.** Перед соответствующим эпиком Muse доводит каждый до полного шаблона: ЧТО, ЗАЧЕМ, проверенный code touchpoint/unknown, зависимости, scope, steps, delete-vs-wrap, positive/negative/E2E acceptance, postconditions, разрешения/запреты, evidence, `make check`, CI и независимый Codex review. До этого в `STATE.md` фиксируется `BLOCKED: ROADMAP_REFINEMENT`, в том числе E07-T04 и однострочные E08.

Ни один следующий тикет не начинается до merge предыдущего. Это прямо следует из рабочего соглашения Muse.

### Epic E00 — baseline после merge PR #151 и разделение worktree

#### E00-T01 — read-only baseline и разделение двух worktree

**ЧТО.** Подтвердить успешный merge PR #151, актуальный `main` и существование ровно двух согласованных linked worktree. **Выполнено 2026-11-16:** PR #151 merged на `470a9a63280f9dac392f559a6fe99115db2e50d0`; второе дерево `/home/ki/Project/Slavikai-rebuild` создано от этого SHA, служебная ветка `muse/slavikai-rebuild`. Muse не выполняла там implementation.

**ЗАЧЕМ.** Исключить путаницу между исходным проектом владельца и специальным рабочим деревом Muse.

**Dependencies.** PR #151 завершён и больше не blocker; исходное дерево остаётся за владельцем.

**Steps.** Read-only `git status --porcelain=v1`, `git worktree list --porcelain`, `git rev-parse` в обоих деревьях, remote `main`, GitHub статус PR #151. Не переключать ветки исходного дерева.

**Acceptance.** В [STATE.md](STATE.md) оба точных пути и branch/HEAD, запрет работы Muse в исходном дереве, запись merge PR #151 и исходный SHA.

**Negative.** Muse не меняет исходный worktree, не чистит untracked, не создаёт третье дерево и не начинает параллельный тикет.

**Delete-vs-wrap.** Не применимо.

**Разрешено.** Read-only проверка и фиксация evidence.

**Запрещено.** `reset/clean/stash`, удаление чужих веток, изменение PR #151.

**Evidence.** `git worktree list`, exact local/remote main SHA, PR #151 merged URL и commit.

#### E00-T02 — канонические ROADMAP, STATE и рабочее соглашение в GitHub

**ЧТО.** Опубликовать `ROADMAP.md`, `STATE.md`, `slavikai-how-to-work-with-muse.md`, `AGENTS.md` и `docs/workflow/dev_workflow.md` в одном docs-only PR #152; убрать весь внутренний ChatGPT citation markup, закрепить два worktree и согласовать merge workflow и Outcome с ADR-0002.

**ЗАЧЕМ.** Muse, Codex и владелец должны читать согласованный source of truth непосредственно в репозитории.

**Dependencies.** E00-T01 завершён. Docs PR #152 открыт; branch создан до merge PR #151 и обязан быть синхронизирован с новым `main` перед merge. Исходный и Muse worktree не использовать для публикации документации, если можно обновить docs-ветку через GitHub.

**Steps.** Сверить новый main, обновить пять Markdown файлов в docs branch, проверить ссылки и 62 ticket IDs, провести Codex review/CI последнего commit, согласовать base (docs-branch от старого main → текущий main) без merge-коммитов.

**Acceptance.** Ровно пять Markdown docs-файлов в PR, без runtime code; ссылки относительные; 62 уникальных тикета; E00–E03 имеют исполнимые gates, E04–E08 NEEDS_REFINEMENT; нет ChatGPT citation markers; worktree и merge/Outcome правила согласованы с AGENTS, dev_workflow и ADR-0002; свежие CI+Codex review без blockers.

**Negative.** GitHub auto-merge не включён; нет изменения runtime, Makefile, tests или локальных worktree; Muse не работает в исходном дереве.

**Delete-vs-wrap.** Не применимо.

**Разрешено.** Только отдельный docs-only PR.

**Запрещено.** Merge до чистого review/CI, `make git-check` и base synchronization; auto-merge; E01+ до закрытия E00.

**Evidence.** Branch, commit SHA, PR, diffstat, citation scan, checks/review status (не заявлять заранее).

#### E00-T03 — создать architecture evidence checklist

**ЧТО.** Добавить в roadmap таблицу `claim → verified file/call path/test/e2e/unknown`.

**ЗАЧЕМ.** Не позволить предыдущим аудитам превратиться в догму.

**Acceptance.** Каждое deletion ticket ниже ссылается либо на verified production call graph, либо сначала содержит discovery step.

**Negative.** Нельзя помечать subsystem dead только по отсутствию упоминания в docs.

##### Evidence checklist — verified 2026-11-16 (baseline `fca51c2`)

**Baseline:** `/home/ki/Project/Slavikai-rebuild`, branch `muse/slavikai-rebuild`, HEAD `fca51c2`, clean tree (verified via `git rev-parse` + `git status --porcelain`). Audits were done on `pr-151` HEAD `728c04c` + 17 uncommitted files, so line numbers drifted slightly; current line numbers are reported below. Memory DB row counts are runtime data (gitignored) — verified against the **original** checkout's live DBs, since the rebuild tree has no runtime DBs (fresh worktree).

###### Evidence table (40 claims)

| # | Claim | Status | File:line / call path (fca51c2) | Test / E2E evidence | Notes |
|---|---|---|---|---|---|
| 1 | MWV `VerifierRuntime.run` always returns ERROR | verified | `core/mwv/verifier_runtime.py:304` `return sandbox_verification_unavailable(command or [])`; `core/mwv/manager.py:200-204` `_is_success` requires `work==SUCCESS and verification==PASSED` | No prod test without mock (audit: tests mock VerifierRuntime) | MWV PASS impossible in principle; `max_retries`/`MAX_MWV_ATTEMPTS` dead params |
| 2 | Chat MWV path gives fake DONE without action | verified | `core/agent_mwv.py:450` one step, `allowed_tool_kinds=[]`; `:1228` `step.result = f"Выполнен: {step.description}"` | none | `grep -c "generate(" core/agent_mwv.py` = 0 → no LLM in chat MWV path |
| 3 | `classify_request` is legacy, ignores context | verified | `core/mwv/routing.py:22-23` comment "Legacy chat/MWV gate only"; `:90` `_ = context` | none | Call sites: `core/agent_routing.py:346,539`, `server/http/handlers/ui_chat.py:242` |
| 4 | Skill-match forces route=mwv | verified | `core/mwv/routing.py:97-98` matched → `_add_flag(flags, "skill")`; `:136-143` `if flags:` → `RouteDecision(route="mwv")`; `docs/agent/ROUTING_POLICY.md:60` matched → always MWV | none | Deprecated/ambiguous → direct mwv (`:99-108`); matched → skill flag → `if flags` → mwv. Audit claim stands; W3 withdrawn after Codex review |
| 5 | Chat `plan`/`act` are the same code; fresh Agent defaults to `act` | verified | `core/agent_routing.py:304` ask, `:306` auto, `:343` desktop — no plan/act branches; fall-through to `classify_request` at `:346`, `:539` (stream `:474/:482/:519` same); `core/agent.py:343` `self.runtime_mode = "act"` | none | Real plan/act live only in UI workflow (`/ui/api/plan/*`), not in chat |
| 6 | One `AgentToolLoop` in chat/auto/desktop (KEEP) | verified | `core/agent_routing.py:803,892`; `core/auto_runtime.py:292`; `core/desktop_runtime.py:343` | none | Canonical loop candidate for E01 |
| 7 | Auto: fake PASSED + planner/shard/merge theater | verified | `core/auto_runtime.py:698,711` `verifier_profile="response_only"`; `:741` `tool_outcomes`; `:266` single shard `tool-loop`; `:176,316` `merge` status idle/completed | none | Budgets real per audit (not re-read) |
| 8 | `MWVSingleAttemptRuntime` / `CodingSkill` / `CodingTaskRuntime` test-only | verified | Defined in `core/mwv/single_attempt.py`, `coding_skill.py`, `coding_task.py`; 0 imports in `core/server/tools`; only `tests/test_coding_skill_e2e.py`, `tests/test_mwv_single_attempt_flow.py` | tests only | Delete candidates E08-T01 |
| 9 | `SessionContext` / `ExecutionPolicy` are phantoms | verified | `grep -rn` → 0 matches in `core server tools llm memory shared config` | — | Docs-only entities |
| 10 | `ExecutionMode` ask≡plan in registry | verified | `tools/tool_registry.py:28` `Literal["ask","plan","act","auto","desktop"]`; `:390` ask → read-only; `:392` plan → read-only (identical rule); plan guard `:403-408` (`plan_id`/`plan_hash`) | none | — |
| 11 | KEEP anchors: Desktop `final_gate` + double `_verify`; UI plan→act workflow | verified | `core/desktop_runtime.py:335-336` `_final_gate`→`_verify`, `:356`, `:391` second `_verify`, `:412` def; `server/http/common/workflow_runtime.py:698` `run_plan_runner`; `server/http/handlers/plan.py:678`; `core/agent_mwv.py:895` | none | Only working verification in repo; plan workflow is the real plan mechanism |
| 12 | Fake execution surface: `shell`, `terminal_run`, `workspace_run` | partial | `tools/shell_tool.py:75` `return sandbox_execution_failure()`; `tools/terminal_tool.py:138` same, `:148/:158/:167` `raise SandboxExecutionUnavailable`; `core/agent_tools.py:1091-1096` `workspace_run` refusal is ONLY the manual legacy entrypoint (comment: new executor via gateway) | tests assert refusal (audit) | `workspace_run` has a real qualified native path: `core/agent.py:578` registers `RunCodeTool(backend=self._python_backend)`; `tools/workspace_tools.py:735-745` calls `backend.execute_python` with managed binding. E08-T02 must NOT delete the qualified native executor — only fence/remove the legacy entrypoint in `Agent.call_tool` |
| 13 | `fs` duplicates `workspace_*`, wider root, exec capability | verified | `tools/filesystem_tool.py:9` `SANDBOX_ROOT = Path("sandbox")`, `:13` mkdir side effect; `core/agent.py:476` `capability="exec"` | none | Root `sandbox/` wider than `sandbox/project`; bypasses workspace budget/`AutoFileEffects` |
| 14 | `project` rebuilds `VectorIndex` per call | verified | `tools/project_tool.py:39-48` new `VectorIndex("memory/vectors.db")` + `close()` in `finally`, inside `handle_project_request` | none | Duplicates `workspace_list/read` + vector search |
| 15 | `create`≈`write`, `rename`≈`move` | partial | `tools/workspace_tools.py:514` Write vs `:536` Create; `:563` Rename vs `:601` Move | none | `create` ≠ `write`: `CreateFileTool` defaults `overwrite=False` and refuses existing files before writing (`:540,551-552`); `WriteFileTool` unconditionally overwrites (`:528`). Merge must preserve create-only semantics in the acceptance of the merged tool (E08-T03) |
| 16 | Browser: no persistent context | verified | `tools/desktop_browser_tool.py:305` `chromium.launch`, `:306` `new_context`; 0 matches for `launch_persistent_context` | none | E02-T01/T02 dependency |
| 17 | Gateway bypass: workspace read path | verified | `server/http/handlers/workspace.py:593` direct `tool.handle(ToolRequest(...))`, called at `:701`, `:788` | none | Second read path without approval/log/budget |
| 18 | Container backend: unqualified mode touches host directly | verified | `core/container_computer_backend.py:120-175`: qualified → `_qualified_file_call`; else `_ensure_in_project` + direct host access | none | Qualified mode goes via gateway (2nd-audit correction confirmed) |
| 19 | `SAFE_MODE_TOOLS_OFF` contains phantoms | verified | `core/agent.py:129-141`: `web_search`, `http_client` listed, no such registered tools | none | — |
| 20 | `update_tools_enabled(persist=True)` TypeError mechanism | partial | `core/agent.py:765-769` `save_tools_config(ToolsConfig(**self.tools_enabled))`; `DEFAULT_TOOLS` (`agent.py:117-128`) contains `image_analyze`/`image_generate`/`workspace_run`; `ToolsConfig` fields (`config/tools_config.py:20-28`) lack them (but have dead `img`) | none | `save_tools_config` filter not re-checked; crash path end-to-end not executed |
| 21 | `DesktopLaunchTool` gives fake success on instant crash | **wrong** | `tools/desktop_system_tools.py:298-303`: immediate `poll()`; if exited → `failure("Process exited before launch verification.")` | none | 1st audit §3.4 wrong; 2nd-audit correction confirmed. Residual: sub-second window + DEVNULL output remain |
| 22 | `local-default` phantom model | verified | `server/http/common/ui_settings.py:422,433` fallback `os.getenv("LOCAL_LLM_DEFAULT_MODEL", "local-default")` | none | Unreachable endpoint returns a fake model instead of an error |
| 23 | `api_key_valid: True` badge without check | verified | `server/http/handlers/settings.py:636` `"api_key_valid": True if key_is_saved else None` | none | Real check exists elsewhere (audit: `ui_settings.py:700-730`, not re-read) |
| 24 | `ChatGPTPlanBrain`: no retry at all | verified | `grep -c RetryPolicy llm/chatgpt_plan_brain.py` = 0 | none | All other brains have retry (audit) |
| 25 | Direct `requests.post` in 4 providers (11 places) | verified | deepseek 2, openrouter 2, xai 4, inception 2, local_http 1 | none | `provider_http` used only by `ChatGPTPlanBrain` + `LocalHttpBrain` (audit) |
| 26 | `mercury-2` fake catalog | verified | `server/http/common/ui_settings.py:103` `INCEPTION_DOCS_MODELS = ["mercury-2"]`; `:408` returned directly | none | Mercury API closes 2026-12-28 (audit) |
| 27 | `BrainManager` is a delegating stub | verified | `llm/brain_manager.py:11-18`: single `build()` → `create_brain`, no storage despite docstring | none | — |
| 28 | `_inject_system` / parser duplication across brains | verified | `_inject_system` occurrences: deepseek 3, openrouter 3, xai 4, inception 2, local_http 3 | none | ~500–600 lines dupes is the audit's estimate, not re-measured |
| 29 | Default model broken (frontend hardcode + no persisted model) | partial | `ui/src/app/components/Settings.tsx:383` `defaultModelProvider: 'deepseek'`; `config/model_store.py:45-48` `load_model_configs` returns `None` when file missing; `core/agent.py:432-434` `RuntimeError("Не выбрана модель...")` | none | File part is stale (see W2): `config/model_config.json` absent in main, not `{}` — outcome identical (`None` → error) |
| 30 | Memory empty + junk vectors | verified | Original checkout live DBs: `canonical_atoms` 0, `memory` 0, `memory_item` 2, `vectors` 120 (all `length(embedding)=12`), `policy_rule` 0, `feedback_event` 0, `batch_review_run` 0, `interaction_log` 25 | none | Rebuild tree has no runtime DBs. Count drifted 117→120 since audit; all rows still 12-byte and dead for search |
| 31 | Silent context deletion, no live compaction | verified | `core/agent_tools.py:88` `MAX_SHORT_TERM_MESSAGES = 20`; `:315-319` `del history[:overflow]`; `grep -rni "compact"` finds only `_compact_summary_text` (summary shortening) and JSON `compact_value` — no history compaction | none | — |
| 32 | `BatchReviewer` / policy contour test-only | verified | `core/agent_memory.py:119` def `run_batch_review`; 0 callers in `core/server` outside tests | tests only | No endpoint/button/cron (audit) |
| 33 | `FeedbackManager` dead | verified | 0 imports in `core/server/tools`; `memory/feedback.db` absent | none | Duplicates companion `feedback_event` |
| 34 | Inbox/triage is write-only | verified | `list_items` consumers only UI pages (`server/http/handlers/memory.py:308`, `memory_management.py:90`); no context reader; store constructed `core/agent.py:308-310` | none | Path inbox → category → nowhere |
| 35 | `decide_request` fail-open for unknown tools | verified | `core/approval_policy.py:186-187` `if not intents: return ApprovalDecision("allow", "no_intent", [], [])` | none | Tool without `risk_classes` bypasses approval in safe_mode |
| 36 | Claims registry: gate checks existence only; counts 54/4/6/1; `openai_subscription` still `target`; 9 `read_view` / 19 projections | verified | `scripts/check_source_of_truth.py` 122 lines, imports `json`/`pathlib` only, `:42` path-exists check, no subprocess/pytest; JSON counter `{'implemented': 54, 'target': 6, 'partial': 4, 'legacy': 1}`; `docs/runtime_contract_claims.json:280-283` `openai_subscription` status `target`; python count: 9 `read_view` implemented | `make check` runs the gate (audit) | Claim stale post-merge: PR #151 merged, claim still `target`. 19 total projections = 10 settings + 6 ui + 1 session + 2 config (audit) |
| 37 | `lane` as domain discriminator | verified | `server/ui_session_storage.py:104,105,151,152,569` (`_normalize_legacy_lane`, `_message_with_legacy_lane`, `_message_table_for_lane`) | none | Canon forbids lane driving storage/API/frontend; code does exactly that |
| 38 | `/v1/chat/completions` is a fake | verified | `server/http/handlers/chat.py:110-113` plan\|act → 400; `:125-130` `stream=true` → 400 `streaming_not_supported`; single proxy model `PUBLIC_PROXY_MODEL_ID = "slavik"` (`server/http/common/proxy_model.py:5`) | none | Non-streaming single-model proxy, not OpenAI-compatible surface |
| 39 | Deploy service example dead | verified | `/opt/slavikai` absent; `deploy/slavikai.service.example:10,14` `WorkingDirectory`/`ExecStart` → `/opt/slavikai` | none | — |
| 40 | Routes 113; `/slavik/*` dupes; dead UI components; fake legacy-UI guard | verified | `grep -c add_* server/http/routes.py` = 113; `:59-60` `/ui/api/runtime/init` = `/ui/api/init`; 5 `slavik/` refs; `ChatArea.tsx`, `ChatCanvas.tsx`, `Sidebar.tsx`, `figma/ImageWithFallback.tsx` exist with 0 importers in `ui/src`; `scripts/check_no_legacy_ui.sh` checks only `ui/src/legacy` + `ui_archive/legacy-ui` | `make check` runs the guard (audit) | — |

###### Also verified (minor, outside the table)

- `config/settings.py` `AppSettings`: 0 references outside its own file + its test — dead.
- `_COMPAT_EXPORTS` exists (`core/agent.py:148`); `img` key validated in `server/ui_hub.py:996` and present as a `ToolsConfig` field but never consumed as a tool.
- `stt`: 0 `call_tool("stt")` callers in `core/server`.
- SSE pseudo-streaming: `server/http/common/streaming.py:194` `_publish_chat_stream_from_text` → `:204` `_publish_chat_stream_delta` (replay semantics per audit's read).
- `Agent.__init__` constructs multiple memory stores (`core/agent.py:306` MemoryManager, `:308-310` CategorizedMemoryStore, `:314` VectorIndex, plus CanonicalAtomStore/ClaimExtractor/SessionSummarizer imports).
- Gate requires `LOCAL_LLM_URL` in `.env.example` (`scripts/check_source_of_truth.py:98`; `.env.example:30`) — removing it breaks `make check`; affects E08-T04.

###### Claims that turned out WRONG

- **W1 — `DesktopLaunchTool` fake success (deep audit §3.4).** Code does an immediate `poll()` after start and returns `failure("Process exited before launch verification.")` (`tools/desktop_system_tools.py:298-303`). Instant-crash fake success does not exist. The 2nd audit's correction is confirmed. Residual issue only: sub-second window + DEVNULL output.
- **W2 — `config/model_config.json` is `{}` (3 bytes).** The file does **not exist** in main at `fca51c2`. `load_model_configs` returns `None` when missing (`config/model_store.py:45-48`), so the outcome is identical (no default → `RuntimeError`), but the file claim is stale — it described a working-tree local file on the pr-151 branch, not repo content. E01 tickets should cite "missing file → None", not "empty file".

###### Unknown / not re-verified in this pass (audit-only)

- **U1** — "113 test files mention MWV" / tests mock `VerifierRuntime`: count not re-run (low value; E01-T08's grep test will re-derive it).
- **U2** — `AgentComputerRuntime` / `LocalComputerBackend` have zero callers of `make_computer_runtime`: not re-verified.
- **U3** — OpenRouter/Inception `raise` on tools: files exist and direct transport confirmed; the raise statements themselves not re-read.
- **U4** — SSE replay line-by-line behavior: function exists; "slices ready text" per audit's read.
- **U5** — `save_tools_enabled` filter behavior: `ToolsConfig(**self.tools_enabled)` with extra keys is confirmed in code; whether `save_tools_config` filters keys before construction was not checked, so the TypeError path is mechanism-confirmed, not end-to-end executed.

###### Consequences for the deletion tickets

- **E01-T08** (remove legacy classifier/MWV routing): fully grounded — claims 2, 3, 4, 5, 8, 10 are verified; no discovery step needed beyond the E01-T01 call-graph matrix.
- **E08-T01** (MWV/legacy removal): grounded — claims 1, 2, 8, 37; plus the registry claim (36) needs a fix (claim `openai_subscription` → `implemented`) as part of the cleanup, not a deletion.
- **E08-T02** (dead/fake shell + PTY): grounded with a correction — claim 12 is partial: `shell`/`terminal_run` are fake, but `workspace_run` has a real qualified native executor (`RunCodeTool` + `backend.execute_python`) that must be kept; only the fenced legacy entrypoint in `Agent.call_tool` is a removal candidate. Note W1 does **not** affect this ticket (`DesktopLaunchTool` is not part of it).
- **E08-T03** (duplicate workspace tools): grounded with a correction — claims 13, 14 verified; claim 15 is partial: `create` ≠ `write` (create-only semantics must be preserved in the acceptance of the merged tool).
- **E08-T04** (provider cleanup): grounded with two corrections — claim 29's file part is stale (use "missing file → None"), and the `LOCAL_LLM_URL` gate requirement must be handled together with any `.env.example` change.
- **E07 memory tickets**: grounded — claims 30–34 verified; vectors count is now 120, not 117.

### Epic E01 — canonical execution contract

#### E01-T01 — inventory production execution entrypoints

**ЧТО.** Трассировать production entrypoints для ask, plan, act, auto, desktop до `AgentToolLoop`, MWV или другого executor.

**Verified touchpoints.** `core/agent_routing.py`, `core/tool_loop.py`, `core/desktop_runtime.py`, `core/tool_gateway.py`, `tools/tool_registry.py`.

**ЗАЧЕМ.** Перед удалением режимов получить доказанный call graph.

**Steps.** Для каждого public route/UI action записать entry handler → Agent method → loop/runtime → gateway → executor → outcome.

**Acceptance.** Machine-readable или markdown matrix; каждый path помечен production/test-only/dead/unknown.

**Negative.** Search-only evidence без call-path не считается достаточным.

**Delete-vs-wrap.** Ничего не удалять в этом тикете.

##### Execution entrypoints matrix — verified 2026-11-17 (baseline c5e5543)

**Baseline:** `/home/ki/Project/Slavikai-rebuild`, branch `pr-e01-t01-entrypoints`, HEAD `c5e5543`, clean tree. Read-only research — no code was changed.

**Method:** every row traces an actual call chain by reading the code (entry handler → Agent method → loop/runtime → gateway → executor → outcome). A name match alone was never enough. Cross-checked against the E00-T03 evidence checklist already in `ROADMAP.md` (claims 1–12, 15, 17, 36) — those verified facts are reused where they apply, but each call path below was re-verified by reading the code for this ticket.

**Status key:**
- `production` — reachable in the running server/UI and actually does what the row says.
- `production (broken)` — reachable and runs, but a core part of the outcome is fake/broken (named in the row).
- `dead` — reachable as an HTTP endpoint but the execution never happens (refused, rejected, or disabled by design).
- `test-only` — only reachable from tests (none found among the HTTP entrypoints; see the notes).
- `unknown` — could not be determined from the code read; the reason is given.

**The one loop:** there is a single `AgentToolLoop` class (`core/tool_loop.py:133`), called from exactly three fresh-start production sites: chat (`core/agent_routing.py:803`, read-only tools; stream variant `agent_routing.py:892`), auto (`core/auto_runtime.py:292`), desktop (`core/desktop_runtime.py:343`). A fourth instantiation (`core/agent_tools.py:586`) only resumes the chat loop via `resume_chat_approval` after an approval (reached through `/ui/api/decision/respond`) — the same chat loop, not a new execution path. Everything else is a different runtime (MWV manager/worker/verifier, or a direct tool call).

###### Matrix

| # | Entrypoint | Mode | Entry handler | Agent method | Loop / runtime | Gateway | Executor | Outcome | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `POST /ui/api/chat/send` (stream or not) | ask (UI «Обсудить») | `server/http/handlers/ui_chat.py:1948` `handle_ui_chat_send` | `Agent.set_runtime_state` via `_apply_agent_runtime_state` (`ui_chat.py:1090,2180` → `core/agent_tools.py:730`) → `Agent.respond_stream` (`core/agent_routing.py:411`) / `Agent.respond` (`core/agent_routing.py:259`) | `Agent._run_chat_response` (`agent_routing.py:623`) → `_run_chat_tool_loop_if_available` (`agent_routing.py:792`); stream variant `_run_chat_response_stream` (`agent_routing.py:833`) | `Agent._build_tool_gateway` (`core/agent_tools.py:773`) → `ToolGateway.call` (`core/tool_gateway.py:180`) | `AgentToolLoop().run` (`agent_routing.py:803`) / `.run_stream_events` (`agent_routing.py:892`) | LLM answer with tool results; no state changes | production | Read-only by construction: `_chat_read_tool_specs` (`agent_routing.py:812`) only exposes tools with `capability == "read"` and `chat_exposed`. The only honest read-only path. |
| 2 | `POST /ui/api/chat/send` | auto (UI «Агент») | `ui_chat.py:1948` | `Agent.respond` / `respond_stream` → `Agent.handle_auto_command` (`core/agent_tools.py:1287`) | `AutoAgent.run_outcome` (`core/auto_agent.py:41`) → `AutoOrchestrator.run_v1` (`core/auto_runtime.py:141`) | `ToolGateway` (built in `run_v1`, passed to the loop) | `AgentToolLoop(...).run` (`core/auto_runtime.py:292`) | Tools execute (writes allowed); verification branches in `_run_verifier` (`core/auto_runtime.py:675`): no tool calls → fake `PASSED` (`verifier_profile="response_only"`, `:685-713`); tool calls → real checks: generic workspace returns FAILED on a failed tool call (`verifier_profile="tool_outcomes"`, `:730-745`), repo workspace returns ERROR without a canonical verifier or calls `VerifierRuntime.run` (`:714-729`, `:756-775`) | production (broken) | «planner / coders / merge» are state labels only (E00-T03 claim 7); the real work is one `AgentToolLoop`. No real postcondition check when no tools were called; after tool calls the verifier does check outcomes (FAILED on tool error, ERROR without a canonical verifier, `VerifierRuntime.run` otherwise). |
| 3 | `POST /ui/api/chat/send` | desktop (UI «Этот компьютер») | `ui_chat.py:1948` | `Agent.respond` / `respond_stream` → `Agent._run_desktop_response` (`core/agent_routing.py:725`) | `DesktopRuntime.run` (`core/desktop_runtime.py:150`) → `_run_with_lease` (`desktop_runtime.py:295`) | `parent._build_tool_gateway()` (`desktop_runtime.py:317`) → `ToolGateway` | `AgentToolLoop(...).run` (`desktop_runtime.py:343`) with `final_gate=_final_gate` (`desktop_runtime.py:335`) | Tools execute; verification is REAL: `_final_gate` → `_verify` (`desktop_runtime.py:412`) → `VerifierRuntime.verify_desktop_observations` (`core/mwv/verifier_runtime.py:218`), which re-observes via a later `desktop_verify` call (`_desktop_change_is_verified`, `verifier_runtime.py:68`); a second `_verify` runs after the loop (`desktop_runtime.py:391`) | production | The only path with honest verification. One active run at a time enforced by a lease (`run_coordinator.try_acquire`, `desktop_runtime.py:180`). |
| 4 | `POST /ui/api/chat/send` | plan or act (chat message — NOT the UI plan workflow) | `ui_chat.py:1948` | `Agent.respond` / `respond_stream` — there is NO plan/act branch (`agent_routing.py:304` ask, `:306` auto, `:343` desktop only) → falls through to `classify_request` (`agent_routing.py:346`; stream `:539`) | `classify_request` → if `route == "mwv"` → `_run_mwv_flow` (`core/agent_mwv.py:266`); else → `_run_chat_response` (same as row 1) | MWV path: `ToolGateway` via `_execute_mwv_plan_step` (`core/agent_mwv.py:1204`) — but the step has no operation and `allowed_tool_kinds=[]`, so no tool is ever called | MWV: `ManagerRuntime` + `WorkerRuntime` + `VerifierRuntime` (`agent_mwv.py:266-300`); the worker builds ONE step with `allowed_tool_kinds=[]` (`agent_mwv.py:430-440`) and `_execute_mwv_plan_step` marks it `DONE` without action (`agent_mwv.py:1228`, `step.result = f"Выполнен: {step.description}"`) | MWV path: fake DONE + verifier always `ERROR` (`core/mwv/verifier_runtime.py:304`) → `verifier_fail` decision packet — BUT only for requests without action-oriented risk_flags; with risk_flags (`code_change`/`filesystem`/`git`) the worker refuses first with ERROR + `REPLAN_REQUIRED` (`core/agent_mwv.py:659-675`, `_mwv_step_requires_operation` `:1247`, `_resolve_packet_step_operation` `:1176`) and `_execute_mwv_plan_step` is never called. Non-MWV path: same as ask (row 1) | production (broken) | THE TRAP: chat «plan»/«act» do NOT run the UI plan workflow. The real plan/act live only in `/ui/api/plan/*` (rows 10–13). A fresh `Agent` defaults to `runtime_mode = "act"` (`core/agent.py:343`), so any path that skips `set_runtime_state` lands in `classify_request`. The fake DONE is conditional: it only happens when no explicit operation is required; action-oriented requests are refused by a real guard (ERROR + `REPLAN_REQUIRED`) — a guard worth keeping when legacy routing is removed. |
| 5 | `POST /ui/api/chat/send` | «/» command | `ui_chat.py:1948` | `Agent.respond` → `handle_tool_command` (`core/agent_tools.py:243`) — any message starting with `/` is routed here before the mode dispatch (`agent_routing.py:277-278`) | none (command lane, no loop) | none | none | `/trace` → tracer log; `/end-session` → session-end packet; any other `/command` → «неактивна и отключена» | production | Only `/trace` and `/end-session` work. `/fs`, `/web`, `/sh`, `/project`, `/plan`, `/auto`, `/imggen`, `/imganalyze` are all disabled by design. |
| 6 | `POST /v1/chat/completions` | ask (default) | `server/http/handlers/chat.py:47` `handle_chat_completions` | `Agent.set_runtime_state(mode=runtime_mode)` (`chat.py:289`) → `Agent.respond` (`chat.py:309`) | same as row 1 (`_run_chat_response` → read-only `AgentToolLoop`) | `ToolGateway` | `AgentToolLoop` (read-only tools) | LLM answer; no state changes | production | OpenAI-compatible API. `runtime_mode` defaults to `"ask"` (`server/http/common/chat_request.py:134`). |
| 7 | `POST /v1/chat/completions` | auto | `chat.py:47` | `Agent.set_runtime_state(mode="auto")` (`chat.py:289`) → `Agent.respond` with `auto_capture` (`chat.py:305`) | `handle_auto_command` → `AutoAgent` → `AutoOrchestrator.run_v1` → `AgentToolLoop` (same as row 2) | `ToolGateway` | `AgentToolLoop` | Tools execute; verification branches as in row 2 (fake `PASSED` only when no tools were called; FAILED/ERROR after tool calls). The run is durably admitted via `run_store` (`chat.py:245-270`) with a `CaptureBinding` | production (broken) | The only `/v1` mode besides ask that executes (desktop is rejected at parse — row 8). The `run_store` admission is durable, but execution is NOT recoverable from it: unfinished runs go to `RECOVERY_REQUIRED` (`core/task_run_storage.py:143-163`), recovery endpoints return evidence only (`server/http/handlers/recovery.py:18,60`), and the Auto continuation lives in-memory (`_paused_runs`, `core/auto_runtime.py:139`). |
| 8 | `POST /v1/chat/completions` | desktop | `chat.py:47` | rejected at payload parse, before any `Agent` is created | none | none | none | HTTP 400: `slavik_meta.runtime_mode` must be ask\|auto\|plan\|act (`server/http/common/chat_request.py:33` `RUNTIME_MODES`, `:147-150` error for any other mode; `handle_chat_completions` turns it into 400 at `chat.py:69-77`) | dead | Desktop is NOT reachable via /v1. Only ask and auto execute through /v1; plan/act are allowed at parse but rejected later (`chat.py:110-119`). |
| 9 | `POST /v1/chat/completions` | plan or act | `chat.py:47` | rejected before any `Agent` is created | none | none | none | HTTP 400: «runtime_mode=plan\|act для /v1 пока не поддерживается» (`chat.py:110-119`), directing to `/ui/api/plan/*` and `/ui/api/mode` | dead | Rejected by design. Confirms plan/act are UI-workflow-only. Desktop is rejected even earlier (row 8, at parse) — /v1 executes only ask and auto. |
| 10 | `POST /ui/api/plan/draft` | plan (UI «План») | `server/http/handlers/plan.py:112` `handle_ui_plan_draft` | `_apply_agent_runtime_state` → `_run_plan_readonly_audit` (`server/http/common/workspace_runtime.py:58`, calls `workspace_list` / `workspace_read` via `agent.call_tool`) → `Agent.compile_plan_steps` (`core/agent.py:154`) → `compile_structured_plan_steps` (LLM) | none (no `AgentToolLoop`; the LLM compiles steps directly) | `ToolGateway` (via `agent.call_tool` → `_call_tool_logged` → `gateway.call`) — for the read-only audit tools only | LLM (`compile_structured_plan_steps`) + read-only tools | A plan draft (`TaskPlan` payload) stored in the session workflow; no execution | production | Requires `mode == "plan"` (409 otherwise, `plan.py:121`). The audit is read-only. Drafting is planning, not execution. |
| 11 | `POST /ui/api/plan/approve` | plan | `server/http/handlers/plan.py:225` `handle_ui_plan_approve` | none (marks the plan approved in the session workflow via `hub.approve_plan_if_current`) | none | none | none | Plan status draft → approved. No execution | production | Plan mutation only. Requires `mode == "plan"` and plan status `draft` (409 otherwise). |
| 12 | `POST /ui/api/plan/edit` | plan | `server/http/handlers/plan.py:301` `handle_ui_plan_edit` | none (edits the plan payload in the session workflow) | none | none | none | Plan updated. No execution | production | Plan mutation only. Requires `mode == "plan"` (409 otherwise). |
| 13 | `POST /ui/api/plan/execute` | act (UI «Выполнение плана») | `server/http/handlers/plan.py:400` `handle_ui_plan_execute` | Spawns a background task `_run_plan_runner` (`server/http/common/workflow_runtime.py:698`) → `agent.run_task_packet` (`core/agent_mwv.py:907`) → `_run_task_packet_frame` (`core/agent_mwv.py:923`) | `ManagerRuntime.run_flow` (`agent_mwv.py:966`) with `WorkerRuntime` (`_mwv_worker_runner`) and `VerifierRuntime` | `ToolGateway` via `_execute_mwv_plan_step` (`core/agent_mwv.py:1204`) → `gateway.call` / `call_approved_once` | MWV worker executes the approved plan steps — REAL tools, REAL effects | Tools DO execute. BUT the verifier is `VerifierRuntime.run`, which always returns `ERROR` (`core/mwv/verifier_runtime.py:304`, `return sandbox_verification_unavailable(command or [])`) → the run can never reach `PASSED` → outcome is a `verifier_fail` decision (`run_result.verification_result.status != PASSED`) | production (broken) | Requires `mode == "act"` and an active task with `status == "running"` (409 `mode_not_act` otherwise). Runner name: `mwv_packet_runner`. This is the REAL act execution (unlike chat act, row 4): the tools run, but the outcome can never be honestly `PASSED` — execution and verification are decoupled in a broken way. |
| 14 | `POST /ui/api/plan/cancel` | (any) | `server/http/handlers/plan.py:788` `handle_ui_plan_cancel` | `PlanCancellationRegistry.request_cancel` + agent lookup; cancels the running plan task | none | none | none | Plan/task cancelled. No execution | production | Control plane. |
| 15 | `POST /ui/api/mode` | (sets ask/plan/act/auto/desktop) | `server/http/handlers/workflow.py:93` `handle_ui_mode` | none (sets the session workflow mode via the hub; validates against `build_mode_transitions`) | none | none | none | Session mode changed. No execution | production | Control plane. The UI calls this from `ui/src/app/use-session-runtime-controller.ts:143`. Mode labels (`ui/src/app/types.ts:104-113`): ask «Обсудить», plan «План», act «Выполнение плана», auto «Агент», desktop «Этот компьютер». |
| 16 | `POST /ui/api/runtime/init` and `POST /ui/api/init` | (resets to ask) | `server/http/handlers/workflow.py:256` `handle_ui_runtime_init` | Resolves the agent, sets `mode="ask"`, resets transient state (`reset_runtime_transient_state` → `runtime_mode = "ask"`, `core/agent_tools.py:355`) | none | none | none | Runtime initialised to ask mode. No execution | production | Init / control plane. Both routes point at the same handler (`server/http/routes.py:59-60`). |
| 17 | `POST /ui/api/decision/respond` | (approval resume) | `server/http/handlers/decision.py:153` `handle_ui_decision_respond` | Resumes based on `resume_payload`: `plan_id`/`task_id` → `run_plan_runner` (resume); `checkpoint_id` → MWV checkpoint resume; tool call → `agent.call_tool` → gateway; `project.command` → re-run | Depends on the resume target (MWV packet runner / MWV checkpoint / `ToolGateway`) | `ToolGateway` (for tool-call resumes) | Depends on the target | Resumes a paused run. Not a fresh execution | production | Approval-continuation entrypoint. It does not start a new mode execution; it continues a paused one. |
| 18 | `POST /ui/api/workspace/run` | (n/a — direct tool call) | `server/http/handlers/workspace.py:1339` `handle_ui_workspace_run` | `_call_workspace_tool` (`workspace.py:598`) → `agent.call_tool("workspace_run")` (`core/agent_tools.py:1083`) | none | none — `call_tool` intercepts `"workspace_run"` BEFORE the gateway | none — returns `sandbox_execution_failure()` (`core/agent_tools.py:1091-1096`, «Новый executor доступен native Gateway/continuation, не этому entrypoint») | Always fails (400 `workspace_run_failed`). No execution | production (dead end) | The qualified native executor (`RunCodeTool`, `core/agent.py:578` → `backend.execute_python`, `tools/workspace_tools.py:735-745`) is reachable ONLY via `AgentToolLoop` → gateway, never through this endpoint. This endpoint is a fenced manual entrypoint. |
| 19 | `POST /ui/api/workspace/terminal/run` | (n/a — direct tool call) | `server/http/handlers/workspace.py:1416` `handle_ui_workspace_terminal_run` | `_call_workspace_tool(tool_name="workspace_terminal_run")` → `agent.call_tool` → `_call_tool_logged` (`core/agent_tools.py:1102`) → `gateway.call` → `WorkspaceTerminalRunTool.handle` (`tools/workspace_tools.py:847`) → `TerminalTool.run_oneshot` (`tools/terminal_tool.py:101`) | none (direct tool call via gateway) | `ToolGateway` (yes — this one DOES go through the gateway) | `TerminalTool.run_oneshot` → `return sandbox_execution_failure()` (`tools/terminal_tool.py:138`) | Always fails. No execution | production (dead end) | Unlike row 18 it passes the gateway, but the tool itself always refuses. |
| 20 | `POST /ui/api/terminal` (create), `GET /ui/api/terminal`, `GET /ui/api/terminal/stream`, `POST /ui/api/terminal/input`, `POST /ui/api/terminal/resize`, `POST /ui/api/terminal/close` | (n/a — PTY) | `server/http/handlers/terminal.py:92` (create), `:136`, `:154`, `:215`, `:279`, `:299` | none (no `Agent` involved) | none | none | `TerminalTool` (PTY): `create_or_get` raises `SandboxExecutionUnavailable` (`tools/terminal_tool.py`, `create_or_get`); `write_input` / `resize` raise it too | Always 503 `execution_unavailable`. No PTY ever starts | production (dead end) | Requires the YOLO profile (`_require_terminal_yolo`, `terminal.py:97`), but even in YOLO the terminal raises `SandboxExecutionUnavailable`. The UI terminal is a dead surface. |
| 21 | `GET/PUT/POST/DELETE /ui/api/workspace/file`, `GET /ui/api/workspace/tree`, `GET /ui/api/workspace/browse`, `GET /ui/api/workspace/git-diff`, etc. (workspace file ops) | (n/a — file ops) | `server/http/handlers/workspace.py` (various) | Read ops → `_call_workspace_read_tool` (`workspace.py:593`); write ops → `_call_workspace_tool` (`workspace.py:598`) → `agent.call_tool` | none | BYPASSED for reads; used for writes | The workspace tools directly (reads) or via gateway (writes) | File reads/writes. Reads bypass approval, budget, and logging | production | THE GATEWAY BYPASS (E00-T03 claim 17): `_call_workspace_read_tool` calls `tool.handle(ToolRequest(...))` DIRECTLY (`workspace.py:593`). Used for `workspace_list` (`workspace.py:701`), `workspace_read` (`workspace.py:788`), and a pre-write read (`workspace.py:874`). Write operations DO go through the gateway. |
| 22 | `POST /ui/api/tools/project` — `find` / `index` | (n/a — command) | `server/http/handlers/project.py:72` `handle_ui_project_command` | Appends `"/project ..."` to the workspace lane (`project.py:193`) → `agent.respond(llm_messages)` (`project.py:418`) → `handle_tool_command` (`core/agent_tools.py:243`) | none | none | none | Returns a «неактивна и отключена» message. `find` / `index` do nothing | dead | The `/project` command lane is disabled (only `/trace` and `/end-session` work, row 5). The endpoint accepts `find`/`index`/`github_import`, but none of the three executes: `find`/`index` hit the disabled lane, `github_import` is fenced at the clone (row 23). |
| 23 | `POST /ui/api/tools/project` — `github_import` | (n/a — command) | `project.py:72` | none (executed in-handler, no `Agent`) | none | none | `api._clone_github_repository` (`project.py:307`) — fenced | Always fails: `clone_github_repository` unconditionally returns `(False, SANDBOX_EXECUTION_MESSAGE)` (`server/http/common/github_import.py:99-105`, bound at `server/http_api.py:151`), so `project.py:312-314` takes the failure branch and `_index_imported_project` is never reached. No execution | dead (fenced) | NOT a working execution path: the clone is fenced by the sandbox contract. The approval flow for the required categories runs, but the clone itself always refuses. |

###### Non-execution routes (for completeness — no Agent/mode executor)

These are registered in `server/http/routes.py` but do not reach an execution entrypoint: `/healthz`, `/ui/api/auth/*`, `/v1/models`, `/ui/api/settings*`, `/ui/api/memory/*`, `/ui/api/sessions*`, `/ui/api/session-model`, `/ui/api/workspace/git/*`, `/ui/api/workspace/root*`, `/ui/api/workspace/index`, `/ui/api/chatgpt-plan/*`, `/ui/api/provider-instances/*`, `/ui/api/embeddings/*`, `/ui/api/tts/speak`, `/ui/api/stt/transcribe`, `/ui/api/desktop/approvals*` (CRUD for approval rules), `/slavik/trace`, `/slavik/tool-calls/*`, `/slavik/feedback`, `/slavik/approve-session` (legacy session approval), `/ui/api/chat/events/{session_id}` (SSE stream), `/ui/api/chat/cancel` (cancellation, control plane), `/ui/api/session/security`, `/ui/api/session/paused-model-request`, `/ui/api/settings/chats/*`, `/ui/api/folders`. They are session/settings/memory/file bookkeeping, not execution.

Only three places in `server/` call `agent.respond` at all: `project.py:418`, `ui_chat.py:1450`, `chat.py:305/309` (plus `respond_stream` in `ui_chat.py:1173`). All Agent execution goes through one of those.

###### Mode map — what each UI mode actually does

| UI label | `runtime_mode` | Actual code path | Executes? | Verifies? |
|---|---|---|---|---|
| «Обсудить» | `ask` | `_run_chat_response` → read-only `AgentToolLoop` | read-only tools only | n/a (no state change) |
| «План» | `plan` | `/ui/api/plan/draft` → LLM compiles steps, read-only audit | no | n/a |
| «Выполнение плана» | `act` | `/ui/api/plan/execute` → MWV packet runner → real tools via gateway | YES | NO — `VerifierRuntime.run` always `ERROR`, a `PASSED` outcome is impossible |
| «Агент» | `auto` | `AutoAgent` → `AgentToolLoop` | YES | NO real postcondition: fake `PASSED` only when no tools were called (`verifier_profile="response_only"`); after tool calls, FAILED on tool error / ERROR without a canonical verifier |
| «Этот компьютер» | `desktop` | `DesktopRuntime` → `AgentToolLoop` + `_final_gate` + `_verify` | YES | YES — observation-based re-verification via `desktop_verify` |
| (chat message, mode plan/act) | `plan` / `act` | falls through to `classify_request` → MWV (fake) or chat | MWV branch: NO (fake DONE); chat branch: read-only | NO |
| (chat message, mode ask/auto/desktop) | as set | same as the UI rows above | as above | as above |

###### Unknowns (could not be traced)

1. **Exact tool allowlist of the auto and desktop loops.** The chat loop is proven read-only (`_chat_read_tool_specs`, `agent_routing.py:812`). The auto loop (`auto_runtime.py:292`) and the desktop loop (`desktop_runtime.py:343`) both go through the `ToolGateway` with the session's enabled tools, but I did not enumerate the precise tool set each one exposes (the `tool_specs` construction in `run_v1` and the desktop gateway's tool list were not read line-by-line). The key fact — both execute real tools via the gateway — is verified; the per-mode allowlist is not.
2. **Mode transition table.** `handle_ui_mode` validates the requested mode against `build_mode_transitions` (`server/http/handlers/workflow.py:93`), but I did not read the transition table itself, so I cannot state exactly which mode switches the UI allows or forbids (e.g. whether plan → act requires an approved plan — the plan endpoints enforce that separately with 409s). The modes themselves are all reachable; the transition rules are not traced.

###### Resolved during Codex review (was unknown #2)

**Durable run recovery for `/v1` auto — resolved as ABSENT.** The `/v1` auto path admits a durable run via `run_store` (`chat.py:245-270`) with a `CaptureBinding`, but that does NOT make execution recoverable: on server restart `TaskRunStore._mark_unfinished_for_recovery` moves unfinished runs to `RECOVERY_REQUIRED` (`core/task_run_storage.py:143-163`, reason `server_restart_requires_reconciliation`), the recovery endpoints return evidence only (`server/http/handlers/recovery.py:18,60`), and the Auto continuation lives in-memory (`_paused_runs`, `core/auto_runtime.py:139,643-652`). The store gives durable admission/evidence, not execution recovery.

###### Surprising findings

1. **Chat «plan»/«act» are a trap.** There is no plan/act branch in `Agent.respond` — only ask/auto/desktop (`core/agent_routing.py:304,306,343`). A chat message sent while the session mode is plan or act falls through to the legacy `classify_request` and then to the fake MWV flow or to the plain chat response — with one guard: for action-oriented requests (risk_flags `code_change`/`filesystem`/`git`) the worker refuses with ERROR + `REPLAN_REQUIRED` before any fake DONE (`core/agent_mwv.py:659-675`). The real plan/act live only in the `/ui/api/plan/*` workflow endpoints. A fresh `Agent` defaults to `runtime_mode = "act"` (`core/agent.py:343`), so any code path that constructs an `Agent` and calls `respond` without `set_runtime_state` lands in the classifier, not in ask.
2. **One loop, three honesties.** The single `AgentToolLoop` is called from three sites with three different verification regimes: chat is read-only (no verification needed), auto executes with a fake `PASSED` only when no tools were called (after tool calls it checks outcomes — FAILED on tool error, ERROR without a canonical verifier), desktop executes with real observation-based verification. Same loop, same gateway — the difference is entirely in the wrapper.
3. **The plan→act path executes but can never pass.** `run_task_packet` genuinely runs the approved plan steps through the `ToolGateway` (real tools, real effects), yet the outcome can never be `PASSED` because the MWV verifier is `VerifierRuntime.run`, which unconditionally returns `sandbox_verification_unavailable` (`core/mwv/verifier_runtime.py:304`). Execution and verification are decoupled in a broken way: the work happens, the verdict is always an error.
4. **Two methods, opposite honesty, same class.** `VerifierRuntime.run` (used by all MWV flows) always returns `ERROR`; `VerifierRuntime.verify_desktop_observations` (used only by `DesktopRuntime`) is a real verifier that re-observes state changes via later `desktop_verify` calls. The desktop mode is the only honest path in the repo.
5. **`workspace_run` is real but unreachable from the UI.** The qualified native executor (`RunCodeTool` → `backend.execute_python`) exists and is reachable only via `AgentToolLoop` → `ToolGateway`. The UI endpoint `/ui/api/workspace/run` always fails because `Agent.call_tool` fences the manual entrypoint (`core/agent_tools.py:1091-1096`). The endpoint and the executor share a name but are different paths.
6. **The workspace read path bypasses the gateway.** `_call_workspace_read_tool` (`server/http/handlers/workspace.py:593`) calls `tool.handle()` directly — no approval policy, no budget, no logging. Writes go through the gateway; reads do not. This is a live second read path, not a legacy shim.
7. **All three `/project` commands are dead.** `find`/`index` end up in the disabled command lane («неактивна и отключена»); `github_import` is executed directly in the handler without the `Agent`, but its clone is fenced — `clone_github_repository` unconditionally returns `(False, SANDBOX_EXECUTION_MESSAGE)` (`server/http/common/github_import.py:99-105`), so `_index_imported_project` is never reached. No project command executes anything.
8. **The PTY terminal is a dead surface.** All `/ui/api/terminal/*` operations raise `SandboxExecutionUnavailable` (503), even in YOLO mode. The same holds for `/ui/api/workspace/terminal/run` (the tool always returns `sandbox_execution_failure`).
9. **`/v1/chat/completions` exposes only two modes for execution.** It refuses plan/act with 400 (`chat.py:110-119`) and rejects desktop even earlier, at payload parse (`chat_request.py:33,147-150` — only ask|auto|plan|act are allowed). So only ask and auto execute through /v1; plan/act exist only as a UI workflow, desktop only in the UI chat.
10. **The approval-resume endpoint is the only way back into a paused run.** `/ui/api/decision/respond` is the single entrypoint that continues a paused MWV packet, a paused auto/desktop run, or an approved tool call. There is no other way to resume; a fresh `respond` does not pick up a paused continuation.

###### E01-T01 acceptance check

- Every production path above cites an actual call chain (entry handler → Agent method → loop/runtime → gateway → executor → outcome) verified by reading the code. No row rests on a name match or a grep hit alone.
- Every path is marked `production`, `production (broken)`, or `dead`. No production execution path was found that is `test-only` (the test-only runtimes from E00-T03 claim 8 — `MWVSingleAttemptRuntime`, `CodingSkill`, `CodingTaskRuntime` — have no HTTP entrypoint at all and do not appear in this matrix).
- The two `unknown` items above are the only gaps; each names what was not traced and why. One former unknown (durable run recovery for `/v1` auto) was resolved during Codex review as ABSENT — see the Resolved section.

#### E01-T02 — ввести immutable RunSpec/RunPolicy contract

**ЧТО.** Добавить минимальный run-level contract `intent/target/policy/budget`.

**ЗАЧЕМ.** Разделить то, что сейчас смешано в ExecutionMode.

**Implementation.** Использовать существующие типы где возможно; не создавать hierarchy framework. Policy snapshot immutable на протяжении run.

**Acceptance positive.** Два одновременных runs с разными target/policy не влияют друг на друга.

**Acceptance negative.** Mutation global ToolRegistry mode во время другого run не меняет его effective policy.

**Delete-vs-wrap.** Существующие mode fields временно остаются только как callers; новый contract становится source of truth.

**Forbidden.** Изменение пользовательского UX в этом PR.

##### E01-T02 implementation — verified 2026-11-17 (baseline 58ce9f4)

**Status:** DONE. `make check` green on branch `pr-e01-t02-runspec` (2478 passed, coverage 83.52%), incl. the Codex review P2 fix (recursive freeze of nested plan/task data).

**Files.**
- New: `core/run_contract.py` — `RunIntent` (`"answer" | "plan" | "execute"`), `RunBudget` (frozen; `loop_kwargs()` → `AgentToolLoop` kwargs, `expires_at_monotonic` from `started_monotonic + max_runtime_seconds`), `RunPolicy` (frozen; recursively freezes `active_plan`/`active_task` in `__post_init__` — `FrozenDict`/`FrozenList`, dict/list subclasses so `ToolRegistry._plan_guard_error` `isinstance` checks still work), `RunSpec` (frozen: `intent/target/policy/budget`), `intent_for_mode` / `target_for_mode` / `build_run_spec`. Reuses `ExecutionMode`/`ExecutionTarget` from `tools/tool_registry.py`; no new hierarchy or framework. Safe mode deliberately NOT in the snapshot (global user setting, enforced by the registry independently).
- `tools/tool_registry.py` — `call(..., policy=None)` and `list_tool_specs(*, policy=None)` resolve mode/target/plan-guard from the snapshot when given, else the global state (unchanged legacy behaviour). `RunPolicy` imported only under `TYPE_CHECKING` (no runtime cycle: `run_contract` imports the mode/target types from here).
- `core/tool_gateway.py` — `run_policy: RunPolicy | None = None` field (preserved by `replace()`); `call()` passes `policy=self.run_policy` to both `registry.call()` sites.
- `core/agent_tools.py` — `_run_spec(*, budget=None)` derives the spec from runtime state; `_build_tool_gateway(..., budget=None)` builds the spec internally and attaches `run_policy=spec.policy`. The `set_execution_policy` call is REMOVED from the run path — the global registry mode is no longer mutated when a run starts. The only remaining `set_execution_policy` caller is `reset_runtime_transient_state` (resets to ask, intentional).
- `core/auto_runtime.py` — builds one `RunBudget` from `AutoBudgets`, passes it to `_build_tool_gateway(budget=...)`, lists specs with `policy=gateway.run_policy`, builds the loop from `run_budget.loop_kwargs(started_monotonic=...)`.
- `core/desktop_runtime.py` — `RunBudget(max_iterations=self.max_iterations)`, same wiring; `list_specs(policy=gateway.run_policy)`.
- Chat sites (`core/agent_routing.py`) unchanged: `_build_tool_gateway()` with no args builds the spec internally (default budget); the loop stays `AgentToolLoop()`.
- TYPE_CHECKING stubs (`core/agent_routing.py`, `core/agent_mwv.py`, `core/agent_memory.py`) and `DesktopAgentProtocol` declare the new `budget` param so mypy accepts the calls.
- Test doubles updated to the new interface (fake `_build_tool_gateway` methods and `_OkRegistry`/`_BoomRegistry` accept `budget`/`policy`; 8 `_build_tool_gateway` monkeypatch lambdas accept `budget=None`).

**Tests.** `tests/test_run_contract.py` (17 tests):
- Contract: frozen spec/policy/budget, deep-copy of plan/task, recursive freeze of nested plan/task collections (Codex review fix, see below), intent/target derivation for all 5 modes, `loop_kwargs` (default unbounded; expiry from `started_monotonic`).
- Acceptance positive (`test_two_gateways_with_different_policies_are_independent`, `test_loop_run_enforces_snapshot_policy`): ask gateway blocks the write while act gateway executes it on the same registry; same at the loop level.
- Acceptance negative (`test_global_mode_mutation_does_not_change_run_policy`, `test_loop_run_ignores_global_mode_mutation`): mutating the global registry mode mid-run / between runs does not change the run's effective policy; a gateway without a snapshot (legacy path) still follows the global mode.
- Target listing (`test_list_tool_specs_uses_snapshot_target`): a desktop-only tool is listed for a desktop snapshot even when the global target is sandbox, and hidden for a sandbox snapshot when the global target is desktop.
- Agent wiring (`test_build_tool_gateway_snapshots_without_mutating_global_registry`): real `Agent._build_tool_gateway()` attaches the snapshot and leaves the global registry mode untouched.
- Verified to fail on the old code: with the production changes stashed, the 6 isolation tests fail (the 9 contract unit tests pass — `core/run_contract.py` is a standalone new module).
- Codex review fix (P2, `core/run_contract.py:97`): nested plan/task data was mutable through the snapshot — `policy.active_plan['steps'][0]['allowed_tool_kinds'].append('workspace_write')` changed the gateway's effective policy (`_plan_guard_error` reads the plan from the policy). Fixed with a recursive freeze (`FrozenDict`/`FrozenList`); a `MappingProxyType`/tuple freeze was rejected because `_plan_guard_error` requires `isinstance(active_plan, dict)` and `isinstance(steps, list)` — that freeze would have silently disabled the guard. Tests: `test_run_policy_nested_collections_immutable` (the finding's exact repro + variants) and `test_snapshot_nested_mutation_cannot_change_gateway_policy` (gateway keeps blocking the write after the failed mutation).

**Delete-vs-wrap.** Mode fields (`runtime_mode`, `runtime_active_plan`, `runtime_active_task`, `runtime_plan_guard_enabled`) remain only as the callers from which `_run_spec` derives the contract; the spec is the source of truth while the run executes. No UX changes.

**Note (design fork, resolved).** The first wiring passed a pre-built `RunSpec` into `_build_tool_gateway(run_spec=...)` and had the runtimes call `parent._run_spec(...)`. That broke the test doubles: fake parent agents (`_AutoV1Agent`, `DesktopParent`) have no `_run_spec` and their `_build_tool_gateway` takes no kwargs, and 8 monkeypatch lambdas take no args. The final wiring passes only a `RunBudget` (`_build_tool_gateway(budget=...)`) and builds the spec inside, from runtime state — the spec's budget is the same `RunBudget` the runtime uses for the loop, so the contract stays coherent, and the only test-double updates are the new optional params.

#### E01-T03 — перевести ToolRegistry listing на RunSpec

**ЧТО.** Tool availability вычисляется из target/policy/qualification, а не из глобальной mutable mode state.

**ЗАЧЕМ.** Model видит только инструменты, реально допустимые в конкретном run.

**Acceptance.** `answer` не получает write tools; `execute` получает только qualified tools target-а; unknown target fail-closed.

**Negative.** Нельзя вручную вызвать tool через registry и обойти target check.

##### E01-T03 implementation — verified 2026-11-17 (baseline c198659)

**Status:** DONE. `make check` green on branch `pr-e01-t03-registry-listing` (2484 passed, coverage 83.53%).

**Files.**
- `tools/tool_registry.py` — `list_tool_specs(*, policy=None)` now derives availability from the RunSpec: the execution target, the mode capability (ask/plan list read-only tools) and, when the plan guard is on, the current step's allowed tools. The policy check is the same `_mode_policy_error` `call()` applies, so listing and calling can never disagree. An unknown target lists nothing (fail-closed, warning `tool_list_unknown_target` logged). Without a policy (legacy: plan drafting) the listing is the sandbox catalog and never follows the global mutable target. `call(..., policy=None)` no longer uses the global target: without a policy the effective target is sandbox, so a desktop tool requires an explicit desktop policy. `RunPolicy` still imported only under `TYPE_CHECKING`.
- No other production file changed. `core/agent.py:164` (`compile_plan_steps`) keeps using the legacy listing (plan drafting needs the full sandbox catalog); the chat loop keeps its own `_chat_read_tool_specs` (read-only + `chat_exposed`).

**Tests.** `tests/test_tool_registry_listing.py` (6 tests):
- Acceptance (a) (`test_answer_spec_lists_no_write_tools`): ask and plan specs list no write/exec tools, read tools listed.
- Acceptance (b) (`test_execute_spec_lists_only_target_qualified_tools`): act/auto on sandbox list only sandbox-qualified tools; desktop lists only desktop-qualified tools (+ both-targets tools).
- Acceptance (c) (`test_unknown_target_lists_nothing_and_call_is_blocked`): a bogus target lists nothing and the call is EXECUTION_TARGET_BLOCK.
- Acceptance (d) (`test_manual_call_without_policy_cannot_bypass_target`): with the global target set to desktop, a direct no-policy call, a sandbox-policy call and a sandbox-gateway call all refuse the desktop-only tool; an explicit desktop policy allows it.
- Plan guard (`test_plan_guard_narrows_listing_to_current_step`): with the guard on, the listing is narrowed to the current step's allowed tools; a broken plan lists nothing; without the guard the write tool is listed.
- Legacy (`test_legacy_listing_is_sandbox_catalog`): the no-policy listing is the sandbox catalog whatever the global target is, and a legacy no-policy sandbox call still works.
- Verified to fail on the old code: with the production change stashed, 8 tests fail (4 new behaviour tests + 4 updated tests: snapshot target, loop policy, desktop isolation, phase2 registration); the 2 remaining new tests (execute-spec target qualification, unknown target) already passed on the old code — they pin acceptance (b)/(c) that E01-T02 already provided.
- Updated existing tests to the new listing contract: `test_run_contract.py::test_list_tool_specs_uses_snapshot_target` (legacy listing no longer follows the global target), `test_run_contract.py::test_loop_run_enforces_snapshot_policy` (the ask loop now refuses the write at the listing layer: MODEL_TOOL_NOT_EXPOSED + `tool_policy_denied`; the gateway-level ASK_READ_ONLY_BLOCK is still covered by `test_two_gateways_with_different_policies_are_independent`), `test_desktop_runtime.py::test_execution_target_isolation_between_chat_agent_and_desktop` (rewritten to drive isolation via RunPolicies + the no-policy bypass negative), `test_desktop_agent_registration.py::test_phase2_tools_are_registered_only_for_desktop_execution_target` (rewritten to use RunPolicies + a legacy-listing assertion), `test_approval_gate.py::test_confirmed_server_tool_bypasses_only_desktop_action_policy` (setup change only: the gateway now carries an explicit desktop policy instead of relying on the global mode; passes on both old and new code). The `DesktopParent` test double in `test_desktop_runtime.py` now builds a desktop policy (like the real `Agent._build_tool_gateway` when `runtime_mode == "desktop"`); the global-mode setup was removed from the desktop-loop tests and the fixture.

**Delete-vs-wrap.** No production code deleted. Mode fields remain only as callers; the spec is the source of truth for listing and calling. No UX changes.

**E08 candidates (not deleted here).**
- `ToolRegistry._execution_target` is now write-only (set by `set_execution_policy`, read by nothing on the run path) — remove once `set_execution_policy`'s last caller (`reset_runtime_transient_state`) is gone.
- `Agent._chat_read_tool_specs` (`core/agent_routing.py:812`) is a parallel hand-rolled listing (read-only + `chat_exposed`) that bypasses `list_tool_specs` — unify it with the registry listing (e.g. a `chat_exposed` filter in the listing) in a later ticket; the chat loop is the only run path not deriving its listing from the RunSpec.
- `_call_workspace_read_tool` (`server/http/handlers/workspace.py:593`) bypasses the gateway for reads — already recorded in E01-T01; E01-T04 owns the gateway unification.

#### E01-T04 — сделать ToolGateway единственной execution gate

**ЧТО.** Проверить все production tool calls; прямые model-visible calls мимо gateway перевести через gateway или доказанно удалить.

**ЗАЧЕМ.** Одна policy/approval/effect точка.

**Acceptance.** Instrumentation test подтверждает: каждый model-originated state-changing call проходит gateway.

**Negative.** Добавленный synthetic tool, вызываемый напрямую, должен падать test gate.

**Delete-vs-wrap.** Прямые обходные paths удалять, не оборачивать.

##### E01-T04 implementation — verified 2026-11-17 (baseline 637d3ed)

**Status:** DONE (acceptance met). Fork resolved by owner decision 2026-11-17 (option C); see below. `make check` green on branch `pr-e01-t04-gateway` (2488 passed, coverage 83.54%).

**Audit — every production tool call, verified by reading the code.** Model-originated calls go through `ToolGateway` on all paths:
- The one loop: `AgentToolLoop` dispatches only via `gateway.call` / `gateway.call_approved_once` (`core/tool_loop.py:840`, `:344`) — chat (`core/agent_routing.py:803,892`), auto (`core/auto_runtime.py:292`), desktop (`core/desktop_runtime.py:343`).
- `Agent.call_tool` → `_call_tool_logged` → `_build_tool_gateway` → `gateway.call` (`core/agent_tools.py:1094-1140`) — plan audit (`server/http/handlers/plan.py:172`), workspace writes (`server/http/handlers/workspace.py:623`), decision resume (`server/http/handlers/decision.py:87,884`), settings TTS (`server/http/handlers/settings.py:994`), agent TTS/STT (`core/agent.py:733,739`), confirmed memory save (`core/agent_memory.py:337`).
- MWV packet runner: `_execute_mwv_plan_step` → `gateway.call` / `call_approved_once` (`core/agent_mwv.py:1217-1221`).
- Computer backends: `LocalComputerBackend` → `gateway.call` (`core/computer_backend.py:72-95`); `ContainerComputerBackend._qualified_file_call` → `file_gateway.call` (`core/container_computer_backend.py:83-89`); `execute_local_commit` → `gateway.call` (`core/agent_computer.py:127,134`).
- `AgentComputerRuntime` / `make_computer_runtime` have zero production callers (verified by grep) — the direct-FS methods of `ContainerComputerBackend` are unreachable from model paths. Resolves E00-T03 unknown U2 as ABSENT in production.
- Verifiers (`_run_verifier`, `DesktopRuntime._verify`, `verify_desktop_observations`) analyze already-executed calls and invoke no tools.

**Proven not-a-bypass (listing-only or tool-internal):**
- `_chat_read_tool_specs` (`core/agent_routing.py:812`) — builds the chat loop's read-only ToolSpec list; execution still goes through the gateway.
- `compile_plan_steps` (`core/agent.py:164`) — passes the legacy listing to the plan compiler as `available_tools`; no execution; plan execution goes through the MWV runner → gateway.
- `DesktopBrowserTool.handle` → `backend.handle` (`tools/desktop_browser_tool.py:63`) and `DesktopProcessTool.handle` (op=launch) → `DesktopLaunchTool.start` (`tools/desktop_system_tools.py:266-267`) — tool-internal composition; `desktop_launch` is NOT a registered tool (only constructed at `core/agent.py:592` and passed to the process tool); the model-visible entry `desktop_process` is gateway-evaluated (capability exec, risk [execute]).
- `CodingSkill` / `CodingTaskRuntime` / `MWVSingleAttemptRuntime` — test-only, no production callers (confirms E00-T03 claim 8); the skill itself routes through its own gateway (`build_workspace_gateway`, `core/mwv/coding_skill.py:124`).

**Dead/fenced (not bypasses in production):**
- `Agent.call_tool("workspace_run")` fence — refuses before the gateway; the real executor is reachable only via loop → gateway.
- `handle_project_request` direct call from `github_import.index_imported_project` — reachable only after a clone that unconditionally returns False (`server/http/common/github_import.py:99-105`); both call sites (`server/http/handlers/project.py:321`, `server/http/handlers/decision.py:729`) unreachable in production. Latent if the clone fence is ever lifted.
- PTY terminal endpoints (no Agent, always 503); `/project find/index` (disabled command lane).

**Resolved fork — UI workspace reads (owner decision 2026-11-17, option C).** The single remaining live direct `tool.handle()` call is `_call_workspace_read_tool` (`server/http/handlers/workspace.py:593`): it constructs `ListFilesTool()` / `ReadFileTool()` fresh and calls `.handle()` directly, bypassing approval policy, budget, logging and RunSpec. Called from three UI handlers: tree GET (`:701`), file GET (`:788`), pre-write version check (`:874`). It is user-originated (the user's browser on click), read-only, and unreachable by the model (the model has no HTTP path to these handlers; model reads go through the loop → gateway). Tree/file GET provably work with `agent=None` (`tests/ui_api/test_workspace_and_tools.py::test_ui_workspace_tree_and_file_get_without_selected_model`): the file browser is usable before model selection, a deliberate product behavior.

**Owner decision (2026-11-17): option C.** The UI file browser is a separate user-initiated read-only path. No second ToolGateway is created, and file browsing is NOT bound to the selected model. Existing access checks are preserved as the path's own controls: the session workspace root (`_workspace_root_for_session` + `set_runtime_workspace_root`, cleared after the call), paths resolved inside the session root (`tools/workspace_tools.py::_ensure_in_workspace` — outside-workspace paths rejected, private runtime paths rejected), read size capped (`MAX_FILE_BYTES`). Rejected: (A) routing reads through `agent.call_tool` would break no-model file browsing; (B) a second agent-less read gateway would be a second gate and wrapping, not deleting. Follow-up: E08-T08 (UI read authorization audit) verifies this path's authorization explicitly instead of assuming it.

This ticket's acceptance (instrumentation test for model-originated state-changing calls) is met; the read path is NOT a model-originated call.

**Files.** No production code changed in this ticket (the read path stays as the documented option-C exception; see the resolved fork above). New: `tests/test_tool_gateway_gate.py` (4 tests).

**Tests.** `tests/test_tool_gateway_gate.py` (4 tests):
- Acceptance (`test_loop_dispatch_goes_through_gateway`): the one AgentToolLoop dispatches a model write + read via the gateway (recording pre_call); covers chat/auto/desktop loop sites (same class, same dispatch helper).
- Acceptance (`test_agent_call_tool_goes_through_gateway`): real Agent; `call_tool("workspace_write")` writes the file and is observed by a monkeypatched `ToolGateway.call`; `call_tool("workspace_read")` likewise; the `workspace_run` fence refuses before the gateway (never observed).
- Acceptance (`test_mwv_plan_step_goes_through_gateway`): `_execute_mwv_plan_step` with an operation executes via the passed gateway; step DONE.
- Negative (`test_direct_tool_call_fails_gateway_gate`): a synthetic tool called directly via its handler DOES execute (dispatched proves it) but the gate trips (AssertionError "gateway gate"); the same tool through `gateway.call` passes the gate.
- Fail-on-old-code: N/A as a stash round-trip — no production code changed, so there is nothing to stash. Gate sensitivity is proven directly by the negative test (a direct call trips the gate).

**Delete-vs-wrap.** No production code deleted. No bypass path wrapped. No UX changes.

**E08 candidates (not deleted here).**
- `_call_workspace_read_tool` (`server/http/handlers/workspace.py:593`) — resolved fork above (owner decision 2026-11-17, option C): separate user-initiated read-only path, no second gateway, no model binding; authorization audit is E08-T08.
- `make_computer_runtime` / `AgentComputerRuntime` (`core/agent.py:402`) — zero production callers; dead code, remove in cleanup.
- `index_imported_project` (`server/http/common/github_import.py:108`) — direct `handle_project_request` call, dead while the clone is fenced; route through the gateway if the clone fence is ever lifted, else delete with the fenced flow.
- `ToolRegistry._execution_target` — still write-only (E01-T03); remove once `set_execution_policy`'s last caller is gone.
- `Agent._chat_read_tool_specs` — parallel hand-rolled listing (E01-T03); unify with the registry listing in a later ticket.

#### E01-T05 — канонизировать Outcome

**ЧТО.** Канонизировать structured outcome, **не переписывая** принятый [ADR-0002](docs/architecture/decisions/ADR-0002-terminal-outcome-and-partial-result.md): отдельно `TerminalReason=completed|failed|cancelled|aborted`, `Disposition=full|accepted_partial|none`, `RunPhase` и `Verification` с evidence.

**ЗАЧЕМ.** Исключить fake success, не выдавать административный `aborted` за failure и не терять факт частично принятого результата.

**Acceptance.** State-changing dummy task без подтверждённой postcondition не может стать full `DONE`. Механизм сохраняет отдельно причину завершения и disposition для `aborted + none`, `failed + accepted_partial`, `cancelled + accepted_partial` и `completed + accepted_partial`; `final_partial` допустим только для `completed + accepted_partial`. Не-терминальные `blocked` / `waiting_approval` не выдаются за завершение; UI projection полностью проверена.

**Negative.** `tool.ok=true` сам по себе не proof of completion; `unknown verification` не превращается ни в full success, ни автоматически в failed, не переписывает terminal_reason. Контракт ADR-0002 нельзя менять молча; если нужна реальная развилка — `BLOCKED: REAL_IMPLEMENTATION_FORK` к владельцу.

##### E01-T05 implementation — verified 2026-11-17 (baseline 125da46)

**Status:** DONE (acceptance met). ADR-0002 не переписан: таблица §3 сохранена как есть; добавлен только gate публикации `final_success` (не публикуется без подтверждённой postcondition, по §4 — acceptance transaction проверяет критерии до публикации финала).

**Новый модуль `core/run_outcome.py`.** `TerminalReason`, `Disposition`, `RunPhase`, `VerificationOutcome` (`passed|failed|unknown|not_applicable`) + `PostconditionCheck` / `Verification` с evidence + frozen `RunOutcome` с инвариантами в `__post_init__` (terminal <=> reason; `completed+none` и не-completed `+full` — invalid; `accepted_partial` требует `accepted_by`). `communication_class` — маппинг ADR-0002 §3; `is_full_done` — `completed + full + verification=passed` (или `not_applicable` для задачи без изменения состояния); `ui_label` — полная проекция (9 меток), `DONE` тогда и только тогда, когда `is_full_done`. Деривации: `outcome_from_mwv` (work SUCCESS + verification PASSED -> `completed+full`; SUCCESS + FAILED -> `failed+none` — проверка опровергла результат; SUCCESS + ERROR -> `completed+full+unknown` — `UNVERIFIED`, финал удерживается; FAILURE -> `failed`) и `outcome_from_loop_result` (cancelled/error -> причина из loop, verification unknown; чистый финиш state-changing задачи без проверки -> `UNVERIFIED`, не `DONE`).

**Production wiring.** `AgentMWVMixin._mwv_outcome_label` (`core/agent_mwv.py`) теперь выводит русскую строку из `RunOutcome.ui_label` через `outcome_from_mwv` (state-changing); поведение на достижимом пути не изменилось (`"Итог: проверки пройдены"` при work SUCCESS + verification PASSED — закреплено тестом).

**Anti-pseudo audit (до implementation).** Проверен контур outcome: типы — строго типизированные, без парсинга прозы; проекция — чистая функция структуры, не классификатор; fallback/адаптеров/миграций не добавлено; `_is_success` в `core/mwv/manager.py` (work SUCCESS + verification PASSED) — не дубль, а то же условие, что и в `outcome_from_mwv`; `RunOutcome` — единственный канонический outcome, старых параллельных путей не осталось, кроме известного ниже.

**Известный остающийся fake-DONE (не в scope тикета, E01-T08).** `Agent._execute_mwv_plan_step` (`core/agent_mwv.py`) ставит `PlanStepStatus.DONE` из одного `result.ok` и из прозы при отсутствии operation (`acceptance_checks` из `TaskStepContract` не проверяются). Это step-level статус legacy MWV-роутинга чата, а не `RunOutcome`; исправление — в рамках удаления legacy MWV routing (E01-T08), не здесь. Претензия ADR-0002 (`docs/runtime_contract_claims.json`) остаётся `target`: полный lifecycle-контракт (durable task identity, acceptance transaction) — это E05, этот тикет канонизирует только структуру outcome.

**Files.** Новый `core/run_outcome.py`; `core/agent_mwv.py` (rewire `_mwv_outcome_label` + русские метки); новый `tests/test_run_outcome.py` (36 тестов: инварианты, маппинг `communication_class`, `final_partial` только для `completed+accepted_partial`, state-changing dummy без postcondition — не DONE, `tool.ok` alone — не proof, unknown verification не переписывает причину, exhaustive UI projection matrix, деривации из MWV/loop, зафиксированная строка `"Итог: проверки пройдены"`).


**Codex review fix (2 P2 findings, 2026-11-17).** `outcome_from_loop_result`: clean finish with an explicit `accepted_by` now keeps `accepted_partial` (was unconditionally `FULL` — an accepted partial became DONE/`final_success`); a known policy stop (`error="tool_policy_denied"`, `core/tool_loop.py`) now maps to `ABORTED` (was unconditionally `FAILED`, per ADR-0002 §1). New tests: clean finish with accepted partial (state-changing and not), policy stop aborted (with and without acceptance), other loop errors still failed. Re-review of the fix (2026-11-17) found the same class in the other converter: `outcome_from_mwv` SUCCESS+PASSED also ignored `accepted_by` (was unconditionally `FULL`); fixed the same way with a regression test (`test_mwv_success_passed_with_accepted_partial`).

#### E01-T06 — Plan как artifact, Execute как обычный run

**ЧТО.** Сохранить plan drafting/version/approval, но execution approved plan направить в canonical loop.

**ЗАЧЕМ.** Пользовательская ценность Plan остаётся, отдельный Act runtime исчезает.

**Acceptance.** `plan` не вызывает write tools; approved plan запускается как `intent=execute` через тот же loop.

**Negative.** Подмена plan hash/revision блокируется.

##### E01-T06 implementation — verified 2026-11-17 (baseline 09cbd53)

**Status:** DONE (acceptance met).

**Mini-audit (read-only, до implementation).** Предпосылки тикета держатся. Plan drafting/version/approval живут в `server/http/handlers/plan.py` + `server/http/common/workflow_runtime.py` (`build_plan_draft`, `compile_plan_to_task_packet`, `plan_with_status`, `run_plan_runner`) с хешем sha256 по контенту плана и CAS-привязкой plan_id/hash/revision; execution сегодня — MWV packet runner (`run_plan_runner` → `agent.run_task_packet` → `_execute_mwv_plan_step`, DONE из одного `result.ok`); canonical loop — `core/tool_loop.py` `AgentToolLoop` с gateway из RunSpec (`intent=execute`, plan guard в `tools/tool_registry.py:443-520`); outcome — `core/run_outcome.py` `outcome_from_loop_result`. Пробелы, найденные аудитом: `_compile_plan_to_task_packet` не пересчитывает plan_hash из содержимого (полагается на привязку вызывающего); `normalize_task_payload` не несёт `plan_revision` — ревизия только в `task_packet.context`; `plan_revision` — CAS-счётчик, растёт при каждом обновлении статуса шага (`plan_mark_step`, `plan_with_status`), а не привязка контента.

**Новый модуль `core/plan_execute.py`.** `ValidatedPlan`, `PlanStepExecution`, `PlanExecuteResult`, `PlanLoopCheckpoint`/`PlanLoopCheckpoints`, `PlanLoopApprovalPending`, `validate_approved_plan` (статус approved/running; task-привязка plan_id/hash; хеш пересчитан из содержимого; структура шагов как в compile — ровно один инструмент на шаг, operation совпадает; ревизия пакета — только монотонность: пакет из ревизии новее живой отклоняется), `execute_approved_plan`/`resume_approved_plan` (один `AgentToolLoop` на шаг; runtime state mode=act + active_plan/active_task + guard; gateway из RunSpec; done — только чистый loop + `is_full_done`; write-шаг без postcondition — не full_done, это E05; scope resume — plan_id/hash/task_id/step_id + идентичность brain/main_config, ревизия на равенство не сравнивается, контент привязан хешем).

**Production wiring.** `run_plan_runner` (`server/http/common/workflow_runtime.py`): свежие запуски — только canonical loop, без fallback на MWV; resume старых MWV-чекпоинтов — legacy-ветка (E01-T08 удалит); resume loop-пауз — `resume_approved_plan`; отмена/reject/reset забирают loop-чекпоинт (`plan.py`, `decision.py`, `workflow.py`); store `plan_loop_checkpoints` регистрируется в `create_app` (`server/http/app.py`), а не ленивой записью в запущенный app. `_run_owned_packet_thread` переименован в `_run_owned_sync_thread` (generic, PEP 695).

**Исправленный баг (найден тестами тикета).** Прямой act-путь `handle_ui_plan_execute` компилировал пакет из approved-плана, а сохранял running-план с инкрементированной ревизией — контекст пакета отставал от сохранённого плана. Порядок исправлен как в decision-пути: сначала `running_plan`, потом compile из него.

**Scope.** MWV не удаляется (E01-T08). Dual-path (свежие запуски canonical, resume старых MWV-чекпоинтов) — осознанно, как legacy-ветка; если Codex сочтёт недопустимым — развилка, не молчаливая правка. Fake step-level DONE (`_execute_mwv_plan_step`) не тронут — E01-T08.

**Files.** Новый `core/plan_execute.py` (820 строк); `server/http/common/workflow_runtime.py` (перепроводка runner); `server/http/app.py` (store); `server/http/handlers/plan.py` (cancel + порядок compile); `server/http/handlers/decision.py` (reject); `server/http/handlers/workflow.py` (reset); новый `tests/test_plan_execute.py` (11 тестов: read-шаг через canonical loop, тот же gateway, что прямой execute, подмена hash/task hash/пакета из будущего, рост ревизии после compile — норма, guard, approval pause/resume, resume отклоняет правку контента, write-шаг не full_done, отмена до старта); `tests/ui_api/test_modes_and_plan.py` (20-кейсовый тест переведён на canonical loop, `changed_root` исключён — привязки packet scope у loop нет; lifecycle + draft/approve никогда не вызывает write tools); `tests/ui_api/fakes.py` (DummyAgent для canonical loop); `tests/ui_api/test_runtime_init.py` (rename `_run_owned_packet_thread` → `_run_owned_sync_thread`).

**Tests.** `make check` зелёный (2538 passed, coverage 83.73%).

#### E01-T07 — Auto → policy preset

**ЧТО.** Убрать Auto как самостоятельный orchestration engine после доказательства equivalent canonical execution.

**ЗАЧЕМ.** Autonomous behavior — budget+policy, а не отдельная архитектура.

**Acceptance.** Существующий полезный Auto scenario проходит через canonical run с теми же или более строгими budgets/approval.

**Negative.** Нет planner/shard/merge labels без реальной реализации.

**Delete-vs-wrap.** Старую orchestration оболочку удалить после parity test.

##### E01-T07 implementation — verified 2026-11-18 (baseline 649bf90)

**Status:** DONE (acceptance met).

**Mini-audit (read-only, до implementation).** Предпосылки тикета держатся. Auto сегодня — тонкая оболочка вокруг канонического `AgentToolLoop`: `run_v1` (`core/auto_runtime.py`, 981 строка) фабрикует `AutoPlan` (один шард `"tool-loop"`, prose acceptance checks), пишет театральный state (planner/coders/merge, `pool_size: 1`), затем запускает `AgentToolLoop` с `RunBudget` из `AutoBudgets` (900с / 80 вызовов / 120 файлов / 120k токенов / 0 retries, env-переопределяемо `AUTO_MAX_*`) + gateway из `_build_tool_gateway(budget=run_budget)` + системный промпт `AUTO_V1_SYSTEM_PROMPT` + skill instruction. Реальные механизмы (сохранены): native-tool capability gate; fail-closed на любом failed tool call; budget exhaustion (runtime/tool_calls/tokens); approval pause/resume через `_PausedRun` + `validate_resume` (проверки session/user/mode/workspace/brain/config); capture scope check; skill metadata/report + инъекция skill instruction per-run; usage accounting; file-effects cap; реальный `VerifierRuntime` для repo workspace. Театр (удалён): `planner`/`plan`/`coders`/`merge` state labels, `AutoPlan`/`AutoShard`/`AutoCoderState`/`AutoVerifierState`, `AUTO_CODER_POOL_*`, статусы PLANNING/CODING/MERGING. Фейковые PASSED в `_run_verifier`: профиль `response_only` (непустой текст модели → PASSED) и профиль `tool_outcomes` (все tool calls ok → PASSED). Каноническая модель (E01-T05): response-only → NOT_APPLICABLE, state-changing без postcondition → UNKNOWN, а не PASSED. Production entrypoints: `Agent.handle_auto_command` (`core/agent_tools.py:1307`, из `core/agent_routing.py:335` и `:512` при `runtime_mode == "auto"`) → `AutoAgent.run_outcome` → `AutoOrchestrator.run_v1`; approval plumbing в `agent_tools.py`; `cancel_auto_run` из `server/http/handlers/decision.py:1610`; auto_state UI-контракт — `normalize_auto_state` (`shared/auto_models.py`). Других вызовов orchestrator нет (тест `test_auto_orchestrator_has_no_legacy_run_entrypoint` это уже утверждал). Что сломалось бы при наивном удалении: потребители `AutoRunOutcome` (`core/agent_response.py:53`, `server/http/common/chat_payload.py:175`); approval pause/resume registry; нормализация auto_state; импорт `AUTO_V1_SYSTEM_PROMPT` в `scripts/goal_behavior_eval.py:13`; 5 тестовых файлов + ui_api тесты. UI-контракт: `normalize_auto_state` требует только run_id/status/goal/started_at/updated_at; frontend `parseAutoState` требует статус из известного списка — добавлен `running`; `autoStepLabel` — честный `'Working on task'` для running.

**Новый модуль `core/auto_preset.py`.** `AutoPresetRunner` (`run`/`resume`/`cancel`/`pending_run_id`/`validate_resume`, конструктор `(parent_agent, *, workspace_root=None, progress_callback=None)`); сохранены имена `AutoRunOutcome`, `AutoContinuationUnavailable`, `AutoBudgets`, `AUTO_V1_SYSTEM_PROMPT` и env-имена `AUTO_MAX_*` с теми же дефолтами (900/80/120/120000/0) — минимальный churn потребителей; `AutoBudgets.to_run_budget()` → `RunBudget(max_iterations=80, max_tool_calls=80, max_model_tokens=120_000, max_runtime_seconds=900)`; `run()` — capture scope check, run_root resolution и state dict как в `run_v1`, но БЕЗ planner/plan/coders/merge/pool_size (ключи присутствуют пустыми: `planner={}`, `plan=None`, `coders=[]`, `merge={}` — та же форма state, без фабрикации), начальный статус `RUNNING`; `_execute_frame` — та же логика, что у orchestrator (native-tool gate, budget checks, loop, fail-closed, budget branches, ApprovalRequired → `_PausedRun` + re-raise, Exception → FAILED_INTERNAL), минус театр; **`_verify_after_loop`** — честный verifier: нет tool calls → `None` (без claim); repo workspace + canonical verifier → реальный `VerifierRuntime`; repo без verifier → ERROR `canonical_repo_verifier_unavailable`; non-repo → `None` (фейковый `tool_outcomes` PASSED удалён); **завершение гейтится канонической деривацией** `outcome_from_loop_result(loop_result, state_changing=bool(loop_result.tool_calls))` — COMPLETED только при `terminal_reason is COMPLETED`, иначе FAILED_INTERNAL `canonical_outcome_not_completed`; `state_changing = bool(tool_calls)` — консервативно (без effect-классификации отдельных инструментов, это E02/out-of-scope; любой вызов ⇒ может менять состояние ⇒ UNKNOWN, не NOT_APPLICABLE). Статусы preset'а: RUNNING, VERIFYING, WAITING_APPROVAL, COMPLETED, FAILED_*, CANCELLED; PLANNING/CODING/MERGING никогда не эмитируются.

**Production wiring.** `core/agent.py` — `self.auto_preset = AutoPresetRunner(self, progress_callback=self._record_auto_progress)`; `core/agent_tools.py` — `handle_auto_command`/`cancel_auto_run`/approval plumbing на `auto_preset` (`run`/`resume`/`cancel`/`validate_resume`/`pending_run_id`); `core/agent_response.py`, `core/agent_routing.py`, `server/http/common/chat_payload.py` — `AutoRunOutcome` из `core.auto_preset`; `scripts/goal_behavior_eval.py` — `AUTO_V1_SYSTEM_PROMPT` из `core.auto_preset`.

**Удалено (не обёрнуто).** `core/auto_runtime.py` (981 строка: `AutoOrchestrator`, `run_v1`, `AutoPlan`/`AutoShard`/`AutoCoderState`/`AutoVerifierState`, фабрикация плана, фейковые verifier-профили); `core/auto_agent.py` (`AutoAgent` — тонкий wrapper); `tests/test_auto_runtime.py`. В `shared/auto_models.py`: `AutoShard`, `AutoPlan`, `AutoCoderState`, `AutoVerifierState`, `AUTO_CODER_POOL_*` удалены; в `AutoRunStatus` добавлен `RUNNING = "running"` (PLANNING/CODING/MERGING оставлены в enum для back-compat старых states in flight, preset их не эмитирует); `_normalize_execution_metrics` сохраняет честные счётчики `tool_calls_used`/`iterations` (non-negative int, иначе dropped — нормализация их раньше выбрасывала); `_normalize_pool_size` переписан (positive int as-is, иначе 1). Frontend (минимально, честный статус): `ui/src/app/types.ts` (`'running'` в union `AutoRunStatus`), `ui/src/app/session-payload.ts` (`'running'` в `isKnownStatus`), `ui/src/features/workspace/computer-status.ts` (`AUTO_STATUS_MAP` + `running: 'running'`, `autoStepLabel` → `'Working on task'`); старые статусы оставлены парситься.

**Claims registry.** `docs/runtime_contract_claims.json` (65 claims): claim `runtime.auto.v1_tool_loop` заменён на `runtime.auto.preset_canonical_loop` (implemented, verified_by `tests/test_auto_preset.py` + `tests/test_tool_loop.py` + `tests/test_verifier_runtime.py`); в 5 claims (`runtime.auto.native_call_cap`, `runtime.auto.run_expiry_admission`, `runtime.auto.file_effects_truth`, `runtime.auto.staged_file_budget`, `runtime.tools.generic_process_outcome`) путь `tests/test_auto_runtime.py` заменён на `tests/test_auto_preset.py`; `skills.per_run_instructions`: пути `core/auto_runtime.py` → `core/auto_preset.py` и тест, summary «Auto v1» → «Auto preset», version 2 → 3. Gate `check-contracts` зелёный (65 claims validated).

**Parity test.** `test_auto_preset_parity_write_scenario` (`tests/test_auto_preset.py`): настоящий `WriteFileTool` через canonical loop (ScriptedBrain вызывает `workspace_write`, файл создаётся с контентом в ManagedWorkspace; gateway с `WorkspaceAccess` + `ApprovalContext(False, ...)` — binding как у настоящего Agent; approval pause/resume покрыты портированными тестами): файл существует, outcome COMPLETED, `verifier is None` (non-repo workspace), `files_touched == 1`, budgets в state == preset defaults 80/900/120000/120, нет фабрикации planner/coders/merge. Три находки при портировании тестов (все — механика, не поведение preset'а): (1) `WriteFileTool` через preset-gateway требует managed workspace binding (`_workspace_binding` в `ToolGateway`: writer без binding → `workspace_unavailable`); (2) `normalize_auto_state` выбрасывала `tool_calls_used`/`iterations` из `execution_metrics` — исправлено в `shared/auto_models.py`; (3) `files_touched` для fake echo — `is None`, а не `0` (неподтверждённый count нельзя выводить в ноль).

**Scope.** MWV не удаляется (E01-T08). `Agent` в `desktop`/`chat` не тронут. `AUTO_RUN_ACTIVE_STATUSES`/`TERMINAL_STATUSES` в Python-коде нигде не используются — оставлены как есть. Старые frontend-статусы (`planning`/`coding`/`merging`) оставлены парситься для старых states in flight.

**Files.** Новый `core/auto_preset.py`; новый `tests/test_auto_preset.py` (порт `tests/test_auto_runtime.py` 1:1 с честными ассертами + parity-тест); удалены `core/auto_runtime.py`, `core/auto_agent.py`, `tests/test_auto_runtime.py`; `core/agent.py`, `core/agent_tools.py`, `core/agent_response.py`, `core/agent_routing.py`, `scripts/goal_behavior_eval.py`, `server/http/common/chat_payload.py`, `shared/auto_models.py`, `docs/runtime_contract_claims.json`; портированы `tests/test_auto_native_continuation.py`, `tests/test_auto_usage_accounting.py`, `tests/test_run_expiry.py`, `tests/test_token_admission.py`, `tests/test_agent_modes_logic.py`, `tests/test_agent_response_projection.py`, `tests/test_run_submission.py`, `tests/ui_api/test_stream_and_events.py`, `tests/ui_api/test_auto_native_approval.py`; frontend `ui/src/app/types.ts`, `ui/src/app/session-payload.ts`, `ui/src/features/workspace/computer-status.ts`.

**Tests.** `make check` зелёный (2553 passed, coverage 83.75%).


#### E01-T08 — удалить legacy classifier/MWV production routing

**ЧТО.** После трассировки перевести production callers с legacy `classify_request` и псевдо-MWV path.

**Verified evidence.** `AgentRoutingMixin` всё ещё импортирует `core.mwv.routing.classify_request`.

**ЗАЧЕМ.** Один execution universe.

**Acceptance.** grep/test не показывает production call from user request → legacy MWV executor.

**Negative.** Tests MWV не считаются production reachability.

**Delete-vs-wrap.** Если MWV concepts понадобятся позже, они могут существовать как library, но production routing удаляется.

##### E01-T08 implementation — verified 2026-11-18 (baseline 9a431ed)

**Status:** DONE (acceptance met).

**Mini-audit (read-only, до implementation).** Предпосылки тикета держатся. Живой production-путь в legacy MWV: `Agent` по умолчанию `runtime_mode="act"` (`core/agent.py:343`); `respond`/`respond_stream` (`core/agent_routing.py:346,539`) падали в `classify_request` (`core/mwv/routing.py:83`) для любого режима вне {ask, auto, desktop} — regex-триггеры или любой skill match давали route=mwv; `_run_mwv_flow` (`core/agent_mwv.py:285`) строил `ManagerRuntime`+`WorkerRuntime`+`VerifierRuntime` (псевдо-MWV: verifier всегда ERROR, `_execute_mwv_plan_step` ставил DONE из одного `result.ok` и из прозы); второй caller — `_request_requires_root_gate` (`server/http/handlers/ui_chat.py:230,703`) использовал `classify_request` как root-gate эвристику; третий — legacy MWV resume в `run_plan_runner` (`server/http/common/workflow_runtime.py:1062–1089`) + `validate_mwv_approval`/`cancel_mwv_approval` + `MWVExecutionCancelled`. `core/mwv/coding_skill.py`, `coding_task.py`, `single_attempt.py`, `worker.py` — без production callers вне `core/mwv/` (только тесты). Матрица E01-T01 (строки 610–624) описывает состояние до E01-T08: строки 4, 13, 17 (MWV-пути чата, UI plan execute, approval resume) больше не действуют.

**Удалено (не обёрнуто).** `core/agent_mwv.py` целиком (1607 строк: `_run_mwv_flow`, `_mwv_worker_runner`, `_mwv_task_builder`, `run_task_packet`/`resume_task_packet`, `validate/cancel_mwv_approval`, `_execute_mwv_plan_step`, `_mwv_checkpoints`, `MWVWorkerCheckpoint`, `MWVExecutionCancelled`, `TaskPacketApprovalPending`, `MWVContinuationUnavailable`); `core/mwv/routing.py` (`classify_request`, `RouteDecision`), `core/mwv/manager.py` (`ManagerRuntime`, `decide_retry`/`build_retry_task`), `core/mwv/coding_skill.py`, `coding_task.py`, `single_attempt.py`, `worker.py`, `interfaces.py`; `AgentMWVMixin` из `core/agent.py` + `MAX_MWV_ATTEMPTS`, `_COMPAT_EXPORTS`, `last_mwv_result`, `_mwv_checkpoints`; `MWVRunResult` из union `RuntimeResult` (`core/agent_response.py`) и ветка `ProducerKind.MWV` оттуда же; `run_task_packet`/`cancel_mwv_approval` из `AgentProtocol` (`server/http/common/runtime_contract.py`); legacy MWV resume-ветка, обработка `TaskPacketApprovalPending`/`MWVExecutionCancelled`, `mwv_packet_runner` labels (12 замен → `canonical_tool_loop`) в `server/http/common/workflow_runtime.py`; 18 тестовых файлов, тестировавших удалённое поведение.

**Решения (отступления от брифа — задокументированы).**
- `set_runtime_state` НЕ нормализует plan/act → "ask": `core/plan_execute.py:403-412` вызывает `agent.set_runtime_state(mode="act")` и требует `policy.mode == "act"` для шагов canonical loop; нормализация сломала бы выполнение планов (E01-T06, DO NOT TOUCH). Вместо этого гарантия «нет маршрута в MWV» — default `runtime_mode="ask"` + fallthrough в обычный chat loop для любого режима кроме ask/auto/desktop. Зафиксировано gate-тестом `test_set_runtime_state_keeps_act_for_canonical_loop`.
- Root-gate (`_request_requires_root_gate`): ОТСТУПЛЕНИЕ ОТ БРИФА. Бриф требовал «auto всегда gated», но чтение кода и тесты это опровергли: gate срабатывал только для запросов, которые classifier отправлял в MWV (тест `test_ui_chat_send_in_auto_mode_chat_like_request_skips_root_gate` — chat-like запросы в auto намеренно обходили gate, коммит 8e85037), а 40 тестов кодируют прямой запуск auto без gate. Без classifier'а честно отличить chat-like от execution-like нельзя; gate на каждый auto-запрос — подтверждение на каждое сообщение, что противоречит и тикету (удаление, а не добавление трений), и позиции пользователя против карточек на каждый шаг. Поэтому gate больше не срабатывает (всегда False); функция и call site сохранены — gated-путь (202 + `chat.run_root` decision) остаётся рабочим и покрыт тестами через monkeypatch, fingerprint-проверка при `root_context` не тронута. Auto preset сам управляет бюджетами и approval'ами; desktop-режим gate не имел никогда.
- Skill handling сохранён в fallthrough: `_apply_skill_decision`, deprecated-блок, `decision_handler.evaluate` с route="chat"; `_record_unknown_inbox`/`_record_unknown_skill_candidate` — inbox/candidate writers, не исполнение MWV, вызываются при `no_match` в chat-пути (сигнатуры переведены с `RouteDecision` на `SkillMatchDecision | None`).
- Прямые `call_tool` write-операции в ask блокируются (`ASK_READ_ONLY_BLOCK`); skill deprecated/ambiguous — только в legacy fallthrough (act/plan) и auto, не в ask.

**Намеренно оставлено.** `core/mwv/models.py` (типы TaskPacket/RunContext/VerificationResult — auto_preset, desktop, plan_execute, run_outcome); честный verifier (`core/mwv/verifier.py`, `core/mwv/verifier_runtime.py` — auto preset, desktop, plan loop); `core/run_outcome.py::outcome_from_mwv` (чистая деривация work+verification); `ProducerKind.MWV` в `shared/task_run.py` — только исторический вид для чтения старых stored submissions; поле `mwv_report` в UI payloads + парсинг `MWV_REPORT_PREFIX` (всегда None, API-совместимость); `StopReasonCode.MWV_INTERNAL_ERROR`/`VERIFIER_FAILED` (используются auto preset и chat/desktop путями); `SESSION_MODES` с plan/act (миграция словаря UI — E06); комментарий в `core/plan_execute.py:17` про fake-DONE `_execute_mwv_plan_step` — исторический контраст.

**Claims registry.** `docs/runtime_contract_claims.json` (65 claims): `runtime.approval.native_continuation` v8→v9 (убран `tests/test_mwv_native_continuation.py` из verified_by, MWV-предложения в summary заменены на canonical loop); `runtime.agent.request_local_response` v7→v8 (убран `tests/test_mwv_response_observation.py`, удалены MWV-предложения); `skills.per_run_instructions` v3→v4 (summary «MWV reporting» → «Response reporting»). Gate `check-contracts` зелёный.

**Доки.** `docs/agent/ROUTING_POLICY.md` переписан под ask/auto/desktop + chat loop; `docs/agent/MWV_FLOW.md` — stub «удалён в E01-T08»; `docs/agent/README.md`, `docs/agent/STOP_RESPONSES.md`, `docs/SOURCE_OF_TRUTH.md`, `docs/workflow/TEST_TAXONOMY.md`, `docs/architecture/TASK_RUN_STORAGE_CONTRACT.md` (§ «Typed MWV observation» — пометка E01-T08).

**Files.** Production: удалены `core/agent_mwv.py`, `core/mwv/routing.py`, `core/mwv/manager.py`, `core/mwv/coding_skill.py`, `core/mwv/coding_task.py`, `core/mwv/single_attempt.py`, `core/mwv/worker.py`, `core/mwv/interfaces.py`; изменены `core/agent.py`, `core/agent_routing.py`, `core/agent_response.py`, `core/agent_tools.py`, `core/mwv/__init__.py`, `core/plan_execute.py` (комментарии), `core/run_outcome.py` (комментарий), `shared/task_run.py` (комментарий), `server/http/handlers/ui_chat.py`, `server/http/common/workflow_runtime.py`, `server/http/common/runtime_contract.py`, `server/http/handlers/plan.py`, `server/http/handlers/decision.py`. Тесты: удалены 18 файлов, переписаны/урезаны 18 файлов, новый `tests/test_legacy_mwv_routing_removed.py`.

**Tests.** `make check` зелёный (2435 passed, coverage 83.19%). Новый gate-тест `tests/test_legacy_mwv_routing_removed.py` (9 тестов: AST-скан production на запрещённые импорты/имена, отсутствие 8 удалённых модулей и 14 MWV-атрибутов у `Agent`, default ask, fallthrough act/plan → chat loop sync+stream, deprecated/ambiguous в fallthrough, root-gate только по режиму, resume legacy чекпоинта отклоняется без эффектов).

#### E01-T09 — regression E2E canonical loop

**ЧТО.** Первый немоканный acceptance test: модель/fixture вызывает read tool, write tool, observation, postcondition.

**Acceptance.** Fail tool → FAILED/BLOCKED; missing verifier → UNVERIFIED; confirmed postcondition → DONE.

**Gate.** Только после этого E02.

**Статус:** READY.

**Зависимость.** Gate требует штатного process verifier: узкий срез E04-T03 — prerequisite; до его merge E01-T09 не закрывается, E02 не начинается. Работа по тикету сохранена в PR #163 (ветка `pr-e01-t09-e2e-canonical-loop`); после merge среза — возврат к #163 и прогон E2E без `_ExecutingVerifierRuntime` через штатный `VerifierRuntime`.

### Epic E02 — Persistent Browser Session Resource

#### E02-T01 — отделить browser session lifecycle от DesktopBrowserTool

**ЧТО.** Спроектировать/ввести BrowserProfile + BrowserService minimal API.

**Verified touchpoint.** `tools/desktop_browser_tool.py` уже содержит generic Playwright backend с pages, navigate, snapshot/read и другими operations.

**ЗАЧЕМ.** Не переписывать browser actions; заменить disposable lifecycle на durable resource.

**Acceptance.** Existing generic actions работают через service.

**Delete-vs-wrap.** Action implementation reuse; lifecycle ownership заменить.

#### E02-T02 — dedicated persistent Chromium profile

**ЧТО.** Перейти на Playwright persistent context с отдельным `user_data_dir`.

**ЗАЧЕМ.** Cookies/local storage/login переживают restart. Playwright предоставляет этот mechanism напрямую.

**Acceptance positive.** Test site login → stop SlavikAI/browser → restart → authenticated page remains authenticated.

**Negative.** Основной Chrome profile владельца нельзя выбрать.

**Forbidden.** Копирование password DB из Chrome владельца.

#### E02-T03 — Browser profile lease

**ЧТО.** Один active writer lease на profile.

**ЗАЧЕМ.** Избежать concurrent corruption и неопределённого tab state.

**Acceptance.** Второй run получает BLOCKED/profile_busy.

**Negative.** Crash holder → lease recoverable после health validation.

#### E02-T04 — manual login handoff

**ЧТО.** UI/API состояние `NEEDS_USER_AUTH`; видимое окно Chromium; owner finishes login; session возвращается agent.

**ЗАЧЕМ.** Пользователь логинится сам, agent не получает password.

**Acceptance.** Password никогда не появляется в tool log/LLM message/effect log.

**Negative.** MFA timeout даёт NEEDS_USER, не fake success.

#### E02-T05 — effect classification browser actions

**ЧТО.** Классифицировать generic operations.

Пример:

```text
observe/read → read
navigate → read/network
type → draft/local mutation
submit → external side effect
download → write
upload → external/write
```

**ЗАЧЕМ.** Авторизованный browser не должен означать безусловное право действовать от имени владельца.

**Acceptance.** Reading profile page не требует send approval; actual form submit может требовать.

**Negative.** Click, приводящий к destructive/external action, не должен автоматически наследовать `read`.

#### E02-T06 — browser postconditions

**ЧТО.** После state change агент обязан re-observe relevant DOM/state.

**Acceptance.** «Пост опубликован» только если page observation подтверждает опубликованный entity/state.

**Negative.** `click()` success без observation → UNVERIFIED.

#### E02-T07 — profile secret boundary

**ЧТО.** Сделать profile directory недоступным обычному workspace shell principal.

**ЗАЧЕМ.** Session cookies являются credentials-equivalent state.

**Acceptance.** Workspace executor не может прочитать browser profile path.

**Negative.** Symlink из workspace в profile root блокируется.

#### E02-T08 — browser real E2E acceptance

**Scenario.**

1. Создать тестовый profile.
2. Вручную login в test authenticated app.
3. Stop browser/runtime.
4. Restart.
5. Agent читает private page.
6. Agent меняет безопасное test value.
7. Re-observe подтверждает изменение.
8. Logout/session invalidation корректно даёт NEEDS_USER, а не DONE.

После этого persistent browser capability считается реализованной.

### Epic E03 — Owner Host Bridge / Desktop Commander

#### E03-T01 — protocol ADR и threat model

**ЧТО.** Зафиксировать AF_UNIX transport, principals, capabilities, boundaries.

**ЗАЧЕМ.** Не начинать daemon до решения trust model.

**Owner gate.** Только если возникает необходимость bridge работать root; рекомендуемый v1 — **не root**.

**Acceptance.** В ADR перечислены assets, attacker classes, allowed roots, forbidden operations.

#### E03-T02 — минимальный AF_UNIX daemon

**ЧТО.** systemd-managed Host Bridge с `ping/version/capabilities`.

**ЗАЧЕМ.** Доказать lifecycle/auth без dangerous operations.

**Acceptance.** Только expected UID может подключиться; другой local user получает отказ. Linux peer credentials для Unix sockets доступны через `SO_PEERCRED`.

**Negative.** World-writable socket невозможен.

#### E03-T03 — protocol validation

**ЧТО.** Versioned request schema, frame limit, request ID, timeout.

**Acceptance.** Oversized frame, malformed JSON, unknown version/capability → deterministic rejection.

**Negative.** Неизвестные поля не превращаются в command arguments.

#### E03-T04 — bounded host filesystem API

**ЧТО.** `stat/list/read/write/patch/move/delete/search` только для approved roots.

**ЗАЧЕМ.** Реальная работа с файлами владельца без arbitrary host shell.

**Acceptance.** Test root works.

**Negative.** `../`, absolute escape, symlink escape, destination escape, protected file access reject.

**Delete-vs-wrap.** Если существующие Desktop filesystem tools имеют хороший verified implementation, extractor может reuse их internals; не сохранять отдельный second API после bridge adoption.

#### E03-T05 — process run/poll/kill

**ЧТО.** argv-only process API.

**Acceptance.** stdout, stderr, exit code, timing доступны; long process получает stable process ID.

**Negative.** `/bin/sh -c`, env secret inheritance, timeout escape запрещены по умолчанию.

#### E03-T06 — STOP and process ownership

**ЧТО.** Каждый process ref связан с run ID; cancel убивает только processes этого run.

**Acceptance.** STOP long-running test command; process group исчезает; partial state logged.

**Negative.** STOP run A не убивает run B.

#### E03-T07 — host diagnostics/service read-only

**ЧТО.** Добавить read-only service status, process listing, disk/host diagnostics.

**Acceptance.** Агент способен подтвердить состояние service/process без shell parsing.

**Negative.** `service.restart` отсутствует в registry.

#### E03-T08 — audit/effect log

**ЧТО.** Bridge возвращает structured effect; SlavikAI сохраняет request metadata + effect + observation.

**ЗАЧЕМ.** Пользователь видит, что реально изменилось на его ПК.

**Negative.** stdout secrets redacted; raw env never logged.

#### E03-T09 — systemd hardening

**ЧТО.** Service sandboxing по реально нужным capabilities: `NoNewPrivileges`, restricted filesystem view, `PrivateDevices` где возможно, address-family restrictions, explicit writable paths. systemd поддерживает эти controls, но они должны проверяться фактическим `systemd-analyze security`/runtime tests, а не считаться магической защитой.

**Acceptance.** Daemon не имеет доступа к запрещённым roots.

**Negative.** Hardening не ломает socket lifecycle/restart.

#### E03-T10 — optional desktop observation/control split

**ЧТО.** Вынести GUI observation и GUI control в две capabilities.

**ЗАЧЕМ.** Наблюдение экрана и управление мышью/клавиатурой имеют разный риск.

**Owner gate.** До включения GUI control.

**Acceptance.** `desktop.observe` можно разрешить без `desktop.control`.

#### E03-T11 — bridge E2E

**Scenario.**

1. Agent target=`owner_host`.
2. Создаёт test file в approved host root.
3. Перечитывает его.
4. Запускает safe process.
5. Poll получает exit code.
6. Проверяет file postcondition.
7. Попытка `/etc/shadow` отвергнута.
8. Symlink escape отвергнут.
9. STOP long process подтверждён.
10. Restart bridge → old finished effects не повторяются.

Только после этого Desktop Commander capability считается настоящей.

### Epic E04 — собственная рабочая среда агента

> **NEEDS_REFINEMENT:** нижеследующие пункты — предварительные цели. Перед началом эпика обязателен полный ticket refinement и owner decision по развилкам.

#### E04-T01 — выбрать bounded workspace deployment

**ЧТО.** На основании threat model и измерений выбрать: isolated OS user или rootless container.

**Рекомендация.** Начать isolated OS user + bounded directory, если shell/process capability не требует сильной untrusted-code isolation; если агенту разрешается выполнять произвольный скачанный код, перейти на rootless container.

**Owner gate.** Только здесь окончательный выбор.

**Решение владельца 2026-11-18 (для среза E04-T03, не для всего эпика).** Deployment штатного process verifier — rootless container на уже квалифицированной boundary (`core/rootless_podman.py`): агент исполняет произвольный код workspace, поэтому isolated OS user для process execution не выбран. Отдельный pinned image с make — `python:3.12.16-bookworm` (тот же Python 3.12.16 и та же Debian bookworm база, что и квалифицированный Python-образ, плюс build tools; `make` 4.3 проверен на хосте, Id `28d30c15b9e9c3b1d557441aad9592226e852f7716451d6c8df5b4f0c3ce3d1b`, digest `sha256:3c3ba8ad4c85eb63b7568f806434f14e33370056985da7fdf2673621cb1d7e93`). Python profile (`…-python-bounded-v3`, образ `37ccb536…`) не трогается. Выбор образа для полного `make check` репозитория (node/venv) — отдельное решение, не этот срез.

**Не выбирать VM без измеренной необходимости.**

#### E04-T02 — canonical workspace filesystem

**ЧТО.** Один API list/read/write/patch/move/delete/search.

**ЗАЧЕМ.** Убрать дублированные fs/project/workspace names после call-path audit.

**Acceptance.** Все model-visible workspace file actions идут в один backend.

#### E04-T03 — real workspace executor

**ЧТО.** Один qualified shell/process backend.

**Acceptance.** `argv`, cwd, timeout, stdout, stderr, exit code.

**Negative.** Если backend health-check fails, tool вообще не показывается модели.

##### E04-T03 slice — штатный process verifier (prerequisite E01-T09, узкий срез, 2026-11-18)

**Решение владельца 2026-11-18.** Разрешён отдельный pinned image с make и его квалификация в рамках узкого среза; scope — только штатный process verifier (`VerifierRuntime` + `make check` в managed workspace). Переиспользуются существующие rootless Podman boundary и staging; Python profile сохраняется; verifier запускается на staged snapshot без записи артефактов обратно в workspace (no-publish).

**Условие 1 — явное разрешение на конкретный run.** Запуск `make check` должен быть явно разрешён для конкретного run и привязан к его principal, workspace и qualified environment (доверенный `WorkspaceBinding`: principal/session/environment revision + квалифицированный профиль и staging pool). Само наличие Makefile такого разрешения не даёт: без квалифицированного backend и без binding, совпадающего с run, `VerifierRuntime.run()` возвращает ERROR `sandbox_execution_unavailable`, процесс не запускается.

**Условие 2 — критерий задаёт доверенная сторона.** Критерий успешной проверки (содержимое `Makefile`, цель `check`) задаётся доверенной стороной до запуска агента и защищён от подмены агентом: хэш `Makefile` фиксируется доверенным кодом раннера до начала loop и хранится вне agent-writable workspace; при проверке хэш staged snapshot обязан совпасть с предзапусковым — несовпадение даёт ERROR (`verifier_criterion_changed`), никогда PASSED. Агент может писать `result.txt`, но не может переопределить, что считается пройденной проверкой.

**Механизм.** Отдельный pinned image `python:3.12.16-bookworm` (см. E04-T01), отдельный профиль `i1-ubuntu-podman-5.7-command-bounded-v1`, `create_command_container` с allowlist ровно `["make", "check"]` (отказ до любого обращения к engine при любом другом argv), тот же confinement, что у Python-пути (network none, private namespaces, cap-drop ALL, no-new-privileges, readonly rootfs, 256 МиБ / 64 PID, ровно один bind mount staged копии, ровно те же 7 env-записей), `finished_exit` для реального exit code, no-publish освобождение stage (`release_unpublished`). `VerifierRuntime`: опциональный `executor` (default `None` — поведение без исполнителя не меняется); маппинг: finished+0 → PASSED, finished+≠0 → FAILED (`check_failed`), unavailable/unknown/conflict → ERROR, никогда фейковый PASSED. `_verify_after_loop`: canonical verifier есть, но backend недоступен → ERROR, а не `None` (fail-closed в FAILED_VERIFIER вместо COMPLETED).

**Доказательство (отдельный PR).** Штатный `VerifierRuntime` на реальном квалифицированном backend: верный `result.txt` → PASSED, неверный → FAILED, недоступный backend → ERROR без запуска процесса. Полный `make check` самого SlavikAI внутри sandbox — вне scope среза (репозиторий нельзя заснять в managed workspace: symlink/размеры; в образе нет node/venv). Публикация shell tools модели — E04-T04, полная семантика UNVERIFIED/BLOCKED — E05-T03.

**Порядок.** Срез — prerequisite перед закрытием E01-T09: после merge среза возврат к PR #163, E2E без `_ExecutingVerifierRuntime` через штатный путь; E01-T09 закрывается только после DONE через штатный путь + негативные сценарии без DONE. До этого #163 остаётся BLOCKED, E02 ждёт.

**Статус:** IN_PROGRESS, ветка `pr-e04-t03-slice-process-verifier` от `origin/main` `d36fd64`.

#### E04-T04 — tool qualification

**ЧТО.** Startup/runtime health check для executor capabilities.

**Acceptance.** Model tool list меняется в соответствии с реальной availability.

**Negative.** Не должно быть «tool registered but always returns sandbox unavailable».

#### E04-T05 — long-running process registry

**ЧТО.** Stable process IDs, poll/log/kill и persistence metadata.

**Acceptance.** Ответ модели может завершиться, process остаётся trackable.

**Negative.** Restart не объявляет исчезнувший PID живым.

#### E04-T06 — real workspace E2E

**Scenario.** Agent создаёт test project file, запускает deterministic command, изменяет файл, выполняет validation, re-reads result и только после этого получает DONE.

### Epic E05 — verification, durable run state and recovery

> **NEEDS_REFINEMENT:** нижеследующие пункты — предварительные цели. Перед началом эпика обязателен полный ticket refinement и owner decision по развилкам.

#### E05-T01 — postcondition model

**ЧТО.** Создать маленький набор typed postconditions:

```text
file_exists
file_content_matches
process_running
process_exit
service_state
browser_state
custom_command
```

**ЗАЧЕМ.** Проверять effect, а не tool return.

#### E05-T02 — effect→postcondition binding

**ЧТО.** Каждый state-changing tool может вернуть suggested postcondition; agent/run может добавить expected condition.

**Acceptance.** Write file автоматически даёт verify candidate.

#### E05-T03 — удалить fake verifier semantics

**ЧТО.** Ни один verifier не может вернуть PASSED, если он ничего не проверил.

**Acceptance negative.** unavailable executor → UNVERIFIED/BLOCKED.

**Delete-vs-wrap.** Псевдо-verifier удалить, не прятать за adapter.

#### E05-T04 — persist RunState

**ЧТО.** Записывать goal, policy hash, effects, approvals, processes, continuation, postconditions.

**Acceptance.** Kill/restart runtime не уничтожает run history.

#### E05-T05 — idempotent recovery

**ЧТО.** После restart runtime сначала reconstruct state и observations, а не повторяет последний non-idempotent action.

**Acceptance.** Test «создать запись/файл один раз → crash → resume» не создаёт duplicate.

#### E05-T06 — approval recovery

**ЧТО.** Pending approval переживает restart с exact subject binding.

**Acceptance.** Изменённые args invalidates approval.

#### E05-T07 — unified STOP

**ЧТО.** Run cancellation propagates workspace, browser, owner-host bridge.

**Acceptance.** Новые tool calls после STOP запрещены.

### Epic E06 — modes, policy and UI simplification

> **NEEDS_REFINEMENT:** нижеследующие пункты — предварительные цели. Перед началом эпика обязателен полный ticket refinement и owner decision по развилкам.

#### E06-T01 — UI vocabulary migration

**ЧТО.** Отображать intent/target/policy вместо internal modes.

**Acceptance.** User sees `Ответить / План / Выполнить`, target и permission preset.

#### E06-T02 — remove Act runtime concept

**ЧТО.** Approved Plan → execute Run.

**Acceptance.** Нет самостоятельного act orchestrator в production path.

#### E06-T03 — remove Auto runtime concept

**ЧТО.** Auto UI option, если нужен, становится preset «разрешить агенту продолжать в рамках policy/budget».

**Acceptance.** Один loop.

#### E06-T04 — Desktop → owner_host target

**ЧТО.** Desktop больше не влияет на reasoning loop; выбирает executor/tool set/trust boundary.

**Acceptance.** Один и тот же task contract способен работать workspace и owner_host с разными tools.

#### E06-T05 — effect-based approvals

**ЧТО.** Approval policy зависит от effect + target + principal.

**Acceptance.** External send отдельно от write; destructive отдельно от exec.

#### E06-T06 — hard external-send rule

**ЧТО.** Публикации, сообщения третьим лицам, отправка формы с существенным последствием получают отдельную approval category.

**Acceptance negative.** Browser session наличие login не является approval на send.

### Epic E07 — memory and context

> **NEEDS_REFINEMENT:** нижеследующие пункты — предварительные цели. Перед началом эпика обязателен полный ticket refinement и owner decision по развилкам.

#### E07-T01 — inventory readers/writers stores

**ЧТО.** Для каждого current store определить writer, reader, source-of-truth, production use.

**Acceptance.** Store без reader/writer помечается removal candidate.

#### E07-T02 — canonical curated memory

**ЧТО.** Выбрать один durable source для user/project long-term memory.

**ЗАЧЕМ.** Убрать несколько параллельных истин.

#### E07-T03 — conversation compaction

**ЧТО.** При budget threshold создавать summary и продолжать с bounded recent context.

**Acceptance.** Long conversation не просто silently drops oldest messages.

#### E07-T04 — run history separate from memory

**ЧТО.** Не смешивать «что агент вчера выполнял» с «долговременный факт о пользователе».

**ЗАЧЕМ.** Восстановление запуска не должно зависеть от неточного memory summary.

**Предварительный acceptance.** После restart история RunState доступна по run_id с effects/observations/outcome; правка curated memory не меняет историю. **Negative.** Memory не может подменить evidence. Полная спецификация и negative E2E дописываются перед E07.

#### E07-T05 — derived search index only if justified

**ЧТО.** Если semantic retrieval реально нужен, index rebuildable из canonical corpus.

**Negative.** Vector DB никогда не становится единственным источником пользовательской памяти.

### Epic E08 — cleanup and operational hardening

> **NEEDS_REFINEMENT:** нижеследующие пункты — предварительные цели. Перед началом эпика обязателен полный ticket refinement и owner decision по развилкам.

#### E08-T01 — removal pass: MWV/legacy routing

Удалять только после E01 parity evidence.

#### E08-T02 — removal pass: dead/fake shell and PTY

Удалять model-visible tools без qualified backend; PTY не возвращать до реального consumer.

#### E08-T03 — removal pass: duplicate workspace tools

Схлопнуть fs/project/workspace aliases после caller migration.

#### E08-T04 — provider cleanup after PR #151 stability

**ЧТО.** Только после стабилизации subscription integration переоценить brain/provider duplication.

**Forbidden.** Не смешивать provider refactor с архитектурой tool loop.

#### E08-T05 — deployment service

**ЧТО.** Runtime service restart policy, health endpoint, graceful shutdown, explicit state directories.

#### E08-T06 — observability screen

**ЧТО.** Runs, target, browser, bridge, approvals, effects, postconditions, STOP.

#### E08-T07 — scheduler readiness gate

Scheduler разрешено проектировать **только когда** browser, owner-host bridge, durable run state и recovery прошли acceptance.

Только тогда monitoring становится настоящим, а не cron вокруг ненадёжного агента.

#### E08-T08 — UI read authorization audit

**ЧТО.** Аудит отдельного user-initiated read-only пути файлового браузера (`_call_workspace_read_tool`, `server/http/handlers/workspace.py:593`): session workspace root, sandbox path checks (`_ensure_in_workspace`), private-runtime guard, read size cap; кто может вызвать tree/file GET и pre-write version check, что видит сессия без выбранной модели.

**ЗАЧЕМ.** Путь намеренно вне ToolGateway (решение владельца 2026-11-17, опция В) — его authorization должна быть проверена явно, а не подразумеваться.

**Acceptance.** Перечислены все проверки доступа на пути + негативные тесты (путь вне workspace, private runtime, чужая сессия).

**Forbidden.** Не создавать второй ToolGateway и не привязывать просмотр файлов к выбранной модели.

## Verification matrix and deletion rules

Capability считается существующей не по наличию класса, а по реальному E2E.

| Capability | Positive E2E | Negative E2E | Postcondition |
|---|---|---|---|
| Canonical loop | task→tool→observation | tool error | verified outcome |
| Workspace file | create/change test file | path escape | re-read |
| Workspace exec | run safe command | timeout/invalid cwd | exit+artifact |
| Persistent browser | login survives restart | invalidated session | authenticated observation |
| Browser action | modify test web entity | submit without approval | re-observe entity |
| Host bridge FS | write approved root | `/etc`, `..`, symlink escape | host re-read |
| Host bridge process | launch/poll | shell injection | exit/process state |
| STOP | cancel long run | unaffected other run | no managed process |
| Approval | approve exact request | mutate args | binding match |
| Recovery | crash/resume | duplicate side effect | no duplicate |
| Memory | remember curated fact | irrelevant old chat | retrieval from canonical |
| Context compaction | continue long chat | summary corruption | bounded context |
| Tool qualification | healthy backend visible | failed backend | registry absence |
| No fake DONE | verified task completes | unavailable verifier | UNVERIFIED |

Удаление subsystem допускается только если выполнены все четыре условия:

1. доказан production caller inventory;
2. replacement capability уже проходит E2E либо capability признана ненужной владельцем;
3. никакой уникальный user-visible behavior не теряется;
4. после removal есть regression test, предотвращающий возврат второго production path.

Начальный removal inventory — **кандидаты, не приговоры**:

- legacy `classify_request` production routing — высокий приоритет, потому что current `AgentRoutingMixin` реально его импортирует;
- самостоятельный Auto orchestration — удалить после переноса budgets/resume;
- Act runtime — убрать после approved-plan→canonical-run;
- production MWV executor — удалить, если после call graph не найдётся уникальной необходимой capability;
- fake/unqualified shell registrations;
- dead PTY scaffolding;
- duplicate fs/project/workspace tools;
- duplicate approval layers;
- memory stores без production reader;
- vector stores без canonical source;
- provider-specific duplicate code — поздний cleanup, не foundation.

Нельзя удалять только по старому аудиту:

- `AgentToolLoop` — подтверждён актуальным `main`;
- `ToolGateway` — подтверждён как реальный enforcement layer;
- useful parts `ToolRegistry`;
- verified Desktop observation/control primitives — использовать как reference/extractable implementation;
- generic Playwright operations — менять lifecycle, а не переписывать без причины.

## `STATE.md`, Muse workflow and decision gates

Исторический шаблон `STATE.md` (актуальные значения всегда брать из [STATE.md](STATE.md)):

```md
# SlavikAI Worker State

Updated: YYYY-MM-DD HH:MM Europe/Vilnius

## Baseline

Main SHA:
Origin/main SHA:
Original worktree: /home/ki/Project/Slavikai (owner/main; Muse read-only)
Muse worktree: /home/ki/Project/Slavikai-rebuild
Muse service branch: muse/slavikai-rebuild
Current ticket branch:
Working tree: clean|dirty
PR #151 status:
Last merged roadmap ticket:

## Current ticket

ID:
Title:
Branch:
PR:
Started:
Status: READY|IN_PROGRESS|BLOCKED|AWAITING_REVIEW|AWAITING_CI|AWAITING_OWNER|DONE

Goal:

Allowed scope:

Forbidden scope:

Acceptance checklist:
- [ ]
- [ ]
- [ ]

Required E2E:
- [ ]

## Review state

make check:
CI:
Codex review:
Blocking findings:
Last reviewed commit SHA:

## Owner decision required

None

or

Question:
Options:
Muse recommendation:
Why this is architectural:
Exact consequences:

## Next ticket

ID:
Blocked by:

## Runtime / environment notes

Host reachable:
Relevant services:
Browser profile state:
Host bridge state:

## Handoff

What was done:
What remains:
Exact next command/read:
Known risks:
Do not repeat:
```

Resume rule:

Muse начинает новую сессию не с памяти о чате, а с:

```text
1. Read repository instructions.
2. Read ROADMAP.md.
3. Read STATE.md.
4. Verify worktree/branch/HEAD still match STATE.md.
5. If mismatch → STOP and reconcile.
6. Continue only current ticket.
```

Перед созданием каждой новой ветки:

```text
working tree clean
previous ticket merged
main updated
branch from exact current main
STATE updated
```

После implementation:

```text
make check
targeted tests
required real E2E
git diff review by Muse for accidental scope
push
open/update PR
Codex independent review
CI green
fix blockers
re-review changed commit
manual merge (NO GitHub auto-merge)
update STATE
next ticket
```

Решением владельца от 2026-11-16 разрешены **ровно два linked worktree**: `/home/ki/Project/Slavikai` — исходный checkout владельца и `/home/ki/Project/Slavikai-rebuild` — единственная рабочая директория Muse. Один тикет/ветка/PR, без параллельной реализации. После Codex review последнего HEAD, зелёного CI и `make git-check` Muse выполняет **ручной fast-forward push** своей PR-ветки в `origin/main` из второго worktree, предварительно проверив что текущий `origin/main` — предок PR HEAD. Original checkout не трогать. При GitHub protection rejection или non-FF — STOP, решение владельца. **GitHub auto-merge и merge queue не включать.**

Muse **не спрашивает владельца** о названиях внутренних функций, разбиении helper-ов, тестовой структуре, formatting, выборе очевидного стандартного Python API или локальном refactor внутри утверждённого тикета.

Muse **обязана спросить при любой реальной развилке реализации «так или иначе»** (значимые варианты A/B, влияющие на архитектуру, контракты, security, продукт, backlog или поддержку), а также при расширении host roots, privileged operations, external send, root daemon, VM/container/OS-user, удалении пользовательских данных, создании третьего worktree, вмешательстве в исходный checkout и любом sudo с точной командой. Требуются trade-offs, рекомендация и последствия; до ответа — `BLOCKED: REAL_IMPLEMENTATION_FORK`.

Весь roadmap имеет один строгий critical path:

```text
PR #151 merged (470a9a6) + two worktrees verified
→ docs PR #152 accepted and E00 complete
→ repository baseline
→ canonical Run contract
→ one production execution loop
→ honest outcomes
→ persistent authenticated browser
→ Owner Host Bridge / Desktop Commander
→ real workspace executor
→ postconditions + durable run/recovery
→ modes/policy/UI simplification
→ memory/context consolidation
→ cleanup/deployment
→ only then scheduler/subagents/proactive monitoring
```

Ключевой критерий успеха перестройки: после foundation SlavikAI должен пройти четыре реальные демонстрации без специальных обходов.

**Первая:** «Создай файл в своей рабочей среде, измени его, запусти проверку и докажи результат».

**Вторая:** «Открой свой браузер, где владелец входил вчера, зайди на авторизованную страницу и выполни разрешённое действие; после restart сессия должна сохраниться».

**Третья:** «На этом компьютере владельца создай test file через Host Bridge, запусти безопасный process, проверь результат; попытку выйти за разрешённый root отвергни».

**Четвёртая:** «Начни длительную операцию, получи STOP, перезапусти SlavikAI и честно покажи, что было сделано, что остановлено и что осталось непроверенным».

Если эти четыре сценария работают, появляется настоящий фундамент агента. Если вместо них система имеет красивые `Auto`, `Act`, `MWV`, `Verifier`, `MemoryManager` и двадцать классов, но не может пройти эти сценарии, перестройка считается неуспешной.

## Канонические документы и проверяемые источники

- [Соглашение Muse](slavikai-how-to-work-with-muse.md) и [STATE.md](STATE.md).
- [SlavikAI repository](https://github.com/Phonezzzz/Slavikai), [PR #151](https://github.com/Phonezzzz/Slavikai/pull/151).
- [Playwright persistent Chromium context](https://playwright.dev/python/docs/api/class-browsertype#browser-type-launch-persistent-context), [Playwright Authentication](https://playwright.dev/docs/auth).
- [Linux unix(7)](https://man7.org/linux/man-pages/man7/unix.7), [systemd.socket](https://www.freedesktop.org/software/systemd/man/latest/systemd.socket.html).

Исследовательские гипотезы нужно проверять на фактическом main HEAD, production call path, test и postcondition до изменения кода.
