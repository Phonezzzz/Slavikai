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

Текстовые метки UI (`DONE`, `FAILED`, `CANCELLED`, `BLOCKED`, `UNVERIFIED`) — только **проекции этой структуры**, не источник истины и не подмена `TerminalReason`. `DONE` для state-changing задачи требует `completed + full + verification=passed`. `completed + accepted_partial` отражает частично принятый результат; его нельзя представить полным `DONE`. `aborted` сохраняется отдельно от `failed`, а `blocked/waiting_approval` не являются terminal completion. При `Verification=unknown` результат честно отображается как unverified **с сохранением причины завершения**, а не подменяет её. Никакой fake PASS из `tool.ok`. Точное mapping и acceptance authority проверяются в E01-T05.

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

**Программа: 62 тикета, строго последовательно**, один тикет = одна ветка = один PR. Список пересматривается по подтверждённым фактам после эпиков: sunk cost не является основанием сохранять механизм.

| Epic | Tickets | Статус |
|---|---:|---|
| E00 — baseline/docs | 3 | E00-T01 DONE; E00-T02 PR #152 review; E00-T03 pending |
| E01 — canonical execution | 9 | READY после нового preflight |
| E02 — persistent browser | 8 | READY после E01 |
| E03 — Host Bridge / Desktop Commander | 11 | READY после E02 |
| E04 — workspace/executor | 6 | NEEDS_REFINEMENT |
| E05 — verification/recovery | 7 | NEEDS_REFINEMENT |
| E06 — modes/policy/UI | 6 | NEEDS_REFINEMENT |
| E07 — memory/context | 5 | NEEDS_REFINEMENT |
| E08 — cleanup/operations | 7 | NEEDS_REFINEMENT |
| **Всего** | **62** | **31 READY, 31 предварительный** |

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

### Epic E01 — canonical execution contract

#### E01-T01 — inventory production execution entrypoints

**ЧТО.** Трассировать production entrypoints для ask, plan, act, auto, desktop до `AgentToolLoop`, MWV или другого executor.

**Verified touchpoints.** `core/agent_routing.py`, `core/tool_loop.py`, `core/desktop_runtime.py`, `core/tool_gateway.py`, `tools/tool_registry.py`.

**ЗАЧЕМ.** Перед удалением режимов получить доказанный call graph.

**Steps.** Для каждого public route/UI action записать entry handler → Agent method → loop/runtime → gateway → executor → outcome.

**Acceptance.** Machine-readable или markdown matrix; каждый path помечен production/test-only/dead/unknown.

**Negative.** Search-only evidence без call-path не считается достаточным.

**Delete-vs-wrap.** Ничего не удалять в этом тикете.

#### E01-T02 — ввести immutable RunSpec/RunPolicy contract

**ЧТО.** Добавить минимальный run-level contract `intent/target/policy/budget`.

**ЗАЧЕМ.** Разделить то, что сейчас смешано в ExecutionMode.

**Implementation.** Использовать существующие типы где возможно; не создавать hierarchy framework. Policy snapshot immutable на протяжении run.

**Acceptance positive.** Два одновременных runs с разными target/policy не влияют друг на друга.

**Acceptance negative.** Mutation global ToolRegistry mode во время другого run не меняет его effective policy.

**Delete-vs-wrap.** Существующие mode fields временно остаются только как callers; новый contract становится source of truth.

**Forbidden.** Изменение пользовательского UX в этом PR.

#### E01-T03 — перевести ToolRegistry listing на RunSpec

**ЧТО.** Tool availability вычисляется из target/policy/qualification, а не из глобальной mutable mode state.

**ЗАЧЕМ.** Model видит только инструменты, реально допустимые в конкретном run.

**Acceptance.** `answer` не получает write tools; `execute` получает только qualified tools target-а; unknown target fail-closed.

**Negative.** Нельзя вручную вызвать tool через registry и обойти target check.

#### E01-T04 — сделать ToolGateway единственной execution gate

**ЧТО.** Проверить все production tool calls; прямые model-visible calls мимо gateway перевести через gateway или доказанно удалить.

**ЗАЧЕМ.** Одна policy/approval/effect точка.

**Acceptance.** Instrumentation test подтверждает: каждый model-originated state-changing call проходит gateway.

**Negative.** Добавленный synthetic tool, вызываемый напрямую, должен падать test gate.

**Delete-vs-wrap.** Прямые обходные paths удалять, не оборачивать.

#### E01-T05 — канонизировать Outcome

**ЧТО.** Канонизировать structured outcome, **не переписывая** принятый [ADR-0002](docs/architecture/decisions/ADR-0002-terminal-outcome-and-partial-result.md): отдельно `TerminalReason=completed|failed|cancelled|aborted`, `Disposition=full|accepted_partial|none`, `RunPhase` и `Verification` с evidence.

**ЗАЧЕМ.** Исключить fake success, не выдавать административный `aborted` за failure и не терять факт частично принятого результата.

**Acceptance.** State-changing dummy task без подтверждённой postcondition не может стать full `DONE`. Механизм сохраняет отдельно причину завершения и disposition для `aborted + none`, `failed + accepted_partial`, `cancelled + accepted_partial` и `completed + accepted_partial`; `final_partial` допустим только для `completed + accepted_partial`. Не-терминальные `blocked` / `waiting_approval` не выдаются за завершение; UI projection полностью проверена.

**Negative.** `tool.ok=true` сам по себе не proof of completion; `unknown verification` не превращается ни в full success, ни автоматически в failed, не переписывает terminal_reason. Контракт ADR-0002 нельзя менять молча; если нужна реальная развилка — `BLOCKED: REAL_IMPLEMENTATION_FORK` к владельцу.

#### E01-T06 — Plan как artifact, Execute как обычный run

**ЧТО.** Сохранить plan drafting/version/approval, но execution approved plan направить в canonical loop.

**ЗАЧЕМ.** Пользовательская ценность Plan остаётся, отдельный Act runtime исчезает.

**Acceptance.** `plan` не вызывает write tools; approved plan запускается как `intent=execute` через тот же loop.

**Negative.** Подмена plan hash/revision блокируется.

#### E01-T07 — Auto → policy preset

**ЧТО.** Убрать Auto как самостоятельный orchestration engine после доказательства equivalent canonical execution.

**ЗАЧЕМ.** Autonomous behavior — budget+policy, а не отдельная архитектура.

**Acceptance.** Существующий полезный Auto scenario проходит через canonical run с теми же или более строгими budgets/approval.

**Negative.** Нет planner/shard/merge labels без реальной реализации.

**Delete-vs-wrap.** Старую orchestration оболочку удалить после parity test.

#### E01-T08 — удалить legacy classifier/MWV production routing

**ЧТО.** После трассировки перевести production callers с legacy `classify_request` и псевдо-MWV path.

**Verified evidence.** `AgentRoutingMixin` всё ещё импортирует `core.mwv.routing.classify_request`.

**ЗАЧЕМ.** Один execution universe.

**Acceptance.** grep/test не показывает production call from user request → legacy MWV executor.

**Negative.** Tests MWV не считаются production reachability.

**Delete-vs-wrap.** Если MWV concepts понадобятся позже, они могут существовать как library, но production routing удаляется.

#### E01-T09 — regression E2E canonical loop

**ЧТО.** Первый немоканный acceptance test: модель/fixture вызывает read tool, write tool, observation, postcondition.

**Acceptance.** Fail tool → FAILED/BLOCKED; missing verifier → UNVERIFIED; confirmed postcondition → DONE.

**Gate.** Только после этого E02.

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

**Не выбирать VM без измеренной необходимости.**

#### E04-T02 — canonical workspace filesystem

**ЧТО.** Один API list/read/write/patch/move/delete/search.

**ЗАЧЕМ.** Убрать дублированные fs/project/workspace names после call-path audit.

**Acceptance.** Все model-visible workspace file actions идут в один backend.

#### E04-T03 — real workspace executor

**ЧТО.** Один qualified shell/process backend.

**Acceptance.** `argv`, cwd, timeout, stdout, stderr, exit code.

**Negative.** Если backend health-check fails, tool вообще не показывается модели.

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
