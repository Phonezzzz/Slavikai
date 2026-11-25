# AGENTS — обязательная точка входа

Репозиторий работает по принципу rules-first/policies-first.
Любой агент обязан прочитать и применить правила из этого файла до начала работы.

## Initialization protocol (обязательно)

Перед любыми действиями агент обязан:

1. Прочитать все документы из раздела **Canonical rules**.
2. Определить релевантные контекстные документы и прочитать их.
3. Сформировать **Rules + Context Snapshot** и считать его жёсткими ограничениями на всю сессию.

Работу над задачей начинать только после этого.

## Canonical rules (must read)

- `docs/SOURCE_OF_TRUTH.md` — иерархия контрактов и статусы current/target/legacy.
- `docs/architecture/ARCH_CANON.md` — обязательные runtime-инварианты.
- `docs/agent/DevRules.md` — глобальные инварианты проекта.
- `docs/workflow/dev_workflow.md` — git-процесс (режим A).
- `docs/agent/COMMAND_LANE_POLICY.md` — границы командного режима.
- `docs/agent/ROUTING_POLICY.md` — маршрутизация chat/mwv.
- `docs/agent/STOP_RESPONSES.md` — единый формат остановки.

## Contextual references (читать по релевантности)

- `docs/architecture/Architecture.md`
- `docs/agent/COMMAND_LANE_POLICY.md`
- `docs/workflow/CONTRIBUTING.md`
- `docs/agent/MWV_FLOW.md`

## Rules + Context Snapshot (формат)

- Какие канонические правила применяются в текущей задаче.
- Какие контекстные документы прочитаны и почему они релевантны.
- Какие ограничения являются жёсткими (sandbox, safe-mode, MWV, git-flow, approvals).

## Product deployment: closed owner/member group

- Целевой deployment — небольшая закрытая группа за Cloudflare Access email OTP.
- Один настроенный owner имеет административные полномочия; остальные допущенные
  пользователи являются members.
- Browser principal выводится только из криптографически проверенного Cloudflare Access JWT;
  подтверждённый нормализованный email является `principal_id`.
- Browser identity и Bearer auth для `/v1` automation — разные auth lanes.
- Sessions, runtime state, Memory, vectors и approvals обязаны быть principal-scoped.
- Текущая реализация этого контракта неполна. Статус определяется только через
  `docs/runtime_contract_claims.json`; нельзя считать legacy token-cookie auth достаточной
  multi-user isolation.
- Не добавлять compatibility/migration layers «на всякий случай». Миграция существующих
  локальных данных к owner допустима только как явно спроектированная часть security PR.

## Non-negotiable rules

- **Именованные Git worktrees (решение владельца 2026-11-25, переход от двух
  деревьев к dependency-driven parallel development):**
  - `/home/ki/Project/Slavikai` — исходное дерево владельца, обычно `main`;
    Muse не переключает в нём ветки и не меняет файлы (read-only).
  - `/home/ki/Project/Slavikai-rebuild` — стрим A (Host Bridge), служебная ветка
    `muse/slavikai-rebuild`.
  - `/home/ki/Project/Slavikai-foundation` — стрим C (Foundation), служебная ветка
    `muse/slavikai-foundation`.
  - `/home/ki/Project/Slavikai-verification` — стрим D (Verification & Recovery),
    служебная ветка `muse/slavikai-verification`.
  - `/home/ki/Project/Slavikai-workspace` — стрим E (Workspace Executor),
    создаётся после решения E04-T01 и refinement E04-T02.
- Все деревья принадлежат одному Git-репозиторию, имеют общую историю и
  отдельные HEAD/index/working files. Это не независимые копии и не backup:
  исходный baseline закреплён Git SHA.
- **Один активный implementation-тикет на worktree.** Два worktree никогда не
  работают на одной PR-ветке и не получают пересекающийся scope без согласования
  координатора. Служебные ветки `muse/slavikai-*` — не PR-ветки; каждая PR-ветка
  создаётся от актуального `origin/main` в своём worktree.
- Новое дерево — только решением владельца (новый стрим), список выше — закрытый.
- Параллельные read-only subagents допустимы, но изменения исходного дерева
  владельца и рабочих деревьев Muse не смешиваются.
- Работа никогда не ведётся напрямую в `main`.
- Каждая задача выполняется в отдельной PR-ветке.
- На новой чистой PR-ветке до изменений: `make preflight`.
- Перед завершением: `make check`.
- После commit и push открыть PR. До merge обязательны успешный CI, независимый Codex
  review текущего опубликованного HEAD и прочие требования GitHub и проекта. Порядок
  проверки Qodo, других reviewers и замечаний описан в `docs/workflow/dev_workflow.md`.
- После commit и push перед merge: `make git-check`.
- `main` сохраняет линейную историю: **только fast-forward**, без merge-коммитов. В исходном дереве допустим старый маршрут `git merge --ff-only` → `git push origin main`; **Muse исходное дерево не трогает** и после всех gates выполняет явный `git push origin <pr-branch>:refs/heads/main` из своего worktree (Git отклоняет non-fast-forward). Перед push обязательно `make git-check`, актуальный fetch, проверка `origin/main` как предка PR HEAD; при запрете push защитой ветки — STOP и вопрос владельцу, без обхода.
- После подтверждённого merge Muse возвращается в служебную ветку `muse/slavikai-rebuild` и fast-forward синхронизирует её с `origin/main`, **не переходя на `main` во втором дереве**.
- **Merge queue — централизованная, с just-in-time rebase (решение 2026-11-25).
  Массового rebase всех открытых PR после каждого merge НЕТ.**
  - Единственный координатор (главный агент) управляет merge в `main`,
    docs-sync и распределением задач между стримами.
  - В `main` вливается ровно один PR за раз, только fast-forward, только после
    зелёного CI и чистого независимого Codex review именно того SHA, который
    вливается (инвариант из правила выше — без изменений).
  - Перед merge выбранного PR координатор делает его just-in-time rebase на
    актуальный `origin/main` → `make check` → Codex ре-ревью нового head.
    Остальные открытые PR НЕ трогаются при каждом merge: они обновляются
    только при реальной зависимости, конфликте либо перед их собственной
    интеграцией.
  - Порядок merge — по §6 аудита параллельности (`parallel-audit-2026-11-25.md`)
    и DAG в ROADMAP.md; порядок merge не является новой искусственной
    последовательностью всего roadmap. Hard dependencies, semantic dependencies
    и временные конфликты файлов различаются: временный конфликт не блокирует
    чужой стрим, а решается в merge queue.
  - Общие contracts между E03, E04 и E05 согласовываются до реализации
    зависимых компонентов (refinement-контракт до имплементации).
  - Refinement-конвейер идёт на несколько тикетов вперёд независимо от
    implementation (read-only работа на этой VM, worktree не требует).
- **Runtime isolation: worktree — это не изоляция runtime (проверено пробой
  2026-11-25).**
  - Git branches — отдельные; working directories — отдельные; runtime sockets —
    отдельные (`XDG_RUNTIME_DIR` свой на worktree, socket dir демона
    `$XDG_RUNTIME_DIR/slavikai/` — свой на worktree).
  - Systemd scopes — общий user manager на весь UID; изоляция — только через
    уникальные имена unit (`slavikai-bridge-<uuid>.scope`) и orphan sweep строго
    в своём namespace (T06, PR #203).
  - Test databases — отдельные: `TaskRunStore`/`PreferencesStore` изолированы
    в `tmp_path` (conftest), `.run/task_runs.db` в тестах — только в tmp.
  - Runtime ports — без пересечения: все тестовые HTTP-серверы биндятся на
    порт 0 (эфемерный); фиксированных портов в тестах нет.
  - Build caches — отдельные: `.pytest_cache`, `.coverage`, `__pycache__`,
    `.ruff_cache`, `.mypy_cache` живут внутри каждого worktree.
  - Python venv — общий `/home/ki/Project/Slavikai/venv` (override
    `VENV_DIR=... PYTHON=.../bin/python`) только при отсутствии изменений
    requirements. `make check` его только читает. `pip install` / `make venv`
    в двух worktrees одновременно запрещён; перед первой параллельной проверкой
    venv один раз приводится в порядок последовательно.
  - Тестовые ресурсы — `tmp_path`/`tmp_path_factory` на каждый запуск; исключение:
    opt-in verifier E2E (`SLAVIK_RUN_QUALIFIED_VERIFIER_E2E=1`) использует
    фиксированный `~/.cache/slavikai-qualified-verifier-e2e` — при запуске
    задавать `SLAVIK_VERIFIER_E2E_STATE` свой на worktree.
  - `main` — единственный координатор merge (см. merge queue выше).
  - **Чеклист перед первым параллельным `make check`:**
    1. Один последовательный `make venv` с override на общий venv (или сверка +
       `touch` маркеров `.installed` / `.installed-dev`) — иначе первый `make check`
       в новом worktree начнёт `pip install` в общий venv.
    2. `npm ci` в `ui/` каждого нового worktree.
    3. Отдельный `XDG_RUNTIME_DIR` на worktree для любого запуска демона/тестов
       (в т.ч. live-тестов).
    4. Конкурентные live bridge-тесты — только после мержа фикса P1 4687943835
       (T06) + namespaced probe scope: оба в PR #203 (main `658a638`, 2026-11-25).
    5. Opt-in verifier E2E — только с `SLAVIK_VERIFIER_E2E_STATE` свой на worktree.
- **Codex-бюджет — приоритет стримам A (Host Bridge) и C (Foundation).**
  При исчерпании лимита review gates для security-critical и
  architecture-critical PR **не ослабляются автоматически**. В это время
  продолжается refinement и другая доступная работа; маленькие PR — Muse
  self-review (standing rule). Открытые архитектурные решения (§4 ADR-0013 и
  др.) остаются открытыми: переход к параллельной разработке не является их
  автоматическим утверждением.
- Ветки завершённых тикетов Muse пока сохраняются; удаление только по отдельному решению владельца. GitHub auto-merge и встроенную merge queue GitHub не включать.
- Полный цикл согласованного PR до merge разрешён без дополнительного подтверждения,
  если владелец прямо поручил такой цикл; иначе требуется решение владельца.
- В утверждённом объёме обычные инженерные решения принимать самостоятельно по
  имеющимся свидетельствам. Остановить затронутую часть и обратиться к владельцу
  при новой продуктовой цели, существенном конфликте контрактов или архитектуры,
  выходе за согласованный объём, угрозе незавершённой работе или необходимости
  обязательного разрешения.
- Детальный порядок и безопасное обращение с существующими worktrees:
  `docs/workflow/dev_workflow.md`.

## Принципы разработки

- Сохранять продуктовую цель, проверенное поведение и инварианты. Предпочитать
  связную замену, когда её преимущества подтверждены. Инженерные критерии —
  в [DevRules.md](docs/agent/DevRules.md#принятие-инженерных-решений).
- Владелец задаёт архитектурный замысел и утверждает значимые изменения. Обсуждение
  идеи, принятие решения и поручение на реализацию — отдельные шаги.
- Сначала устойчивость и проверка реальных пользовательских сценариев. Для согласованной перестройки Muse актуальный source of truth — [ROADMAP.md](ROADMAP.md) и [STATE.md](STATE.md); [исторический roadmap](docs/architecture/roadmap/master-roadmap.md) не должен переписывать решения владельца.
- Незавершённую работу и уникальную историю сохранять. Правила для локальных
  данных разработки — в [dev_workflow.md](docs/workflow/dev_workflow.md#данные-локальной-разработки).

## Anti-pseudo audit перед implementation

Перед implementation каждого PR worker обязан выполнить mini non-mutating audit не только по файлам/entrypoints, но и по признакам pseudo-runtime / pseudo-agent behavior.

Проверить изменяемый контур на:

- regex/prose extraction вместо explicit structured input / ToolRequest / ToolSpec;
- classifier/router, который принимает runtime decision вместо основного runtime/tool loop;
- fallback path, который обходит основной runtime;
- adapter/compatibility/migration layer без явного разрешения;
- tests, которые проверяют только итоговый эффект, но не проверяют механизм;
- direct file/db/tool action в Python вместо ToolGateway / explicit tool call;
- lane/type discriminator, который протекает как domain model;
- legacy entrypoint, который остаётся reachable после нового path;
- duplicate runtime path, где новый path есть, но старый всё ещё используется production-кодом.

Правило:

- если найденный pseudo-path входит в scope текущего PR — удалить/исправить его в рамках PR;
- если исправление требует архитектурного решения вне scope — остановиться и вернуть BLOCKED report;
- не добавлять adapters/fallbacks/compatibility/migrations “чтобы тесты прошли” без явного разрешения.

## Язык (обязательно)

- Все ответы, планы, объяснения и комментарии — на русском.
- Английский допускается только в коде, командах, логах, именах файлов и точных цитатах ошибок/документации.
- Размышления оформлять как: **Краткий план (RU)** + **Проверки/риски (RU)**.


## Workflow (кратко)

1. Для работ Muse: использовать только назначенное координатором worktree
(см. список выше), прочитать `ROADMAP.md`, `STATE.md` и соглашение, проверить
`git worktree list`, fetch актуального `origin/main`. Исходное
`/home/ki/Project/Slavikai` остаётся деревом владельца. Незавершённый PR
продолжать в его рабочей ветке и его worktree.
2. `git checkout -b pr-<id>-<name>`; `make preflight`.
3. Реализация + targeted tests; `make check`.
4. Commit + `git push -u origin HEAD`; открыть PR.
5. Проверить обязательный CI, независимый Codex review текущего HEAD и остальные
   требования по `docs/workflow/dev_workflow.md`. Исправить подтверждённые замечания,
   после push повторить применимые проверки и review.
6. Перед merge: fetch; если `origin/main` продвинулся — rebase PR-ветки, повторить
   `make check`, push и необходимые review/checks.
7. `make git-check`.
8. После одобрения и при наличии мандата владельца на полный цикл Muse выполняет **ручной fast-forward push** `git push origin <pr-branch>:refs/heads/main` из своего дерева, не трогая исходный `main`; при non-FF или защите ветки остановиться. Для других исполнителей в исходном дереве действует старый `git merge --ff-only` + push.
9. Подтвердить remote `main` SHA и статус PR `merged` на GitHub; для Muse переключиться на служебную `muse/slavikai-rebuild` и fast-forward синхронизироваться с `origin/main`. Ветки Muse не удалять до отдельного разрешения. Не начинать следующий PR автоматически.

## Code Review Rules

Правила применяются к GitHub/Codex code review. Reviewer проводит независимую оценку PR; review не даёт разрешения менять код, создавать commits или выполнять merge.

- Проверяй PR относительно его фактической base branch, учитывая затронутые reachable execution paths, а не только изменённые строки.
- Сопоставляй реализацию с применимыми canonical contracts, `docs/runtime_contract_claims.json` и существующими runtime-механизмами. PR description и успешный CI сами по себе не доказывают архитектурную корректность.
- Ищи конкретные регрессии correctness и security в изменяемом контуре: authorization, principal/session isolation, approvals, sandbox, state ownership, concurrency, persistence, budgets, errors и unknown outcomes — только где релевантно.
- Проверяй отсутствие обходов typed interfaces, `ToolGateway`, policy enforcement и verifier, а также новых duplicate/legacy execution paths и silent fallbacks.
- Проверяй, покрывают ли тесты реально изменённый механизм и значимые failure paths, а не только ожидаемый итоговый эффект.
- Отличай дефекты, внесённые этим PR, от существующих проблем. Не выдавай unrelated legacy debt за blocker этого PR; существенные обнаруженные риски вне scope отмечай отдельно.
- Для каждого actionable finding указывай файл и строку, воспроизводимое условие или конкретный путь отказа, ожидаемое и фактическое поведение, затронутый контракт и серьёзность. Не выдвигай гипотезу как подтверждённый дефект.
- Не требуй рефакторинга, форматирования или дополнительных abstractions без доказанного влияния на корректность. Если замечаний нет, сообщи об этом, указав границы выполненной проверки.
