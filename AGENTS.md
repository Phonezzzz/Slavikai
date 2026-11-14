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

- Штатная локальная рабочая папка — `/home/ki/Project/Slavikai`. Для последовательных PR
  используется один checkout; создавать дополнительные `git worktree` или копии репозитория
  без конкретной необходимости и явного согласования владельца запрещено.
- Параллельные read-only subagents могут использовать этот checkout без отдельных worktrees;
  параллельные изменения файлов требуют координации.
- Работа никогда не ведётся напрямую в `main`.
- Каждая задача выполняется в отдельной PR-ветке.
- На новой чистой PR-ветке до изменений: `make preflight`.
- Перед завершением: `make check`.
- После commit и push открыть PR. До merge обязательны успешный CI, независимый Codex
  review текущего опубликованного HEAD и прочие требования GitHub и проекта. Порядок
  проверки Qodo, других reviewers и замечаний описан в `docs/workflow/dev_workflow.md`.
- После commit и push перед merge: `make git-check`.
- В `main` только `git merge --ff-only` с последующим `git push origin main`.
- После merge обязательно вернуться в `main`, проверить результат и удалить только
  подтверждённо завершённую PR-ветку локально и на GitHub.
- Полный цикл согласованного PR до merge разрешён без дополнительного подтверждения,
  если владелец прямо поручил такой цикл; иначе требуется решение владельца.
- Если что-то неясно — остановиться и запросить решение у человека.
- Детальный порядок и безопасное обращение с существующими worktrees:
  `docs/workflow/dev_workflow.md`.

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

1. Для новой задачи в чистом `/home/ki/Project/Slavikai` перейти на `main` и обновить её
   через `git pull --ff-only origin main`. Незавершённый PR продолжать в его ветке.
2. `git checkout -b pr-<id>-<name>`; `make preflight`.
3. Реализация + targeted tests; `make check`.
4. Commit + `git push -u origin HEAD`; открыть PR.
5. Проверить обязательный CI, независимый Codex review текущего HEAD и остальные
   требования по `docs/workflow/dev_workflow.md`. Исправить подтверждённые замечания,
   после push повторить применимые проверки и review.
6. Перед merge: fetch; если `origin/main` продвинулся — rebase PR-ветки, повторить
   `make check`, push и необходимые review/checks.
7. `make git-check`.
8. После одобрения и при наличии мандата владельца на полный цикл:
   `git merge --ff-only <pr-branch>` в `main` + `git push origin main`.
9. Подтвердить вливание текущего HEAD в удалённый `main` и статус `merged` на GitHub,
   удалить безопасно завершённую PR-ветку локально и на GitHub,
   завершить на актуальном `main`. Не начинать следующий PR автоматически.

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
