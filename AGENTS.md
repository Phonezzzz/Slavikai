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

- Работа никогда не ведётся напрямую в `main`.
- Каждая задача выполняется в отдельной PR-ветке.
- На новой чистой PR-ветке до изменений: `make preflight`.
- Перед завершением: `make check`.
- После commit и push перед merge: `make git-check`.
- В `main` только `git merge --ff-only`.
- После merge обязательно вернуться в `main`.
- Если что-то неясно — остановиться и запросить решение у человека.
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

1. `git checkout main` + `git pull --ff-only origin main`
2. `git checkout -b pr-<id>-<name>`
3. `make preflight`
4. реализация + targeted tests
5. `make check`
6. commit + `git push -u origin HEAD`
7. перед merge: fetch; если `origin/main` продвинулся — rebase PR-ветки, повторный
   `make check` и push
8. `make git-check`
9. после одобрения: `git merge --ff-only` в `main`
10. `git checkout main`
