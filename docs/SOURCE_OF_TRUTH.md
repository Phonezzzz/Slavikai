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

Cloudflare browser identity contract реализован в отдельном production mode; token-cookie
browser auth остаётся reachable legacy для локального запуска. Полная principal isolation
Memory/vector/runtime state ещё имеет статус `partial`, поэтому multi-user deployment пока
нельзя считать полностью изолированным.

## Binding runtime decisions

- Ask: целевой invariant — zero write и no hidden vector initialization; current status
  `partial`.
- Plan: current contract — read-only tools только через `ToolGateway`; write/exec запрещены.
- Auto: Auto v1 через `AgentToolLoop -> ToolGateway -> verifier` является current;
  `Ask -> Plan -> Act` FSM является target.
- Public OpenAI-compatible API: `slavik` — единственный публичный proxy model id; strict
  request validation ещё `partial`.
- Memory: запись возможна только после отдельного явного confirm/edit; current status
  `target`, автоматический `auto_save_dialogue` не является допустимым целевым поведением.
- Engineering skills: instructions и supporting dependencies должны внедряться per run в
  существующий runtime без глобальной мутации Brain; current status `target`.

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
