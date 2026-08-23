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
- Engineering skills: current `implemented` contract — instructions и supporting dependencies
  внедряются per run в Auto v1; MWV report/UI наблюдают выбранный skill и terminal status. Brain
  не мутируется, tool/approval/sandbox/verifier полномочия не расширяются.

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
