# SlavikAI Worker State

**Source of truth:** [ROADMAP.md](ROADMAP.md) · [Рабочее соглашение Muse](slavikai-how-to-work-with-muse.md) · [Git workflow](docs/workflow/dev_workflow.md). Это зафиксированный снимок от 2026-11-18, синхронизирован с merge PR #161 (main `9a431ed`); перед любым действием Muse перепроверяет Git, CI, review и актуальный `main`.

## Два Git worktree — подтверждено на ПК владельца 2026-11-16

| Назначение | Путь | Ветка на момент фиксации | HEAD | Право Muse |
|---|---|---|---|---|
| Исходный проект владельца | `/home/ki/Project/Slavikai` | `main` | `fca51c2fdcecd5d2b37c554105af44e3d925f348` | **Только read-only без изменения checkout** |
| Перестройка под Muse | `/home/ki/Project/Slavikai-rebuild` | `muse/slavikai-rebuild` | `9a431ed338abcf4affdf41b8c93132bb437ac040` | **Единственная рабочая директория для roadmap** |

- Команда `git worktree list --porcelain` подтвердила обе записи. Второе дерево создано от `main` на `470a9a6` и синхронизировано самой Мусей fast-forward до `fca51c2`, затем до `4c9515a` (merge PR #153) 2026-11-16, до `c5e5543` (merge PR #154) 2026-11-17 и до `58ce9f4` (merge PR #155) 2026-11-17 и до `c198659` (merge PR #156) 2026-11-17 и до `637d3ed` (merge PR #157) 2026-11-17 и до `125da46` (merge PR #158) 2026-11-17 и до `09cbd53` (merge PR #159) 2026-11-17 и до `649bf90` (merge PR #160) 2026-11-18 и до `9a431ed` (merge PR #161) 2026-11-18; **никаких implementation-изменений внутри не выполнялось**.
- `origin/main` на момент проверки: `9a431ed338abcf4affdf41b8c93132bb437ac040`; локальный `main` оригинального checkout — `fca51c2` (позади remote; владелец подтянет сам, Muse исходный checkout не трогает).
- Исходный checkout: tracked-файлы clean, имеется один **посторонний untracked** `ystemctl status cloudflared.service`. Не удалять, не включать в PR и не пытаться очищать.
- Ветку `muse/slavikai-rebuild` использовать как служебную точку возвращения, не коммитить тикеты в неё. На каждый тикет Muse создаёт новую PR-ветку от актуального `origin/main`.
- Третьи worktree или копии без отдельного согласования не создавать; параллельные implementation-тикеты запрещены.
- **Worktree не backup**: исходный code baseline зафиксирован immutable commit `470a9a63280f9dac392f559a6fe99115db2e50d0` (merge PR #151); текущий `main` — `9a431ed338abcf4affdf41b8c93132bb437ac040` (E01-T07 auto preset, PR #161). Историю и изменения исходного checkout не переписывать.

## PR #151 — завершён

- [PR #151](https://github.com/Phonezzzz/Slavikai/pull/151): **MERGED** 2026-11-16, commit `470a9a63280f9dac392f559a6fe99115db2e50d0`.
- GPT subscription и исправления Desktop/approval/UI находятся в `main`; PR #151 больше не блокирует E00 и архитектурную очередь.
- Muse не занимается исправлением/очисткой старого PR без нового поручения.

## Docs bootstrap: PR #152 — merged

- [PR #152](https://github.com/Phonezzzz/Slavikai/pull/152), branch `docs/muse-roadmap-2026-11-16`: **MERGED** 2026-11-16, GitHub Rebase and merge, новый `main` — `fca51c2fdcecd5d2b37c554105af44e3d925f348`.
- Содержимое: `ROADMAP.md`, `STATE.md`, `slavikai-how-to-work-with-muse.md`, `AGENTS.md`, `docs/workflow/dev_workflow.md`. Runtime-код не меняется.
- Gates пройдены на последнем HEAD: GitHub CI success, независимое Codex review без замечаний. Два P1 документационных противоречия (merge-gate и Outcome/ADR-0002), найденные Codex ранее, исправлены в PR.
- Локальный `make git-check` для этой docs-ветки не выполнялся; проверками были успешный GitHub CI и чистое Codex review.
- **Статус:** `DONE`. E01 не начинать до завершения E00-T03.

## STATE sync: PR #153 — merged

- [PR #153](https://github.com/Phonezzzz/Slavikai/pull/153), branch `pr-e00-t02-state-sync`: **MERGED** 2026-11-16, ручной fast-forward push, новый `main` — `4c9515aed2aa2a509656ba1ca927e7440de649d8`.
- Содержимое: только `STATE.md` (16 insertions / 16 deletions) — сверка с merged PR #152: E00-T02 DONE, E00-T03 IN_PROGRESS.
- Gates пройдены на последнем HEAD `4c9515a`: GitHub CI success (оба `quality` pass), независимое Codex review без замечаний (commit `4c9515a`, `has_any_findings: false`), `make git-check` OK. Qodo недоступен (подписка неактивна) — необязательный reviewer, merge не блокирует.
- Служебная ветка `muse/slavikai-rebuild` обновлена fast-forward `fca51c2..4c9515a`.
- **Статус:** `DONE`. E00-T02 done.

## E00-T03: PR #154 — merged

- [PR #154](https://github.com/Phonezzzz/Slavikai/pull/154), branch `pr-e00-t03-evidence-checklist`: **MERGED** 2026-11-17, ручной fast-forward push, новый `main` — `c5e5543ed495d5c4a1d1393544cc5c6988b45dbf`.
- Содержимое: evidence checklist (40 утверждений, baseline `fca51c2`) в ROADMAP.md + STATE.md sync. Codex review первого HEAD дал 3 P2-finding'а (claim 12, W3, claim 15) — все подтвердились по коду и исправлены коммитом `c5e5543` (W3 отменена: skill-match действительно всегда форсит route=mwv).
- Gates пройдены на последнем HEAD `c5e5543`: GitHub CI success (оба `quality` pass), независимое Codex review без замечаний (`has_any_findings: false`), `make git-check` OK. Qodo недоступен (подписка неактивна) — необязательный reviewer, merge не блокирует.
- Служебная ветка `muse/slavikai-rebuild` обновлена fast-forward `4c9515a..c5e5543`.
- **Статус:** `DONE`. E00 закрыт полностью (T01, T02, T03).

## E01-T01: PR #155 — merged

- [PR #155](https://github.com/Phonezzzz/Slavikai/pull/155), branch `pr-e01-t01-entrypoints`: **MERGED** 2026-11-17, ручной fast-forward push, новый `main` — `58ce9f4b572f0a52c3b90bcd2477ac0a0f74596b`.
- Содержимое: entrypoints matrix (23 строки, baseline `c5e5543`) в ROADMAP.md + STATE.md sync. Codex review первого HEAD дал 5 P2-finding'ов — все подтвердились по коду и исправлены коммитом `58ce9f4` (github_import fenced, auto verifier, durable admission ≠ recovery, worker REPLAN_REQUIRED, /v1 desktop 400).
- Gates пройдены на последнем HEAD `58ce9f4`: GitHub CI success (quality), независимое Codex review без замечаний (`has_any_findings: false`), `make git-check` OK. Qodo недоступен (подписка неактивна) — необязательный reviewer, merge не блокирует.
- Служебная ветка `muse/slavikai-rebuild` обновлена fast-forward `c5e5543..58ce9f4`.
- **Статус:** `DONE`. E01-T01 done.

## E01-T02: PR #156 — merged

- [PR #156](https://github.com/Phonezzzz/Slavikai/pull/156), branch `pr-e01-t02-runspec`: **MERGED** 2026-11-17, ручной fast-forward push, новый `main` — `c1986596cafc558921a8748d4eda84c221c5597e`.
- Содержимое: immutable RunSpec/RunPolicy contract (`core/run_contract.py`, 256 строк; recursive freeze FrozenDict/FrozenList), snapshot policy в `ToolGateway`, `set_execution_policy` убран из run path. Codex review первого HEAD дал 1 P2-finding (nested RunPolicy data mutable через snapshot) — подтвердился по коду и исправлен коммитом `9a142b1`; status-line коммит `c198659`.
- Gates пройдены на последнем HEAD `c198659`: GitHub CI success (оба `quality` pass), независимое Codex review без замечаний (`has_any_findings: false`, `status: completed`), `make check` зелёный (2478 passed, coverage 83.52%). Qodo недоступен (подписка неактивна) — необязательный reviewer, merge не блокирует.
- Служебная ветка `muse/slavikai-rebuild` обновлена fast-forward `58ce9f4..c198659`.
- **Статус:** `DONE`. E01-T02 done.

## E01-T03: PR #157 — merged

- [PR #157](https://github.com/Phonezzzz/Slavikai/pull/157), branch `pr-e01-t03-registry-listing`: **MERGED** 2026-11-17, ручной fast-forward push, новый `main` — `637d3ed1bf89d92452d70f924ac3ab83bd315a43`.
- Содержимое: ToolRegistry listing на RunSpec (`tools/tool_registry.py`; legacy no-policy path — sandbox catalog, `call()` без policy — effective target sandbox), 6 новых тестов в `tests/test_tool_registry_listing.py`, 5 обновлённых тестов, ROADMAP.md record + STATE.md sync.
- Gates пройдены на последнем HEAD `637d3ed`: GitHub CI success (оба `quality` pass), независимое Codex review без замечаний (`has_any_findings: false`, `status: completed`), `make check` зелёный (2484 passed, coverage 83.53%), `make git-check` OK. Qodo недоступен (подписка неактивна) — необязательный reviewer, merge не блокирует.
- Служебная ветка `muse/slavikai-rebuild` обновлена fast-forward `c198659..637d3ed`.
- **Статус:** `DONE`. E01-T03 done.

## E01-T06: PR #160 — merged

- [PR #160](https://github.com/Phonezzzz/Slavikai/pull/160), branch `pr-e01-t06-plan-artifact`: **MERGED** 2026-11-18, ручной fast-forward push, новый `main` — `649bf906a9cf6b829176cfd7a5a97f312c02f398`.
- Содержимое: Plan как artifact, Execute как обычный run — новый `core/plan_execute.py` (validate/execute/resume approved plan через canonical loop, `PlanLoopCheckpoint` с привязкой workspace root), перепроводка `run_plan_runner` на canonical loop, ROADMAP.md record + STATE.md sync.
- Codex review первого HEAD дал 5 finding'ов (3 P1 + 2 P2: повторное выполнение после ошибки resume, workspace-контекст loop, сверка args шага, потеря выполненных вызовов при отмене, очистка чекпоинтов при удалении сессии) — все подтвердились по коду и исправлены; ещё 4 раунда re-review дали по 1–2 новых finding'а (потеря выполненного шага при недоступности следующего инструмента, повтор вызова шага, потеря вывода неподтверждённого шага, привязка continuation к workspace root, потеря данных read-only шага) — все подтверждены и исправлены, всего 5 раундов.
- Gates пройдены на последнем HEAD `649bf90`: GitHub CI success (оба `quality` pass), независимое Codex review без замечаний (`has_any_findings: false`), `make check` зелёный (2551 passed, coverage 83.80%; известный флейк `tests/test_embedding_download.py` — сетевой download-тест, не регрессия). Qodo недоступен (подписка неактивна) — необязательный reviewer, merge не блокирует.
- Служебная ветка `muse/slavikai-rebuild` обновлена fast-forward `09cbd53..649bf90`.
- **Статус:** `DONE`. E01-T06 done.

## E01-T07: PR #161 — merged

- [PR #161](https://github.com/Phonezzzz/Slavikai/pull/161), branch `pr-e01-t07-auto-policy-preset`: **MERGED** 2026-11-18, ручной fast-forward push, новый `main` — `9a431ed338abcf4affdf41b8c93132bb437ac040`.
- Содержимое: Auto → policy preset — новый `core/auto_preset.py`, удалены `core/auto_runtime.py` / `core/auto_agent.py` / `tests/test_auto_runtime.py`, новый `tests/test_auto_preset.py` с parity-тестом, перепроводка `Agent`/`AgentToolsMixin`/UI-контракта, claims registry sync, ROADMAP.md record + STATE.md sync.
- Локальные гейты: `make preflight` на чистой ветке зелёный (2552 passed, coverage 83.82%); `make check` зелёный (2553 passed, coverage 83.75%).
- Служебная ветка `muse/slavikai-rebuild` обновлена fast-forward `649bf90..9a431ed`.
- **Статус:** `DONE`. E01-T07 done; E01-T08 next.

## E01-T08: ветка pr-e01-t08-remove-legacy-mwv-routing — PR #162 (открыт, gates pending)

- Ветка `pr-e01-t08-remove-legacy-mwv-routing` от `origin/main` `9a431ed` (merge PR #161). PR #162 открыт; merge — после GitHub CI и чистого Codex review, ручной fast-forward push по соглашению.
- Содержимое: удаление legacy classifier/MWV production routing — удалены `core/agent_mwv.py`, `core/mwv/routing.py`, `core/mwv/manager.py`, `core/mwv/coding_skill.py`, `coding_task.py`, `single_attempt.py`, `worker.py`, `interfaces.py`; `classify_request`/`ManagerRuntime`/`MWVRunResult`/`run_task_packet`/`cancel_mwv_approval` удалены из production; default `runtime_mode` — `ask`, fallthrough — в обычный chat loop; root-gate больше не срабатывает (его триггером был legacy classifier); новый gate-тест `tests/test_legacy_mwv_routing_removed.py`; claims registry sync (3 claims); доки переписаны (`ROUTING_POLICY.md`, `MWV_FLOW.md` — stub); ROADMAP.md record + STATE.md sync.
- Локальные гейты: `make check` зелёный (2435 passed, coverage 83.19%).
- **Статус:** IN_REVIEW. E01-T08 done (по локальным гейтам); E01-T09 next.

## Текущая очередь

- **E00-T01 — DONE** (read-only baseline, разрешение PR #151, проверка двух worktree). Evidence: Git `worktree list`, local/remote `main` SHA, merge status PR #151.
- **E00-T02 — DONE** (публикация и согласование пяти Markdown-файлов PR #152; сверка STATE.md PR #153). Evidence: PR #152 merged 2026-11-16, PR #153 merged 2026-11-16, GitHub CI success, Codex review без замечаний, `main` — `4c9515a`.
- **E00-T03 — DONE** (evidence checklist с verified call paths добавлена в ROADMAP.md). Evidence: таблица 40 утверждений в ROADMAP.md (E00-T03 evidence checklist, baseline `fca51c2`), три поправки W1/W2/W3, пять unknown U1–U5; ветка `pr-e00-t03-evidence-checklist`.
- **E01-T01 — DONE** (матрица production entrypoints: 23 строки, каждый path помечен production/production (broken)/dead, baseline `c5e5543`). Evidence: таблица в ROADMAP.md (E01-T01 execution entrypoints matrix), 10 ключевых выводов, 3 unknown; ветка `pr-e01-t01-entrypoints`.
- **E01-T02 — DONE** (immutable RunSpec/RunPolicy contract: `core/run_contract.py`, snapshot policy в `ToolGateway`, `set_execution_policy` убран из run path; baseline `58ce9f4`). Evidence: запись в ROADMAP.md (E01-T02 implementation), 17 тестов в `tests/test_run_contract.py` (оба acceptance-критерия + fail-on-old-code проверка), `make check` зелёный (2478 passed, coverage 83.52%); ветка `pr-e01-t02-runspec`.
- **E01-T03 — DONE** (ToolRegistry listing на RunSpec: доступность инструментов вычисляется из target/policy/qualification, legacy no-policy path — sandbox catalog и не следует глобальному target; baseline `c198659`). Evidence: запись в ROADMAP.md (E01-T03 implementation), 6 новых тестов в `tests/test_tool_registry_listing.py` (все acceptance-критерии + fail-on-old-code проверка), `make check` зелёный (2484 passed, coverage 83.53%); ветка `pr-e01-t03-registry-listing`.
- **E01-T04 — DONE** (ToolGateway — единственная execution gate для model-originated вызовов: аудит всех production tool calls + instrumentation gate-тесты; baseline `637d3ed`). Evidence: запись в ROADMAP.md (E01-T04 implementation; fork по UI workspace reads resolved решением владельца 2026-11-17, опция В — отдельный user-initiated read-only path, аудит — E08-T08), 4 новых теста в `tests/test_tool_gateway_gate.py` (acceptance по трём механизмам dispatch + negative gate), `make check` зелёный (2488 passed, coverage 83.54%); ветка `pr-e01-t04-gateway`.
- **E01-T05 — DONE** (канонизирован structured outcome без переписывания ADR-0002: `core/run_outcome.py` (TerminalReason/Disposition/RunPhase/Verification с evidence, инварианты в конструкторе, `communication_class`, `is_full_done`, `ui_label` — 9 меток, `final_success` публикуется только при подтверждённой postcondition); деривации `outcome_from_mwv` / `outcome_from_loop_result`; `_mwv_outcome_label` выводит русскую строку из канонического outcome; baseline `125da46`). Evidence: запись в ROADMAP.md (E01-T05 implementation; известный остающийся fake-DONE — `_execute_mwv_plan_step`, E01-T08), 36 новых тестов в `tests/test_run_outcome.py` (инварианты, маппинг, state-changing dummy без postcondition — не DONE, `tool.ok` alone — не proof, exhaustive UI projection matrix); ветка `pr-e01-t05-outcome`.
- **E01-T06 — DONE** (Plan как artifact, Execute как обычный run: новый `core/plan_execute.py` — validate/execute/resume approved plan через canonical loop; `run_plan_runner` — свежие запуски только canonical loop, resume старых MWV-чекпоинтов — legacy-ветка; baseline `09cbd53`). Evidence: запись в ROADMAP.md (E01-T06 implementation + mini-audit; исправлен порядок compile пакета в прямом act-пути; ревизия — CAS-счётчик, контент привязан хешем), 11 новых тестов в `tests/test_plan_execute.py` + переведённый на canonical loop `tests/ui_api/test_modes_and_plan.py` (20 кейсов + lifecycle + draft/approve без write tools), `make check` зелёный (2538 passed, coverage 83.73%); ветка `pr-e01-t06-plan-artifact`.
- **E01-T07 — DONE** (Auto → policy preset: новый `core/auto_preset.py` — Auto как policy preset поверх canonical loop; удалены `core/auto_runtime.py`, `core/auto_agent.py`, `tests/test_auto_runtime.py` (театр planner/coders/merge и фейковые PASSED); честный verifier `_verify_after_loop` + завершение по канонической деривации `outcome_from_loop_result`; baseline `649bf90`). Evidence: запись в ROADMAP.md (E01-T07 implementation + mini-audit; parity-тест `test_auto_preset_parity_write_scenario` — настоящий `WriteFileTool` через canonical loop, файл создан, COMPLETED, `verifier is None`, `files_touched == 1`; claims registry: `runtime.auto.v1_tool_loop` → `runtime.auto.preset_canonical_loop`), новый `tests/test_auto_preset.py` + 9 портированных тестовых файлов, `make check` зелёный (2553 passed, coverage 83.75%); ветка `pr-e01-t07-auto-policy-preset`.
- **E01-T08 — DONE** (удалён legacy classifier/MWV production routing: `core/agent_mwv.py`, `core/mwv/routing.py`, `core/mwv/manager.py` и MWV-путь чата удалены; default `runtime_mode` — `ask`; root-gate больше не срабатывает). Evidence: запись в ROADMAP.md (E01-T08 implementation + mini-audit; отступления от брифа — `set_runtime_state` не нормализует режимы, root-gate не срабатывает), новый `tests/test_legacy_mwv_routing_removed.py` (9 тестов) + переведённые тесты, `make check` зелёный (2435 passed, coverage 83.19%); ветка `pr-e01-t08-remove-legacy-mwv-routing`, PR #162.

## Readiness: 63 последовательных тикета

| Epic | IDs | Count | Status |
|---|---|---:|---|
| E00 | T01–T03 | 3 | E00-T01 done; E00-T02 done; E00-T03 done |
| E01 | T01–T09 | 9 | E01-T08 done; E01-T09 next |
| E02 | T01–T08 | 8 | READY после E01 |
| E03 | T01–T11 | 11 | READY после E02 |
| E04 | T01–T06 | 6 | NEEDS_REFINEMENT |
| E05 | T01–T07 | 7 | NEEDS_REFINEMENT |
| E06 | T01–T06 | 6 | NEEDS_REFINEMENT |
| E07 | T01–T05 | 5 | NEEDS_REFINEMENT |
| E08 | T01–T08 | 8 | NEEDS_REFINEMENT |

## Обязательный цикл Muse

1. Проверить `git worktree list`, `git status`, branch, current HEAD и `origin/main` во **втором** дереве. Исходный checkout не изменять.
2. Прочитать `AGENTS.md`, `docs/workflow/dev_workflow.md`, соглашение, roadmap и `STATE.md`; брать ровно один тикет.
3. На чистой PR-ветке от актуального `origin/main` выполнить `make preflight`, реализовать тикет, targeted/E2E tests, `make check`; push и открыть PR.
4. Дождаться успешного GitHub CI и независимого чистого Codex review для **последнего** commit, устранить blockers; выполнить `make git-check` и повторить fast-forward eligibility checks.
5. **Вручную** выполнить FF push PR-ветки в `origin/main` из дерева Muse (`git push origin <pr-branch>:refs/heads/main`) только если политика GitHub допускает. Если нет — STOP и обратиться к владельцу, без обхода.
6. Проверить точный SHA удалённого `main`, GitHub `merged`, обновить служебную ветку в worktree Muse fast-forward, записать доказательства, только затем следующий тикет. PR-ветки пока не удалять.
7. Любая содержательная архитектурная развилка — `BLOCKED: REAL_IMPLEMENTATION_FORK`, варианты/trade-offs владельцу. `sudo` — отдельное явное согласование команды.

## Resume

**Next active:** E01-T09 — regression E2E canonical loop. **Do not repeat:** PR #151, `git reset`, `git clean`, `git stash`, GitHub auto-merge, создание третьего worktree, правки в исходном дереве.
