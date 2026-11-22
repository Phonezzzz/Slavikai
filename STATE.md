# SlavikAI Worker State

**Source of truth:** [ROADMAP.md](ROADMAP.md) · [Рабочее соглашение Muse](slavikai-how-to-work-with-muse.md) · [Git workflow](docs/workflow/dev_workflow.md). Это зафиксированный снимок от 2026-11-18, синхронизирован с merge PR #162 (main `d36fd64`); перед любым действием Muse перепроверяет Git, CI, review и актуальный `main`.

## Два Git worktree — подтверждено на ПК владельца 2026-11-16

| Назначение | Путь | Ветка на момент фиксации | HEAD | Право Muse |
|---|---|---|---|---|
| Исходный проект владельца | `/home/ki/Project/Slavikai` | `main` | `fca51c2fdcecd5d2b37c554105af44e3d925f348` | **Только read-only без изменения checkout** |
| Перестройка под Muse | `/home/ki/Project/Slavikai-rebuild` | `muse/slavikai-rebuild` | `d36fd64e23e444c1304932d6a5c0000e6abfa8ef` | **Единственная рабочая директория для roadmap** |

- Команда `git worktree list --porcelain` подтвердила обе записи. Второе дерево создано от `main` на `470a9a6` и синхронизировано самой Мусей fast-forward до `fca51c2`, затем до `4c9515a` (merge PR #153) 2026-11-16, до `c5e5543` (merge PR #154) 2026-11-17 и до `58ce9f4` (merge PR #155) 2026-11-17 и до `c198659` (merge PR #156) 2026-11-17 и до `637d3ed` (merge PR #157) 2026-11-17 и до `125da46` (merge PR #158) 2026-11-17 и до `09cbd53` (merge PR #159) 2026-11-17 и до `649bf90` (merge PR #160) 2026-11-18 и до `9a431ed` (merge PR #161) 2026-11-18 и до `d36fd64` (merge PR #162) 2026-11-18; **никаких implementation-изменений внутри не выполнялось**.
- `origin/main` на момент проверки: `d36fd64e23e444c1304932d6a5c0000e6abfa8ef`; локальный `main` оригинального checkout — `fca51c2` (позади remote; владелец подтянет сам, Muse исходный checkout не трогает).
- Исходный checkout: tracked-файлы clean, имеется один **посторонний untracked** `ystemctl status cloudflared.service`. Не удалять, не включать в PR и не пытаться очищать.
- Ветку `muse/slavikai-rebuild` использовать как служебную точку возвращения, не коммитить тикеты в неё. На каждый тикет Muse создаёт новую PR-ветку от актуального `origin/main`.
- Третьи worktree или копии без отдельного согласования не создавать; параллельные implementation-тикеты запрещены.
- **Worktree не backup**: исходный code baseline зафиксирован immutable commit `470a9a63280f9dac392f559a6fe99115db2e50d0` (merge PR #151); текущий `main` — `d36fd64e23e444c1304932d6a5c0000e6abfa8ef` (E01-T08 legacy MWV removal, PR #162). Историю и изменения исходного checkout не переписывать.

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

## E01-T08: ветка pr-e01-t08-remove-legacy-mwv-routing — PR #162 (MERGED, main d36fd64)

- Ветка `pr-e01-t08-remove-legacy-mwv-routing` от `origin/main` `9a431ed` (merge PR #161). PR #162 MERGED 2026-11-18, ручной fast-forward push по соглашению, новый `main` — `d36fd64`.
- Содержимое: удаление legacy classifier/MWV production routing — удалены `core/agent_mwv.py`, `core/mwv/routing.py`, `core/mwv/manager.py`, `core/mwv/coding_skill.py`, `coding_task.py`, `single_attempt.py`, `worker.py`, `interfaces.py`; `classify_request`/`ManagerRuntime`/`MWVRunResult`/`run_task_packet`/`cancel_mwv_approval` удалены из production; default `runtime_mode` — `ask`, fallthrough — в обычный chat loop; root-gate больше не срабатывает (его триггером был legacy classifier); новый gate-тест `tests/test_legacy_mwv_routing_removed.py`; claims registry sync (3 claims); доки переписаны (`ROUTING_POLICY.md`, `MWV_FLOW.md` — stub); ROADMAP.md record + STATE.md sync.
- Локальные гейты: `make check` зелёный (2435 passed, coverage 83.19%).
- **Статус:** DONE. E01-T08 done; E01-T09 DONE (PR #163 MERGED, main 0f894c6).

## E04-T03 (slice): штатный process verifier — ветка pr-e04-t03-slice-process-verifier (DONE — PR #164 MERGED, main 7affb3f)

- Ветка `pr-e04-t03-slice-process-verifier` от `origin/main` `d36fd64`. Узкий срез E04-T03 как prerequisite перед закрытием E01-T09; один активный тикет.
- Содержимое: отдельный pinned image `python:3.12.16-bookworm` с make (Id `28d30c15…`, digest `sha256:3c3ba8ad…`) и его квалификация (профиль `i1-ubuntu-podman-5.7-command-bounded-v1`, Python profile не тронут); `create_command_container` с allowlist ровно `["make", "-B", "-r", "-f", "Makefile", "check"]`; no-publish исполнитель на staged snapshot (`release_unpublished`); опциональный `executor` в `VerifierRuntime` с маппингом PASSED/FAILED/ERROR; fail-closed `_verify_after_loop` (backend недоступен → ERROR, не `None`); два условия владельца в спеках: (1) запуск `make check` явно разрешён для конкретного run и привязан к principal/workspace/qualified environment — наличие Makefile разрешения не даёт; (2) критерий проверки задаёт доверенная сторона до запуска агента, хэш `Makefile` фиксируется до loop, несовпадение → ERROR. Доказательство: верный `result.txt` → PASSED, неверный → FAILED, недоступный backend → ERROR без запуска процесса; полный `make check` репозитория в sandbox — вне scope. ROADMAP.md record (E04-T01 решение, E04-T03 slice, E01-T09 зависимость) + STATE.md sync.
- **Статус:** DONE. PR #164 MERGED 2026-11-19 (main 7affb3f); возврат к PR #163 выполнен — E2E без `_ExecutingVerifierRuntime` через штатный путь. E01-T09 закрыт merge PR #163 (main 0f894c6).

- **Статус:** `DONE`. E01-T08 done; E01-T09 DONE (PR #163 MERGED, main 0f894c6); E02 READY.

## E01-T09: ветка pr-e01-t09-e2e-canonical-loop — PR #163 (MERGED, main 0f894c6)

- Ветка `pr-e01-t09-e2e-canonical-loop` от `origin/main` `7affb3f` (merge PR #164, срез E04-T03). PR открыт; merge — после GitHub CI и чистого Codex review, ручной fast-forward push по соглашению.
- Содержимое: первый немоканный acceptance test канонического цикла — новый `tests/test_canonical_loop_e2e.py` (7 тестов): scripted brain вызывает настоящие read/write tools через настоящие ToolRegistry + ToolGateway + AgentToolLoop; ветви acceptance: fail tool → FAILED_WORKER (preset, fail-closed gate), missing verifier → UNVERIFIED (не DONE), confirmed postcondition → DONE через production-деривацию; preset-level: COMPLETED + `verifier=None` без fake verification, preset run со штатным canonical verifier → DONE на `canonical_outcome`, preset run без backend → FAILED_VERIFIER (`sandbox_execution_unavailable`), failing verifier → FAILED_VERIFIER. Production-проводка (коммит d5b1cca): `_execute_frame()` передаёт `_verify_after_loop()` в `outcome_from_loop_result()` через `Verification.from_verification_result()`; новый E2E доказывает цепочку runner → verifier result → DONE штатным `VerifierRuntime` на квалифицированном backend. ROADMAP.md record + STATE.md sync; E01 T01–T09 DONE, E02 READY.
- Локальные гейты: `make check` зелёный (2486 passed, coverage 83.11%); real-backend E2E: DONE 36.48s, failing 37.08s; оба sensitivity-красных прогона на штатном пути падают.
- **Статус:** DONE. PR #163 MERGED 2026-11-19 (main 0f894c6; CI зелёный, Codex review без findings). E01 9/9 DONE; E02 READY.


## E02-T01: ветка pr-e02-t01-browser-service — PR #166 (MERGED, main d8f0498)

- Ветка `pr-e02-t01-browser-service` от `origin/main` (`eed97a7`, docs E01 9/9 DONE, E02 READY; PR #165 MERGED).
- Содержимое: новый `core/browser_service.py` (`BrowserProfile` + `BrowserService`, протокол `BrowserAutomationBackend` переехал из tool); `DesktopBrowserTool` — тонкий адаптер над сервисом (injectable `service`/`profile`); production wiring — `BrowserProfile.default(owner_id=self.user_id)` в `core/agent.py`; generic actions reused без переписывания; claim `runtime.browser.session_service` (partial). ROADMAP.md record (implementation + мини-аудит) + STATE.md sync.
- Локальные гейты: `make check` зелёный (2497 passed, coverage 83.16%); новый `tests/test_browser_service.py` (11 тестов).
- **Статус:** DONE. PR #166 MERGED 2026-11-19 (main d8f0498; CI зелёный, Codex review без findings).

## E02-T02: ветка pr-e02-t02-persistent-profile — PR #167 (MERGED, main a4932bc)

- Ветка `pr-e02-t02-persistent-profile` от `origin/main` (`d8f0498`, PR #166 MERGED).
- Содержимое: dedicated persistent Chromium profile — `resolve_browser_profile_dir` / `validate_browser_profile_dir` в `core/browser_service.py` (`~/.local/state/slavikai/browser-profiles/<owner>/<profile>`), `launch_persistent_context` в `PlaywrightBrowserBackend` (disposable путь без `user_data_dir` сохранён), profile-aware default factory в `DesktopBrowserTool`; claim `runtime.browser.session_service` version 1 → 2. ROADMAP.md record (implementation + мини-аудит) + STATE.md sync.
- Локальные гейты: `make check` зелёный (2506 passed, 4 skipped, coverage 83.24%); новый `tests/test_browser_persistence.py` (10 тестов, в т.ч. opt-in real-browser E2E: login → restart → `cookie=yes storage=yes`, 1.44s).
- **Статус:** DONE. PR #167 MERGED 2026-11-19 (main a4932bc; CI зелёный, Codex review без findings — 1×P1 про production principal исправлен quote-кодированием, коммит a4932bc).

## E02-T03: ветка pr-e02-t03-profile-lease — PR #180 (MERGED, main f2e94ba)

- Ветка `pr-e02-t03-profile-lease` от `origin/main` (`a4932bc`, PR #167 MERGED).
- Содержимое: один active writer lease на profile — новый `core/browser_lease.py` (`flock` на `<lease-root>/<owner>/<profile>.lease` в отдельном lease root `<profile-root>.leases`, `fstat`-квалификация как в `ManagedWorkspacePool`, `profile_busy` с pid держателя, `recovered_from_pid` после health validation каталога), `lease_provider` в `BrowserService` (lease на время сессии, release после `backend.close()`), lease provider в default persistent factory `DesktopBrowserTool` (injected factory — без lease); claim `runtime.browser.session_service` version 2 → 3. ROADMAP.md record (implementation + мини-аудит + дизайн-решение про рестарт) + STATE.md sync.
- Локальные гейты: `make check` зелёный (2519 passed, 5 skipped, coverage 83.27%); новый `tests/test_browser_profile_lease.py` (14 тестов, в т.ч. opt-in real-browser E2E: login → второй run `profile_busy` → close → второй run видит залогиненный state; negative — SIGKILL держателя → recoverable после health validation).
- Rework после Codex review (round 1, 4×P2): отдельный lease root (без коллизий с profile id), отказ symlink-алиасам профиля/owner'а/lease-каталога, единственное открытие lease с `O_NOFOLLOW | O_NONBLOCK` и чтение holder record через тот же fd (FIFO → явный отказ без зависания), каталог профиля `0700` (новый — при любом umask, существующий открытый — приводится к `0700`). `tests/test_browser_profile_lease.py` — 23 теста; `make check` зелёный (2528 passed, 5 skipped, coverage 83.27%).
- Rework после Codex review (round 2, 1×P2): lease root и `user_data_dir` выводятся из одного канонического profile root — алиас root (symlink на него) получает тот же lease, второй run через алиас → `profile_busy`. `tests/test_browser_profile_lease.py` — 25 тестов; `make check` зелёный (2530 passed, 5 skipped, coverage 83.27%).
- **Статус:** DONE. PR #180 MERGED 2026-11-19 (main f2e94ba).

## E02-T04: ветка pr-e02-t04-login-handoff (MERGED, main 30bd0da)

- Ветка `pr-e02-t04-login-handoff` от `origin/main` (`f2e94ba`, PR #180 MERGED).
- Решение владельца 2026-11-19 — вариант A: существующий continuation, фаза `WAITING_USER_INPUT`, причина `NEEDS_USER_AUTH`; тот же persistent profile headed, lease через перезапуск и всю попытку; «Готово» — только владельцу run, ровно один раз, затем re-observation; таймаут 5 мин → окно закрыто, lease отпущен после подтверждённого закрытия, явный retry.
- Содержимое: `core/user_auth.py`, операция `login_handoff` в `DesktopBrowserTool` (headed-перезапуск, одноразовое продолжение, маскировка password/OTP перед `aria_snapshot`), `user_auth`-пауза в `ToolLoopContinuation`/`DesktopRuntime`/`ToolGateway`, lease отпускается только после успешного close, decision `user_auth` в server/UI; claim `runtime.browser.session_service` version 3 → 4. ROADMAP.md record (решение, implementation, находка про `aria_snapshot`, тесты, мутации).
- Локальные гейты: `make check` зелёный (2545 passed, 6 skipped, coverage 83.39%); новые `tests/test_login_handoff.py` (12 + opt-in E2E) и `tests/ui_api/test_login_handoff_api.py` (3).
- Rework после Codex review (round 1, 2×P1 + 2×P2; решения владельца 2026-11-19 19:11): маскировка/восстановление рекурсивно по открытым shadow roots + подтверждение маскировки до snapshot (неподтверждена → отказ, snapshot модели не уходит; real headless shadow-тест); Desktop policy перечитывается на «Готово»/retry (ask/deny во время паузы действуют); `ApprovalRequest.user_auth_attempt_id` — approval на продолжение завершает ту же попытку; ошибка неподтверждённого закрытия сохраняется на backend (в т.ч. при перезапуске headless → headed), lease не отпускается; таймаут 5+5 мин принят, отмена run после второго дедлайна отражается в API/UI (decision `expired`); streaming-путь отказывает `login_handoff` до открытия окна. Тесты: +5 в `tests/test_login_handoff.py` (+ opt-in real shadow E2E), +3 в `tests/ui_api/test_login_handoff_api.py`, +3 фронтенд; мутации (7) убиты; `make check` зелёный (2553 passed, 7 skipped, coverage 83.39%).

- Rework после Codex review (round 2, 1×P1 + 1×P2; план владельца 2026-11-20): observation после handoff маскирует все непустые текстовые поля (MFA-код в обычном поле без autocomplete не уходит модели; неподтверждённая маскировка → явный отказ; real headless-тесты plain MFA/shadow/textarea + getter-only поле); `UIHub` публикует переход decision в `expired` по `expires_at` событием `decision.packet` без отдельного GET (открытая вкладка показывает «Вход не выполнен»); respond на истёкший decision — `410 decision_expired` независимо от источника перехода. Тесты: +2 fake и +2 opt-in real в `tests/test_login_handoff.py`, +1 в `tests/ui_api/test_login_handoff_api.py`; мутации (4) убиты; `make check` зелёный (2556 passed, 9 skipped, coverage 83.39%).
- Rework после Codex review (round 3, 1×P1; уточнение владельца 2026-11-20): строгая маскировка — состояние сессии, а не операция `_finish_login_handoff()`: страница ручного входа остаётся в строгом режиме (`_strict_masking_pages`) с открытия окна до закрытия вкладки, все её observations (`snapshot` и завершение handoff — единственный механизм `_observation`) идут через строгую маскировку с подтверждением; снятие — только явное закрытие вкладки, другие страницы не затронуты; observation сообщает `strict_masking`. Тесты: +3 fake (цепочка handoff → done → snapshot; новая вкладка не строгая; конец режима по close_tab) и +1 opt-in real headless в `tests/test_login_handoff.py`; мутации (3) убиты; `make check` зелёный (2559 passed, 10 skipped, coverage 83.40%).
- **Статус:** DONE. PR #181 MERGED 2026-11-22 (main 30bd0da; CI зелёный, Codex review без findings).

## E02-T05: ветка pr-e02-t05-effect-classification (IN_REVIEW)

- Ветка `pr-e02-t05-effect-classification` от `origin/main` (`30bd0da`, PR #181 MERGED).
- Содержимое: новый `core/browser_effects.py` — явная таблица effect-классов generic browser operations (read / network / local_mutation / write / external_side_effect / user_handoff; неизвестная операция fail-closed в external_side_effect); `core/desktop_policy.py` берёт risk class из таблицы вместо inline-sets; claim `runtime.browser.session_service` version 4 → 5. Поведение: чтение страницы и ввод в поля без approval по умолчанию (явные ask/deny действуют), submit/click — external side effect, download — write (destination sub-action без изменений), click не наследует read. ROADMAP.md record (implementation + мини-аудит + тесты + мутации).
- Локальные гейты: `make check` зелёный (2588 passed, 10 skipped, coverage 83.41%); новый `tests/test_browser_effects.py` (29 тестов); мутации (4) убиты.
- **Статус:** IN_REVIEW. Merge — после GitHub CI и чистого Codex review.

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
- **E01-T09 — DONE** (regression E2E canonical loop: первый немоканный acceptance test — scripted brain, настоящие read/write tools, ToolRegistry + ToolGateway + AgentToolLoop, observation, postcondition; ветви: fail tool → FAILED_WORKER, missing verifier → UNVERIFIED, confirmed postcondition → DONE; preset run со штатным canonical verifier → DONE на `canonical_outcome`). Evidence: запись в ROADMAP.md (E01-T09 implementation; находка — `workspace_read` обязан регистрироваться с `capability="read"`, иначе file-effects accounting preset'а отклоняет write; BLOCKED как терминал не покрыт — нетерминальная фаза, approval-пауза в `tests/test_auto_preset.py`; P1 round 5 закрыт — E2E на штатном пути без подмены), новый `tests/test_canonical_loop_e2e.py` (7 тестов), `make check` зелёный (2486 passed, coverage 83.11%); ветка `pr-e01-t09-e2e-canonical-loop`; PR #163 MERGED (main 0f894c6), см. ROADMAP.md.
- **E02-T01 — DONE** (BrowserProfile + BrowserService minimal API: lifecycle сессии отделён от DesktopBrowserTool; generic actions идут через service). Evidence: запись в ROADMAP.md (E02-T01 implementation + мини-аудит), новый `tests/test_browser_service.py` (11 тестов), `make check` зелёный (2497 passed, coverage 83.16%); ветка `pr-e02-t01-browser-service`; PR #166 MERGED (main d8f0498).
- **E02-T02 — DONE** (dedicated persistent Chromium profile: `user_data_dir` на профиль, `launch_persistent_context`, login переживает restart; основной Chrome-профиль владельца выбрать нельзя). Evidence: запись в ROADMAP.md (E02-T02 implementation + мини-аудит), новый `tests/test_browser_persistence.py` (10 тестов), `make check` зелёный (2506 passed, 4 skipped, coverage 83.24%); ветка `pr-e02-t02-persistent-profile`; PR #167 MERGED (main a4932bc).
- **E02-T03 — DONE** (один active writer lease на profile: `flock` на lease-файле в отдельном lease root, второй run → `profile_busy`, crash holder → recoverable после health validation, алиасы профилей отклоняются). Evidence: запись в ROADMAP.md (E02-T03 implementation + мини-аудит + дизайн-решение + rework rounds 1–2), новый `tests/test_browser_profile_lease.py` (25 тестов), `make check` зелёный (2530 passed, 5 skipped, coverage 83.27%); ветка `pr-e02-t03-profile-lease`; PR #180 MERGED (main f2e94ba).
- **E02-T04 — DONE** (manual login handoff по решению владельца, вариант A: `NEEDS_USER_AUTH` / `WAITING_USER_INPUT`, headed-перезапуск того же профиля под lease, одноразовое «Готово» владельца → re-observation, таймаут 5 мин → окно закрыто, lease после подтверждённого закрытия, явный retry; пароль/MFA не в model messages, tool log, effect log и snapshot). Evidence: запись в ROADMAP.md, `tests/test_login_handoff.py` + `tests/ui_api/test_login_handoff_api.py` + фронтенд-тест, rework round 1 (2×P1 + 2×P2 закрыты), round 2 (1×P1 + 1×P2 закрыты) и round 3 (1×P1 закрыт), `make check` зелёный (2559 passed, 10 skipped, coverage 83.40%); ветка `pr-e02-t04-login-handoff`; PR #181 MERGED (main 30bd0da).
- **E02-T05 — IN_REVIEW** (effect classification browser actions: одна таблица `core/browser_effects.py` — read/network/local_mutation/write/external_side_effect/user_handoff, неизвестная операция fail-closed; ввод в поля — local mutation без approval по умолчанию, submit/click — external side effect, download — write, click не наследует read). Evidence: запись в ROADMAP.md (implementation + мини-аудит), новый `tests/test_browser_effects.py` (29 тестов, мутации 4/4), `make check` зелёный (2588 passed, 10 skipped, coverage 83.40%); ветка `pr-e02-t05-effect-classification`.

## Readiness: 63 последовательных тикета

| Epic | IDs | Count | Status |
|---|---|---:|---|
| E00 | T01–T03 | 3 | E00-T01 done; E00-T02 done; E00-T03 done |
| E01 | T01–T09 | 9 | E01 T01–T09 DONE |
| E02 | T01–T08 | 8 | T01 DONE (PR #166); T02 DONE (PR #167); T03 DONE (PR #180); T04 DONE (PR #181); T05 IN_REVIEW |
| E03 | T01–T11 | 11 | READY после E02 |
| E04 | T01–T06 | 6 | T03 slice DONE (PR #164); остальное NEEDS_REFINEMENT |
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

**Next active:** E02-T05 — IN_REVIEW (ветка `pr-e02-t05-effect-classification`). **Do not repeat:** PR #151, `git reset`, `git clean`, `git stash`, GitHub auto-merge, создание третьего worktree, правки в исходном дереве.
