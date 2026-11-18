# MWV_FLOW — удалён в E01-T08

Manager → Worker → Verifier путь чата удалён: `core/agent_mwv.py`,
`core/mwv/routing.py` (`classify_request`), `core/mwv/manager.py`
(`ManagerRuntime`), `core/mwv/coding_skill.py`, `coding_task.py`,
`single_attempt.py`, `worker.py`, `interfaces.py` удалены в E01-T08
(ветка `pr-e01-t08-remove-legacy-mwv-routing`, PR #162).

Что заменило:
- чат/skill запросы — обычный chat loop (`AgentToolLoop`) или Auto preset;
- утверждённые планы — canonical loop (`core/plan_execute.py`);
- resume старых MWV-чекпоинтов отклоняется fail-closed, без эффектов.

Что осталось (не routing):
- типы `TaskPacket`/`RunContext`/`VerificationResult` (`core/mwv/models.py`);
- честный verifier (`core/mwv/verifier.py`, `core/mwv/verifier_runtime.py`) —
  его используют auto preset, desktop runtime и plan loop;
- `outcome_from_mwv` (`core/run_outcome.py`) — чистая деривация
  work+verification, без `MWVRunResult`.

Подробности — в `docs/agent/ROUTING_POLICY.md` и в записи E01-T08 в `ROADMAP.md`.
