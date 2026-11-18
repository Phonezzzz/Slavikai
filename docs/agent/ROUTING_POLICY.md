# ROUTING_POLICY — текущая маршрутизация (E01-T08)

E01-T08 удалил legacy classifier/MWV routing: `core/mwv/routing.py`
(`classify_request`, `RouteDecision`), `core/mwv/manager.py` (`ManagerRuntime`),
`core/agent_mwv.py` и весь MWV-путь чата удалены. Keyword-маршрутизации по
тексту запроса больше нет: ни regex-триггеры, ни skill match не запускают
исполнение.

Действующие режимы (`Agent.respond` / `respond_stream`, `core/agent_routing.py`):
- `runtime_mode=ask` (default): прямой chat-ответ через `AgentToolLoop`
  (`_run_chat_response` / `_run_chat_response_stream`).
- `runtime_mode=auto`: Auto preset (`handle_auto_command`); skill deprecated —
  блок-ответ, ambiguous skill — decision-пакет, иначе auto preset.
- `runtime_mode=desktop`: Desktop runtime.
- Любой другой режим (`act`, `plan`, неизвестный) — fail-closed в обычный
  chat loop: skill-решение берётся напрямую из `SkillIndex.match_decision`
  (deprecated — блок, ambiguous — decision-пакет, no_match — запись
  в inbox/candidate), исполнения MWV нет.

`set_runtime_state` режимы НЕ нормализует: `core/plan_execute.py` вызывает
`set_runtime_state(mode="act")` и требует `policy.mode == "act"` для шагов
canonical loop (E01-T06). Гарантия «нет маршрута в MWV» — default `ask` +
fallthrough в chat loop, а не нормализация.

## Skill routing (жёстко)

- `matched` → chat loop (skill фиксируется в `_last_skill_match`, исполнения
  навыка как MWV-задачи нет).
- `deprecated` → **блок‑ответ** с инструкцией (указать замену или переформулировать запрос).
- `ambiguous` → структурированный decision‑пакет с выбором навыка (`select_skill`), без исполнения.
- `no_match` → chat loop; в legacy fallthrough (act/plan) запрос пишется
  в inbox/candidate.

## Root-gate

`_request_requires_root_gate` (`server/http/handlers/ui_chat.py`) больше не
срабатывает: его единственным триггером был legacy classifier — gate решал,
пойдёт ли запрос в MWV-исполнение. MWV удалён, а auto preset сам управляет
бюджетами и approval'ами; без classifier'а честно отличить chat-like запрос
от execution-like нельзя. Функция и call site сохранены: gated-путь
(202 + `chat.run_root` decision) остаётся рабочим механизмом и покрыт тестами
через monkeypatch; fingerprint-проверка при `root_context` не зависит от
этого флага.

## Command lane

См. `docs/agent/COMMAND_LANE_POLICY.md` — command lane теперь debug-only.
Разрешены только `/trace` и `/end-session`; tool-like команды больше не являются маршрутом выполнения.

## Stop responses

Единый формат остановки см. в `docs/agent/STOP_RESPONSES.md`.

## Ограничения

- Политика **не** меняет смысл честного verifier (`core/mwv/verifier_runtime.py`).
- Командный режим остаётся ручным и явно помечается.
- Policy не является tool planner и не извлекает аргументы tools из текста.

Целевой путь исполнения: LLM `tool_calls` -> `ToolGateway` -> `role="tool"`;
утверждённые планы — canonical loop (`core/plan_execute.py`).
Regression-контракт: само наличие tool names в тексте запроса
(`workspace_read`, `workspace_write`, `image_generate`, `web` и т.п.) ничего
не запускает — исполнение только через LLM `tool_calls` или утверждённый план.
