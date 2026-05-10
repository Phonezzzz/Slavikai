# ROUTING_POLICY — current legacy routing

Документ фиксирует текущую legacy-маршрутизацию между режимами:
- `chat` — ответы без инструментов (объяснения/консультации).
- `mwv` — задачи с изменениями/инструментами (Manager → Worker → Verifier).

Политика применяется в runtime (`Agent.respond`) только для режимов, где ещё включена
keyword/classifier маршрутизация.

- `runtime_mode=ask`: классификатор не используется (прямой chat-ответ).
- `runtime_mode=act|plan`: используется `chat|mwv` классификация.
- `runtime_mode=auto`: классификатор не используется; запрос сразу идёт в auto v1 path
  (`AutoAgent.run_outcome() -> run_v1()`).

Новые tool-capabilities не должны добавляться в этот keyword router. Целевой путь:
LLM `tool_calls` -> `ToolGateway` -> `role="tool"` message.
Regression-контракт: native tool names сами по себе (`workspace_read`,
`workspace_write`, `image_generate`, `web` и т.п.) остаются `chat/no_triggers`;
router не является tool planner.

## Правила (жёстко, без магии)

Маршрут `mwv`, если обнаружены явные триггеры:
- “изменить/написать код”, “рефактор”, “почини тесты”, “добавь фичу”.
- команды/инструменты: `git`, `pip`, `npm`, `docker`, `systemctl`, `sudo`.
- файловые действия: patch/write/delete/overwrite и т.п.

Маршрут `chat`, если запрос — объяснение/теория/консультация, без явных действий.

## Что отдаёт политика

`RouteDecision`:
- `route`: `"chat"` или `"mwv"`.
- `reason`: короткая причина (для логов).
- `risk_flags`: список флагов (например `code_change`, `tools`, `filesystem`, `install`, `git`, `sudo`).

## Примеры (таблица)

| Пример запроса | Route | Почему |
| --- | --- | --- |
| “что такое git” | chat | объяснение, нет действий |
| “как работает sudo” | chat | объяснение, нет действий |
| “исправь тесты” | mwv | кодовые изменения (`code_change`) |
| “рефактор модуля оплаты” | mwv | кодовые изменения (`code_change`) |
| “добавь фичу логирования” | mwv | кодовые изменения (`code_change`) |
| “git commit -m 'x'” | mwv | git‑действие (`git`, `tools`) |
| “сделай PR” | mwv | git‑действие (`git`, `tools`) |
| “pip install requests” | mwv | установка deps (`install`, `tools`) |
| “npm install” | mwv | установка deps (`install`, `tools`) |
| “sudo systemctl restart nginx” | mwv | sudo/системная команда (`sudo`, `tools`) |
| “systemctl restart nginx” | mwv | системная команда (`tools`) |
| “удали файл README.md” | mwv | файловое действие (`filesystem`, `tools`) |
| “перезапиши файл config.yaml” | mwv | файловое действие (`filesystem`, `tools`) |
| “применить патч для README” | mwv | файловое действие (`filesystem`, `tools`) |
| “объясни, как работает TCP” | chat | объяснение, без инструментов |

## Skill routing (жёстко)

- `matched` → **всегда MWV**, без скрытого переключения.
- `deprecated` или `ambiguous` → **блок‑ответ** с инструкцией (указать skill, переформулировать запрос).
- `no_match` → маршрут определяется триггерами; если триггеры есть, MWV запускается и может создать candidate.

## Command lane

См. `docs/agent/COMMAND_LANE_POLICY.md` — command lane теперь debug-only.
Разрешены только `/trace` и `/end-session`; tool-like команды больше не являются маршрутом выполнения.

## Stop responses

Единый формат остановки см. в `docs/agent/STOP_RESPONSES.md`.

## Ограничения

- Политика **не** меняет смысл MWV/Verifier.
- Командный режим остаётся ручным и явно помечается как «без MWV».
- Policy не является tool planner и не извлекает аргументы tools из текста.
