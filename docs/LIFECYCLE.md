# LIFECYCLE — жизненный цикл запроса

## 1) Вход

- HTTP: `/v1/chat/completions` или UI-endpoints `/ui/api/*`.
- Точка обработки: `Agent.respond(...)`.

## 2) Разветвление

- Если сообщение начинается с `/` -> command lane (`handle_tool_command`), без MWV.
- Иначе -> deterministic routing (`core/mwv/routing.py`): `chat` или `mwv`.

## 3) Chat lane

1. Сбор контекста (memory, feedback hints, workspace context, vector search).
2. Применение approved policies.
3. Вызов `Brain.generate(...)`.
4. Логирование trace и interaction.

## 4) MWV lane

1. `ManagerRuntime` формирует `TaskPacket`.
2. `WorkerRuntime` исполняет план и инструменты.
3. `VerifierRuntime` запускает проверку.
4. При fail возможен bounded retry.
5. Результат возвращается в формате MWV-ответа/stop-ответа.

## 5) Tool path

- Все вызовы идут через `ToolGateway` -> `ToolRegistry`.
- Safe-mode/approval-policy применяются до выполнения опасных действий.
- Каждый вызов инструмента логируется в `logs/tool_calls.log`.

## 6) Трассировка и аудит

- Trace: `logs/trace.log`.
- Tool calls: `logs/tool_calls.log`.
- Interaction/feedback/policies: `memory/memory_companion.db`.
