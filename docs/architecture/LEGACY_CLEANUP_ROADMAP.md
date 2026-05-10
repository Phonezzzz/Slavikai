# Legacy Cleanup Roadmap After PR-17

Этот roadmap описывает оставшуюся расчистку legacy после PR-17. Он не объявляет
будущие PR уже реализованными: текущая фактическая архитектура описана в
`Architecture.md`, целевые границы — в `ARCH_CANON.md`.

## Already implemented

- PR-0..PR-17 находятся в `main`.
- Документация разложена по `docs/architecture`, `docs/agent`, `docs/for-humans`,
  `docs/workflow`, `docs/archive`.
- `AgentToolLoop` и native tool-calling contract существуют:
  `LLMMessage.role="tool"`, `LLMResult.tool_calls`, `Brain.generate(..., tools=...)`.
- `runtime_mode=ask` может использовать read-only chat tool loop.
- Primary `local` OpenAI-compatible provider поддерживает generic native tools.
- Chat/workspace send и SSE endpoints разделены.
- Command lane оставлен debug-only: `/trace`, `/end-session`.
- Terminal backend объединён в `TerminalTool`.
- Новые auto-запуски через `AutoAgent.run_outcome()` идут в auto v1:
  `AgentToolLoop -> ToolGateway -> verifier`.
- Runtime tools имеют LLM descriptions/JSON Schema.
- `runtime_mode=auto` больше не вызывает `classify_request(...)`; auto сразу идёт в
  `AutoAgent.run_outcome() -> run_v1()`.
- Legacy auto pipeline `planner -> coder pool -> merge -> verifier` больше не имеет
  runtime entrypoint.

## Known legacy

- `classify_request(...)` ещё участвует в legacy `plan|act` routing. Он не является
  tool planner.
- Storage физически ещё хранит `ui_messages.lane`.
- `lane` остаётся временным marker-ом старой модели, но не должен считаться
  domain discriminator.
- MWV/CodingTask legacy path ещё содержит regex target extraction и append-comment
  behavior.
- `Planner`/`Executor` ещё существуют как runtime entrypoints в части legacy tests/code.
- `server/http/handlers/ui_chat.py`, `server/ui_hub.py`,
  `use-session-runtime-controller.ts`, `use-session-transport.ts` и
  `workspace-ide.tsx` остаются крупными монолитами.
- `/ui/api/events/stream` ещё зарегистрирован как hard-failing legacy route.

## Cleanup workflow

Перед каждым implementation PR:

1. Обновить `main`: `git checkout main && git pull --ff-only`.
2. Создать отдельную ветку `pr-<n>-<slug>`.
3. Запустить baseline `make git-check`.
4. Выполнить mini non-mutating audit изменяемого контура.
5. Только после audit начинать implementation.
6. Перед merge запустить полный `make check`.

Mini audit нужен для поиска code/API/frontend/test references, а не для сохранения
старых локальных данных.

## Local UI storage policy

- Проект single-user/single-admin.
- Локальные UI sessions/chats/history disposable by default.
- Старую `.run/ui_sessions.db` сохранять не нужно.
- Если для упрощения storage/API нужен destructive reset/recreate локальной DB/schema,
  это допустимо.
- Не добавлять migrations, backward compatibility, import/export adapters или
  compatibility layers ради сохранения старых sessions/chats/history.
- Если audit показывает, что direct destructive split не проходит текущий runtime/tests
  без adapter/compatibility/migration, implementation должен остановиться и вернуть
  BLOCKED report. Compatibility layer автоматически не реализуется.

BLOCKED report должен включать:

1. какой PR / section заблокирован;
2. какой код/тест требует compatibility/adaptation;
3. почему direct destructive split не проходит;
4. варианты решения;
5. рекомендацию;
6. `git status --short`.

## Planned cleanup

1. PR-15 `pr-tool-descriptors-complete` — done
   - Заполнить descriptions и JSON Schema для runtime tools.
   - Проверить, что tools, видимые LLM, имеют usable `ToolSpec`.

2. PR-16 `pr-auto-no-classifier-gate` — done
   - Убрать `classify_request(...)` из auto path.
   - Auto должен идти в `AutoAgent.run_outcome() -> run_v1()` без route classifier.

3. PR-17 `pr-remove-legacy-auto-pipeline` — done
   - Удалить или изолировать legacy `AutoOrchestrator.run()` pipeline.
   - Оставить один auto execution path: `AgentToolLoop -> ToolGateway -> verifier`.

4. PR-18 `pr-mwv-worker-tool-loop`
   - Перевести MWV worker на real tool loop/gateway execution.
   - Убрать regex target extraction и append-comment fake worker behavior.

5. PR-19 `pr-kill-planner-executor-entrypoints`
   - Удалить `Planner`/`Executor` как runtime entrypoints.
   - Сохранить только typed contracts/adapters, если они нужны текущему TaskPacket flow.

6. PR-20 `pr-storage-destructive-physical-split`
   - Destructive/direct storage physical split.
   - Заменить `ui_messages(session_id, lane, ...)` на физические `chat_messages` и
     `workspace_messages`.
   - Не добавлять migrations, import/export adapters, backward compatibility или
     compatibility layers.
   - Destructive reset/recreate local DB/schema допустим.
   - Audit ищет references к `ui_messages`, `lane`, `/history?lane`,
     persisted session contracts, которые нужно удалить или переписать.

7. PR-21 `pr-ui-chat-workspace-handler-split`
   - Разделить chat/workspace handlers.
   - Убрать shared `payload_override` path как основной runtime path.

8. PR-22 `pr-remove-legacy-event-route`
   - Удалить `/ui/api/events/stream` route полностью.
   - Можно выполнить раньше только если audit докажет, что route уже не используется.

9. PR-23 `pr-frontend-runtime-split`
   - Разделить frontend chat/workspace runtime hooks/transports.
   - `lane` не должен управлять frontend runtime flow.

10. PR-24 `pr-agent-core-decompose`
    - Сжать `core/agent*.py` вокруг отдельных runtimes.
    - Новые capabilities добавляются через registry/schema, не через routing branches.

11. PR-25 `pr-decision-gateway-cleanup`
    - Свести decision cleanup к gateway approval/stop layer.
    - Удалить недостижимые decision branches после split runtimes.

12. PR-26 `pr-workspace-ide-decompose`
    - Разбить `workspace-ide.tsx` на меньшие компоненты без redesign.
    - Цель: снизить риск дальнейших workspace UI изменений.
