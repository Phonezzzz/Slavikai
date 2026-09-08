# Open Questions

- **Для чего:** фиксировать открытые вопросы, требующие ответа до или в процессе проектирования.
- **Сюда:** вопрос, почему он важен, варианты/статус, кто или что может дать ответ.
- **Не сюда:** уже отвеченные или решённые вопросы.
- **Обновлять:** при появлении или закрытии вопроса.

## Task / Run Lifecycle Audit — communication boundary

**Статус research:** lifecycle-facing semantic findings зафиксированы; полный
`User Interaction / Communication Architecture Audit` остаётся отдельным будущим
scope. Отдельный lifecycle contract в текущем repository ещё не выделен.

### Установленные findings

- Authoritative lifecycle state принадлежит task/run workflow, а не communication
  layer. Направление зависимости: `authoritative lifecycle state -> communication
  decision`; текст assistant сам по себе не переводит task в `completed`, `failed`
  или `cancelled`.
- `progress_update` и `final_response` — разные semantic output classes. Token
  streaming, tool-call events, Computer activity events и background notifications
  могут быть transport/observability signals, но не являются progress update по
  умолчанию и не являются вторым conversational channel.
- Progress допускается при существенном изменении user-visible execution picture:
  significant intermediate result, переход в long-running/background phase,
  `blocked`/`waiting_user_input`/`waiting_approval`, recovery/replan, partial
  verification, существенный plan/ETA change или degraded execution mode. Нужна
  semantic filtering policy; каждый internal event не публикуется пользователю.
- Long-running определяется совокупностью semantic признаков, а не фиксированным
  числом секунд. Thresholds остаются implementation-defined.
- Two-phase означает два класса коммуникации, а не ровно два сообщения:
  `progress -> progress -> final` допустимо, как и один `final` для короткой задачи.
- Final response разрешён только после принятого authoritative terminal outcome:
  verification/acceptance должны быть завершены настолько, насколько требует
  workflow, и lifecycle должен быть `completed` либо соответствующим terminal outcome.
  Model `done`, artifact creation, конец tool loop или отсутствие tool call сами по
  себе недостаточны.
- Terminal communication различает successful, failed, cancelled и partial/incomplete
  outcomes. `waiting_*` и `blocked` — не completion.
- Final readiness, authoritative completion, response generation, delivery и delivery
  acknowledgement — разные состояния. Crash/reconnect/retry требуют idempotency и
  replay policy, которые остаются отдельным design scope.
- Workers/siblings не получают права напрямую публиковать user-facing progress:
  typed internal coordination state/events агрегируются orchestrator/communication
  decision layer с duplicate suppression и явным отражением unresolved conflicts.
- User-visible communications могут возвращаться в context как history, referenced
  artifact или bounded projection, но progress prose не становится authoritative state.
- User message во время run классифицируется как clarification, requirement change,
  cancel, status request или unrelated question; оно не означает автоматическую отмену.

### Current implementation gap

Текущий runtime публикует `response.ready`, `agent.respond.*`, token deltas,
`chat.*` stream/tool events и auto-progress через UIHub; workflow state отдельно
нормализуется как `running/completed/failed/cancelled`, а approval может остановить
execution в `waiting_approval`. Не найдены отдельные authoritative
`progress_update`/`final_response` artifacts, semantic publication gate, terminal
acceptance gate или recovery/idempotent delivery contract. Текущий streaming поэтому
не считается реализацией discovered requirement.

### Следующий architecture scope

Нужен отдельный `User Interaction / Communication Architecture Audit` для semantic
filtering, communication artifacts, delivery/replay/idempotency, interruption,
multi-agent aggregation и context projection. Он не должен создавать новый
conversational lane или превращать Computer activity events во второй чат.
