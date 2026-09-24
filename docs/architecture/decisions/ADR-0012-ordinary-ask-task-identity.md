# ADR-0012 — Сохраняемая task identity для обычного Ask

- **Статус:** принятое продуктовое решение
- **Дата:** 24.09.2026
- **Владелец решения:** владелец продукта SlavikAI
- **Затронутые области:** Task/Run Lifecycle, User Communication, persistence/recovery,
  Context, Identity/Policy, observability

## Контекст и проблема

Неинтегрированный Lifecycle snapshot `6040640`, §26, оставляет открытым,
получает ли обычный короткий Ask durable logical task. Если у Ask есть только
UI session/message identity, после restart нельзя надёжно связать принятый
outcome, revision и semantic final с той же пользовательской работой.
Одновременно полный checkpoint исполнения каждого короткого ответа не нужен.

## Решение

1. Каждый принятый обычный Ask создаёт сохраняемую logical `task_id`, initial
   accepted task revision и отдельный run/attempt identity по общему Lifecycle
   contract. Они не выводятся из `session_id`, текста ответа или trace.
2. User-facing terminal outcome и **один semantic final** этой ревизии
   сохраняются для history/replay через restart в пределах принятой retention
   policy. Если execution оборвалось до принятого terminal outcome, сохранённая
   запись показывает unresolved/recovery state; система не восстанавливает
   `completed` из последнего текста или UI snapshot.
3. Полный durable execution checkpoint и автоматическое продолжение после
   crash обязательны только там, где это требует выбранный execution mode,
   side-effect/policy contract или обещание background continuation. Короткий
   Ask может завершиться после restart через явное recovery/abort decision,
   не притворяясь продолженным run.
4. Видимость history и final ограничена principal/audience и retention;
   хранение task identity само по себе не разрешает model/tool action, Memory
   promotion или повторный effect. ADR-0002/0010 определяют terminal cause,
   disposition и supersession final rule.

## Причина и альтернативы

Владелец продукта выбрал сохраняемые identity и final для каждого Ask с
избирательной долговечностью execution checkpoint. Полностью ephemeral Ask
теряет надёжную историю и final после restart. Полный background-style
checkpoint для каждого короткого Ask добавляет стоимость и recovery
обязательства без продуктовой необходимости.

## Последствия

- Lifecycle contract обязан охватывать Ask, Plan/Act, Auto и background с
  одинаковыми identity/final semantics, но разным checkpoint profile.
- Communication хранит и доставляет revision-bound semantic final, UI session
  остаётся projection/attachment.
- Точные schema, retention duration и recovery executor — последующее ТЗ.
  Решение не утверждает, что текущий Ask runtime это реализует.
