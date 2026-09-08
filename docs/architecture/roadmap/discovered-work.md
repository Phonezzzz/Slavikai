# Discovered Work

- **Для чего:** вести backlog обнаруженных работ, которые ещё не запланированы.
- **Сюда:** описание работы, источник, приоритет, связь с capability.
- **Не сюда:** committed roadmap items, дубли.
- **Обновлять:** при обнаружении новой работы или переводе/удалении элемента.

## Two-phase communication for long-running tasks

- **Статус:** discovered requirement.
- **Источник:** product requirement, сформулированный владельцем проекта.
- **Приоритет:** не определён до появления общего механизма приоритизации discovered requirements.
- **Требование:** при длительной задаче SlavikAI должен разделять пользовательскую
  коммуникацию на две семантически разные фазы:
  1. Краткие промежуточные progress updates: что уже выяснено, что выполняется сейчас,
     есть ли препятствия или изменения плана. Они нужны для visibility/progress и не
     означают завершение задачи.
  2. Отдельный самодостаточный final response после завершения: значимый результат,
     важные решения и существенные ограничения с явным обозначением завершения текущей
     задачи. Он не должен требовать от пользователя перечитывать промежуточные сообщения.
- **Связанные capability candidates:** task lifecycle, user interaction/communication,
  long-running execution, progress reporting, final-result semantics.
- **Ограничение existing canon:** Chat остаётся единственным conversational entrypoint;
  требование не должно создавать новый lane или превращать Computer activity events во
  второй чат.
- **Граница:** detailed capability design отложен до завершения Capability Discovery.
- **Lifecycle/contract result:** requirement остаётся `discovered requirement`, но теперь
  покрыт нормативным target contract:
  `docs/architecture/USER_INTERACTION_COMMUNICATION_CONTRACT.md`. В repository нет
  установленного перехода `incorporated/resolved`, поэтому статус не выдумывается и не
  означает runtime implementation.
