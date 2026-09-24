# Decisions

- **Для чего:** вести реестр принятых архитектурных решений в формате, близком к ADR.
- **Сюда:** решение с контекстом, рассмотренными вариантами, выбором, последствиями, статусом и датой.
- **Не сюда:** current-state документация, research, roadmap.
- **Обновлять:** при принятии, изменении или замене решения.

## Accepted decisions

- [`ADR-0001-subscription-backed-openai-access.md`](ADR-0001-subscription-backed-openai-access.md)
  — subscription-backed OpenAI/ChatGPT access является обязательной Target capability;
  конкретный compliant auth/transport contract ещё требует research и отдельного design.
- [`ADR-0002-terminal-outcome-and-partial-result.md`](ADR-0002-terminal-outcome-and-partial-result.md)
  — terminal cause и принятый частичный результат различаются; final сохраняет причину остановки.
- [`ADR-0003-preapproved-autonomous-work-initiation.md`](ADR-0003-preapproved-autonomous-work-initiation.md)
  — mature Target включает запуск новых задач по расписанию/событию только по заранее утверждённым пользователем правилам; подробный firing contract ещё открыт.
- [`ADR-0004-work-initiation-ownership-boundary.md`](ADR-0004-work-initiation-ownership-boundary.md)
  — Work Initiation владеет approved rules и firing history, Lifecycle принимает task-creation intent; детали rule/firing protocol ещё открыты.
- [`ADR-0005-local-inference-operations-boundary.md`](ADR-0005-local-inference-operations-boundary.md)
  — операции локального inference engine имеют отдельного владельца от model route selection; provisioning и tuning остаются открытыми вопросами.
- [`ADR-0006-resource-event-evaluation-authority.md`](ADR-0006-resource-event-evaluation-authority.md)
  — shared resource allocation и system evaluation имеют отдельных владельцев; material history принадлежит authority факта, а UI/trace являются projections.
- [`ADR-0007-owner-provider-credential-delegation.md`](ADR-0007-owner-provider-credential-delegation.md)
  — owner provider key доступен только owner по умолчанию; use другими principals требует явного правила делегирования.
- [`ADR-0008-request-scoped-media-interaction.md`](ADR-0008-request-scoped-media-interaction.md)
  — voice/visual interaction остаётся набором отдельных запросов, без непрерывной media-session capability.
- [`ADR-0009-governed-external-tool-providers.md`](ADR-0009-governed-external-tool-providers.md)
  — внешние tool providers входят в Target и управляются отдельно от skills-инструкций.
- [`ADR-0010-supersession-lineage-without-separate-final.md`](ADR-0010-supersession-lineage-without-separate-final.md)
  — замена принятой task revision создаёт status/lineage link без отдельного final старой ревизии.
- [`ADR-0011-system-fact-ownership-and-cross-cutting-protocols.md`](ADR-0011-system-fact-ownership-and-cross-cutting-protocols.md)
  — каждый durable факт имеет одного logical owner; persistence/recovery/events и observability остаются сквозными протоколами.
