# North Star

- **Для чего:** фиксировать долгосрочную целевую точку, на которую выравниваются архитектурные решения.
- **Сюда:** желаемое конечное состояние, целевую ценность для пользователя и владельца, ключевые ограничения направления.
- **Не сюда:** текущая реализация, roadmap, разбивка задач, детальный дизайн.
- **Обновлять:** только при реальном изменении продуктового или стратегического направления; редко.

## Mandatory target outcomes

- SlavikAI использует несколько явно различимых model access routes и не связывает capability
  с одним provider или billing mode.
- Пользователь может подключить собственный поддерживаемый OpenAI/ChatGPT account и использовать
  доступ, предоставленный его consumer subscription, как отдельный first-class access route.
- Subscription-backed route не подменяется OpenAI API key/billing и не переключается молча на
  другой account, provider, trust/privacy или data-egress boundary.
- Если compliant access path или entitlement не подтверждены, SlavikAI честно показывает route
  как unavailable/blocked; скрытый Web UI bypass не считается выполнением Target.
- Новая работа может запускаться по расписанию или внешнему событию без нового сообщения
  пользователя только в пределах заранее утверждённых им правил. Trigger не выдаёт себе
  права на инструменты и не объявляет задачу завершённой.

Нормативное решение и границы неизвестного зафиксированы в
[`../decisions/ADR-0001-subscription-backed-openai-access.md`](../decisions/ADR-0001-subscription-backed-openai-access.md).
Это обязательная продуктовая capability, но не утверждение о наличии текущей runtime
implementation или о выбранном auth/transport механизме.

Автономный запуск работы принят отдельно в
[`../decisions/ADR-0003-preapproved-autonomous-work-initiation.md`](../decisions/ADR-0003-preapproved-autonomous-work-initiation.md).
Его owner закреплён в
[`../decisions/ADR-0004-work-initiation-ownership-boundary.md`](../decisions/ADR-0004-work-initiation-ownership-boundary.md),
а подробный firing contract остаётся для следующего этапа;
это не утверждение о наличии scheduler в текущем runtime.
