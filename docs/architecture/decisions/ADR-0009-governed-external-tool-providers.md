# ADR-0009 — Внешние tool providers входят в Target

- **Статус:** принятое продуктовое решение и системная граница
- **Дата:** 24.09.2026
- **Владелец решения:** владелец продукта SlavikAI
- **Затронутые области:** external extensions, skills, tool execution,
  identity/policy, credentials, provider egress, audit, artifacts

## Контекст

Текущий skill manifest выбирает версионированные инструкции; сам по себе
skill не выдаёт permissions. Внешний tool provider несёт исполняемые
capabilities, credentials, обновления и отдельную trust boundary. Gate 0
должен был решить, входит ли подключение таких providers в зрелый Target.

## Решение

1. Target **включает внешние tool providers/extensions**. Их подключение,
   версия, происхождение, доступные capabilities, enabled/revoked status и
   principal scope имеют отдельного владельца — External Extension
   Governance. Это граница ответственности, не выбор конкретного protocol
   или package manager.
2. Skills/Reusable Procedures владеют инструкциями и их eligibility/version.
   Skill text не устанавливает tool provider, не меняет permissions и не
   превращается в credential или executable authority. Skill может ссылаться
   только на уже разрешённые capabilities.
3. Каждый вызов внешнего инструмента проходит typed Tool Execution и
   effective principal/policy/approval/budget проверки для конкретного
   action/target. Provider output и metadata являются untrusted input.
   Credentials выдаются по отдельным правилам, включая ADR-0007 для owner
   provider keys; extension не получает их автоматически.
4. Installation/update/revocation, version pinning, audit, data egress,
   result provenance и failure/recovery должны быть определены до runtime
   активации. Отзыв provider делает новые вызовы недопустимыми; ранее
   произведённые artifacts сохраняют provenance и собственный retention.

## Почему

Владелец продукта включил внешние extensions. Версионированная инструкция
и исполняемый provider имеют разные права, секреты и жизненный цикл;
объединение их в один owner позволило бы тексту skill неявно расширять
полномочия исполнения.

## Рассмотренные варианты

- Только встроенные tools и skills: отклонено владельцем продукта.
- Управлять providers как частью skill manifest: отклонено из-за смешения
  instructions и executable trust boundary.
- Позволить внешнему provider самому объявлять effective permissions:
  отклонено; authority остаётся у SlavikAI principal/policy layer.

## Последствия

- OQ-CD-02 закрыт. System map содержит отдельные Skills/Procedures и
  External Extension Governance owners.
- Разделы 7/8/9/10/16/19 должны провести аудит transport, supply chain,
  credential use, egress, revocation и post-condition verification. ADR не
  утверждает наличие runtime интеграции или выбор MCP/другого протокола.
