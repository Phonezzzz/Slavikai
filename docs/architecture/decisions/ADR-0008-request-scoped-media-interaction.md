# ADR-0008 — Медиа остаются отдельными запросами

- **Статус:** принятое продуктовое решение
- **Дата:** 24.09.2026
- **Владелец решения:** владелец продукта SlavikAI
- **Затронутые области:** interaction, voice, visual input, communication,
  artifacts, verification, policy, model access

## Контекст

Текущий runtime имеет отдельные STT/TTS endpoints и пути для изображений и
вложений. Это не доказывает непрерывную двустороннюю voice/visual session с
собственным turn-taking, interruption и media clock. Во время Gate 0 было
нужно решить, является ли такая сессия обязательной для зрелого Target.

## Решение

1. Зрелому SlavikAI достаточно **отдельных request/response media actions**:
   транскрибации, генерации речи, анализа изображения и передачи вложения.
   Непрерывная realtime voice/visual session с отдельным turn/media-session
   owner не входит в текущий Target.
2. Каждый media request привязывается к principal, task/session context,
   источнику, consent, retention и provider/egress policy. Media input и
   полученный artifact остаются evidence или untrusted data согласно
   общим Artifact/Context/Verification границам.
3. Результат STT или анализа изображения является входом/наблюдением, а не
   автоматическим изменением goal, approval или lifecycle state. TTS и
   доставка аудио проецируют принятую Communication семантику; сам звук не
   создаёт второй final.
4. Если позже понадобится постоянный duplex stream, barge-in или совместная
   визуальная сессия, это новое продуктовое решение с отдельным owner и
   границей отмены tool effects, а не неявное расширение этих adapters.

## Почему

Владелец продукта выбрал отдельные запросы. Это сохраняет единые task,
policy, artifact и communication contracts без дополнительного независимого
состояния непрерывной медиа-сессии. Текущие STT/TTS endpoints не должны
создавать ложное утверждение, что такой session owner уже реализован.

## Рассмотренные варианты

- Добавить постоянную voice/visual session как обязательную capability:
  отклонено для текущего Target.
- Считать STT/TTS полноценным realtime session runtime: отклонено, потому
  что отдельные запросы не задают turn-taking, interruption и recovery.

## Последствия

- OQ-CD-01 закрыт. Gate 0 классифицирует voice/visual как request-scoped
  interaction adapters и media artifacts, а не отдельную continuous-session
  capability.
- Разделы 5/7/16/18/19 позже задают media provenance, consent, provider
  egress, retention и проверку результата. ADR не подтверждает, что каждый
  текущий путь уже соблюдает эти требования.
