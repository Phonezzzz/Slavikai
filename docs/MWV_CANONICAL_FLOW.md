# MWV_CANONICAL_FLOW — канонический поток Manager → Worker → Verifier

Документ фиксирует единственный допустимый runtime‑поток для MWV.
LLM‑критики и DualBrain не используются как арбитр.

## 1) Маршрутизация (chat vs mwv)

| Сценарий запроса | Route | Почему |
| --- | --- | --- |
| Объяснение/теория/консультация без инструментов | chat | Не требует изменений кода или действий инструментами |
| Любые изменения кода/файлов/инструментов | mwv | Нужна проверка Verifier, иначе нет успеха |
| Командный lane (fs/shell/git/etc.) | mwv | Инструменты = действия, нужна верификация |

Источник правил: `core/mwv/routing.py` (deterministic, rule‑based).

## 2) Канонический поток MWV

```
User Input
  -> Routing (chat | mwv)
     -> chat: LLM‑ответ, без planner/executor/MWV
     -> mwv:
          Manager -> TaskPacket
          Worker  -> WorkResult (diff/steps/tools)
          Verifier -> VerificationResult (scripts/check.sh)
          if FAIL and retries left: Manager уточняет задачу -> Worker -> Verifier
          if FAIL and no retries: STOP
```

## 3) Verifier — единственный арбитр

- Verifier детерминированный: выполняет `scripts/check.sh` (или зафиксированный эквивалент).
- Успех = только зелёный Verifier.
- Если Verifier FAIL → итог НЕ может быть “сделано”.

## 4) Retry (bounded)

- Максимум попыток: фиксированный лимит (см. `core/mwv/manager.py`).
- Каждая попытка отражается в ответе (attempt X/Y).
- Причина повторной попытки: только FAIL от Verifier.
- После исчерпания лимита → STOP с отчётом.

## 5) Остановка и блокировки

MWV обязан остановиться, если:
- Требуется approval (safe‑mode gate).
- Verifier FAIL после лимита попыток.
- Worker вернул FAILURE.

## 6) Выходные данные (UX‑минимум)

MWV‑ответ пользователю должен содержать:
- Короткий итог (PASS/FAIL).
- Статус Verifier.
- Изменения (файлы + краткий summary).
- Если FAIL — что делать дальше (1–3 пункта).
- Детали (логи/шаги) — отдельным блоком, коротко.

Chat‑ответы не включают MWV‑отчёт.
