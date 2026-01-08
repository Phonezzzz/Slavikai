# RUNTIME_MWV_FLOW — план интеграции (без wiring)

Документ описывает будущий runtime‑контур Manager → Worker → Verifier.
Текущий статус: **код‑каркас без подключения к agent.py**.

## 1) Как routing решает маршрут

Источник: `core/mwv/routing.py` (M2).

Решение:
- `route = "chat"` → обычный ответ без инструментов.
- `route = "mwv"` → запуск MWV‑контура (в будущем).

Никакого автомата в runtime сейчас нет — это только политика.

## 2) Будущий MWV‑контур (описание потока)

1) **Manager**
   - Получает `messages`, `user_input`, `RunContext`.
   - Формирует `TaskPacket`.
2) **Worker**
   - Получает `TaskPacket` + `RunContext`.
   - Работает в изолированной среде и возвращает `WorkResult`.
3) **Verifier (детерминированный)**
   - Запускает `scripts/check.sh` (или канонический набор команд).
   - Возвращает `VerificationResult`.
4) **Решение**
   - `passed` → изменения применяются.
   - `failed` → bounded‑retry (будет позже).
   - `error` → остановка и отчёт.

## 3) Где будет точка входа (позже)

Кандидаты (после утверждения M4):
- `core/agent.py` — на уровне обработки запроса (после routing).
- `core/executor.py` — если нужно контролировать применение изменений.

**Важно:** сейчас эти файлы не трогаются.

## 4) Какие данные передаются между ролями

Минимальные модели из `core/mwv/models.py`:
- `TaskPacket`
- `WorkResult`
- `VerificationResult`
- `RunContext`

## 5) Retry‑модель (без цикла)

Структуры в `core/mwv/models.py`:
- `RetryPolicy`
- `RetryDecision`

Реальный цикл не включён; это только каркас.

## 6) Ограничения текущего этапа

- Нет wiring в runtime.
- Нет запуска verifier из agent.py.
- Никаких side‑effects на import.
