# MIGRATION_MWV_PLAN — переход на Manager → Worker → Verifier

Документ задаёт план миграции и минимальные контракты. Код НЕ меняет текущий runtime.
Verifier — детерминированный, НЕ LLM. Истина проверки — `scripts/check.sh`.

## 0) Этапы миграции (M0–M3)

### M0 — Freeze + Clean Interfaces (без удалений)
- Никаких новых фич вокруг старых режимов.
- Контракты ролей на бумаге (TaskPacket/WorkResult/VerificationResult/RunContext).
- Эталонные e2e‑сценарии (через `scripts/check.sh`).
- Scaffold в `core/mwv/*` без подключения к runtime.
- **DoD:** документы есть, тесты зелёные, core‑pipeline не тронут.

### M1 — Verifier как арбитр (минимально)
- Детерминированный VerifierRunner (обёртка над `scripts/check.sh`).
- Структурированный отчёт (exit code, stdout/stderr, команда, длительность).
- **Без** интеграции в Agent/Executor.
- **DoD:** модуль есть, тесты есть, no runtime wiring.

### M2 — Один цикл Manager → Worker → Verifier
- Manager формирует `TaskPacket`.
- Worker производит изменения в изоляции и возвращает `WorkResult`.
- Verifier запускается, и **только зелёный** результат допускает применение изменений.
- Bounded‑retry + диагностический отчёт при фейле.
- **DoD:** минимум один e2e‑сценарий в тестах/ручной проверке.

### M3 — Удаление старых режимов
- После подтверждения устойчивого M2‑цикла.
- Чистка старых режимов из кода/доков/тестов.
- **DoD:** ни одного упоминания старых режимов, все проверки зелёные.

## 1) Роли

### Manager
- Принимает вход (запрос пользователя/системы).
- Формирует `TaskPacket` и управляет циклом попыток.
- Сохраняет trace/лог и сводку для пользователя.

### Worker
- Исполняет задачу (инструменты/код/правки) в изолированной рабочей области.
- Возвращает `WorkResult` с описанием изменений и диагностикой.

### Verifier (детерминированный)
- Запускает проверку `scripts/check.sh` (или строго заданный набор команд).
- Возвращает `VerificationResult`.
- «Зелёный» статус — единственный критерий успеха.

## 2) Контракты (JSON schemas + dataclass‑план)

Ниже минимальные схемы. Дополнительные поля допускаются только при явной фиксации.

### 2.1 TaskPacket

**JSON schema (минимум):**
```json
{
  "type": "object",
  "required": ["task_id", "session_id", "trace_id", "goal"],
  "properties": {
    "task_id": {"type": "string"},
    "session_id": {"type": "string"},
    "trace_id": {"type": "string"},
    "goal": {"type": "string"},
    "messages": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["role", "content"],
        "properties": {
          "role": {"type": "string", "enum": ["system", "user", "assistant", "tool"]},
          "content": {"type": "string"}
        }
      }
    },
    "constraints": {"type": "array", "items": {"type": "string"}},
    "context": {"type": "object"}
  }
}
```

**Dataclass‑план:**
- `TaskPacket(task_id, session_id, trace_id, goal, messages, constraints, context)`

### 2.2 WorkResult

**JSON schema (минимум):**
```json
{
  "type": "object",
  "required": ["task_id", "status", "summary"],
  "properties": {
    "task_id": {"type": "string"},
    "status": {"type": "string", "enum": ["success", "failure"]},
    "summary": {"type": "string"},
    "changes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["path", "change_type", "summary"],
        "properties": {
          "path": {"type": "string"},
          "change_type": {"type": "string", "enum": ["create", "update", "delete", "rename"]},
          "summary": {"type": "string"}
        }
      }
    },
    "tool_summaries": {"type": "array", "items": {"type": "string"}},
    "diagnostics": {"type": "object"}
  }
}
```

**Dataclass‑план:**
- `WorkResult(task_id, status, summary, changes, tool_summaries, diagnostics)`

### 2.3 VerificationResult

**JSON schema (минимум):**
```json
{
  "type": "object",
  "required": ["status", "command", "exit_code", "stdout", "stderr", "duration_seconds"],
  "properties": {
    "status": {"type": "string", "enum": ["passed", "failed", "error"]},
    "command": {"type": "array", "items": {"type": "string"}},
    "exit_code": {"type": ["integer", "null"]},
    "stdout": {"type": "string"},
    "stderr": {"type": "string"},
    "duration_seconds": {"type": "number"},
    "error": {"type": ["string", "null"]}
  }
}
```

**Dataclass‑план:**
- `VerificationResult(status, command, exit_code, stdout, stderr, duration_seconds, error)`

### 2.4 RunContext

**JSON schema (минимум):**
```json
{
  "type": "object",
  "required": ["session_id", "trace_id", "workspace_root", "safe_mode"],
  "properties": {
    "session_id": {"type": "string"},
    "trace_id": {"type": "string"},
    "workspace_root": {"type": "string"},
    "safe_mode": {"type": "boolean"},
    "approved_categories": {"type": "array", "items": {"type": "string"}},
    "max_retries": {"type": "integer", "minimum": 0},
    "attempt": {"type": "integer", "minimum": 1}
  }
}
```

**Dataclass‑план:**
- `RunContext(session_id, trace_id, workspace_root, safe_mode, approved_categories, max_retries, attempt)`

## 3) Правила bounded‑retry (будут применены в M2)

- **Базовый лимит:** `max_retries = 2` (итого максимум 3 попытки: первичная + 2 повтора).
- **Условия остановки:**
  - Verifier = `passed` → успех, изменения принимаются.
  - Verifier = `failed` → следующая попытка Worker с диагностикой, если лимит не исчерпан.
  - Verifier = `error` → остановка (это инфраструктурная ошибка, требует ручного разбора).
  - `approval_required` → остановка до решения пользователя (не считать попыткой).
- **Эскалация:** после исчерпания лимита возвращаем отчёт с последним `VerificationResult` и краткими рекомендациями.

## 4) Стратегия rollback / изоляции (будет применена в M2)

Выбранная стратегия (предпочтительно):
- **Изолированная рабочая область** (tmp‑dir или отдельная git‑ветка).
- Изменения применяются к основному workspace **только после зелёного Verifier**.
- При `failed/error` — изоляция удаляется/откатывается, основной workspace не трогается.

## 5) Будущие точки интеграции (без внедрения сейчас)

Кандидаты на интеграцию Verifier/Manager цикла:
- `core/agent.py` (верхнеуровневая оркестрация запросов).
- `core/executor.py` (шаги работы с инструментами).
- `core/tool_gateway.py` (гейтинг действий, если потребуются доп. проверки).

**Важно:** текущий этап — только план и scaffold. Никакой wiring не выполняется.

## 6) Риски и ограничения (для контроля)
- Нельзя запускать LLM как Verifier — только `scripts/check.sh` или эквивалент.
- Любая авто‑правка должна быть за gated‑проверкой.
- Auto‑save памяти не должен включаться автоматически без явного решения.
