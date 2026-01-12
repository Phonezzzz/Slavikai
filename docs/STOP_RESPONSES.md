# STOP_RESPONSES — M12

Единый формат остановки используется для всех «stop» случаев:

- skill blocked (ambiguous / deprecated)
- approval_required (command lane)
- verifier_failed (MWV)
- mwv_internal_error

## Шаблон (обязательный)

```
Что случилось: <коротко, 1 строка>
Почему: <коротко, 1 строка>
Что делать дальше:
- <пункт 1>
- <пункт 2>
- <пункт 3>
trace_id=<uuid>   # если есть
MWV_REPORT_JSON=<json>
```

## Правила

- Всегда 1–3 пункта «Что делать дальше».
- Если trace_id отсутствует — строка не добавляется.
- Никаких скрытых обходов: stop‑ответ означает остановку выполнения.

## stop_reason_code (строго фиксирован)

| code | когда |
| --- | --- |
| BLOCKED_SKILL_AMBIGUOUS | матч навыка неоднозначен |
| BLOCKED_SKILL_DEPRECATED | навык помечен deprecated |
| APPROVAL_REQUIRED | требуется подтверждение действия |
| VERIFIER_FAILED | верификатор вернул FAIL/ERROR |
| MWV_INTERNAL_ERROR | ошибка MWV‑контуров |
| COMMAND_LANE_NOTICE | командный режим (без MWV) |
| WORKER_FAILED | выполнение шагов завершилось ошибкой |

## MWV_REPORT_JSON (стабильный блок)

Пример для stop‑ответа:

```
MWV_REPORT_JSON={"route":"mwv","trace_id":"...","attempts":{"current":1,"max":2},"verifier":{"status":"fail","duration_ms":1200},"next_steps":["..."],"stop_reason_code":"VERIFIER_FAILED"}
```

Минимум для chat‑route:

```
MWV_REPORT_JSON={"route":"chat","trace_id":null}
```
