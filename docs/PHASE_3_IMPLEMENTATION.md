# PHASE 3 - HTTP Gateway Implementation

## Как запускать сервер

По умолчанию:
```bash
python -m server
```

Параметры (env):
- `SLAVIK_HTTP_HOST` (default: `127.0.0.1`)
- `SLAVIK_HTTP_PORT` (default: `8000`)
- `SLAVIK_HTTP_MAX_REQUEST_BYTES` (default: `1000000`)

Опционально можно задать конфиг-файл:
`config/http_server.json`:
```json
{
  "host": "127.0.0.1",
  "port": 8000,
  "max_request_bytes": 1000000
}
```
Env‑переменные имеют приоритет над файлом.

## Эндпоинты
- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /slavik/trace/{trace_id}`
- `GET /slavik/tool-calls/{trace_id}`
- `POST /slavik/feedback`
- `POST /slavik/approve-session`

## Примеры curl

### /v1/models
```bash
curl http://127.0.0.1:8000/v1/models
```

### /v1/chat/completions
```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "slavik-single",
    "messages": [
      { "role": "user", "content": "Привет" }
    ],
    "stream": false
  }'
```

### /slavik/trace/{trace_id}
```bash
curl http://127.0.0.1:8000/slavik/trace/TRACE_ID
```

### /slavik/tool-calls/{trace_id}
```bash
curl http://127.0.0.1:8000/slavik/tool-calls/TRACE_ID
```

### /slavik/feedback
```bash
curl -s http://127.0.0.1:8000/slavik/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "interaction_id": "TRACE_ID",
    "rating": "good",
    "labels": ["too_long"],
    "free_text": "Ок"
  }'
```

### /slavik/approve-session
```bash
curl -s http://127.0.0.1:8000/slavik/approve-session \
  -H 'Content-Type: application/json' \
  -d '{ "session_id": "SESSION_ID" }'
```

## Как подключать Open WebUI (Phase 6)
- Base URL: `http://127.0.0.1:8000/v1`
- Модели: `slavik-single`, `slavik-dual`, `slavik-critic`

## Известные ограничения (Phase 3)
- Streaming отключён (HTTP 400 `streaming_not_supported`).
- Tool‑calling в запросе возвращает `tool_calling_not_supported`.
- Trace/tool‑calls берутся из текущих логов (`logs/trace.log`, `logs/tool_calls.log`).
- Session approvals хранятся только в памяти процесса и сбрасываются при перезапуске.
