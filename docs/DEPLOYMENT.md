# DEPLOYMENT (production)

Документ описывает минимальный production-сценарий запуска SlavikAI.

## 1) Требования

- Linux/macOS сервер с Python 3.12+.
- `git`, `make`, `venv`.
- Для UI-сборки: Node.js + npm.

## 2) Быстрый запуск (xAI / OpenRouter / Local)

```bash
git clone <repo-url>
cd slavikai
make venv
make ui-build
```

Создай модельный конфиг `config/model_config.json`:

```json
{
  "main": {
    "provider": "xai",
    "model": "your-model-id",
    "temperature": 0.3
  }
}
```

`provider` поддерживает: `xai`, `openrouter`, `local`.

Экспортируй ключи/настройки:

```bash
# xAI
export XAI_API_KEY="..."

# optional: OpenRouter
export OPENROUTER_API_KEY="..."

# optional: Local endpoint
export LOCAL_LLM_URL="http://localhost:11434/v1/chat/completions"
export LOCAL_LLM_API_KEY=""

# важно: whitelist должен содержать выбранный model id
export SLAVIK_MODEL_WHITELIST="your-model-id"
```

Запуск сервера:

```bash
make run-prod PROD_HOST=0.0.0.0 PROD_PORT=8000
```

## 3) HTTP-конфиг (опционально)

Можно задать `config/http_server.json`:

```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "max_request_bytes": 1000000
}
```

Либо переопределять переменными окружения:

- `SLAVIK_HTTP_HOST`
- `SLAVIK_HTTP_PORT`
- `SLAVIK_HTTP_MAX_REQUEST_BYTES`

## 4) Проверка после старта

```bash
curl -sS http://127.0.0.1:8000/ui/api/status
curl -sS http://127.0.0.1:8000/v1/models
```

## 5) Фоновый режим (без systemd)

```bash
make up
make status
make logs
```

## 6) Пример systemd unit

```ini
[Unit]
Description=SlavikAI server
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/slavikai
Environment=SLAVIK_HTTP_HOST=0.0.0.0
Environment=SLAVIK_HTTP_PORT=8000
Environment=SLAVIK_MODEL_WHITELIST=your-model-id
Environment=XAI_API_KEY=your-key
ExecStart=/opt/slavikai/venv/bin/python -m server
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

После добавления:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now slavikai
sudo systemctl status slavikai
```

