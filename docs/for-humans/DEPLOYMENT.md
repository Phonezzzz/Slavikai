# DEPLOYMENT (production)

Документ описывает минимальный production-сценарий запуска SlavikAI в текущей архитектуре backend+UI.

## 1) Требования

- Linux/macOS сервер с Python 3.12+.
- `git`, `make`, `venv`.
- Для UI-сборки: Node.js 20 (см. `.nvmrc`) + npm.

## 2) Быстрый beta-запуск (DeepSeek)

```bash
git clone <repo-url>
cd slavikai
cp .env.example .env
make install-beta
```

Минимальный `.env`:

```dotenv
SLAVIK_API_TOKEN=replace-with-a-long-random-token
DEEPSEEK_API_KEY=replace-with-your-deepseek-key
```

`provider` для chat-модели поддерживает: `xai`, `openrouter`, `local`, `inception`, `deepseek`.
`openai` в runtime используется только для STT-транскрипции (не как chat provider).

После входа выбери `Settings → API Keys → Default chat model`, загрузи live-список
DeepSeek models и сохрани выбранную модель. Конфиг `config/model_config.json` будет
создан автоматически; новые сессии наследуют эту модель.

Дополнительные ключи/настройки:

```bash
# xAI (optional)
export XAI_API_KEY="..."

# optional: OpenRouter
export OPENROUTER_API_KEY="..."

# optional: Local endpoint (OpenAI-compatible)
export LOCAL_LLM_URL="http://localhost:11434/v1/chat/completions"
export LOCAL_LLM_API_KEY=""

# optional: Inception endpoint
export INCEPTION_API_KEY="..."
export INCEPTION_API_URL="https://api.inceptionlabs.ai/v1"

# optional: OpenAI key for STT transcription endpoint
export OPENAI_API_KEY="..."

# выбранная в UI модель должна быть доступна у настроенного провайдера
```

Запуск сервера:

```bash
make run-prod PROD_HOST=0.0.0.0 PROD_PORT=8000
```

При открытии UI введи `SLAVIK_API_TOKEN` в форме входа. Сервер установит подписанную
`HttpOnly`, `SameSite=Strict` cookie; сам token в cookie не сохраняется. Для внешних API
клиентов остаётся Bearer auth: `Authorization: Bearer <SLAVIK_API_TOKEN>`.

Это текущая legacy browser auth, а не реализация целевого owner/member deployment за
Cloudflare Access. Статус security claims см. в `docs/runtime_contract_claims.json`.

Для локальной разработки без auth можно явно задать
`SLAVIK_ALLOW_UNAUTH_LOCAL=true`. На публичном интерфейсе этот режим не использовать.

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
curl -sS http://127.0.0.1:8000/healthz
curl -sS -H "Authorization: Bearer $SLAVIK_API_TOKEN" http://127.0.0.1:8000/ui/api/status
curl -sS -H "Authorization: Bearer $SLAVIK_API_TOKEN" http://127.0.0.1:8000/v1/models
curl -sS -H "Authorization: Bearer $SLAVIK_API_TOKEN" http://127.0.0.1:8000/ui/api/settings
make smoke-prod
```

`/v1/models` возвращает один публичный proxy model id `slavik`. Реальный provider/model
выбирается в Settings и не является публичным model id этого API.

## 5) Фоновый режим (без systemd)

```bash
make up-prod PROD_HOST=127.0.0.1 PROD_PORT=8000
make status-prod
make logs-prod
```

Остановка: `make down-prod`. Development background использует отдельные `make up/down/status/logs`
и окружение `venv`; не используй его как production-команду.

## 6) systemd

Используй `deploy/slavikai.service.example`: он запускает `venv-prod`, читает
секреты из `/opt/slavikai/.env` и по умолчанию слушает только localhost.

После добавления:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now slavikai
sudo systemctl status slavikai
```

## 7) Что важно проверить перед выкладкой

- `make check` проходит локально.
- UI собран (`make ui-build`), `ui/dist` актуален.
- `curl /healthz` возвращает `status=ok` и `ui_built=true`.
- Целевая модель видна в списке моделей выбранного провайдера в UI.
- Для выбранного провайдера выставлен корректный API key.
- `SLAVIK_API_TOKEN` задан, UI login и Bearer smoke-check проходят.
- Plan/Auto используют provider с native tools: `deepseek` или `local`.
- Для local embeddings установлен beta bundle и модель имеет статус `ready` в UI.
