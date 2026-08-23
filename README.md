# SlavikAI

SlavikAI — self-hosted AI workspace с browser UI, сессиями, памятью, инструментами,
режимами Ask/Plan/Auto и OpenAI-compatible HTTP API. Текущий релиз — **beta**:
основной рекомендуемый путь запуска использует DeepSeek, а local embeddings можно
установить и скачать прямо из Settings.

## Что уже можно использовать

- Browser UI с текущим legacy token login и сохранением истории чатов.
- DeepSeek, OpenRouter, xAI, Inception и OpenAI-compatible local chat providers.
- Выбор глобальной chat-модели и API key через `Settings → API Keys`.
- Local или OpenAI embeddings; local model скачивается через
  `Settings → Memory → Advanced indexing`.
- Ask/Plan/Auto workflow, workspace tools, approvals и session-level model override.
- `/v1/chat/completions`, `/v1/models` и `/healthz` для интеграции и мониторинга.
  Публичный API использует единый proxy model id `slavik`; provider/model из Settings
  остаются внутренней конфигурацией runtime.

## Быстрый beta-запуск (DeepSeek)

Требования: Linux/macOS, Python 3.12+, Node.js 20 (см. `.nvmrc`), npm, git и make.

```bash
git clone https://github.com/Phonezzzz/Slavikai.git
cd Slavikai
cp .env.example .env
```

Заполни минимум две переменные в `.env`:

```dotenv
SLAVIK_API_TOKEN=replace-with-a-long-random-token
DEEPSEEK_API_KEY=replace-with-your-deepseek-key
```

Установи production-зависимости, embeddings и собери UI:

```bash
make install-beta
make run-prod PROD_HOST=127.0.0.1 PROD_PORT=8000
```

Открой `http://127.0.0.1:8000`, войди с `SLAVIK_API_TOKEN`, затем:

1. В `Settings → API Keys` проверь DeepSeek key.
2. В `Default chat model` выбери DeepSeek, нажми `Load models`, выбери модель и
   `Save changes`.
3. При необходимости открой `Settings → Memory`, сохрани local model и нажми
   `Download model`.
4. Создай новый чат — выбранная глобальная модель будет применена автоматически.

Проверка работающего сервера:

```bash
SLAVIK_API_TOKEN=replace-with-the-same-token make smoke-prod
```

## Production-заметки

- Не публикуй порт без auth: `SLAVIK_API_TOKEN` обязателен для production.
- За пределами localhost используй reverse proxy с TLS и оставляй backend на
  `127.0.0.1`, если внешний bind не нужен.
- Пример systemd unit: [`deploy/slavikai.service.example`](deploy/slavikai.service.example).
- Полная инструкция: [`docs/for-humans/DEPLOYMENT.md`](docs/for-humans/DEPLOYMENT.md).

## Разработка

```bash
make venv
make ui-ci
make check
```

Работа ведётся только через PR-ветки. На чистой новой ветке используется `make preflight`,
после push перед merge — `make git-check`. Правила:
[`docs/workflow/CONTRIBUTING.md`](docs/workflow/CONTRIBUTING.md).

Иерархия контрактов и честные статусы current/partial/target описаны в
[`docs/SOURCE_OF_TRUTH.md`](docs/SOURCE_OF_TRUTH.md). Архитектура и поведение agent runtime —
в [`docs/architecture`](docs/architecture) и [`docs/agent`](docs/agent).

## Статус beta

Beta предназначена для контролируемого self-hosted использования. Текущая browser auth ещё
не реализует целевой Cloudflare Access owner/member contract и полную principal isolation.
До завершения соответствующего security PR не открывай UI нескольким внешним пользователям.
Не выдавай проекту доступ к критичным workspace или production secrets без ограничений
approvals и OS user.
