# PHASE 5 — Decision Matrix (Safe‑mode + Session Approvals)

Этот документ фиксирует точные правила, по которым backend определяет:
что блокировать, что требовать подтверждения, и как выглядит запрос на подтверждение.
Документ является источником истины для Phase 5.

## A) Concepts

- **safe_mode**: при включении любые опасные действия блокируются до явного подтверждения.
- **approval**: разрешение выполнить действие конкретной категории.
- **session approval**: временное разрешение на категорию, действующее только для текущего `session_id`.
- **reset rule**: все approvals сбрасываются при смене `session_id` или при перезапуске процесса.
- Если `safe_mode=false`, опасные действия разрешены, но всё равно логируются в trace.

## B) Categories (фиксированные)

Используются только эти категории:
- `FS_DELETE_OVERWRITE`
- `FS_OUTSIDE_WORKSPACE`
- `FS_CONFIG_SECRETS`
- `DEPS_INSTALL_UPDATE`
- `GIT_PUBLISH`
- `SYSTEM_IMPACT`
- `SUDO`
- `NETWORK_RISK`
- `EXEC_ARBITRARY`

## C) Decision Matrix

Столбец “Default behavior” всегда относится к `safe_mode=true`.

| Category | Default behavior | Session‑approved? | Example actions | User‑facing risk text | Minimal details (max 3) |
|---|---|---|---|---|---|
| FS_DELETE_OVERWRITE | BLOCK | YES | `workspace_write`, `workspace_patch`, `fs op=write` | “Можно потерять данные/историю файлов.” | path<br>операция (delete/overwrite)<br>объём (файлы/папки) |
| FS_OUTSIDE_WORKSPACE | BLOCK | YES | `fs` с path вне `project/`, `workspace_*` с `../`/абс. путём | “Действие вне рабочей папки проекта.” | path<br>workspace_root<br>операция |
| FS_CONFIG_SECRETS | BLOCK | YES | `workspace_write/patch` или `fs write` по `.env`/config/secret/key | “Риск утечки или поломки конфигурации.” | path/ключ<br>тип секрета<br>операция |
| DEPS_INSTALL_UPDATE | BLOCK | YES | `shell` с `pip install`, `poetry add`, `npm install` | “Изменит зависимости и окружение проекта.” | команда/менеджер<br>пакеты<br>файлы зависимостей |
| GIT_PUBLISH | BLOCK | YES | `shell` с `git commit/push/tag` или publish/release | “Публикация изменений наружу.” | команда<br>ветка/тег<br>remote |
| SYSTEM_IMPACT | BLOCK | YES | `shell` с `systemctl/service/mount/umount` | “Может повлиять на систему.” | команда<br>объект (сервис/диск/сеть)<br>уровень влияния |
| SUDO | BLOCK | YES | `shell` содержит `sudo` | “Повышенные права, риск для системы.” | команда<br>цель<br>причина |
| NETWORK_RISK | BLOCK | YES | `web` инструмент; `shell` с `curl/wget/http(s)` | “Сетевой доступ во внешние сервисы.” | url/host<br>метод/инструмент<br>цель запроса |
| EXEC_ARBITRARY | BLOCK | YES | `shell` любая команда, `workspace_run` | “Выполнение произвольной команды.” | команда/скрипт<br>рабочая папка<br>ожидаемый эффект |

## D) Tool mapping (детерминированные правила)

Инструменты зарегистрированы в `core/agent.py` → `ToolRegistry`.
Ниже правила соответствуют реальным `tool` именам и аргументам:

- **FS_DELETE_OVERWRITE**
  - `workspace_write` → `args.path` (любая запись в workspace)
  - `workspace_patch` → `args.path` (кроме `dry_run=true`)
  - `fs` → `args.op == "write"` и `args.path`

- **FS_OUTSIDE_WORKSPACE**
  - `fs` → `args.path` вне `sandbox/project` (не начинается с `project/`)
  - `workspace_*` → `args.path` содержит `../` или абсолютный путь

- **FS_CONFIG_SECRETS**
  - `workspace_write|workspace_patch|fs(write)` → `args.path` содержит `.env`, `config`, `конфиг`, `secret`, `token`, `apikey`, `api key`, `key`, `password`, `credential`, `ssh`, `*.pem`, `*.key`, `ключ`, `пароль`

- **DEPS_INSTALL_UPDATE**
  - `shell` → команда содержит `pip install`/`pip3 install`, `poetry add/update`, `pipenv install`, `npm install/update`, `yarn add/upgrade`, `requirements.txt`, `package.json`

- **GIT_PUBLISH**
  - `shell` → команда содержит `git commit`, `git push`, `git tag`, `publish`, `release`

- **SYSTEM_IMPACT**
  - `shell` → команда содержит `systemctl`, `service`, `iptables`, `ufw`, `mount`, `umount`, `mkfs`, `reboot`, `shutdown`

- **SUDO**
  - `shell` → команда содержит `sudo`

- **NETWORK_RISK**
  - `web` → любой вызов
  - `shell` → команда содержит `curl`, `wget`, `http://` или `https://`

- **EXEC_ARBITRARY**
  - `shell` → любая непустая команда
  - `workspace_run` → `args.path` (любая непустая)

## E) Human confirmation prompt template (точный формат)

Когда требуется approval, backend должен сформировать структуру:

1) **Что хочу сделать:** `<одна строка>`
2) **Зачем:** `<одна строка>`
3) **Риск:** `<простыми словами, одна строка>`
4) **Что изменится:**  
   - `<пункт 1>`  
   - `<пункт 2>`  
   - `<пункт 3>`
5) **Buttons:** Continue / Cancel / Show details / Edit

## F) “Approve for this session” rule

- Approval даётся **по категориям**, не глобально.
- `POST /slavik/approve-session` **обязан** получать список `categories[]`.
- Если `categories[]` отсутствует или пуст — approvals не выдаются.
- Approval действует **только** для текущего `session_id`.
- Approval не отменяет DevRules и запрет silent‑fallback.

## G) DoD (STEP 1)

- Документ создан и содержит полную таблицу по всем категориям.
- Нет двусмысленных формулировок.
- Никаких изменений кода на этом шаге.
