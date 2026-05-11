# DevRules — правила разработки SlavikAI

Эти правила фиксируют требования, которые уже отражены в настройках проекта (`pyproject.toml`) и в текущей архитектуре (Agent/Tools/Sandbox/Memory).

## 1) Язык, типизация, стиль

- Язык проекта: Python 3.12 (`pyproject.toml` → `tool.ruff.target-version = py312`, `tool.mypy.python_version = 3.12`).
- Типизация: `mypy` в режиме `strict = true` (tests исключены через `exclude = [...]`).
- Форматирование и линт: `ruff` (линт + формат) — конфиг в `pyproject.toml`.
- UI-часть находится в `ui/` и содержит TypeScript/JavaScript (`*.ts`, `*.tsx`).
- Для backend/Python части правило “no any/as/optional-chaining без проверки” трактуется как:
  - **запрещено** “расплывать” типы через `Any`/`cast()` в доменной логике;
  - **запрещено** использовать `Optional[...]` без явной проверки на `None` (или без безопасного значения по умолчанию).
- Для UI/TypeScript части (в `ui/`) это правило трактуется так:
  - **избегать** `any` и небезопасных `as` в прикладной логике;
  - `?.` использовать только там, где `undefined/null` действительно допустимы и есть явная обработка/fallback.

## 2) Никаких “silent-fallback”

- Любой fallback должен быть **явным**: либо с записью в `Tracer`/логгер (см. `core/tracer.py`), либо через понятную ошибку пользователю/в UI.
- Если функция возвращает “значение по умолчанию” из-за ошибки — это должно быть:
  - осознанно (и желательно проверено тестом),
  - наблюдаемо (лог/трейс), если влияет на поведение агента.

## 2.1) Product default: single-user / single-admin

- По умолчанию проект рассматривается как `single-user / single-admin`.
- Агент обязан принимать архитектурные и продуктовые решения, исходя из того, что владелец проекта и основной пользователь — один человек.
- Если в задаче явно не сказано обратное, не нужно добавлять:
  - migration logic,
  - backward compatibility "на всякий случай",
  - multi-user safeguards,
  - специальные меры ради сохранения гипотетически чужих данных.
- Такие меры допустимы только по явному запросу в задаче.

## 3) Инструменты: обязательный `ToolResult`

- Каждый инструмент обязан реализовать интерфейс `Tool.handle(self, request: ToolRequest) -> ToolResult` (см. `tools/protocols.py`).
- Ошибки инструмента возвращаются через `ToolResult.failure(...)`.
- Запрещено “проталкивать” исключения наружу как основной контроль потока; если исключение возможно — ловить и возвращать `ToolResult.failure(...)`.
- Конвенция проекта: если инструмент возвращает человекочитаемый текст, кладите его в `data["output"]`.

## 4) Sandbox restrictions (обязательные)

- Все операции с путями должны быть “заземлены” в песочнице:
  - общий sandbox: `sandbox/` (пример: `tools/filesystem_tool.py`, `tools/shell_tool.py`);
  - workspace: `sandbox/project/` (пример: `tools/workspace_tools.py`, `tools/project_tool.py`).
- Запрещены абсолютные пути и выход за пределы песочницы.
- Для нормализации/проверки пути используйте уже существующие хелперы конкретного инструмента:
  - `tools/filesystem_tool.py::_normalize_path(...)`;
  - `tools/workspace_tools.py::_ensure_in_workspace(...)`;
  - `tools/project_tool.py::_normalize_path(...)`.
- Общий хелпер для нормализации путей в песочнице существует в `shared/sandbox.py` (сейчас используется для `ShellTool.sandbox_root`).

## 5) Safe-mode (обязательный контур безопасности)

- Safe-mode реализован на уровне `ToolRegistry` и включается/выключается агентом (`core/agent.py` → `ToolRegistry.apply_safe_mode`).
- Инструменты, которые дают сетевой/системный доступ, должны быть выключаемыми safe-mode:
  - добавляйте имя инструмента в `SAFE_MODE_TOOLS_OFF` (см. `core/agent.py`);
  - обеспечьте тест, что инструмент реально блокируется.
- Текущий блок-лист safe-mode (источник: `core/agent.py`):
  - `web`, `web_search`, `shell`, `project`,
  - `tts`, `stt`, `http_client`,
  - `image_analyze`, `image_generate`,
  - `workspace_run`.

## 6) Правила работы с LLM

- Сетевые вызовы к LLM должны быть изолированы в слое `llm/*` (см. `Brain` в `llm/brain_base.py`).
- Все LLM-клиенты обязаны:
  - иметь таймаут,
  - валидировать формат ответа (dict/list/choices/message/content),
  - выбрасывать понятные исключения вверх по стеку (агент уже обрабатывает и логирует ошибки).
- Не логировать секреты (API keys). Ключи берутся из env (`OPENROUTER_API_KEY`, `LOCAL_LLM_API_KEY` и т.п.) или из конфигов.

## 7) Где должны быть тесты

- Тесты лежат в `tests/`, именование: `test_*.py`.
- Запрещены реальные сетевые вызовы в тестах. Используйте стабы (пример: `tests/test_tts_stt_tools.py`, `tests/test_web_search_tool.py`).
- Тесты для индекса должны избегать загрузки real `sentence-transformers` модели (пример: monkeypatch `VectorIndex._get_model` в `tests/test_vector_index.py`).
- Для SQLite в тестах используйте `tmp_path`, не пишите в `memory/*.db`.

## 8) Минимальные требования качества (как сейчас настроено)

- `pytest` + `pytest-cov` с порогом покрытия **>= 80%** (см. `pyproject.toml` → `--cov-fail-under=80`).
- `ruff check .` — без предупреждений/ошибок.
- `ruff format --check .` — код должен быть отформатирован.
- `mypy .` — `strict` без ошибок (в текущей конфигурации tests исключены).

## 9) Репозиторий и артефакты

- В проекте есть runtime-артефакты (`logs/*`, `sandbox/*`, `memory/*.db`) и виртуальная среда. Новые артефакты такого рода **не добавлять** в git.

## 10) Git workflow (режим A)

- Работа начинается с `main`: `git checkout main`.
- На PR создаётся ветка: `git checkout -b pr-<номер>-<название>`.
- Перед любой работой в PR-ветке запускай `make git-check`.
- После завершения PR: `git rebase origin/main`, `git merge --ff-only <pr-branch>`, `git push origin main`.
- Перед финализацией изменений запускай `make check`.
- Нельзя продолжать новую фичу в старой ветке.
- После PR — всегда обратно на `main`.

## 11) Memory Companion: инварианты (policies-first, без auto-changes)

- **Запрещены** авто‑апдейты `Memory` в runtime (никаких “сам сохранил важное” без явного approve).
- **Запрещены** авто‑создание/изменение `PolicyRule` в runtime: правила появляются только как **Approved** после ручного review.
- Отсутствие фидбэка после ответа = `unknown` (это **не** `good` и не повод менять поведение).
- BatchReview запускается **только вручную** и генерирует только `PolicyRuleCandidate[]` (не правила).
- Trigger/Action для policies должны быть **типизированы** (никаких “dict на всё”): структуры обязаны валидироваться.

## 12) Документация как источник истины

- `AGENTS.md` и документы из его раздела **Canonical rules** являются обязательными для любой задачи.
- Фактическое поведение должно соответствовать “живым” документам (`architecture/Architecture`, `for-humans/COMMANDS`, `workflow/CONTRIBUTING`).
- Плановые/исторические документы должны быть явно помечены или перенесены в архив, чтобы не смешиваться с актуальными правилами.

## Anti-pseudo audit

### Цель

Каждый implementation PR должен проверять не только список изменяемых файлов, но и риск архитектурной подмены: ситуации, где поведение выглядит как agent/runtime, но фактически выполняется через legacy wrapper, regex, classifier, fallback или прямое Python-действие в обход основного runtime.

Это правило обязательно перед implementation каждого PR и является частью mini non-mutating audit.

### Что считать pseudo-runtime / pseudo-agent behavior

Worker обязан проверить изменяемый контур на следующие признаки:

1. **Regex/prose extraction вместо structured contract**

   Запрещён runtime, где intent/target/action вытаскиваются из обычного текста через regex, string matching или эвристики, если по архитектуре должен использоваться explicit structured input:

   - `ToolRequest`;
   - `ToolSpec`;
   - JSON schema;
   - typed command/payload;
   - explicit tool call.

2. **Classifier/router как runtime decision**

   Запрещено использовать classifier/router как основной механизм выбора tool/action/runtime, если для этого уже существует основной runtime/tool loop.

   Classifier может быть только guard/block layer, если это явно указано в архитектуре.

3. **Fallback path в обход основного runtime**

   Запрещены fallback-и, которые silently обходят основной путь:

   - bypass tool loop;
   - bypass ToolGateway;
   - bypass verifier;
   - bypass approval;
   - bypass structured tool call.

4. **Adapter/compatibility/migration без явного разрешения**

   Не добавлять compatibility layers, adapters, aliases, migrations или dual-support “на всякий случай”.

   Если без такого слоя текущие tests/runtime не проходят — это не разрешение на самостоятельное добавление слоя. Нужно остановиться и вернуть BLOCKED report.

5. **Tests проверяют эффект, но не механизм**

   Если тест проверяет только итоговый эффект, например “файл изменился”, но не проверяет, что изменение прошло через нужный механизм, worker обязан усилить/исправить тест в scope PR.

   Пример плохой защиты:

   - test passes because file changed;
   - but file was changed direct Python append, not через ToolGateway/tool call.

6. **Direct action вместо ToolGateway / explicit tool call**

   В runtime-контуре agent/MWV/auto/workspace запрещено выполнять file/db/tool действия напрямую, если по архитектуре действие должно идти через:

   - explicit tool call;
   - ToolGateway;
   - tool observation;
   - verifier/approval layer.

7. **`lane` / type discriminator как domain model**

   Legacy discriminator вроде `lane` не должен управлять runtime storage/API/frontend flow, если планом предусмотрен physical split.

   Допустимо временно видеть такой marker только во время audit/deletion, но не как новый или сохранённый domain contract.

8. **Reachable legacy entrypoint**

   Если новый path добавлен, но старый production entrypoint остаётся reachable, это считается незавершённым PR или BLOCKED-состоянием.

9. **Duplicate runtime path**

   Запрещена ситуация, где новый runtime существует, но production-код всё ещё может выполнять тот же сценарий через старый legacy path.

### Обязательное поведение worker

Перед implementation PR worker обязан:

1. выполнить mini non-mutating audit;
2. отдельно проверить признаки pseudo-runtime из этого раздела;
3. выписать найденные pseudo-paths;
4. определить, входят ли они в scope текущего PR.

Если pseudo-path входит в scope текущего PR:

- удалить его;
- заменить на канонический runtime/tool path;
- обновить tests так, чтобы они защищали механизм, а не только итоговый эффект.

Если исправление требует архитектурного решения вне scope:

- остановиться;
- не продолжать следующий PR;
- не импровизировать;
- вернуть BLOCKED report.

### BLOCKED report

BLOCKED report должен включать:

1. PR / branch;
2. где найден pseudo-path;
3. факт из кода;
4. почему это архитектурная подмена;
5. какое архитектурное решение требуется;
6. варианты решения;
7. рекомендацию;
8. что уже изменено;
9. какие tests/checks запускались;
10. текущее состояние `git status --short`.

### Запрет

Worker не должен добавлять adapters/fallbacks/compatibility/migrations только ради прохождения тестов.

Если тесты требуют такой слой, а он не был явно разрешён заданием, это BLOCKED, а не повод писать новый compatibility-код.
