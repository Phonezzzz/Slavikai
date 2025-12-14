Полный аудит

Архитектура верхнего уровня (по коду):
Core: core/agent.py (routing запросов, safe-mode), core/planner.py (LLM/эвристический план), core/executor.py (пошаговое выполнение, критик), core/auto_agent.py (простые параллельные подагенты), core/tool_gateway.py, core/tracer.py.
Tools: tools/tool_registry.py + лог logs/tool_calls.log; инструменты fs/shell/web/project/image/tts/stt/workspace, общий tools/http_client.py.
Memory: memory/memory_manager.py (SQLite + валидация записей/предпочтений/фактов), memory/feedback_manager.py, memory/vector_index.py.
Index: VectorIndex c sentence-transformers, неймспейсы code/docs, обрезка по лимитам.
LLM: llm/openrouter_brain.py, llm/local_http_brain.py, llm/dual_brain.py, фабрика llm/brain_factory.py, конфиги config/model_store.py, промпты config/system_prompts.py.
UI (PySide6): ui/main_window.py с панелями Tools/Mode/Reasoning/Trace/Docs/Feedback/Workspace/Chat/Logs; аудио ui/audio_player.py; workspace-редактор ui/workspace_panel.py.
Workspace layer: tools/workspace_tools.py (list/read/write/patch/run в sandbox/project), UI-панель для работы с этими инструментами.
Реальное состояние кода:
Реализовано полностью:
Safe-mode в ToolRegistry с блоком {"web","shell"}, загрузка/сохранение настроек инструментов (config/tools_config.py), shell whitelist и sandbox (tools/shell_tool.py), fs sandbox (tools/filesystem_tool.py).
DualBrain режимы single/dual/critic-only, план переписывается критиком (core/agent.py, tests/test_plan_critic.py), критик шагов (_critic_step).
Контекст LLM: память, feedback hints (major/fatal), user prefs, vector search code/docs, workspace контекст (core/agent._build_context_messages).
VectorIndex с лимитами по неймспейсу и общему объёму, batch insert, детерминированный cosine scoring (memory/vector_index.py).
Project индексация и поиск по векторному индексу (tools/project_tool.py), web search через Serper (tools/web_search_tool.py), HTTP-клиент с лимитами и триммингом (tools/http_client.py).
Workspace инструменты (list/read/write/patch/run) с ограничениями расширений/размеров (tools/workspace_tools.py), UI для редактирования/патчей/запуска.
Feedback loop: сохранение rating/severity/hint, авто-инжект major/fatal в контекст, UI просмотр (memory/feedback_manager.py, ui/feedback_panel.py).
Тесты покрывают основные слои (core/tools/memory/index/web/shell/safe-mode/workspace/llm) и порог покрытия 80% задан в pyproject.toml.
Реализовано частично:
Планировщик: простая эвристика + LLM без строгого маппинга шаг→tool, _execute_with_tools ищет ключевые слова (web/file/shell) — слабая детерминированность.
Safe-mode блокирует только web/shell; TTS/STT/ProjectTool/Web fetch не учитывают safe-mode.
ProjectTool работает по произвольному Path.resolve() без sandbox (tools/project_tool.py), может читать весь диск.
AutoAgent: дробит цель по “и” и просто вызывает brain в потоках, без инструментов/валидации (core/auto_agent.py).
UI запускает Agent(DummyBrain(), DummyCritic()), не подхватывает сохранённые реальные конфиги моделей (ui/main_window.py).
VectorIndex грузит sentence-transformers синхронно при первом вызове, без кеша путей/контроля веса модели.
Workspace patch применяет самописный примитивный diff для одного файла; нет защиты от больших патчей и multi-file.
Shell tool whitelist поддерживает конфиг из UI, но нет явного UI для редактирования списка команд (только через payload).
Реализовано некорректно/уязвимо:
Sandbox для project_index отсутствует — можно обойти песочницу (tools/project_tool.py).
Safe-mode не распространяется на сетевые инструменты TTS/STT/web fetch прямого URL — потенциальный обход ограничений.
Отсутствует / NOT IMPLEMENTED:
CI/CD конфигурация (нет workflow-файлов).
Веб/мобильные клиенты, API-сервер или headless режим — NOT IMPLEMENTED.
Продвинутая критика ответов в single-mode (только обёртка DualBrain, нет внешнего вызова критика для simple brain).
Риски:
Техдолг: слабый планировщик/исполнитель (ключевые слова вместо структурированных действий) может выполнять не тот инструмент.
Безопасность: project tool вне sandbox; safe-mode не охватывает TTS/STT/web-fetch — возможен нежелательный сетевой доступ даже при SafeMode.
UX/Поведение: UI работает с DummyBrain, реальные модели/ключи не подхватываются — рассинхрон с runtime.
Производительность: sentence-transformers грузится блокирующе; индексация больших папок без чанков по файлам может заморозить UI.
Тесты: нет тестов на UI-интеграцию/аудио, нет проверок safe-mode для TTS/STT/ProjectTool.
План исправлений (по коду):
Засандбоксить ProjectTool: проверка путей относительно sandbox/project, лимит глубины/размера, respect safe-mode.
Расширить safe-mode: блокировать сетевые TTS/STT/web-fetch и project index при включении; отразить в ToolsPanel/Agent.
Укрепить Planner/Executor: явная схема шаг→tool, запрет выполнения шагов без разрешения, нормализованные операции вместо поиска по ключевым словам.
Поднимать реальные конфиги моделей в UI: читать config/model_config.json и config/mode.json, отказаться от DummyBrain по умолчанию.
Асинхронно/лениво грузить embeddings модель, кешировать путь, добавлять прогресс/таймауты.
Добавить CI (ruff + mypy + pytest --cov>=80) и тесты для safe-mode охвата новых инструментов и ProjectTool sandbox.
PROJECT_OVERVIEW.md
Проект: локальный многоинструментальный ассистент с PySide6 UI, DualBrain режимами и слоем инструментов/памяти.

Назначение: отвечать на запросы пользователя, строить планы, вызывать инструменты (fs/shell/web/project/workspace/audio/image), вести память/индекс, работать через UI.
Архитектура: Agent (core/agent.py) → Planner (core/planner.py) → Executor (core/executor.py) → ToolRegistry (tools/tool_registry.py) → Memory/Index (memory/*) → UI/Workspace (ui/*, tools/workspace_tools.py).
Pipeline запроса:
Если командный префикс (/fs, /web, /sh, /plan, /auto) — прямой вызов ToolRegistry.
Иначе классификация сложности; для complex строится план (LLM или эвристика), при dual/critic — критика плана/шагов, выполнение через Executor+ToolGateway.
Для простых ответов: Brain.generate с контекстом (memory, feedback hints, user prefs, vector search code/docs, workspace selection/content); сохранение в память, лог трассировки.
TTS/STT: tools/tts_tool.py и tools/stt_tool.py через HTTP-клиент; UI Chat добавляет озвучку и запись/распознавание (sandbox/audio).
Workspace: инструменты workspace_list/read/write/patch/run с корнем sandbox/project, whitelisted расширения (.py/.md/.txt/.json/.toml/.yaml/.yml); UI ui/workspace_panel.py для дерева, редактора, патчей и запуска скриптов.
Ограничения: Safe-mode через ToolRegistry блокирует web/shell; sandbox для fs/shell/workspace; project tool не ограничен sandbox; safe-mode не накрывает tts/stt/web fetch прямого URL.
Структура репозитория: core/ (agent/planner/executor), tools/ (fs/shell/web/project/http/image/tts/stt/workspace), memory/ (sqlite, feedback, vector index), llm/ (brains, configs), ui/ (панели), config/ (mode/tools/model/system_prompts), tests/ (широкое покрытие), sandbox/ (рабочие файлы), logs/.
DevRules.md
Типы и проверки:
Код под mypy strict (см. pyproject.toml), не использовать Any без необходимости; соблюдать dataclass/TypedDict контракты.
Инструменты обязаны принимать ToolRequest и возвращать ToolResult; никакого silent-fallback — ошибки через ToolResult.failure.
Инструменты и sandbox:
fs/shell/workspace работают только внутри sandbox (sandbox / sandbox/project), shell — только whitelist команд, без абсолютных путей и цепочек.
Safe-mode должен блокировать опасные инструменты (сейчас web/shell; при расширении включать tts/stt/project/web-fetch).
ProjectTool требует sandbox-проверок при доработках; web-search — только http/https.
LLM:
Допустимые клиенты: OpenRouterBrain, LocalHttpBrain, DualBrain; конфиги в config/model_config.json, режимы в config/mode.json.
Контекст перед вызовом Brain должен собираться через _build_context_messages (memory, feedback hints, prefs, vector search, workspace контекст).
Planner/Executor:
План минимум 2 шага, максимум 8; шаги валидируются схемой (shared/plan_models.py).
Критик переписывает план и/или отклоняет шаги в dual/critic-only режимах; Executor не выполняет отклонённые шаги.
Память/Индекс/Feedback:
MemoryManager валидирует записи; meta только dict.
VectorIndex ограничен max_records/max_total; использовать неймспейсы code/docs.
Feedback severity/hint сохраняются; major/fatal автоматически попадают в контекст.
Тесты и качество:
Запуск: pytest (порог покрытия 80% из pyproject.toml), ruff, mypy.
Обязательные области: sandbox (fs/shell/workspace), safe-mode, web-search/http limits, vector index pruning, planner/executor критик, feedback/memory валидация, проектный индекс.
UI:
Панели должны отражать режимы, инструменты, логи, планы, память, индекс, workspace; никакой скрытой логики.
Запреты:
Нет silent except; нет обхода safe-mode/sandbox; не подключать новые инструменты без регистрации в ToolRegistry и тестов; не выполнять сетевые вызовы в safe-mode.
CONTRIBUTING.md
Установка и запуск:
pip install -r requirements.txt
Запуск UI: python main.py (PySide6, создаст DummyBrain по умолчанию; для реальных моделей заполните config/model_config.json, config/mode.json).
Тесты и проверки качества:
pytest (порог покрытия 80%), ruff ., mypy . (строгий режим, tests/ исключены).
Нет CI в репозитории — запускайте локально перед коммитом.
Работа с инструментами/Workspace:
Sandbox: все fs/shell/workspace операции внутри sandbox / sandbox/project; разрешённые расширения .py/.md/.txt/.json/.toml/.yaml/.yml.
Shell whitelist редактируется через config/shell_config.json или аргументы shell_config в запросе инструмента.
ProjectTool сейчас не санбоксирован — указывайте осознанные пути при индексации.
Добавление нового инструмента:
Реализуйте handle(self, ToolRequest) -> ToolResult (см. tools/protocols.py).
Зарегистрируйте в Agent._register_tools и при необходимости в UI (ui/tools_panel.py).
Добавьте тесты: успех, ошибки, безопасные ограничения (safe-mode, sandbox).
Добавление моделей:
Реализуйте Brain и подключите через llm/brain_factory.py/BrainManager; учтите DualBrain режимы.
Память/Индекс:
Используйте MemoryManager/FeedbackManager; для индекса — VectorIndex с неймспейсами code/docs.
Коммиты/оформление:
Явных правил нет в репозитории (NOT IMPLEMENTED); используйте осмысленные сообщения и прикладывайте тесты/проверки.
updated_roadmap.md
Stage A — ядро агента
Сделано: Agent/Planner/Executor с DualBrain (перепись плана/шагов), ToolRegistry с safe-mode web/shell, Tracer + tool call лог, память/feedback, vector index code/docs, web/fs/shell/project/image/tts/stt инструменты, авто-применение feedback hints, тесты с порогом 80%.
Частично: планировщик/исполнитель завязаны на ключевые слова; safe-mode не покрывает tts/stt/project/web-fetch; UI не читает реальные конфиги моделей; vector index синхронный тяжёлый загрузчик; auto_agent без инструментов.
Не сделано/NOT IMPLEMENTED: CI/CD; API/CLI/headless режим; продвинутые политики безопасности (RBAC, пер-namespace квоты).
Следующий шаг: санбокс + safe-mode для project/tts/stt/web-fetch; структурированный план→tool маппинг в Executor; загрузка реальных Brain-конфигов в UI; ленивый/cached vector index.
Stage B — Workspace слой
Сделано: workspace list/read/write/patch/run в sandbox/project с ограничениями; UI-редактор с diff preview, запуском скриптов, контекстом для LLM; sandbox для fs/shell.
Частично: патчи только single-file, простой diff-парсер; нет интеграции линтера/тестов в WorkspaceRun; нет квот на размер патча/вывода; project index вне sandbox.
Не сделано/NOT IMPLEMENTED: массовые патчи/commit preview; фоновые операции индексации; контроль версий workspace.
Следующий шаг: усилить patch/apply (многофайловые, размеры), добавить sandbox в project index, интегрировать lint/test команды для workspace_run под whitelist.
Stage C — Расширения
Сделано: TTS/STT через HTTP + UI кнопки/запись.
Частично: управление аудио без safe-mode; нет fallback офлайн TTS/STT.
Не сделано/NOT IMPLEMENTED: Web/Mobile клиенты; REST/WebSocket API; удалённый доступ.
Следующий шаг: определить API слой (REST/WS) или внешний клиент, учесть sandbox/safe-mode для сетевых аудио/веб инструментов.
