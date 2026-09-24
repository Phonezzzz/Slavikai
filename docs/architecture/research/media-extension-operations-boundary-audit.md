# Gate 0 — media, extensions и operations: выборочный Current State audit

**Статус:** выборочный read-only code-path audit, 24.09.2026. Production
checkout `/home/ki/Project/Slavikai` at `920d5f3`; процессы, реальные
provider calls и deployment не запускались. Наблюдения ниже относятся только
к указанным routes/classes. Target boundaries заданы ADR-0007/0008/0009 и
[системной картой](../system/capability-map.md), а не выводятся из текущего кода.

| Область | Проверенный путь | Наблюдение и предел вывода |
| --- | --- | --- |
| Browser authentication | `server/http/app.py::create_app` ставит `auth_gate_middleware`; `server/http/common/auth.py::_resolve_ui_request_identity` получает Cloudflare identity и роль owner/member | Ingress аутентифицирует запросы `/ui/api/`. Это не доказывает, что конкретный media call получает разрешение на использование owner provider key или отдельный consent. |
| STT request | `server/http/routes.py` → `handlers/settings.py::handle_ui_stt_transcribe` | Handler получает application-level OpenAI key через `_resolve_provider_api_key("openai")`, принимает multipart audio с size cap и вызывает `requests.post` к STT endpoint. В исследованном handler нет principal/delegation/consent проверки и вызова `ToolGateway`. Результат — JSON transcript; этот путь сам не создаёт versioned media artifact или task transition. Это статический вывод о handler, не live exploit verdict. |
| TTS request | `routes.py` → `handle_ui_tts_speak` → `TtsTool.handle` | Handler получает application-level key, вызывает tool напрямую, читает созданный файл под `sandbox/audio` и возвращает audio bytes с `Cache-Control: no-store`. В исследованном handler нет principal/delegation/consent или `ToolGateway` call. `tools/tts_tool.py` создаёт файл с timestamp/hash имени; отдельный owner/retention для этих файлов в проверенном пути не установлен. HTTP cache header не является policy хранения файла. |
| Chat attachments | `handlers/ui_chat.py::_handle_ui_send_impl` → `common/chat_payload.py::_parse_ui_chat_attachments` и `_ui_messages_to_llm` | Проверяются количество, длина и форма `{name,mime,content}`; attachment JSON добавляется в model-visible content. Это не versioned artifact record и не отдельное consent/egress решение; session authorization проверяется отдельно. Фактический provider route и хранение сообщения требуют отдельного path audit. |
| Built-in tools and skills | `tools/tool_registry.py::ToolRegistry` хранит in-process `ToolDescriptor`; `core/skills/runtime.py` возвращает skill id/version/status metadata | Проверенные descriptor/skill paths не образуют управляемый каталог внешних tool providers с identity/version/revocation. Поиск по `core/`, `server/`, `tools/`, `config/` не нашёл MCP/provider-extension registration; это предел выборочного поиска, а не доказательство отсутствия интеграций во всём deployment. |
| Server and deployment | `server/http/app.py::create_app` собирает in-process runtime и cleanup hooks; `config/computer_backend_config.py` выбирает local default или explicit container backend; `deploy/slavikai.service.example` — systemd example | Эти файлы показывают несколько operational surfaces и fail-closed выбор backend. Example service и app factory не доказывают, какая конфигурация реально запущена, как выполняются rollout, health supervision, rollback или secret injection. |

## Gate 0 выводы

1. Проверенные routes с отдельными STT/TTS действиями согласуются с
   request-scoped выбором ADR-0008. Target должен иметь явный
   principal/consent/credential/egress/artifact handoff для каждого.
2. ADR-0009 требует отдельный governance owner: текущие static tool descriptors
   и skill metadata не обладают provider identity/revocation semantics.
3. Deployment/Runtime Operations нужен как отдельный system-level candidate
   owner; model-host engine из ADR-0005 покрывает лишь его часть.
4. Текущий direct STT/TTS handler path и owner-key resolver — конкретный
   вход для аудита разделов 7–9/18/19. Не переносить его как Target pattern и
   не менять production code в Gate 0.

**Следующий шаг:** сверить эти пути с Identity/Policy/Tool Execution и
Artifact boundaries при проектировании соответствующих разделов. Для runtime
вывода нужны scope-specific tests и проверка фактического deployment.
