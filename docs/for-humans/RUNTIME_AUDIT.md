# Аудит runtime: режимы, стриминг, TTS

Дата: 2026-08-25. Метод: чтение кода + реальные логи + integration/regression tests.
Все утверждения ниже подтверждены либо code path, либо запущенными тестами, либо логами
реального сервера. Неподтверждённые гипотезы помечены отдельно.

## 1. Режимы (execution modes)

### Карта `mode → policy → действия → approval → execution → failure`

| Mode | Источник policy | Доступные действия | Approval | Execution | Failure behavior |
|------|-----------------|--------------------|----------|-----------|------------------|
| `ask` (Chat) | `profile` сессии (`safe_mode = profile != "yolo"`) | read-tools + ответ | категории вне `approved_categories` → `ApprovalRequired` | `AgentToolLoop` | ошибка в чат |
| `plan` | `ToolRegistry.set_execution_policy("plan")` | только read | — | read-audit | `PLAN_READ_ONLY_BLOCK` |
| `act` | `profile` | write/exec по категориям | approval по категориям | loop → gateway | падение шага |
| `auto` | как `act` | всё через tool loop + verifier | approval по категориям | `AgentToolLoop` → `_run_verifier` | `WORKER_FAILED` / verifier fail |
| `desktop` | `DesktopPolicyRuntime` | desktop tools | Desktop approvals | desktop runtime | block reasons |

Переходы жёстко заданы в `server/http/common/mode_transitions.py`: в `act` только из `plan`
с approved-планом; из `auto` нельзя выйти при активном прогоне; и т.д.

### Подтверждённый дефект №1: YOLO «разрешал» то, что tool-слой молча блокировал

Воспроизведено по логам (2026-08-24 17:56, сессия `a138…`): пользователь в YOLO + auto
попросил переработать Makefile. Политика вернула `policy_allowed` (safe_mode_disabled) для
`shell`/`fs`/`project`, но одна shell-команда была заблокирована фильтром `_is_unsafe`
(`sudo`, `rm -rf`, `shutdown`, `reboot`, `mkfs`, fork bomb, запись в `/etc`/`/dev`) → auto
упал: «Auto-run остановлен: tool loop failed».

Причина: жёсткий фильтр жил только в `tools/shell_tool.py`, политика о нём не знала.

Фикс:
- `shared/command_safety.py` — единый источник правды (`is_hard_unsafe_command`).
- `core/approval_policy.py` — категория `HARD_DENY`, `decide_action` возвращает `block`
  (`command_denied:hard_safety`) независимо от `safe_mode`.
- `core/tool_gateway.py` — понятное сообщение: перечислены запрещённые классы команд,
  «недоступны даже в YOLO».
- Регрессия: `tests/test_policy_semantics.py::TestHardSafetyBlockAppliesEvenInYolo`
  (YOLO блокирует `sudo`/`rm -rf`, разрешает `pwd`).

### Подтверждённый дефект №2: auto-run валил прогон при любом failed tool call

`core/auto_runtime.py` помечал `WORKER_FAILED`, если в истории был хоть один неудачный
вызов, даже когда модель восстановилась и loop завершился финальным ответом — вопреки
дизайну recovery (`AgentToolLoop` подкармливает ошибки обратно в историю).

Фикс: рабочий статус определяется `loop_result.error` (loop не смог завершить); иначе прогон
идёт в verifier, который и судит результат. Неудачные вызовы остаются видны в
`auto_state.coders` (`status: failed`).

Регрессия: `tests/test_auto_runtime.py::test_auto_v1_recovery_from_failed_tool_call_is_not_worker_failed`
— первый tool call падает, второй даёт финальный ответ, verifier проходит → `COMPLETED`.

### Проверено дополнительно

- `agent.call_tool` без session context встречался только в TTS-хендлере (см. ниже);
  STT ходит в OpenAI напрямую (не через gateway) — того же дефекта нет.
- `decision.py` и `plan.py` применяют runtime state перед `call_tool`.

## 2. Стриминг ответов

### Фактический пайплайн

`llm/deepseek_brain.py` (`stream=True`, `iter_openai_sse_events`) →
`core/tool_loop.py::run_stream_events` (yield `TextDelta` сразу) →
`core/agent_routing.py::respond_stream` →
`server/http/handlers/ui_chat.py` → `chat.stream.delta` в hub →
`server/http/handlers/events.py` (SSE) →
`ui/src/app/use-session-transport.ts` (EventSource) → `ChatStreamState` → рендер.

### Причина видимого «буфера»

1. Warmup `CHAT_STREAM_WARMUP_CHARS = 220` — первые секунды фронт не показывал ничего.
   Снижен до 96 (обычный текст — с 48).
2. SSE без анти-buffering заголовков → прокси/Cloudflare могли отдавать поток в конце.
   Добавлены `X-Accel-Buffering: no` + `Cache-Control: no-cache, no-transform` на оба
   SSE-эндпоинта (chat и terminal).
3. Фронт молча глотал ошибки SSE (`onerror = () => {}`) → деградация без сообщения.
   Теперь статус «Live update connection lost…» + сброс при reconnect.

4. Пустой экран во время auto-run: `auto.progress` публиковался только ПОСЛЕ завершения
   прогона (drain в конце send-хендлера). Исправлено двумя частями:
   - `server/http/handlers/ui_chat.py`: параллельная задача `_publish_auto_progress_while`
     публикует `auto.progress` в hub ВО ВРЕМЯ генерации (поллинг
     `drain_auto_progress_events` каждые 0.2с, отмена в `finally`);
   - `ui/src/app/components/canvas.tsx`: индикатор «Agent is working… / SlavikAI is
     thinking…», когда `sending` активен и стримингового текста ещё нет.

   Регрессия: `test_ui_chat_auto_progress_published_during_stream` — `auto.progress`
   доходит до клиента до завершения send.

### Доказательство инкрементальности

`tests/ui_api/test_stream_and_events.py::test_ui_chat_stream_deltas_arrive_incrementally_before_send_completes`
— фейковый провайдер отдаёт дельты с паузами; тест читает первую дельту **до** завершения
send и падает, если она пришла только после. Тест проходит.

Непроверено из песочницы: фактическое поведение Cloudflare (нужен браузер пользователя).
Если после рестарта всё ещё буферизует — в Cloudflare Speed отключить Response Buffering
или переводить на WebSocket.

## 3. TTS

### Root cause «TTS request failed.»

1. Хендлер `/ui/api/tts/speak` вызывал `agent.call_tool("tts")` **без session context** →
   использовал устаревший глобальный `safe_mode`/`approved_categories` → мог кинуть
   `ApprovalRequired` → исключение не обрабатывалось → aiohttp 500 (не-JSON) → фронт показывал
   generic «TTS request failed.». Подтверждено журналом (17:43:55): `ApprovalRequired` в
   `handle_ui_tts_speak`.
2. Фронт не слал session id → бэкенд не мог применить политику нужной сессии.
3. Дополнительно: для генерации не был настроен OpenAI API key.

### Фикс

- `server/http/handlers/settings.py`: изначально — резолв сессии + перехват
  `ApprovalRequired`; после финальной правки TTS вызывается **напрямую** (без
  agent/gateway/approval), т.к. «Listen» — пользовательское действие.
- `ui/src/features/audio/use-tts-audio-player.ts`: шлёт `X-Slavik-Session`; `sessionId`
  добавлен в зависимости `toggle` (актуальная сессия).
- Регрессия: `test_ui_tts_speak_works_without_approval_in_safe_mode` + существующие
  success/failure тесты TTS.

### Подтверждённый дефект №3: TTS читал только первый абзац

Симптом: аудио 31с вместо 84с. Причина: `TtsTool` использовал общий `HttpClient` с
`max_bytes=500_000` — OpenAI возвращал 1.3MB, клиент обрезал до 500KB и молча отдавал
урезанный файл. Прямой запрос того же текста в OpenAI → 84с/1.3MB.

Фикс: TTS получил собственный `HttpClient(HttpConfig(max_bytes=20_000_000))`
(`core/agent.py`); `TtsTool` возвращает ошибку при `meta.truncated`.
Регрессия: `test_tts_rejects_truncated_audio_response`,
`test_tts_http_client_accepts_large_audio`. После фикса — 84с/1.3MB.

### Подтверждённый дефект №4: подтверждение при каждом «Listen» даже в YOLO

Фронтенд-плеер держал `sessionId` в замыкании без зависимости → слал не ту сессию
(sandbox вместо YOLO) → `ApprovalRequired`. Исправлено зависимостью `sessionId` в
`toggle` и прямым вызовом TtsTool (approval не нужен).

Осталось: задать OpenAI API key (Settings → API Keys → OpenAI или `OPENAI_API_KEY` в `.env`).
Модель/голос/формат уже настроены (`tts-1`, `alloy`, `mp3`).

## 4. Web-поиск (по ходу)

Пользователь использовал ключ **SerpApi.com**, а приложение было зашито на **serper.dev** →
403 даже с валидным ключом (проверено прямыми запросами: serpapi.com 200, serper.dev 403).

Фикс: провайдер `serpapi` в `config/web_search_config.py` + `tools/web_search_tool.py`
(`WEB_SEARCH_PROVIDER=serpapi`, `SERPAPI_API_KEY`), распарсинг `organic_results`,
тест `test_web_search_serpapi`. Live-проверка — 5 результатов.

Дополнительно: SerpApi иногда отвечает дольше 10с → таймаут поиска в auto.
Поднят `timeout` до 20с и добавлен один retry на timeout/5xx
(`_request_with_retry`). Регрессия: `test_web_search_serpapi_retries_once_on_timeout`
(таймаут → повтор → успех).

## 5. Прочее

- Заголовок вкладки: `SlavikAI` (было `Slavik UI`).
- Уведомление о skills: показывается только при реальном использовании
  (`completed`/`failed`), скрыто для `skipped`/без совпадения.
- `make deploy`: порт-чек с `SO_REUSEADDR` (TIME_WAIT после остановки больше не мешает).
- Graceful shutdown: `shutdown_timeout=8.0` в `server/http/app.py` (SIGTERM останавливает
  быстро вместо 60с ожидания + SIGKILL).
- npm audit: 0 уязвимостей (обновлён `ui/package-lock.json`).

## 6. Проверки

Backend: 159+ тестов зелёные (streaming/events, auto.progress, settings/TTS, policy, modes, auto, web),
плюс 128 по режимам/tools, плюс 22 по web/app. Frontend: typecheck, 34/34 vitest, build.
Ruff: all checks passed.

## 7. Осталось для финального подтверждения (требует действий пользователя)

1. Обновить `slavikai.online` и проверить: web-поиск возвращает результаты, стриминг идёт
   постепенно, заголовок `SlavikAI`, skills не вылезают для неиспользующих.
2. Задать OpenAI API key для TTS.
3. Если стриминг на Cloudflare всё ещё буферизуется — отключить Response Buffering в
   Cloudflare Speed (или переводить на WebSocket).
