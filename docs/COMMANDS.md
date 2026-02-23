# COMMANDS — команды агента

Команды начинаются с `/` и обрабатываются в command lane (без MWV).
Ответы command lane помечаются строкой `Командный режим (без MWV)`.

## Доступные команды

- `/fs list [path]` — список файлов в `sandbox/`.
- `/fs read <path>` — чтение файла из `sandbox/`.
- `/web <query>` — веб-запрос через `web` инструмент.
- `/sh <command>` — запуск shell-команды с ограничениями.
- `/project index [path]` — индексировать файлы из `sandbox/project/`.
- `/project find <query>` — поиск по индексу `code/docs`.
- `/plan <goal>` — построить и выполнить план.
- `/auto <goal>` — command-lane alias для one-shot auto запуска (без MWV).
- `/imggen <prompt>` — генерация изображения.
- `/imganalyze <path|base64|base64:...>` — анализ изображения.
- `/trace` — последние записи trace.

## Session mode `auto`

- `auto` — отдельный runtime mode (к `ask|plan|act`) с контуром `planner -> N coder -> verifier`.
- В mode-level `auto` запуск идёт через runtime orchestrator, а не через command lane.
- `/auto ...` остаётся именно командой `/...` и всегда помечается `Командный режим (без MWV)`.

## Важно

- Команды не запускают MWV verifier.
- Safe-mode и approvals применяются.
- Если инструмент отключён или заблокирован, возвращается явная ошибка.
