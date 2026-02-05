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
- `/auto <goal>` — запустить auto-agent.
- `/imggen <prompt>` — генерация изображения.
- `/imganalyze <path|base64|base64:...>` — анализ изображения.
- `/trace` — последние записи trace.

## Важно

- Команды не запускают MWV verifier.
- Safe-mode и approvals применяются.
- Если инструмент отключён или заблокирован, возвращается явная ошибка.
