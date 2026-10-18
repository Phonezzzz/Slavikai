# COMMAND_LANE_POLICY

Командный режим (`/…`) — debug-only путь, который **не** проходит через MWV/Verifier.
Он больше не предназначен для прямого вызова обычных инструментов.

## Правило

- Разрешены только `/trace` и `/end-session`.
- Команды `/fs`, `/web`, `/sh`, `/project`, `/plan`, `/auto`, `/imggen`, `/imganalyze`
  отключены.
- Все разрешённые команды `/...` выполняются **без MWV**.
- Ответы командного режима всегда помечаются строкой:
  `Командный режим (без MWV)`.
- Обычные capabilities должны идти через Chat/Workspace runtime, native tool-calling
  contract и `ToolGateway`.

## Причина

Command lane больше не конкурирует с native tool calling. Он оставлен только для
диагностики trace и явного завершения short-term session state.

## Запрещено

- Добавлять новые `/...` команды для tools.
- Возвращать `/fs`, `/web`, `/sh`, `/project`, `/plan`, `/auto`, `/imggen`, `/imganalyze`.
- Использовать command lane как обход approval/policy/gateway.

## Завершение сессии

`/end-session` не пишет Memory и не отменяет выполняющуюся задачу. В browser UI
команда предлагает **Да / Нет / Отмена**:

- **Да** — подготовить summary и показать конкретные данные Memory. Только отдельный
  `confirm` или `edit_and_confirm` сохраняет данные через `memory_save_confirmed` ToolGateway
  и завершает сессию. Отказ на этапе preview возвращает к разговору.
- **Нет** — завершить owned short-term context без записи Memory и без вызова модели.
- **Отмена** — оставить разговор открытым.

Ошибка или отмена подготовки summary не закрывает сессию. Pending decision и активное
выполнение должны быть завершены своим существующим механизмом до `/end-session`.
История чата, artifacts и task evidence не удаляются. Продолжение того же чата может снова
передать сохранённую историю модели; пустой разговор создаётся через New Chat.
В `/v1` команда возвращает только предложение выбора; browser Memory confirm endpoint
не переносится в automation lane.
