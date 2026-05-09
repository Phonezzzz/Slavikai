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
