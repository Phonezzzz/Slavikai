# COMMAND_LANE_POLICY — M11

Командный режим (`/…`) — debug-only путь, который **не** проходит через MWV/Verifier.
Он больше не предназначен для прямого вызова обычных инструментов.

## Правило

- Разрешены только `/trace` и `/end-session`.
- Команды `/fs`, `/web`, `/sh`, `/project`, `/plan`, `/auto`, `/imggen`, `/imganalyze`
  отключены.
- Все разрешённые команды `/...` выполняются **без MWV**.
- Ответы командного режима всегда помечаются строкой:
  `Командный режим (без MWV)`.
- Обычные capabilities должны идти через Chat/Workspace runtime и tool gateway.

## Причина

Command lane больше не конкурирует с native tool calling. Он оставлен только для
диагностики trace и явного завершения short-term session state.
