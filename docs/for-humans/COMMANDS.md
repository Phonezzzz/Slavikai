# COMMANDS

Текущий command lane — только для debug. Он не вызывает обычные инструменты и не заменяет
Chat/Workspace runtime.

Поддерживаемые slash-команды:

- `/trace` — показать последние trace-записи.
- `/end-session` — сохранить короткое резюме текущей сессии в canonical memory и очистить short-term context.

Удалено из command lane:

- `/fs`
- `/web`
- `/sh`
- `/project`
- `/plan`
- `/auto`
- `/imggen`
- `/imganalyze`

Эти возможности должны идти через обычный Chat/Workspace runtime и native
tool-calling/gateway path, а не через прямой slash dispatch.
