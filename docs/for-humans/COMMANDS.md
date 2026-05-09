# COMMANDS

Current command lane is debug-only.

Supported slash commands:

- `/trace` — show recent trace entries.
- `/end-session` — save a short summary of the current session into canonical memory and clear short-term context.

Removed from command lane:

- `/fs`
- `/web`
- `/sh`
- `/project`
- `/plan`
- `/auto`
- `/imggen`
- `/imganalyze`

Those capabilities must go through the normal Chat/Workspace runtime and native tool-calling/gateway path, not direct slash-command dispatch.
