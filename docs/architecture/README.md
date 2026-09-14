# Architecture docs

- `ARCH_CANON.md` — runtime canon и границы legacy.
- `Architecture.md` — текущая runtime inventory.
- `USER_INTERACTION_COMMUNICATION_CONTRACT.md` — target semantic contract пользовательской
  коммуникации; не является текущей runtime implementation.
- `research/task-run-communication-audit.md` — evidence, alternatives и rationale.
- `VERIFICATION_ARCHITECTURE_CONTRACT.md` — target verification/acceptance contract.
- `VERIFICATION_ARCHITECTURE_RESEARCH.md` — current map, external research и alternatives.

---

# Архитектура SlavikAI

Точка входа в долгосрочную архитектурную документацию SlavikAI.

- **Для чего:** ориентироваться в структуре `docs/architecture/`, понимать статус разделов и правила ведения.
- **Сюда:** навигация по разделам, статус каждого раздела, соглашения о ведении документации.
- **Не сюда:** конкретные архитектурные решения, финальный список capabilities, implementation roadmap, выжимки текущего кода.
- **Обновлять:** при добавлении/удалении разделов или файлов, изменении статуса Capability Discovery, изменении правил ведения.

## Правило о Capability Discovery

До завершения Capability Discovery запрещено считать список capabilities полным и начинать детальную архитектурную разработку отдельных capability domains. Каталог `capabilities/` на старте содержит только placeholder-домены и не является окончательным списком.
