# LEGACY_SURFACE_MAP — зоны наследия (pre‑MWV / transitional)

Цель: зафиксировать, какие части системы относятся к прошлой архитектуре и
подлежат изоляции/удалению позже. В M6 ничего не удаляется.

## Легенда
- **status**: deprecated | legacy‑inactive | keep‑for‑compat
- **action**: do‑not‑touch | remove‑later | isolate

## Карта

| Path | Что это | Status | Action | Примечания |
| --- | --- | --- | --- | --- |
| `core/critic_policy.py` | LLM‑критик (A+D) | deprecated | remove‑later | Не используется в MWV‑runtime |
| `llm/dual_brain.py` | DualBrain контейнер | deprecated | remove‑later | Инстанцирование отключено |
| `config/mode_config.py` | Режимы single/dual/critic‑only | keep‑for‑compat | isolate | Режимы dual/critic‑only помечены как legacy |
| `config/mode.json` | Хранилище режима | legacy‑inactive | isolate | Могут встречаться старые значения |
| `config/system_prompts.py` | CRITIC_PROMPT | deprecated | remove‑later | Промпт критика сохранён для истории |
| `llm/types.py` | mode="critic" | legacy‑inactive | isolate | Не используется в MWV‑потоке |
| `server/http_api.py` | Модели slavik‑dual/slavik‑critic | keep‑for‑compat | isolate | Контракт ещё не обновлён |
| `ui/mode_panel.py` | UI‑переключатель DualBrain | legacy‑inactive | do‑not‑touch | UI не в фокусе MWV |
| `ui/dual_chat_view.py` | Dual‑чат | legacy‑inactive | do‑not‑touch | UI не в фокусе MWV |
| `ui/settings_dialog.py` | Настройки критика | legacy‑inactive | do‑not‑touch | UI не в фокусе MWV |
| `ui/main_window.py` | Инициализация критика | legacy‑inactive | do‑not‑touch | UI не в фокусе MWV |
| `tests/test_critic_policy.py` | Тесты критика | deprecated | isolate | Пропускаются как legacy |
| `tests/test_http_api.py` | Тесты dual/critic в API | deprecated | isolate | Пропускаются как legacy |
| `docs/PHASE_4_DUALBRAIN.md` | Документ DualBrain | legacy‑inactive | remove‑later | Исторический документ |
| `updated_roadmap.md` | Старый роадмап | legacy‑inactive | do‑not‑touch | Справочный архив |

## Примечания
- DualBrain и LLM‑критик считаются **логически мёртвыми** в MWV‑runtime.
- Физическое удаление и правка UI — отдельный этап (не M6).
- Legacy‑поверхность исключена из `ruff format --check` через список `LEGACY_FORMAT_EXCLUDES`
  в `Makefile` до фазы удаления.
