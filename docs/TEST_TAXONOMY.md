# TEST TAXONOMY

Этот документ формализует текущую taxonomy тестовых и gate-слоёв как documentation/selection layer.

Важно:

- этот документ не меняет execution semantics;
- этот документ не меняет baseline gate `make check`;
- taxonomy в этом PR не является acceptance enforcement layer;
- markers в этом PR являются vocabulary layer и не являются обязательным runtime selection mechanism.

## Текущие logical layers

В репозитории уже есть устойчивые тестовые слои, которые фактически выражены через path/name conventions:

- `tests/test_mwv_*.py` — MWV, runtime и verifier orchestration.
- `tests/test_auto_*.py` — auto runtime и orchestrator flows.
- `tests/test_http*.py` — HTTP/server/API слой.
- `tests/ui_api/*.py` — UI API, sessions, approvals, settings, events и workspace flows.
- `tests/test_*contract*.py` — contract-oriented проверки.
- `tests/test_*e2e*.py` — end-to-end и cross-flow сценарии.

Эти слои уже можно выбирать детерминированно без дополнительной инфраструктуры, через явные команды по путям и именам файлов.

## Marker vocabulary

В `pyproject.toml` зарегистрирован vocabulary markers:

- `mwv`
- `auto`
- `http_api`
- `ui_api`
- `contract`
- `e2e`

В рамках этого PR markers нужны как единый словарь для обсуждения, документации и следующих узких PR. Они не добавляют auto-tagging, не меняют collection behavior и не требуют массовой разметки существующих тестов.

## Deterministic Selection Commands

Текущий рекомендуемый selection в этом репозитории основан на path/name conventions:

```bash
venv/bin/python -m pytest tests/test_mwv_*.py
venv/bin/python -m pytest tests/test_auto_*.py
venv/bin/python -m pytest tests/test_http*.py
venv/bin/python -m pytest tests/ui_api
venv/bin/python -m pytest tests/test_*contract*.py
venv/bin/python -m pytest tests/test_*e2e*.py
```

Если нужен только просмотр состава suite без запуска:

```bash
venv/bin/python -m pytest --collect-only -q tests/test_mwv_*.py
venv/bin/python -m pytest --collect-only -q tests/test_auto_*.py
venv/bin/python -m pytest --collect-only -q tests/test_http*.py
venv/bin/python -m pytest --collect-only -q tests/ui_api
venv/bin/python -m pytest --collect-only -q tests/test_*contract*.py
venv/bin/python -m pytest --collect-only -q tests/test_*e2e*.py
```

## Risk-Oriented Mapping

Ниже зафиксирован минимальный mapping "тип изменения -> какие suites запускать". Это guidance для selection, а не новый policy gate.

- MWV/runtime/verifier changes -> `tests/test_mwv_*.py`
  Если меняется внешний workflow или пользовательский runtime flow, дополнительно смотреть `tests/test_*contract*.py` или `tests/test_*e2e*.py` по релевантности.
- Auto orchestration changes -> `tests/test_auto_*.py`
  Если change затрагивает cross-flow orchestration, дополнительно смотреть `tests/test_*e2e*.py`.
- HTTP/server endpoint changes -> `tests/test_http*.py`
  Если меняется wire contract, payload shape или response semantics, дополнительно смотреть `tests/test_*contract*.py`.
- UI API, session, approvals, events, settings, workspace changes -> `tests/ui_api`
  Если меняется payload или event contract, дополнительно смотреть `tests/test_*contract*.py`.
- User-visible protocol, report, schema or payload changes -> `tests/test_*contract*.py` плюс lane-specific suite.
- Cross-flow orchestration and end-to-end changes -> `tests/test_*e2e*.py` плюс lane-specific suite.

## Explicit Non-Goals

В этот PR не входят:

- изменения `make check`;
- изменения CI profiles;
- изменения verifier behavior;
- auto-tagging или hooks в `tests/conftest.py`;
- массовое добавление `@pytest.mark...`;
- новый behavior enforcement layer;
- новая архитектура test profiles.
