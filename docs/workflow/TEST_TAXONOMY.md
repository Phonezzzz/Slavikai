# TEST TAXONOMY

Этот документ формализует текущую taxonomy тестовых и gate-слоёв как documentation/selection layer.

Важно:

- этот документ не меняет execution semantics;
- этот документ не меняет baseline gate `make check`;
- taxonomy в этом PR не является acceptance enforcement layer;
- markers в этом PR в основном являются vocabulary layer;
- исключение: `behavior` — это явный minimal curated subset, который можно запускать отдельной focused-командой;
- даже `behavior` в этом PR не вводит CI/profile split и не является полным rollout behavior enforcement.

## Текущие logical layers

В репозитории уже есть устойчивые тестовые слои, которые фактически выражены через path/name conventions:

- `tests/test_mwv_*.py` — MWV, runtime и verifier orchestration.
- `tests/test_auto_*.py` — auto runtime и orchestrator flows.
- `tests/test_http*.py` — HTTP/server/API слой.
- `tests/ui_api/*.py` — UI API, sessions, approvals, settings, events и workspace flows.
- `tests/test_*contract*.py` — contract-oriented проверки.
- `tests/test_*e2e*.py` — end-to-end и cross-flow сценарии.
- `pytest -m behavior` — минимальный curated subset критических сценариев поведения.

Эти слои уже можно выбирать детерминированно без дополнительной инфраструктуры, через явные команды по путям и именам файлов.

## Marker vocabulary

В `pyproject.toml` зарегистрирован vocabulary markers:

- `mwv`
- `auto`
- `http_api`
- `ui_api`
- `contract`
- `e2e`
- `behavior`

В рамках этого PR markers нужны как единый словарь для обсуждения, документации и следующих узких PR. Они не добавляют auto-tagging, не меняют collection behavior и не требуют массовой разметки существующих тестов.

`behavior` — отдельный случай: это не общий vocabulary-only marker, а явный curated marker для узкого набора уже существующих сильных сценарных тестов.

## Deterministic Selection Commands

Текущий рекомендуемый selection в этом репозитории основан на path/name conventions:

```bash
venv/bin/python -m pytest tests/test_mwv_*.py
venv/bin/python -m pytest tests/test_auto_*.py
venv/bin/python -m pytest tests/test_http*.py
venv/bin/python -m pytest tests/ui_api
venv/bin/python -m pytest tests/test_*contract*.py
venv/bin/python -m pytest tests/test_*e2e*.py
make test-behavior
```

Если нужен только просмотр состава suite без запуска:

```bash
venv/bin/python -m pytest --collect-only -q tests/test_mwv_*.py
venv/bin/python -m pytest --collect-only -q tests/test_auto_*.py
venv/bin/python -m pytest --collect-only -q tests/test_http*.py
venv/bin/python -m pytest --collect-only -q tests/ui_api
venv/bin/python -m pytest --collect-only -q tests/test_*contract*.py
venv/bin/python -m pytest --collect-only -q tests/test_*e2e*.py
venv/bin/python -m pytest --no-cov --collect-only -m behavior -q
```

`make test-behavior` намеренно использует `--no-cov`, чтобы focused subset не падал на глобальном coverage threshold, который относится к полному baseline gate `make check`.

## Behavior Layer

Текущий explicit behavior layer в этом репозитории специально узкий. Он включает только curated subset уже существующих сильных сценарных тестов:

- MWV happy path с реальным workspace change.
- Auto approval/resume flow.
- UI approval persistence.
- UI retry/resume flow.
- UI idempotency flow.
- UI plan approval/resume flow.
- Event queue overflow/coalescing.
- Event replay.

Цель этого слоя — сделать критические сценарии явными и запускаемыми отдельной командой без полной реформы test architecture.

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
- Изменения в критических workflow/state transitions/approval-resume/event-ordering -> `make test-behavior`
  Это minimal focused runner, а не замена lane-specific suite.

## Explicit Non-Goals

В этот PR не входят:

- изменения `make check`;
- изменения CI profiles;
- изменения verifier behavior;
- auto-tagging или hooks в `tests/conftest.py`;
- массовое добавление `@pytest.mark...`;
- полный behavior enforcement rollout;
- новая архитектура test profiles.
