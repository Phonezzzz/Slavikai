from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_LEGACY_TESTS = {
    "test_critic_policy.py",
    "test_http_api.py",
}


def pytest_collection_modifyitems(config, items):  # noqa: ANN001
    skip_marker = pytest.mark.skip(
        reason="Legacy DualBrain/critic tests are deprecated in MWV canonical flow."
    )
    _ = config
    for item in items:
        path = Path(str(item.fspath))
        if path.name in _LEGACY_TESTS:
            item.add_marker(skip_marker)
