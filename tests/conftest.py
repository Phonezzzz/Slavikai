from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_skill_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidates_dir = tmp_path / "skills" / "_candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SKILLS_CANDIDATES_DIR", str(candidates_dir))
