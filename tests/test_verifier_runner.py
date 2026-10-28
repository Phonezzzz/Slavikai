from pathlib import Path

import pytest

from core.mwv.models import VerificationStatus
from core.mwv.verifier import VerifierRunner


@pytest.mark.parametrize("body", ["echo ok", "exit 7", "sleep 99"])
def test_verifier_script_is_not_executed(tmp_path, monkeypatch, body):
    script = tmp_path / "scripts" / "check.sh"
    script.parent.mkdir()
    script.write_text(body)
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a))
    result = VerifierRunner(script_path=script).run()
    assert calls == []
    assert result.status == VerificationStatus.ERROR
    assert result.error == "sandbox_execution_unavailable"
    assert result.exit_code is None
    assert result.stdout == ""


def test_missing_script_retains_validation_error(tmp_path: Path):
    result = VerifierRunner(script_path=tmp_path / "missing.sh").run()
    assert result.status == VerificationStatus.ERROR
    assert "not found" in result.error
