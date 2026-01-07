from __future__ import annotations

import logging
from pathlib import Path

import core.tracer as tracer_module
import tools.tool_logger as tool_logger_module
from core.tracer import Tracer
from shared.sanitize import sanitize_record
from tools.tool_logger import ToolCallLogger


def test_sanitizer_masks_secrets() -> None:
    data = {
        "api_key": "supersecret",
        "Authorization": "Bearer token",
        "nested": {"token": "abc"},
        "ok": True,
    }
    sanitized = sanitize_record(data)
    assert sanitized["api_key"] == "[secret]"
    assert sanitized["Authorization"] == "[secret]"
    assert sanitized["nested"]["token"] == "[secret]"


def test_sanitizer_truncates_large_payload() -> None:
    payload = "x" * 600
    data = {"content": payload}
    sanitized = sanitize_record(data)
    content = sanitized["content"]
    assert isinstance(content, dict)
    assert content["bytes_count"] == len(payload.encode())
    assert "…[truncated]" in content["preview"]
    assert len(content["sha256"]) == 64


def test_read_recent_does_not_read_whole_file(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "tool_calls.log"
    logger = ToolCallLogger(path=log_path)
    for i in range(500):
        logger.log(tool="t", ok=True, error=None, meta={"i": i}, args={"v": "x"})

    class NoReadlinesWrapper:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __iter__(self):
            return self

        def __next__(self):
            line = self._wrapped.readline()
            if line == "":
                raise StopIteration
            return line

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self._wrapped.close()
            return False

        def readlines(self, *args, **kwargs):
            raise AssertionError("readlines should not be used")

        def close(self) -> None:
            self._wrapped.close()

        def __getattr__(self, item):
            return getattr(self._wrapped, item)

    orig_open = Path.open

    def fake_open(self, mode="r", encoding=None):  # noqa: ANN001
        f = orig_open(self, mode, encoding=encoding)
        return NoReadlinesWrapper(f)

    monkeypatch.setattr(Path, "open", fake_open, raising=False)

    recent = logger.read_recent(limit=5)
    assert len(recent) == 5
    tracer_path = tmp_path / "trace.log"
    tracer = Tracer(path=tracer_path)
    for i in range(100):
        tracer.log("evt", f"m{i}")
    recent_trace = tracer.read_recent(limit=3)
    assert len(recent_trace) == 3


def test_tool_logger_skips_invalid_json(tmp_path: Path, caplog) -> None:
    log_path = tmp_path / "tool_calls.log"
    log_path.write_text(
        '{"timestamp":"t","tool":"x","ok":true,"error":null,"meta":{},"args":{}}\n{bad json}\n',
        encoding="utf-8",
    )
    logger = ToolCallLogger(path=log_path)
    with caplog.at_level(logging.WARNING):
        records = logger.read_recent(limit=10)
    assert len(records) == 1
    assert any("некорректных строк" in record.message for record in caplog.records)


def test_tracer_skips_invalid_json(tmp_path: Path, caplog) -> None:
    tracer_path = tmp_path / "trace.log"
    tracer_path.write_text(
        '{"timestamp":"t","event":"e","message":"m","meta":{}}\n{bad json}\n',
        encoding="utf-8",
    )
    tracer = Tracer(path=tracer_path)
    with caplog.at_level(logging.WARNING):
        records = tracer.read_recent(limit=10)
    assert len(records) == 1
    assert any("некорректных строк" in record.message for record in caplog.records)


def test_tracer_rotates_by_size(tmp_path: Path, monkeypatch) -> None:
    tracer_path = tmp_path / "trace.log"
    tracer = Tracer(path=tracer_path)
    monkeypatch.setattr(tracer_module, "MAX_TRACE_LOG_BYTES", 200)
    monkeypatch.setattr(tracer_module, "MAX_TRACE_LOG_BACKUPS", 2)

    for i in range(50):
        tracer.log("evt", f"msg-{i}-" + ("x" * 20))

    assert tracer_path.exists()
    assert (tmp_path / "trace.log.1").exists()


def test_tool_logger_rotates_by_size(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "tool_calls.log"
    logger = ToolCallLogger(path=log_path)
    monkeypatch.setattr(tool_logger_module, "MAX_TOOL_LOG_BYTES", 200)
    monkeypatch.setattr(tool_logger_module, "MAX_TOOL_LOG_BACKUPS", 2)

    for i in range(50):
        logger.log(tool="t", ok=True, error=None, meta={"i": i}, args={"v": "x"})

    assert log_path.exists()
    assert (tmp_path / "tool_calls.log.1").exists()
