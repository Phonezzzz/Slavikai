from __future__ import annotations

from pathlib import Path

from config.stt_config import SttConfig
from config.tts_config import TtsConfig
from shared.models import ToolRequest
from tools.http_client import HttpResult
from tools.stt_tool import SttTool
from tools.tts_tool import TtsTool


class DummyHttp:
    def __init__(self, payload: bytes | None = None, json_data=None) -> None:
        self.payload = payload
        self.json_data = json_data
        self.called = False

    def post_bytes(self, url, **kwargs):  # noqa: ANN001
        self.called = True
        return HttpResult(
            ok=True,
            data=self.payload or b"audio",
            status_code=200,
            error=None,
            headers={},
            meta={},
        )

    def _request(self, method, url, expect_json, as_bytes, **kwargs):  # noqa: ANN001
        self.called = True
        return HttpResult(
            ok=True,
            data=self.json_data or {"text": "hello", "confidence": 0.9},
            status_code=200,
            error=None,
            headers={},
            meta={},
        )


def test_tts_tool_writes_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TTS_API_KEY", "key")
    monkeypatch.setenv("TTS_VOICE_ID", "voice")
    http = DummyHttp(payload=b"audio data")
    tool = TtsTool(http, TtsConfig(format="mp3"))
    req = ToolRequest(name="tts", args={"text": "hello"})
    result = tool.handle(req)
    assert result.ok
    file_path = Path(result.data.get("file_path"))
    assert file_path.exists()
    assert http.called


def test_stt_tool_reads_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STT_API_KEY", "key")
    audio_dir = Path("sandbox/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    file_path = audio_dir / "test.wav"
    file_path.write_bytes(b"dummy")
    http = DummyHttp(json_data={"text": "привет", "confidence": 0.5})
    tool = SttTool(http, SttConfig())
    req = ToolRequest(name="stt", args={"file_path": str(file_path)})
    result = tool.handle(req)
    assert result.ok
    assert "привет" in result.data.get("output")
    assert http.called
