from __future__ import annotations

from pathlib import Path

from config.api_keys import save_api_keys
from config.stt_config import SttConfig
from config.tts_config import TtsConfig
from config.ui_embeddings_settings import resolve_openai_api_key
from shared.models import ToolRequest
from tools.http_client import HttpResult
from tools.stt_tool import SttTool
from tools.tts_tool import TtsTool


class DummyHttp:
    def __init__(
        self,
        payload: bytes | None = None,
        json_data=None,
        *,
        ok: bool = True,
        error: str | None = None,
    ) -> None:
        self.payload = payload
        self.json_data = json_data
        self.ok = ok
        self.error = error
        self.called = False
        self.last_kwargs = {}
        self.calls: list[dict[str, object]] = []

    def post_bytes(self, url, **kwargs):  # noqa: ANN001
        self.called = True
        self.last_kwargs = dict(kwargs)
        self.calls.append(dict(kwargs))
        return HttpResult(
            ok=self.ok,
            data=self.payload if self.payload is not None else b"audio",
            status_code=200 if self.ok else 502,
            error=self.error,
            headers={},
            meta={},
        )

    def _request(self, method, url, expect_json, as_bytes, **kwargs):  # noqa: ANN001
        self.called = True
        self.last_kwargs = dict(kwargs)
        return HttpResult(
            ok=True,
            data=(
                self.json_data
                if self.json_data is not None
                else {"text": "hello", "confidence": 0.9}
            ),
            status_code=200,
            error=None,
            headers={},
            meta={},
        )


def test_tts_tool_writes_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_TTS_MODEL", "")
    monkeypatch.setenv("OPENAI_TTS_VOICE", "")
    monkeypatch.setenv("OPENAI_TTS_FORMAT", "")
    http = DummyHttp(payload=b"audio data")
    tool = TtsTool(http, TtsConfig(format="mp3"))
    req = ToolRequest(name="tts", args={"text": "hello"})
    result = tool.handle(req)
    assert result.ok
    file_path = Path(result.data.get("file_path"))
    assert file_path.exists()
    assert http.called
    assert http.last_kwargs.get("json") == {
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "input": "hello",
        "response_format": "mp3",
    }
    headers = http.last_kwargs.get("headers")
    assert isinstance(headers, dict)
    assert headers.get("Authorization") == "Bearer key"


def test_tts_tool_splits_long_text_into_provider_sized_chunks(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    http = DummyHttp(payload=b"audio")
    tool = TtsTool(http, TtsConfig(format="mp3", max_input_chars=1000, max_request_chars=120))
    text = " ".join(f"sentence-{index}" for index in range(80))

    result = tool.handle(ToolRequest(name="tts", args={"text": text}))

    assert result.ok
    assert len(http.calls) > 1
    for call in http.calls:
        payload = call.get("json")
        assert isinstance(payload, dict)
        chunk = payload.get("input")
        assert isinstance(chunk, str)
        assert 0 < len(chunk) <= 120
    assert result.data.get("chunks") == len(http.calls)
    file_path = Path(result.data.get("file_path"))
    assert file_path.read_bytes() == b"audio" * len(http.calls)


def test_tts_tool_rejects_text_above_total_limit(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    http = DummyHttp(payload=b"audio")
    tool = TtsTool(http, TtsConfig(max_input_chars=20, max_request_chars=10))

    result = tool.handle(ToolRequest(name="tts", args={"text": "x" * 21}))

    assert not result.ok
    assert result.error == "Текст для озвучки превышает лимит 20 символов."
    assert not http.called


def test_tts_tool_requires_openai_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tool = TtsTool(DummyHttp(), TtsConfig(api_keys_path=tmp_path / "api_keys.json"))
    result = tool.handle(ToolRequest(name="tts", args={"text": "hello"}))
    assert not result.ok
    assert result.error == "OpenAI API key не задан для TTS (env или Settings → API Keys)."


def test_audio_tools_use_saved_openai_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("STT_API_KEY", raising=False)
    api_keys_path = tmp_path / "api_keys.json"
    save_api_keys({"openai": "saved-openai-key"}, path=api_keys_path)

    assert TtsConfig(api_keys_path=api_keys_path).resolve_api_key() == "saved-openai-key"
    assert SttConfig(api_keys_path=api_keys_path).resolve_api_key() == "saved-openai-key"
    assert resolve_openai_api_key(api_keys_path=api_keys_path) == "saved-openai-key"


def test_tts_tool_rejects_invalid_format(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    tool = TtsTool(DummyHttp())
    result = tool.handle(ToolRequest(name="tts", args={"text": "hello", "format": "ogg"}))
    assert not result.ok
    assert result.error == "Формат должен быть mp3 или wav."


def test_tts_tool_rejects_empty_upstream_audio(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    tool = TtsTool(DummyHttp(payload=b""))
    result = tool.handle(ToolRequest(name="tts", args={"text": "hello"}))
    assert not result.ok
    assert result.error == "Ошибка OpenAI TTS сервиса."


def test_tts_tool_propagates_upstream_error(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    tool = TtsTool(DummyHttp(ok=False, error="upstream boom"))
    result = tool.handle(ToolRequest(name="tts", args={"text": "hello"}))
    assert not result.ok
    assert result.error == "upstream boom"


def test_stt_tool_reads_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
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
    assert http.last_kwargs.get("data") == {
        "model": "whisper-1",
        "language": "ru",
        "response_format": "json",
    }
    headers = http.last_kwargs.get("headers")
    assert isinstance(headers, dict)
    assert headers.get("Authorization") == "Bearer key"


def test_stt_tool_reads_api_key_from_legacy_env(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("STT_API_KEY", "legacy-stt-key")

    audio_dir = Path("sandbox/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    file_path = audio_dir / "stt-legacy-env.wav"
    file_path.write_bytes(b"dummy")

    http = DummyHttp(json_data={"text": "ok"})
    tool = SttTool(http, SttConfig())
    req = ToolRequest(name="stt", args={"file_path": str(file_path)})
    result = tool.handle(req)
    assert result.ok
    headers = http.last_kwargs.get("headers")
    assert isinstance(headers, dict)
    assert headers.get("Authorization") == "Bearer legacy-stt-key"
