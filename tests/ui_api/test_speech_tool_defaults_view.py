import asyncio

import pytest

from config.stt_config import SttConfig
from config.tts_config import TtsConfig
from tests.ui_api.test_memory_management import _isolate, _start


@pytest.mark.parametrize("limits", [(20, 32000, 4096, 20), (7, 12345, 678, 9)])
def test_speech_tool_defaults_are_owner_only_allowlisted_without_resolving_or_activating(
    monkeypatch, tmp_path, limits
):
    _isolate(monkeypatch, tmp_path)
    for name, filename in [
        ("UI_SETTINGS_PATH", "ui.json"),
        ("API_KEYS_PATH", "keys.json"),
        ("MODEL_CONFIG_PATH", "models.json"),
    ]:
        monkeypatch.setattr("server.http_api." + name, tmp_path / filename)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Default diagnostics must not resolve speech options or activate tools"
        )

    for target in [
        "tools.tts_tool.TtsTool",
        "tools.stt_tool.SttTool",
        "tools.http_client.HttpClient",
    ]:
        monkeypatch.setattr(target + ".__init__", forbidden)
    tts = TtsConfig(
        timeout=limits[0],
        max_input_chars=limits[1],
        max_request_chars=limits[2],
        api_key="SECRET_TTS",
        voice="SECRET_VOICE",
        model="SECRET_MODEL",
        endpoint="https://SECRET_ENDPOINT.invalid",
    )
    stt = SttConfig(timeout=limits[3], api_key="SECRET_STT", language="SECRET_LANGUAGE")
    for name in ["resolve_api_key", "resolve_model", "resolve_voice", "resolve_format"]:
        monkeypatch.setattr(tts, name, forbidden)
    monkeypatch.setattr(stt, "resolve_api_key", forbidden)
    # Старую env-диагностику TTS не подменяем; guards относятся к новым snapshot objects.
    monkeypatch.setattr("server.http.common.settings_runtime.TtsConfig", lambda: tts, raising=False)
    monkeypatch.setattr("server.http.common.settings_runtime.SttConfig", lambda: stt, raising=False)

    async def run():
        client = await _start()
        try:

            async def read(who):
                response = await client.get(
                    "/ui/api/settings", headers={"Cf-Access-Jwt-Assertion": who}
                )
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                assert "SECRET_" not in await response.text()
                return (await response.json())["settings"]

            snapshot = (await read("owner"))["speech_tool_defaults_snapshot"]
            assert snapshot == {
                "basis": "speech_tool_class_defaults_at_refresh",
                "values": {
                    "tts_timeout_seconds": limits[0],
                    "tts_input_code_points": limits[1],
                    "tts_request_code_points": limits[2],
                    "stt_timeout_seconds": limits[3],
                },
            }
            assert "speech_tool_defaults_snapshot" not in await read("member")
            before = {p: p.read_bytes() for p in tmp_path.glob("*.json")}
            for who in ["owner", "member"]:
                response = await client.post(
                    "/ui/api/settings",
                    headers={"Cf-Access-Jwt-Assertion": who},
                    json={"speech_tool_defaults_snapshot": snapshot},
                )
                assert response.status == 403
            assert {p: p.read_bytes() for p in tmp_path.glob("*.json")} == before
        finally:
            await client.close()

    asyncio.run(run())
