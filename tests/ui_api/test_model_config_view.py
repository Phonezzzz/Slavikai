import asyncio
import json

from config.model_store import save_model_configs
from llm.types import ModelConfig
from tests.ui_api.test_memory_management import _isolate, _start


def test_saved_model_read_view_is_owner_only_sanitized_and_read_only(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    model_path = tmp_path / "models.json"
    for name, path in [
        ("UI_SETTINGS_PATH", tmp_path / "ui.json"),
        ("API_KEYS_PATH", tmp_path / "keys.json"),
        ("MODEL_CONFIG_PATH", model_path),
    ]:
        monkeypatch.setattr("server.http_api." + name, path)
    save_model_configs(
        ModelConfig(
            provider="local",
            model="fixture",
            temperature=0.23,
            top_p=0.61,
            max_tokens=2345,
            api_key="SECRET_KEY",
            base_url="https://SECRET_URL.invalid",
            extra_headers={"SECRET_HEADER": "SECRET_HEADER_VALUE"},
            system_prompt="SECRET_PROMPT",
            thinking_enabled=True,
            reasoning_effort="high",
            reasoning_summary=True,
            reasoning_summary_wait=False,
            diffusing=True,
            web_search_enabled=True,
        ),
        model_path,
    )
    before = model_path.read_bytes()

    async def run():
        client = await _start()
        try:

            async def read(who):
                response = await client.get(
                    "/ui/api/settings", headers={"Cf-Access-Jwt-Assertion": who}
                )
                assert response.status == 200
                text = await response.text()
                assert "SECRET_" not in text
                return json.loads(text)["settings"]

            owner = await read("owner")
            view = owner["model_config_snapshot"]
            assert view["basis"] == "loaded_default_model_at_refresh"
            assert view["values"]["temperature"] == 0.23
            assert view["values"]["max_tokens"] == 2345
            assert view["values"]["reasoning_effort"] == "high"
            assert all(view["presence"].values())
            assert view["values"]["web_search_enabled"] is True
            assert view["unsupported_fields"] == []
            assert "model_config_snapshot" not in await read("member")
            result = await client.post(
                "/ui/api/settings",
                headers={"Cf-Access-Jwt-Assertion": "owner"},
                json={"model_config_snapshot": view},
            )
            assert result.status == 403
            assert model_path.read_bytes() == before
        finally:
            await client.close()

    asyncio.run(run())


def test_missing_and_malformed_model_options_never_invent_or_echo_values(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    path = tmp_path / "models.json"
    for name, target in [
        ("MODEL_CONFIG_PATH", path),
        ("UI_SETTINGS_PATH", tmp_path / "ui.json"),
        ("API_KEYS_PATH", tmp_path / "keys.json"),
    ]:
        monkeypatch.setattr("server.http_api." + name, target)

    async def run():
        client = await _start()
        try:

            async def read():
                response = await client.get(
                    "/ui/api/settings", headers={"Cf-Access-Jwt-Assertion": "owner"}
                )
                assert response.status == 200
                text = await response.text()
                assert "SECRET_" not in text and "NaN" not in text
                return json.loads(text)["settings"]["model_config_snapshot"]

            assert (await read())["state"] == "unavailable"
            path.write_text(
                json.dumps(
                    {
                        "main": {
                            "provider": "local",
                            "model": "fixture",
                            "temperature": float("nan"),
                            "top_p": "SECRET_TOP_P",
                            "mode": "SECRET_MODE",
                            "reasoning_summary": "SECRET_SUMMARY",
                            "api_key": {"SECRET_KEY": True},
                            "extra_headers": "SECRET_HEADERS",
                        }
                    }
                )
            )
            view = await read()
            for key in ["temperature", "top_p", "mode", "reasoning_summary"]:
                assert view["values"][key] is None
                assert key in view["unavailable_fields"]
            assert view["presence"]["api_key"] is None
            assert view["presence"]["extra_headers"] is None
        finally:
            await client.close()

    asyncio.run(run())
