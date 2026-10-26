import asyncio

import pytest

from config.web_search_config import WebSearchConfig
from tests.ui_api.test_memory_management import _isolate, _start


@pytest.mark.parametrize("limits", [(5, 20, 200000), (3, 7, 12345)])
def test_search_backend_defaults_are_owner_only_read_only_and_do_not_construct_tools(
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
        raise AssertionError("Settings diagnostics must not construct search tools or HTTP clients")

    monkeypatch.setattr("tools.web_search_tool.WebSearchTool.__init__", forbidden)
    monkeypatch.setattr("tools.http_client.HttpClient.__init__", forbidden)
    config = WebSearchConfig(
        provider="serpapi",
        api_key="SECRET_FIXTURE",
        top_k=limits[0],
        timeout=limits[1],
        max_bytes=limits[2],
    )
    # GET читает существующий config owner, а не копию числовых defaults.
    monkeypatch.setattr("server.http.common.settings_runtime.WebSearchConfig", lambda: config)

    async def run():
        client = await _start()
        try:

            async def read(who):
                response = await client.get(
                    "/ui/api/settings", headers={"Cf-Access-Jwt-Assertion": who}
                )
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                text = await response.text()
                assert "SECRET_FIXTURE" not in text
                return (await response.json())["settings"]

            snapshot = (await read("owner"))["search_backend_defaults_snapshot"]
            assert snapshot == {
                "basis": "backend_class_defaults_at_refresh",
                "values": {
                    "top_k": limits[0],
                    "timeout_seconds": limits[1],
                    "response_content_bytes": limits[2],
                },
            }
            assert "search_backend_defaults_snapshot" not in await read("member")
            before = {p: p.read_bytes() for p in tmp_path.glob("*.json")}
            for who in ["owner", "member"]:
                response = await client.post(
                    "/ui/api/settings",
                    headers={"Cf-Access-Jwt-Assertion": who},
                    json={"search_backend_defaults_snapshot": snapshot},
                )
                assert response.status == 403
            assert {p: p.read_bytes() for p in tmp_path.glob("*.json")} == before
        finally:
            await client.close()

    asyncio.run(run())
