import asyncio
import json

from config.memory_config import ContextBudgetConfig, load_memory_config
from tests.ui_api.test_memory_management import _isolate, _start


def test_configured_context_budget_is_owner_only_read_only_snapshot(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    for name, filename in [
        ("UI_SETTINGS_PATH", "ui.json"),
        ("API_KEYS_PATH", "keys.json"),
        ("MODEL_CONFIG_PATH", "models.json"),
    ]:
        monkeypatch.setattr("server.http_api." + name, tmp_path / filename)
    config = tmp_path / "memory.json"
    expected = {key: 7000 + n for n, key in enumerate(ContextBudgetConfig().to_dict())}
    config.write_text(json.dumps({"context_budget": expected}))
    monkeypatch.setattr("server.http_api.load_memory_config", lambda: load_memory_config(config))
    before = config.read_bytes()

    async def run():
        client = await _start()
        try:

            async def read(who):
                response = await client.get(
                    "/ui/api/settings", headers={"Cf-Access-Jwt-Assertion": who}
                )
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                return (await response.json())["settings"]

            owner = await read("owner")
            snapshot = owner["context_budget_snapshot"]
            assert snapshot["values"] == expected
            assert snapshot["basis"] == "server_configuration_at_refresh"
            assert snapshot["source"] == "Memory configuration with server defaults"
            assert "context_budget_snapshot" not in await read("member")
            assert "context_budget" not in owner["memory"]
            for payload in [
                {"context_budget_snapshot": {"values": {"total_chars": 1}}},
                {"memory": {"context_budget": {"total_chars": 1}}},
                {"context_budget": {"total_chars": 1}},
            ]:
                response = await client.post(
                    "/ui/api/settings",
                    headers={"Cf-Access-Jwt-Assertion": "owner"},
                    json=payload,
                )
                assert response.status == 400
                assert (await response.json())["error"]["code"] == "context_budget_read_only"
            assert config.read_bytes() == before
            # A refresh reads the loader again; it is not a cached running-Agent claim.
            expected["total_chars"] = 9123
            config.write_text(json.dumps({"context_budget": expected}))
            assert (await read("owner"))["context_budget_snapshot"]["values"] == expected
        finally:
            await client.close()

    asyncio.run(run())
