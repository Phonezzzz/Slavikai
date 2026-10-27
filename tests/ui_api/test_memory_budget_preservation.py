import asyncio

import pytest

from config.memory_config import (
    ContextBudgetConfig,
    MemoryConfig,
    load_memory_config,
    save_memory_config,
)
from tests.ui_api.test_memory_management import _isolate, _start


def _configure(monkeypatch, root, custom=True):
    _isolate(monkeypatch, root)
    monkeypatch.setattr("server.http.app._DOTENV_LOAD_ATTEMPTED", True)
    for name, filename in [
        ("UI_SETTINGS_PATH", "ui.json"),
        ("API_KEYS_PATH", "keys.json"),
        ("MODEL_CONFIG_PATH", "models.json"),
    ]:
        monkeypatch.setattr("server.http_api." + name, root / filename)
    path = root / "memory.json"
    expected = (
        ContextBudgetConfig.from_dict(
            {key: 7000 + n for n, key in enumerate(ContextBudgetConfig().to_dict())}
        )
        if custom
        else ContextBudgetConfig()
    )
    save_memory_config(MemoryConfig(context_budget=expected), path)
    monkeypatch.setattr("server.http_api.load_memory_config", lambda: load_memory_config(path))
    monkeypatch.setattr(
        "server.http_api.save_memory_config", lambda value: save_memory_config(value, path)
    )
    return path, expected


@pytest.mark.parametrize("custom", [True, False])
@pytest.mark.parametrize(
    "update",
    [
        {"inbox_max_items": 211},
        {"inbox_ttl_days": 31},
        {"inbox_writes_per_minute": 7},
        {"embeddings": {"provider": "local"}},
    ],
)
def test_memory_updates_preserve_unedited_context_budget(monkeypatch, tmp_path, custom, update):
    path, expected = _configure(monkeypatch, tmp_path, custom)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Memory settings preservation must not initialize Agent or call providers"
        )

    async def run():
        client = await _start()
        try:
            client.app["agent_provider"]._factory = forbidden
            monkeypatch.setattr("requests.get", forbidden)
            monkeypatch.setattr("requests.post", forbidden)
            headers = {"Cf-Access-Jwt-Assertion": "owner"}
            response = await client.post(
                "/ui/api/settings", headers=headers, json={"memory": update}
            )
            assert response.status == 200
            reloaded = load_memory_config(path)
            assert reloaded.context_budget == expected
            for key, value in update.items():
                if key != "embeddings":
                    assert getattr(reloaded, key) == value
            fresh = await client.get("/ui/api/settings", headers=headers)
            assert fresh.status == 200
            assert (await fresh.json())["settings"]["context_budget_snapshot"][
                "values"
            ] == expected.to_dict()
        finally:
            await client.close()

    asyncio.run(run())


def test_memory_budget_preservation_does_not_enable_writes_or_member_authority(
    monkeypatch, tmp_path
):
    path, _ = _configure(monkeypatch, tmp_path)
    before = path.read_bytes()

    async def run():
        client = await _start()
        try:
            for who, payload, status in [
                ("member", {"memory": {"inbox_ttl_days": 31}}, 403),
                ("owner", {"memory": {"context_budget": {"total_chars": 1}}}, 400),
            ]:
                response = await client.post(
                    "/ui/api/settings", headers={"Cf-Access-Jwt-Assertion": who}, json=payload
                )
                assert response.status == status
                assert path.read_bytes() == before
        finally:
            await client.close()

    asyncio.run(run())
