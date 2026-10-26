import asyncio

import pytest

from core import agent_tools
from memory import session_summarizer
from tests.ui_api.test_memory_management import _isolate, _start


@pytest.mark.parametrize("limits", [(20, 40, 500), (7, 13, 109)])
def test_runtime_context_limits_are_owner_only_source_values_and_read_only(
    monkeypatch, tmp_path, limits
):
    _isolate(monkeypatch, tmp_path)
    for name, filename in [
        ("UI_SETTINGS_PATH", "ui.json"),
        ("API_KEYS_PATH", "keys.json"),
        ("MODEL_CONFIG_PATH", "models.json"),
    ]:
        monkeypatch.setattr("server.http_api." + name, tmp_path / filename)
    monkeypatch.setattr(agent_tools, "MAX_SHORT_TERM_MESSAGES", limits[0])
    monkeypatch.setattr(session_summarizer, "MAX_SUMMARY_MESSAGES", limits[1])
    monkeypatch.setattr(session_summarizer, "MAX_TRANSCRIPT_MESSAGE_CHARS", limits[2])

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
            snapshot = owner["runtime_context_limits_snapshot"]
            assert snapshot == {
                "basis": "runtime_context_constants_at_refresh",
                "values": {
                    "short_term_max_messages": limits[0],
                    "summary_max_messages": limits[1],
                    "summary_message_code_points": limits[2],
                },
            }
            assert "runtime_context_limits_snapshot" not in await read("member")
            before = {p: p.read_bytes() for p in tmp_path.glob("*.json")}
            for who in ["owner", "member"]:
                response = await client.post(
                    "/ui/api/settings",
                    headers={"Cf-Access-Jwt-Assertion": who},
                    json={"runtime_context_limits_snapshot": snapshot},
                )
                assert response.status == 403
            assert {p: p.read_bytes() for p in tmp_path.glob("*.json")} == before
        finally:
            await client.close()

    asyncio.run(run())
