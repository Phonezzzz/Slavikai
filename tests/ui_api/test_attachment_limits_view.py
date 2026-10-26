import asyncio

import pytest

from server.http.common import chat_payload
from tests.ui_api.test_memory_management import _isolate, _start


@pytest.mark.parametrize("limits", [(8, 80000, 160000), (3, 17, 29)])
def test_attachment_limits_are_owner_only_live_validator_values_and_read_only(
    monkeypatch, tmp_path, limits
):
    _isolate(monkeypatch, tmp_path)
    for name, filename in [
        ("UI_SETTINGS_PATH", "ui.json"),
        ("API_KEYS_PATH", "keys.json"),
        ("MODEL_CONFIG_PATH", "models.json"),
    ]:
        monkeypatch.setattr("server.http_api." + name, tmp_path / filename)
    for name, value in zip(
        ["MAX_ATTACHMENTS_PER_MESSAGE", "MAX_ATTACHMENT_CHARS", "MAX_TOTAL_ATTACHMENTS_CHARS"],
        limits,
        strict=True,
    ):
        monkeypatch.setattr(chat_payload, name, value)

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
            snapshot = owner["attachment_limits_snapshot"]
            assert snapshot == {
                "basis": "server_request_validation_at_refresh",
                "values": {
                    "max_items": limits[0],
                    "max_item_code_points": limits[1],
                    "max_total_code_points": limits[2],
                },
            }
            assert "attachment_limits_snapshot" not in await read("member")
            before = {p: p.read_bytes() for p in tmp_path.glob("*.json")}
            for who in ["owner", "member"]:
                response = await client.post(
                    "/ui/api/settings",
                    headers={"Cf-Access-Jwt-Assertion": who},
                    json={"attachment_limits_snapshot": snapshot},
                )
                assert response.status == 403
            assert {p: p.read_bytes() for p in tmp_path.glob("*.json")} == before
        finally:
            await client.close()

    asyncio.run(run())
