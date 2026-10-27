import asyncio

import pytest

from config.shell_config import ShellConfig
from tests.ui_api.test_memory_management import _isolate, _start


@pytest.mark.parametrize("limits", [(10, 6000), (7, 12345)])
def test_shell_numeric_defaults_are_owner_only_without_policy_io_or_execution(
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
        raise AssertionError("Default diagnostics must not read or mutate policy, or execute")

    for target in [
        "config.shell_config.load_shell_config",
        "config.shell_config.save_shell_config",
        "config.shell_config.normalize_shell_sandbox_root",
        "tools.shell_tool.load_shell_config",
        "tools.terminal_tool.load_shell_config",
        "tools.terminal_tool.subprocess.run",
    ]:
        monkeypatch.setattr(target, forbidden)
    config = ShellConfig(
        timeout_seconds=limits[0],
        max_output_chars=limits[1],
        allowed_commands=["SECRET_COMMAND"],
        sandbox_root="SECRET_ROOT",
    )
    monkeypatch.setattr(
        "server.http.common.settings_runtime.ShellConfig", lambda: config, raising=False
    )

    async def run():
        client = await _start()
        monkeypatch.setattr("tools.shell_tool.ShellTool.__init__", forbidden)
        monkeypatch.setattr("tools.terminal_tool.TerminalTool.__init__", forbidden)
        try:

            async def read(who):
                response = await client.get(
                    "/ui/api/settings", headers={"Cf-Access-Jwt-Assertion": who}
                )
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                assert "SECRET_" not in await response.text()
                return (await response.json())["settings"]

            snapshot = (await read("owner"))["shell_numeric_defaults_snapshot"]
            assert snapshot == {
                "basis": "shell_class_defaults_at_refresh",
                "values": {"timeout_seconds": limits[0], "output_code_points": limits[1]},
            }
            assert "shell_numeric_defaults_snapshot" not in await read("member")
            before = {p: p.read_bytes() for p in tmp_path.glob("*.json")}
            for who in ["owner", "member"]:
                response = await client.post(
                    "/ui/api/settings",
                    headers={"Cf-Access-Jwt-Assertion": who},
                    json={"shell_numeric_defaults_snapshot": snapshot},
                )
                assert response.status == 403
            assert {p: p.read_bytes() for p in tmp_path.glob("*.json")} == before
        finally:
            await client.close()

    asyncio.run(run())
