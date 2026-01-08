from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.http_server_config import (
    DEFAULT_HOST,
    DEFAULT_MAX_REQUEST_BYTES,
    DEFAULT_PORT,
    HttpServerConfig,
    load_http_server_config,
    resolve_http_server_config,
)


def test_load_http_server_config_defaults(tmp_path: Path) -> None:
    path = tmp_path / "http_server.json"
    config = load_http_server_config(path)
    assert config.host == DEFAULT_HOST
    assert config.port == DEFAULT_PORT
    assert config.max_request_bytes == DEFAULT_MAX_REQUEST_BYTES


def test_load_http_server_config_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "http_server.json"
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_http_server_config(path)


def test_load_http_server_config_invalid_types(tmp_path: Path) -> None:
    path = tmp_path / "http_server.json"
    path.write_text(json.dumps(["oops"]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_http_server_config(path)

    path.write_text(
        json.dumps({"host": "", "port": 8000, "max_request_bytes": 100}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_http_server_config(path)

    path.write_text(
        json.dumps({"host": "127.0.0.1", "port": "8000", "max_request_bytes": 100}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_http_server_config(path)

    path.write_text(
        json.dumps({"host": "127.0.0.1", "port": 8000, "max_request_bytes": "x"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_http_server_config(path)


def test_resolve_http_server_config_env_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "http_server.json"
    config = HttpServerConfig(host="0.0.0.0", port=9000, max_request_bytes=123)
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    monkeypatch.setenv("SLAVIK_HTTP_HOST", "localhost")
    monkeypatch.setenv("SLAVIK_HTTP_PORT", "7001")
    monkeypatch.setenv("SLAVIK_HTTP_MAX_REQUEST_BYTES", "2048")

    resolved = resolve_http_server_config(path)
    assert resolved.host == "localhost"
    assert resolved.port == 7001
    assert resolved.max_request_bytes == 2048


def test_resolve_http_server_config_env_invalid_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "http_server.json"
    path.write_text(json.dumps(HttpServerConfig().to_dict()), encoding="utf-8")

    monkeypatch.setenv("SLAVIK_HTTP_PORT", "oops")
    with pytest.raises(ValueError):
        resolve_http_server_config(path)


def test_resolve_http_server_config_env_invalid_max_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "http_server.json"
    path.write_text(json.dumps(HttpServerConfig().to_dict()), encoding="utf-8")

    monkeypatch.setenv("SLAVIK_HTTP_MAX_REQUEST_BYTES", "oops")
    with pytest.raises(ValueError):
        resolve_http_server_config(path)
