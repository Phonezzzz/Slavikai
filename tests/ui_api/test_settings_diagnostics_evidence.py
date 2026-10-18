from __future__ import annotations

# ruff: noqa: F403,F405
import asyncio
import json
from pathlib import Path

import pytest
import requests

from config.api_keys import save_api_keys

from .fakes import *


@pytest.mark.parametrize("source", ["missing", "file", "env"])
def test_tts_reports_configuration_without_speech_probe(
    monkeypatch, tmp_path: Path, source: str
) -> None:
    keys_path = tmp_path / "api_keys.json"
    monkeypatch.setattr("server.http_api.API_KEYS_PATH", keys_path)
    monkeypatch.setattr("server.http_api.UI_SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if source in {"file", "env"}:
        save_api_keys({"openai": "saved-test-key"}, path=keys_path)
    if source == "env":
        monkeypatch.setenv("OPENAI_API_KEY", "environment-test-key")

    def unexpected_request(*args, **kwargs):
        raise AssertionError("settings snapshot must not probe speech or models")

    monkeypatch.setattr("server.http.common.ui_settings.requests.get", unexpected_request)
    monkeypatch.setattr("server.http.common.ui_settings.requests.post", unexpected_request)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.get("/ui/api/settings")
            assert response.status == 200
            payload = await response.json()
            tts = payload["settings"]["audio"]["tts"]
            assert tts["api_key_source"] == source
            assert tts["api_key_set"] is (source != "missing")
            assert "backend_ready" not in tts
            assert "saved-test-key" not in json.dumps(payload)
            assert "environment-test-key" not in json.dumps(payload)
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["local", "inception"])
@pytest.mark.parametrize(
    "outcome", ["catalog", "empty", "transport", "auth", "malformed", "shape", "entry"]
)
def test_discovery_catalog_evidence_excludes_configured_fallback(
    monkeypatch, provider: str, outcome: str
) -> None:
    monkeypatch.setenv("INCEPTION_API_KEY", "inception-test-key")
    monkeypatch.setenv("LOCAL_LLM_DEFAULT_MODEL", "configured-only-model")
    calls: list[str] = []

    class Response:
        def raise_for_status(self) -> None:
            if outcome == "auth":
                raise requests.HTTPError("401 unauthorized")

        def json(self):
            if outcome == "malformed":
                raise ValueError("invalid catalog JSON")
            if outcome == "entry":
                return {"data": [{"name": "missing-id"}]}
            if outcome == "shape":
                return {"unexpected": []}
            return {"data": [{"id": "endpoint-model"}]} if outcome == "catalog" else {"data": []}

    def get(url, **kwargs):
        calls.append(url)
        if outcome == "transport":
            raise requests.ConnectionError("endpoint offline")
        return Response()

    monkeypatch.setattr("server.http.common.ui_settings.requests.get", get)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.get(f"/ui/api/models?provider={provider}&discovery_only=1")
            assert response.status == 200
            item = (await response.json())["providers"][0]
            assert item["models"] == (["endpoint-model"] if outcome == "catalog" else [])
            assert item["catalog_source"] == (
                "endpoint" if outcome in {"catalog", "empty"} else "unverified"
            )
            assert bool(item["error"]) is (outcome not in {"catalog", "empty"})
            assert "inception-test-key" not in json.dumps(item)
            assert len(calls) == 1
        finally:
            await client.close()

    asyncio.run(run())


def test_discovery_inception_without_key_does_not_probe_or_return_documented_models(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("INCEPTION_API_KEY", raising=False)
    monkeypatch.setattr("server.http_api.API_KEYS_PATH", tmp_path / "keys.json")

    def unexpected_request(*args, **kwargs):
        raise AssertionError("missing credentials must not start a network probe")

    monkeypatch.setattr("server.http.common.ui_settings.requests.get", unexpected_request)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.get("/ui/api/models?provider=inception&discovery_only=1")
            item = (await response.json())["providers"][0]
            assert item["models"] == []
            assert item["catalog_source"] == "unverified"
            assert "INCEPTION_API_KEY" in item["error"]
        finally:
            await client.close()

    asyncio.run(run())


def test_non_strict_local_keeps_configured_candidate_without_discovery_claim(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_LLM_DEFAULT_MODEL", "configured-only-model")

    def offline(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("server.http.common.ui_settings.requests.get", offline)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.get("/ui/api/models?provider=local")
            item = (await response.json())["providers"][0]
            assert item["models"] == ["configured-only-model"]
            assert item["catalog_source"] == "unverified"
            assert item["error"] is None
        finally:
            await client.close()

    asyncio.run(run())


def test_existing_strict_inception_picker_keeps_documented_options(monkeypatch) -> None:
    def unexpected_request(*args, **kwargs):
        raise AssertionError("documented options must not be reported as endpoint discovery")

    monkeypatch.setattr("server.http.common.ui_settings.requests.get", unexpected_request)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.get("/ui/api/models?provider=inception&strict=1")
            item = (await response.json())["providers"][0]
            assert item["models"] == ["mercury-2"]
            assert item["catalog_source"] == "documented"
            assert item["error"] is None
        finally:
            await client.close()

    asyncio.run(run())
