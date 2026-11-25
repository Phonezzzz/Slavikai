"""GET /ui/api/bridge/effects (E03-T08): the read API for the
audit journal. Authorization: a caller sees only their own
records; the principal_id filter must match the caller; gap
markers are owner-only and only over an unfiltered listing
(owner condition 7).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from core.bridge_effect_store import BridgeEffectStore
from server.http.common.request_identity import RequestIdentity
from server.http.handlers import effects

_OWNER = "owner"
_MEMBER = "member"
_BOOT = "b" * 32


@pytest.fixture()
def store(tmp_path: Path) -> BridgeEffectStore:
    return BridgeEffectStore(tmp_path / "effects.db")


def _effect(*, seq: int, boot_id: str = _BOOT) -> dict[str, object]:
    return {
        "seq": seq,
        "boot_id": boot_id,
        "kind": "fs.write",
        "at": "2026-11-25T00:00:00+00:00",
        "principal_id": _OWNER,
        "run_id": None,
        "root_id": "work",
        "path": "a.txt",
        "size": 1,
        "bytes_written": 1,
        "committed": True,
        "content_sha256": "0" * 64,
    }


def _seed(
    store: BridgeEffectStore,
    *,
    request_id: str,
    effect: dict[str, object] | None = None,
    outcome: str = "effect",
    principal_id: str = _OWNER,
    run_id: str | None = None,
    recorded_at: str | None = None,
) -> int:
    return int(
        store.record_call(
            capability="host.fs.write",
            request_id=request_id,
            principal_id=principal_id,
            run_id=run_id,
            params_json=json.dumps({"root_id": "work", "path": "a.txt"}, sort_keys=True),
            effect_json=json.dumps(effect, sort_keys=True) if effect is not None else None,
            daemon_boot_id=effect["boot_id"] if isinstance(effect, dict) else None,
            daemon_seq=effect["seq"] if isinstance(effect, dict) else None,
            outcome=outcome,
            error_code=None,
            unknown_reason=None,
            recorded_at=recorded_at,
        )["seq"]
    )


def _request(
    store: BridgeEffectStore,
    *,
    principal_id: str | None = _OWNER,
    role: str = "owner",
    query: dict[str, str] | None = None,
) -> web.Request:
    request = make_mocked_request(
        "GET",
        "/ui/api/bridge/effects" + (f"?{urlencode(query)}" if query else ""),
        app=web.Application(),
    )
    request.app["bridge_effect_store"] = store
    if principal_id is not None:
        request["principal_id"] = principal_id
        request["request_identity"] = RequestIdentity(
            principal_id=principal_id, role=role, auth_method="local"
        )
    return request


def _call(request: web.Request) -> dict[str, object]:
    """The repo has no pytest-asyncio: async handlers are driven
    with asyncio.run (the test_principal_runtime_isolation.py
    pattern)."""
    response = asyncio.run(effects.handle_bridge_effects(request))
    assert response.status == 200, response.status
    return json.loads(response.text)


def _status(request: web.Request) -> int:
    return asyncio.run(effects.handle_bridge_effects(request)).status


def test_requires_principal(store: BridgeEffectStore) -> None:
    assert _status(_request(store, principal_id=None)) == 401


def test_principal_filter_must_match_caller(store: BridgeEffectStore) -> None:
    assert _status(_request(store, query={"principal_id": "someone-else"})) == 403


def test_lists_only_own_records(store: BridgeEffectStore) -> None:
    _seed(store, request_id="r1", effect=_effect(seq=1))
    _seed(store, request_id="r2", effect=_effect(seq=2), principal_id=_MEMBER)
    body = _call(_request(store))
    assert [row["seq"] for row in body["records"]] == [1]
    body = _call(_request(store, principal_id=_MEMBER, role="member"))
    assert [row["seq"] for row in body["records"]] == [2]


def test_filters_and_pagination(store: BridgeEffectStore) -> None:
    _seed(store, request_id="r1", effect=_effect(seq=1), run_id="run-A")
    _seed(store, request_id="r2", effect=_effect(seq=2), run_id="run-B")
    _seed(store, request_id="r3", outcome="no_effect", run_id="run-A")
    body = _call(_request(store, query={"run_id": "run-A"}))
    assert [row["seq"] for row in body["records"]] == [1, 3]
    body = _call(_request(store, query={"outcome": "no_effect"}))
    assert [row["seq"] for row in body["records"]] == [3]
    body = _call(_request(store, query={"limit": "1"}))
    assert [row["seq"] for row in body["records"]] == [1]
    assert body["next_cursor"] == 1
    body = _call(_request(store, query={"cursor": "1", "limit": "1"}))
    assert [row["seq"] for row in body["records"]] == [2]
    assert body["next_cursor"] == 2


def test_invalid_params_are_400(store: BridgeEffectStore) -> None:
    for query in (
        {"limit": "0"},
        {"limit": "501"},
        {"limit": "not-a-number"},
        {"cursor": "-1"},
        {"cursor": "not-a-number"},
        {"outcome": "bogus"},
    ):
        assert _status(_request(store, query=query)) == 400, query


def test_gaps_owner_only_and_unfiltered(store: BridgeEffectStore) -> None:
    old = (datetime.now(UTC) - timedelta(seconds=300)).isoformat()
    _seed(store, request_id="r1", effect=_effect(seq=1), recorded_at=old)
    _seed(store, request_id="r3", effect=_effect(seq=3), recorded_at=old)
    # Owner, no filters: gaps present.
    body = _call(_request(store, query={"include_gaps": "true"}))
    assert body["gaps"] == [
        {"daemon_boot_id": _BOOT, "missing_from": 2, "missing_to": 2, "suspected": True}
    ]
    # Member: no gaps even when requested.
    body = _call(
        _request(store, principal_id=_MEMBER, role="member", query={"include_gaps": "true"})
    )
    assert "gaps" not in body
    # Owner with any filter: no gaps (the listing is not the whole
    # sequence, so a gap reading would be wrong).
    for query in (
        {"include_gaps": "true", "run_id": "run-A"},
        {"include_gaps": "true", "outcome": "effect"},
        {"include_gaps": "true", "principal_id": _OWNER},
    ):
        body = _call(_request(store, query=query))
        assert "gaps" not in body, query


def test_observations_are_included_in_output(store: BridgeEffectStore) -> None:
    seq = _seed(store, request_id="r1", effect=_effect(seq=1))
    observation = {
        "subject": "a.txt",
        "observed_at": "2026-11-25T00:01:00+00:00",
        "state": "exists",
    }
    observation_json = json.dumps(observation, sort_keys=True)
    store.attach_observation(
        effect_seq=seq,
        source="poll-loop",
        source_ref=None,
        observation_json=observation_json,
        observation_sha256=hashlib.sha256(observation_json.encode()).hexdigest(),
    )
    body = _call(_request(store))
    record = body["records"][0]
    assert record["observations"][0]["observation"] == observation
    assert record["observations"][0]["source"] == "poll-loop"


def test_record_without_observation_has_empty_observations(store: BridgeEffectStore) -> None:
    _seed(store, request_id="r1", effect=_effect(seq=1))
    body = _call(_request(store))
    assert body["records"][0]["observations"] == []
