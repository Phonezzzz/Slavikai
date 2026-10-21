from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from server.preferences_store import Preferences, PreferencesConflictError, PreferencesStore


def test_principal_defaults_and_restart(tmp_path: Path) -> None:
    store = PreferencesStore(tmp_path / "preferences.db")
    assert (tmp_path / "preferences.db").stat().st_mode & 0o777 == 0o600
    owner = store.get("owner")
    member = store.get("member")
    assert owner.etag != member.etag
    updated = store.update(
        "owner", expected_etag=owner.etag, payload={"personalization": {"system_prompt": "private"}}
    )
    assert updated.value.system_prompt == "private"
    restored = PreferencesStore(tmp_path / "preferences.db")
    assert restored.get("owner") == updated
    assert restored.get("member").value == Preferences()
    with pytest.raises(PreferencesConflictError):
        store.update(
            "member", expected_etag=updated.etag, payload={"appearance": {"theme": "oled"}}
        )
    assert store.get("member") == member


def test_atomic_validation_reset_and_stale_cas(tmp_path: Path) -> None:
    store = PreferencesStore(tmp_path / "preferences.db")
    first = store.get("owner")
    with pytest.raises(ValueError):
        store.update(
            "owner",
            expected_etag=first.etag,
            payload={
                "appearance": {"theme": "oled"},
                "composer": {"long_paste_threshold_chars": True},
            },
        )
    assert store.get("owner") == first
    updated = store.update(
        "owner", expected_etag=first.etag, payload={"appearance": {"theme": "oled"}}
    )
    reset = store.update("owner", expected_etag=updated.etag, payload={"reset_preferences": True})
    assert reset.value == Preferences()
    assert reset.revision == 2
    with pytest.raises(PreferencesConflictError):
        store.update("owner", expected_etag=first.etag, payload={"appearance": {"theme": "oled"}})


def test_concurrent_writers_preserve_principal_isolation(tmp_path: Path) -> None:
    store = PreferencesStore(tmp_path / "preferences.db")
    etag = store.get("same").etag

    def update(theme):
        try:
            store.update("same", expected_etag=etag, payload={"appearance": {"theme": theme}})
            return "ok"
        except PreferencesConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(update, ["oled", "default"])) == ["conflict", "ok"]
    assert store.get("same").revision == 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                store.update,
                p,
                expected_etag=store.get(p).etag,
                payload={"personalization": {"system_prompt": p}},
            )
            for p in ["a", "b"]
        ]
        assert [f.result().value.system_prompt for f in futures] == ["a", "b"]
