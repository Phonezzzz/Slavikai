from __future__ import annotations

import os
from pathlib import Path

import pytest

from server.http.common.workspace_index import index_workspace_root


class DummyVectorIndex:
    def __init__(self, db_path: str = "memory/vectors.db", **_: object) -> None:  # noqa: ARG002
        self.upserted: list[tuple[str, str]] = []
        self.closed = False

    def ensure_runtime_ready(self) -> None:
        return

    def upsert_text(self, path: str, content: str, namespace: str = "default", meta=None) -> None:
        self.upserted.append((path, namespace))

    def close(self) -> None:
        self.closed = True


def _load_settings() -> object:
    class _Settings:
        provider = "local"
        local_model = "all-MiniLM-L6-v2"
        openai_model = "text-embedding-3-small"

    return _Settings()


def _index(
    root: Path,
    monkeypatch,
    *,
    index_class: type | None = None,
) -> DummyVectorIndex:
    dummy_class = index_class or DummyVectorIndex
    dummy = dummy_class()
    monkeypatch.setattr(
        "server.http.common.workspace_index.VectorIndex",
        lambda *args, **kwargs: dummy,
    )
    index_workspace_root(
        root=root,
        load_embeddings_settings=_load_settings,
        resolve_provider_api_key=lambda provider: None,
        index_enabled_env="INDEX_ENABLED",
        ignored_dirs={"venv", "__pycache__", ".git", "node_modules", "dist", "build"},
        allowed_extensions={".py", ".md", ".txt"},
        max_file_bytes=1_000_000,
    )
    return dummy


def test_index_workspace_root_indexes_allowed_files(tmp_path, monkeypatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("print('hi')", encoding="utf-8")
    (root / "doc.md").write_text("# doc", encoding="utf-8")
    (root / "skip.bin").write_bytes(b"\x00\x01")

    dummy = _index(root, monkeypatch)
    assert dummy is not None
    paths = {path for path, _ in dummy.upserted}
    assert any(str(path).endswith("a.py") for path in paths)
    assert any(str(path).endswith("doc.md") for path in paths)
    assert not any(str(path).endswith("skip.bin") for path in paths)


def test_index_workspace_root_skips_symlink_escape(tmp_path, monkeypatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "ok.py").write_text("print('ok')", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.py").write_text("print('leak')", encoding="utf-8")
    link = root / "escape.py"
    try:
        link.symlink_to(outside / "leak.py")
    except OSError:
        pytest.skip("Symlink недоступен в этом окружении.")

    dummy = _index(root, monkeypatch)
    paths = [str(path) for path, _ in dummy.upserted]
    assert not any(path.endswith("escape.py") for path in paths), paths
    assert any(path.endswith("ok.py") for path in paths)


def test_index_workspace_root_reports_disabled(monkeypatch) -> None:
    root = Path("unused")
    monkeypatch.setenv("INDEX_ENABLED_DISABLED_BY_ENV", "0")
    result = index_workspace_root(
        root=root,
        load_embeddings_settings=_load_settings,
        resolve_provider_api_key=lambda provider: None,
        index_enabled_env="INDEX_ENABLED_DISABLED_BY_ENV",
        ignored_dirs=set(),
        allowed_extensions={".py"},
        max_file_bytes=1_000_000,
    )
    assert result["ok"] is False
    assert "INDEX disabled" in str(result.get("message"))


def test_index_workspace_root_closes_vector_index(tmp_path, monkeypatch) -> None:
    """VectorIndex connections must be closed after indexing (no connection leak)."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("print('hi')", encoding="utf-8")

    dummy = _index(root, monkeypatch)
    assert dummy.closed is True
    assert os.environ.get("INDEX_ENABLED") is None
