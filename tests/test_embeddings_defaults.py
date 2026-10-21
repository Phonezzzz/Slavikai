import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from config.ui_embeddings_settings import (
    UIEmbeddingsSettings,
    load_ui_embeddings_settings,
    save_ui_embeddings_settings,
)
from memory.vector_index import VectorIndex


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"memory": None},
        {"memory": {}},
        {"memory": {"embeddings": {}}},
        {"memory": {"embeddings": {"provider": "invalid"}}},
        {"memory": {"embeddings": {"provider": None}}},
    ],
)
def test_implicit_embeddings_shapes_use_local(tmp_path, payload):
    path = tmp_path / "settings.json"
    if payload is not None:
        path.write_text(json.dumps(payload))
    assert UIEmbeddingsSettings().provider == "local"
    assert load_ui_embeddings_settings(path).provider == "local"


@pytest.mark.parametrize("provider", ["local", "openai"])
def test_explicit_embeddings_choice_roundtrips(tmp_path, provider):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "memory": {
                    "embeddings": {
                        "provider": provider,
                        "local_model": "local-choice",
                        "openai_model": "remote-choice",
                    }
                }
            }
        )
    )
    selected = load_ui_embeddings_settings(path)
    assert selected.provider == provider
    save_ui_embeddings_settings(selected, path)
    assert load_ui_embeddings_settings(path) == selected


def test_runtime_model_loading_is_cache_only(tmp_path, monkeypatch):
    calls = []

    def constructor(model, **kwargs):
        calls.append((model, kwargs))
        if kwargs.get("local_files_only") is not True:
            raise AssertionError("Runtime must not download model files")
        raise OSError("Model files are not cached")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor)
    )
    monkeypatch.setattr(VectorIndex, "_model_cache", {})
    index = VectorIndex(str(tmp_path / "vectors.db"))
    with pytest.raises(RuntimeError, match="Model files are not cached"):
        index.search("query")
    assert calls == [("all-MiniLM-L6-v2", {"local_files_only": True})]


def test_local_runtime_recovers_after_files_become_available_without_deleting_vectors(
    tmp_path, monkeypatch
):
    available = False

    class Model:
        def encode(self, texts):
            return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def load(_self, _name):
        if not available:
            raise OSError("Model files are not cached")
        return Model()

    monkeypatch.setattr(VectorIndex, "load_local_model", load)
    index = VectorIndex(str(tmp_path / "vectors.db"))
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes()
    with index.conn:
        index.conn.execute(
            "INSERT INTO vectors (namespace,path,content,embedding,meta) VALUES (?,?,?,?,?)",
            ("default", "saved", "content", vector, "{}"),
        )
    before = index.conn.execute("SELECT * FROM vectors").fetchall()
    with pytest.raises(RuntimeError):
        index.search("query")
    available = True
    assert index.search("query")[0].path == "saved"
    assert index.conn.execute("SELECT * FROM vectors").fetchall() == before


def test_chat_works_without_local_embeddings_and_never_falls_back_to_remote(tmp_path, monkeypatch):
    from core.agent import Agent
    from llm.brain_base import Brain
    from llm.types import LLMResult
    from shared.models import LLMMessage

    class ChatBrain(Brain):
        def generate(self, messages, config=None, tools=None):
            return LLMResult(text="offline-chat-ok")

    network_calls = []

    def reject_network(*args, **kwargs):
        network_calls.append((args, kwargs))
        raise AssertionError("No network fallback allowed")

    monkeypatch.setattr("requests.sessions.Session.request", reject_network)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key-must-not-be-used")
    settings = load_ui_embeddings_settings(tmp_path / "absent.json")
    agent = Agent(
        brain=ChatBrain(),
        embeddings_provider=settings.provider,
        embeddings_openai_api_key="synthetic-key-must-not-be-used",
        memory_db_path=str(tmp_path / "memory.db"),
        vectors_db_path=str(tmp_path / "vectors.db"),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    agent.set_runtime_state(
        mode="plan", active_plan=None, active_task=None, enforce_plan_guard=False
    )
    try:
        response = agent.respond([LLMMessage(role="user", content="Hello")])
        assert response.failure is None
        assert response.text.split("MWV_REPORT_JSON=", 1)[0].strip() == "offline-chat-ok"
        assert agent.vectors.provider == "local"
        assert not agent.vectors.is_runtime_ready()
        assert network_calls == []
    finally:
        agent.close()


@pytest.mark.parametrize("failure", ["embedding", "metadata", "insert"])
def test_failed_upsert_preserves_existing_vector(tmp_path, monkeypatch, failure):
    import sqlite3

    class Model:
        def encode(self, texts):
            return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    monkeypatch.setattr(VectorIndex, "load_local_model", lambda _self, _name: Model())
    index = VectorIndex(str(tmp_path / "vectors.db"))
    index.index_text("atom", "old", namespace="atoms")
    before = index.conn.execute("SELECT * FROM vectors").fetchall()
    meta = {}
    if failure == "embedding":

        def broken(*args, **kwargs):
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(index, "_encode_texts", broken)
    elif failure == "metadata":
        meta = {"bad": object()}
    else:
        index.conn.execute(
            "CREATE TRIGGER fail_new BEFORE INSERT ON vectors WHEN NEW.content = 'new' "
            "BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
        index.conn.commit()
    with pytest.raises((RuntimeError, TypeError, sqlite3.IntegrityError)):
        index.upsert_text("atom", "new", namespace="atoms", meta=meta)
    assert index.conn.execute("SELECT * FROM vectors").fetchall() == before


@pytest.mark.parametrize("files_cached", [False, True])
def test_explicit_download_activates_existing_and_future_ask_indexes(
    tmp_path, monkeypatch, files_cached
):
    import asyncio

    from server.embedding_download import EmbeddingDownloadManager

    loads = []

    class Model:
        def __init__(self, name, **kwargs):
            loads.append((name, kwargs))

        def encode(self, texts):
            return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    monkeypatch.setattr(VectorIndex, "_model_cache", {})
    index = VectorIndex(str(tmp_path / "vectors.db"), local_model="activation-test")
    with index.conn:
        index.conn.execute(
            "INSERT INTO vectors (namespace,path,content,embedding,meta) VALUES (?,?,?,?,?)",
            (
                "default",
                "saved",
                "content",
                np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
                "{}",
            ),
        )
    assert index.search("query", allow_runtime_init=False) == []
    assert loads == []

    async def run():
        manager = EmbeddingDownloadManager(
            package_available=lambda: True, model_cached=lambda _: files_cached
        )
        await manager.status("activation-test")
        assert loads == []
        await manager.start("activation-test")
        for _ in range(100):
            snapshot = await manager.status("activation-test")
            if snapshot.state != "downloading":
                break
            await asyncio.sleep(0.01)
        assert snapshot.state == "ready"
        await manager.shutdown()

    asyncio.run(run())
    assert len(loads) == 1
    assert index.search("query", allow_runtime_init=False)[0].path == "saved"
    future = VectorIndex(index.db_path, local_model="activation-test")
    assert future.search("query", allow_runtime_init=False)[0].path == "saved"
    assert len(loads) == 1

    from core.agent import Agent
    from llm.brain_base import Brain
    from llm.types import LLMResult
    from shared.models import LLMMessage

    seen = []

    class ChatBrain(Brain):
        def generate(self, messages, config=None, tools=None):
            seen.extend(messages)
            return LLMResult(text="ok")

    index.index_text("project.py", "VERIFIED_RETRIEVAL", namespace="code")

    def forbid_load(*args, **kwargs):
        raise AssertionError("Ask must not initialize a model")

    monkeypatch.setattr(VectorIndex, "load_local_model", forbid_load)
    agent = Agent(
        brain=ChatBrain(),
        embeddings_local_model="activation-test",
        memory_db_path=str(tmp_path / "memory.db"),
        vectors_db_path=index.db_path,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    agent.runtime_mode = "ask"
    try:
        assert agent.respond([LLMMessage(role="user", content="Hello")]).failure is None
        assert any("VERIFIED_RETRIEVAL" in m.content for m in seen)
        assert len(loads) == 1
    finally:
        agent.close()
