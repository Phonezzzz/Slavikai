from memory.canonical_atom_store import CanonicalAtomStore
from shared.canonical_atom_models import ClaimType


def test_deprecated_pin_is_not_retrieved(tmp_path):
    store = CanonicalAtomStore(str(tmp_path / "atoms.db"))
    store.create_atom(
        atom_id="fixture",
        stable_key="fact:fixture",
        claim_type=ClaimType.FACT,
        value_json={"value": "synthetic"},
        confidence=0.9,
        summary_text="synthetic",
    )
    store.set_pinned("fact:fixture", True)
    store.delete_atom("fact:fixture")
    assert store.list_pinned() == []


def test_delete_receipt_and_late_vector_cannot_delete_recreated_atom(tmp_path):
    from memory.atom_embedding_index import AtomEmbeddingIndex
    from memory.vector_index import VectorIndex

    store = CanonicalAtomStore(str(tmp_path / "atoms.db"))
    vectors = VectorIndex(str(tmp_path / "vectors.db"))
    AtomEmbeddingIndex(vectors, store)
    atom = store.create_atom(
        atom_id="fixture",
        stable_key="fact:fixture",
        claim_type=ClaimType.FACT,
        value_json={"value": "synthetic"},
        confidence=0.9,
        summary_text="synthetic",
    )
    assert (
        store.delete_confirmed(
            atom.stable_key,
            atom.revision,
            "operation-1",
            expires_at=__import__("time").time() + 3600,
        )
        == "deleted"
    )
    assert store.get_by_stable_key(atom.stable_key) is None
    recreated = store.create_atom(
        atom_id="new",
        stable_key=atom.stable_key,
        claim_type=ClaimType.FACT,
        value_json=atom.value_json,
        confidence=0.9,
        summary_text="synthetic",
    )
    assert recreated.revision != atom.revision
    assert (
        store.delete_confirmed(
            atom.stable_key,
            atom.revision,
            "operation-1",
            expires_at=__import__("time").time() + 3600,
        )
        == "deleted"
    )
    assert (
        store.delete_confirmed(
            atom.stable_key,
            atom.revision,
            "operation-2",
            expires_at=__import__("time").time() + 3600,
        )
        == "conflict"
    )
    assert not store.publish_vector(
        atom, b"old", content="old", namespace="atoms", max_records=5000, max_total_records=20000
    )
    assert store.get_by_stable_key(atom.stable_key).revision == recreated.revision


def _pair(tmp_path):
    from memory.atom_embedding_index import AtomEmbeddingIndex
    from memory.vector_index import VectorIndex

    store = CanonicalAtomStore(str(tmp_path / "atoms.db"))
    vectors = VectorIndex(str(tmp_path / "vectors.db"))
    index = AtomEmbeddingIndex(vectors, store)
    atom = store.create_atom(
        atom_id="fixture",
        stable_key="fact:fixture",
        claim_type=ClaimType.FACT,
        value_json={"value": "synthetic"},
        confidence=0.9,
        summary_text="synthetic",
    )
    assert store.publish_vector(
        atom,
        b"old",
        content="synthetic",
        namespace="atoms",
        max_records=5000,
        max_total_records=20000,
    )
    return store, vectors, index, atom


def test_receipt_failure_rolls_back_canonical_and_vector_then_retry_after_reopen(tmp_path):
    import sqlite3
    import time

    import pytest

    store, vectors, _, atom = _pair(tmp_path)
    store.conn.execute(
        "CREATE TRIGGER fail_receipt BEFORE INSERT ON atom_delete_receipt "
        "BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END"
    )
    store.conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match="synthetic failure"):
        store.delete_confirmed(atom.stable_key, atom.revision, "op", expires_at=time.time() + 3600)
    assert store.get_by_stable_key(atom.stable_key).revision == atom.revision
    assert vectors.conn.execute("SELECT count(*) FROM vectors").fetchone()[0] == 1
    store.conn.execute("DROP TRIGGER fail_receipt")
    store.conn.commit()
    store.close()
    from memory.atom_embedding_index import AtomEmbeddingIndex

    reopened = CanonicalAtomStore(str(tmp_path / "atoms.db"))
    AtomEmbeddingIndex(vectors, reopened)
    assert (
        reopened.delete_confirmed(
            atom.stable_key, atom.revision, "op", expires_at=time.time() + 3600
        )
        == "deleted"
    )
    assert reopened.get_by_stable_key(atom.stable_key) is None
    assert vectors.conn.execute("SELECT count(*) FROM vectors").fetchone()[0] == 0
    reopened.close()
    again = CanonicalAtomStore(str(tmp_path / "atoms.db"))
    AtomEmbeddingIndex(vectors, again)
    assert (
        again.delete_confirmed(atom.stable_key, atom.revision, "op", expires_at=time.time() + 3600)
        == "deleted"
    )


def test_late_encoder_after_delete_does_not_resurrect_vector(tmp_path, monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    store, vectors, index, atom = _pair(tmp_path)
    started, finish = threading.Event(), threading.Event()

    def encode(_):
        started.set()
        assert finish.wait(5)
        return b"late"

    monkeypatch.setattr(vectors, "encode_text", encode)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(index.sync_atom, atom)
        assert started.wait(5)
        assert (
            store.delete_confirmed(
                atom.stable_key, atom.revision, "op", expires_at=time.time() + 3600
            )
            == "deleted"
        )
        finish.set()
        assert future.result(timeout=5) is False
    assert vectors.conn.execute("SELECT count(*) FROM vectors").fetchone()[0] == 0


def test_two_session_connections_reject_stale_upsert_and_preserve_principal_isolation(tmp_path):
    import time
    from dataclasses import replace

    import pytest

    from memory.atom_embedding_index import AtomEmbeddingIndex

    store, vectors, _, atom = _pair(tmp_path)
    second = CanonicalAtomStore(str(tmp_path / "atoms.db"))
    AtomEmbeddingIndex(vectors, second)
    changed = second.upsert(replace(atom, summary_text="changed"))
    assert changed.revision != atom.revision
    with pytest.raises(ValueError, match="revision_conflict"):
        store.upsert(replace(atom, summary_text="stale"))
    assert (
        store.delete_confirmed(atom.stable_key, atom.revision, "op", expires_at=time.time() + 3600)
        == "conflict"
    )
    other = CanonicalAtomStore(str(tmp_path / "other-atoms.db"))
    from memory.vector_index import VectorIndex

    other_vectors = VectorIndex(str(tmp_path / "other-vectors.db"))
    AtomEmbeddingIndex(other_vectors, other)
    assert (
        other.delete_confirmed(
            atom.stable_key, changed.revision, "op", expires_at=time.time() + 3600
        )
        == "not_found"
    )
    assert store.get_by_stable_key(atom.stable_key).summary_text == "changed"


def test_wal_fails_before_mutation_without_changing_journal_mode(tmp_path):
    import time

    import pytest

    store, vectors, _, atom = _pair(tmp_path)
    vectors.conn.execute("PRAGMA journal_mode=WAL")
    with pytest.raises(ValueError, match="atomicity_unavailable"):
        store.delete_confirmed(atom.stable_key, atom.revision, "op", expires_at=time.time() + 3600)
    assert store.get_by_stable_key(atom.stable_key).revision == atom.revision
    assert vectors.conn.execute("SELECT count(*) FROM vectors").fetchone()[0] == 1
    assert vectors.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_concurrent_claim_batches_do_not_lose_support(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from memory.atom_embedding_index import AtomEmbeddingIndex
    from memory.canonical_aggregator import CanonicalAggregator
    from shared.canonical_atom_models import Claim, utc_now_iso

    store, vectors, _, atom = _pair(tmp_path)
    second = CanonicalAtomStore(str(tmp_path / "atoms.db"))
    AtomEmbeddingIndex(vectors, second)
    barrier = threading.Barrier(2)
    claim = Claim(
        claim_type=ClaimType.FACT,
        stable_key=atom.stable_key,
        value_json=atom.value_json,
        confidence=0.9,
        summary_text="synthetic",
        is_explicit=True,
        source_kind="chat.explicit_remember",
        source_id="fixture",
        created_at=utc_now_iso(),
    )

    def save(storage):
        barrier.wait(timeout=5)
        return CanonicalAggregator(storage).upsert_claims([claim])[0]

    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(save, s) for s in (store, second)]
        revisions = {f.result(timeout=5).revision for f in futures}
    assert len(revisions) == 2
    assert store.get_by_stable_key(atom.stable_key).support_count == 3


def test_canonical_failure_preserves_previous_vector_and_revision(tmp_path):
    import sqlite3
    from dataclasses import replace

    import pytest

    store, vectors, _, atom = _pair(tmp_path)
    store.conn.execute(
        "CREATE TRIGGER fail_atom BEFORE UPDATE ON canonical_atom "
        "BEGIN SELECT RAISE(ABORT, 'synthetic'); END"
    )
    store.conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match="synthetic"):
        store.upsert(replace(atom, summary_text="new"))
    assert store.get_by_stable_key(atom.stable_key).revision == atom.revision
    assert vectors.conn.execute("SELECT content FROM vectors").fetchone()[0] == "synthetic"


def test_receipt_cannot_be_rebound_to_other_item_and_expired_confirmation_has_no_effect(tmp_path):
    import time

    store, vectors, _, atom = _pair(tmp_path)
    assert (
        store.delete_confirmed(
            atom.stable_key, atom.revision, "expired", expires_at=time.time() - 1
        )
        == "expired"
    )
    assert store.get_by_stable_key(atom.stable_key) is not None
    assert (
        store.delete_confirmed(atom.stable_key, atom.revision, "op", expires_at=time.time() + 3600)
        == "deleted"
    )
    other = store.create_atom(
        atom_id="other",
        stable_key="fact:other",
        claim_type=ClaimType.FACT,
        value_json={"value": "other"},
        confidence=0.9,
        summary_text="other",
    )
    assert (
        store.delete_confirmed(
            other.stable_key, other.revision, "op", expires_at=time.time() + 3600
        )
        == "conflict"
    )
    assert store.get_by_stable_key(other.stable_key) is not None


def test_failed_encoding_keeps_previous_vector_until_revision_checked_publication(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    import pytest

    store, vectors, index, atom = _pair(tmp_path)
    changed = store.upsert(replace(atom, summary_text="new"))

    def fail(_):
        raise RuntimeError("synthetic encoder failure")

    monkeypatch.setattr(vectors, "encode_text", fail)
    with pytest.raises(RuntimeError, match="encoder failure"):
        index.sync_atom(changed)
    assert vectors.conn.execute("SELECT content FROM vectors").fetchone()[0] == "synthetic"
    assert store.get_by_stable_key(atom.stable_key).summary_text == "new"


def test_process_exit_before_commit_recovers_both_files(tmp_path):
    import subprocess
    import sys

    store, vectors, _, atom = _pair(tmp_path)
    child = """
import os,sys,time
from memory.canonical_atom_store import CanonicalAtomStore
from memory.vector_index import VectorIndex
from memory.atom_embedding_index import AtomEmbeddingIndex
s=CanonicalAtomStore(sys.argv[1]);v=VectorIndex(sys.argv[2]);AtomEmbeddingIndex(v,s)
execute=s.conn.execute
def crash(sql, parameters=()):
    if sql.startswith("INSERT INTO atom_delete_receipt"):
        os._exit(7)
    return execute(sql,parameters)
s.conn.execute=crash
s.delete_confirmed("fact:fixture",sys.argv[3],"crash-operation",expires_at=time.time()+3600)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            child,
            str(tmp_path / "atoms.db"),
            str(tmp_path / "vectors.db"),
            atom.revision,
        ],
        timeout=10,
        check=False,
    )
    assert result.returncode == 7
    assert store.get_by_stable_key(atom.stable_key).revision == atom.revision
    assert vectors.conn.execute("SELECT count(*) FROM vectors").fetchone()[0] == 1
