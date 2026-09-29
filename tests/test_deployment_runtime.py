"""Focused checks for the configured index and Railway preflight."""

from types import SimpleNamespace

import chromadb
import pytest

from config import PROJECT_DIR, config, resolve_chroma_path
from index_contract import INDEX_EMBEDDING_MODEL, inspect_existing_index
import startup_check


class FakeCollection:
    name = "it_support_tickets"
    configuration = {"hnsw": {"space": "l2"}}
    metadata = {}

    def __init__(self, count=750, dimension=1536, space="l2"):
        self._count = count
        self._dimension = dimension
        self.configuration = {"hnsw": {"space": space}}

    def get_model(self):
        return SimpleNamespace(dimension=self._dimension)

    def count(self):
        return self._count


def test_chroma_path_is_rooted_at_project_even_from_another_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert resolve_chroma_path("chroma_db") == str(PROJECT_DIR / "chroma_db")
    assert resolve_chroma_path("indexes/current") == str(PROJECT_DIR / "indexes/current")
    assert resolve_chroma_path(str(tmp_path / "volume")) == str(tmp_path / "volume")
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "configured-volume"))
    assert resolve_chroma_path() == str(tmp_path / "configured-volume")
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_chroma_path(" ")


def test_retrieval_and_ingestion_share_the_configured_path():
    import agent_pipeline
    import ingest
    import ingest_all
    import rag

    assert {
        agent_pipeline.CHROMA_DIR, ingest.CHROMA_DIR,
        ingest_all.CHROMA_DIR, rag.CHROMA_DIR,
    } == {config.CHROMA_DB_PATH}


def test_missing_index_fails_without_creating_a_collection(monkeypatch, tmp_path):
    monkeypatch.setattr(
        chromadb, "PersistentClient",
        lambda **_kwargs: pytest.fail("missing index must not create a client"),
    )
    with pytest.raises(RuntimeError, match="index is missing"):
        inspect_existing_index(str(tmp_path / "missing"), INDEX_EMBEDDING_MODEL)
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("collection, error", [
    (FakeCollection(), None),
    (FakeCollection(count=0), "empty"),
    (FakeCollection(space="cosine"), "distance metric"),
    (FakeCollection(dimension=3072), "dimensions"),
])
def test_preflight_checks_existing_collection_without_embedding_calls(
    monkeypatch, tmp_path, collection, error,
):
    db_dir = tmp_path / "index"
    db_dir.mkdir()
    (db_dir / "chroma.sqlite3").touch()
    observed_paths = []

    class Client:
        def get_collection(self, name):
            assert name == "it_support_tickets"
            return collection

    def get_client(path):
        observed_paths.append(path)
        return Client()

    monkeypatch.setattr(chromadb, "PersistentClient", get_client)
    if error:
        with pytest.raises((RuntimeError, ValueError), match=error):
            inspect_existing_index(str(db_dir), INDEX_EMBEDDING_MODEL)
    else:
        assert inspect_existing_index(str(db_dir), INDEX_EMBEDDING_MODEL) == 750
    assert observed_paths == [str(db_dir)]


def test_missing_or_unopenable_collection_has_safe_startup_error(monkeypatch, tmp_path):
    db_dir = tmp_path / "index"
    db_dir.mkdir()
    (db_dir / "chroma.sqlite3").touch()

    class MissingClient:
        def get_collection(self, _name):
            raise chromadb.errors.NotFoundError("private diagnostic")

    monkeypatch.setattr(chromadb, "PersistentClient", lambda **_kwargs: MissingClient())
    with pytest.raises(RuntimeError, match="collection is missing") as missing:
        inspect_existing_index(str(db_dir), INDEX_EMBEDDING_MODEL)
    assert "private diagnostic" not in str(missing.value)

    def corrupt_client(**_kwargs):
        raise RuntimeError("private diagnostic")

    monkeypatch.setattr(chromadb, "PersistentClient", corrupt_client)
    with pytest.raises(RuntimeError, match="could not be opened") as corrupt:
        inspect_existing_index(str(db_dir), INDEX_EMBEDDING_MODEL)
    assert "private diagnostic" not in str(corrupt.value)


def test_startup_requires_model_key_after_validating_index(monkeypatch):
    observed = []
    monkeypatch.setattr(startup_check, "check_pii_model", lambda: None)
    monkeypatch.setattr(
        startup_check, "inspect_existing_index",
        lambda path, model: observed.append((path, model)) or 750,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        startup_check.check_startup()
    assert observed == [(config.CHROMA_DB_PATH, config.EMBEDDING_MODEL)]

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-local-preflight-only")
    assert startup_check.check_startup() == 750


def test_missing_pii_model_fails_startup_without_diagnostic_leak(monkeypatch):
    import spacy

    monkeypatch.setattr(spacy, "load", lambda *_args: (_ for _ in ()).throw(
        RuntimeError("private diagnostic")
    ))
    with pytest.raises(RuntimeError, match="spaCy PII model is unavailable") as error:
        startup_check.check_pii_model()
    assert "private diagnostic" not in str(error.value)


def test_optional_feedback_storage_failure_is_reported_without_crashing(monkeypatch, tmp_path):
    import app

    monkeypatch.setattr(app, "FEEDBACK_FILE", str(tmp_path / "missing" / "feedback.jsonl"))
    assert app.log_feedback("positive") is False


def test_unreadable_audit_history_does_not_break_status_summary(monkeypatch, tmp_path):
    import security

    path = tmp_path / "invalid-audit.jsonl"
    path.write_text("not json\n")
    monkeypatch.setattr(security, "LOG_FILE", str(path))
    assert security.get_audit_summary()["total_queries"] == 0
