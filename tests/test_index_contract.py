"""Fast checks for the tracked index contract; no embedding API calls."""

import json
import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import index_contract


ROOT = Path(__file__).resolve().parents[1]


class FakeCollection:
    name = index_contract.COLLECTION_NAME

    def __init__(self, space="l2", dimension=1536, model=None, version=None):
        self.configuration = {"hnsw": {"space": space}}
        self.metadata = {}
        if model:
            self.metadata["embedding_model"] = model
        if version is not None:
            self.metadata["index_contract_version"] = version
        self._dimension = dimension

    def get_model(self):
        return SimpleNamespace(dimension=self._dimension)


def test_tracked_snapshot_has_declared_metric_dimensions_and_count():
    path = ROOT / "chroma_db" / "chroma.sqlite3"
    with sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True) as db:
        name, count, dimension, schema_text = db.execute("""
            SELECT c.name, COUNT(e.id), c.dimension, c.schema_str
            FROM collections c
            JOIN segments s ON s.collection = c.id
            JOIN embeddings e ON e.segment_id = s.id
            WHERE c.name = ?
            GROUP BY c.id
        """, (index_contract.COLLECTION_NAME,)).fetchone()

    manifest = index_contract.MANIFEST
    assert name == manifest["collection_name"]
    assert count == manifest["tracked_record_count"]
    assert dimension == manifest["embedding_dimensions"]
    schema = json.loads(schema_text)
    space = schema["keys"]["#embedding"]["float_list"]["vector_index"]["config"]["space"]
    assert space == manifest["distance_metric"] == "l2"


def test_relevance_conversion_matches_existing_langchain_l2_behavior():
    from langchain_core.vectorstores import VectorStore

    for distance in (0.0, 0.5, 1.0, math.sqrt(2)):
        assert index_contract.l2_relevance_score(distance) == pytest.approx(
            VectorStore._euclidean_relevance_score_fn(distance)
        )
    assert index_contract.COLLECTION_CONFIGURATION == {"hnsw": {"space": "l2"}}


def test_supported_embedding_model_and_existing_collection_are_accepted():
    model = index_contract.INDEX_EMBEDDING_MODEL
    index_contract.validate_embedding_model(model)
    index_contract.validate_collection(FakeCollection(), model)
    index_contract.validate_collection(FakeCollection(model=model), model)


def test_new_collection_records_explicit_metric_and_embedding_model(tmp_path):
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    class LocalEmbeddings:
        def embed_documents(self, texts):
            return [[1.0] + [0.0] * 1535 for _ in texts]

        def embed_query(self, text):
            return [1.0] + [0.0] * 1535

    store = Chroma.from_documents(
        documents=[Document(page_content="Sample VPN ticket")],
        embedding=LocalEmbeddings(),
        collection_name=index_contract.COLLECTION_NAME,
        persist_directory=str(tmp_path),
        collection_configuration=index_contract.COLLECTION_CONFIGURATION,
        collection_metadata=index_contract.COLLECTION_METADATA,
        relevance_score_fn=index_contract.l2_relevance_score,
    )
    index_contract.validate_collection(
        store._collection, index_contract.INDEX_EMBEDDING_MODEL
    )
    assert store.similarity_search_with_relevance_scores("VPN", k=1)[0][1] == 1.0


@pytest.mark.parametrize("collection, model", [
    (FakeCollection(space="cosine"), index_contract.INDEX_EMBEDDING_MODEL),
    (FakeCollection(dimension=3072), index_contract.INDEX_EMBEDDING_MODEL),
    (FakeCollection(model="other-model"), index_contract.INDEX_EMBEDDING_MODEL),
    (FakeCollection(version=999), index_contract.INDEX_EMBEDDING_MODEL),
    (FakeCollection(), "other-model"),
])
def test_incompatible_collection_or_model_fails_closed(collection, model):
    with pytest.raises(ValueError):
        index_contract.validate_collection(collection, model)


def test_batch_ingestion_rejects_wrong_model_before_network_or_deletion(monkeypatch):
    import ingest_all

    monkeypatch.setattr(ingest_all.config, "EMBEDDING_MODEL", "other-model")
    monkeypatch.setattr(
        ingest_all, "run_kaggle_connector",
        lambda: pytest.fail("ingestion must stop before connectors run"),
    )
    with pytest.raises(ValueError, match="incompatible"):
        ingest_all.run_multi_source_ingest()


def test_batch_ingestion_rejects_incompatible_existing_index_before_delete(monkeypatch):
    import chromadb
    import ingest_all

    class ExistingClient:
        def get_collection(self, _name):
            return FakeCollection(space="cosine")

        def delete_collection(self, _name):
            pytest.fail("incompatible index must not be deleted")

    monkeypatch.setattr(chromadb, "PersistentClient", lambda **_kwargs: ExistingClient())
    monkeypatch.setattr(ingest_all, "run_kaggle_connector", lambda: [])
    monkeypatch.setattr(ingest_all, "run_github_connector", lambda: [])
    monkeypatch.setattr(ingest_all, "run_huggingface_connector", lambda: [])
    monkeypatch.setattr(ingest_all, "servicenow_configured", lambda: False)
    monkeypatch.setattr(ingest_all, "normalize_all_sources", lambda *_args: [])

    with pytest.raises(ValueError, match="distance metric"):
        ingest_all.run_multi_source_ingest()
