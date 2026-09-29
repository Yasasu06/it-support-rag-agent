"""Compatibility contract for the tracked Chroma index and future rebuilds."""

import json
import math
from pathlib import Path


MANIFEST = json.loads((Path(__file__).with_name("index_manifest.json")).read_text())
COLLECTION_NAME = MANIFEST["collection_name"]
INDEX_EMBEDDING_MODEL = MANIFEST["embedding_model"]
INDEX_DIMENSIONS = MANIFEST["embedding_dimensions"]
INDEX_DISTANCE_METRIC = MANIFEST["distance_metric"]
COLLECTION_CONFIGURATION = {"hnsw": {"space": INDEX_DISTANCE_METRIC}}
COLLECTION_METADATA = {
    "embedding_model": INDEX_EMBEDDING_MODEL,
    "index_contract_version": MANIFEST["schema_version"],
}


def l2_relevance_score(raw_distance: float) -> float:
    """Preserve the LangChain conversion used with the tracked l2 collection."""
    return 1.0 - raw_distance / math.sqrt(2)


def validate_embedding_model(model: str) -> None:
    """An embedding-model change requires an intentional index rebuild."""
    if model != INDEX_EMBEDDING_MODEL:
        raise ValueError(
            f"Embedding model {model!r} is incompatible with the tracked index; "
            f"expected {INDEX_EMBEDDING_MODEL!r}. Rebuild intentionally before changing it."
        )


def validate_collection(collection, model: str) -> None:
    """Fail before retrieval when a stored collection disagrees with the index contract."""
    validate_embedding_model(model)
    if collection.name != COLLECTION_NAME:
        raise ValueError("Unexpected Chroma collection name")
    hnsw = collection.configuration.get("hnsw") or {}
    if hnsw.get("space") != INDEX_DISTANCE_METRIC:
        raise ValueError("Chroma distance metric does not match the tracked index")
    dimension = collection.get_model().dimension
    if dimension is not None and dimension != INDEX_DIMENSIONS:
        raise ValueError("Chroma embedding dimensions do not match the tracked index")
    metadata = collection.metadata or {}
    recorded_model = metadata.get("embedding_model")
    if recorded_model is not None and recorded_model != model:
        raise ValueError("Chroma collection embedding model does not match configuration")
    recorded_version = metadata.get("index_contract_version")
    if recorded_version is not None and recorded_version != MANIFEST["schema_version"]:
        raise ValueError("Chroma index contract version does not match configuration")


def inspect_existing_index(path: str, model: str) -> int:
    """Validate an existing index without creating or rebuilding a collection."""
    validate_embedding_model(model)
    if not (Path(path) / "chroma.sqlite3").is_file():
        raise RuntimeError("Chroma index is missing at CHROMA_DB_PATH")

    import chromadb
    from chromadb.errors import NotFoundError

    try:
        collection = chromadb.PersistentClient(path=path).get_collection(COLLECTION_NAME)
    except NotFoundError:
        raise RuntimeError("Chroma collection is missing at CHROMA_DB_PATH") from None
    except Exception:
        raise RuntimeError("Chroma index could not be opened") from None

    try:
        validate_collection(collection, model)
    except ValueError:
        raise
    except Exception:
        raise RuntimeError("Chroma index is corrupt or unreadable") from None
    try:
        count = collection.count()
    except Exception:
        raise RuntimeError("Chroma index could not be counted") from None
    if count == 0:
        raise RuntimeError("Chroma collection is empty")
    return count
