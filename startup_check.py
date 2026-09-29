"""Railway preflight for the configured, existing Chroma collection."""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from index_contract import inspect_existing_index


def check_pii_model() -> None:
    """Confirm the build-installed model required for fail-closed masking."""
    try:
        import spacy

        spacy.load("en_core_web_sm")
    except Exception:
        raise RuntimeError("Pinned spaCy PII model is unavailable") from None


def check_startup() -> int:
    from config import config

    count = inspect_existing_index(config.CHROMA_DB_PATH, config.EMBEDDING_MODEL)
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is required for model calls")
    check_pii_model()
    return count


if __name__ == "__main__":
    try:
        count = check_startup()
    except (RuntimeError, ValueError) as error:
        print(f"Startup check failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"ChromaDB ready with {count} documents")
