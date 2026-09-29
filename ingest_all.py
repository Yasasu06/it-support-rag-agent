"""
Multi-source ingestion pipeline for the IT Support RAG Agent.

Replaces ingest.py for ingestion combining synthetic, Kaggle, and GitHub
Issues data sources.

Run with:
    python3 ingest_all.py
"""

import os
import sys
import logging
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INGEST] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

from data.tickets import TICKETS as SYNTHETIC_TICKETS
from connectors.kaggle_connector import (
    run_kaggle_connector
)
from connectors.github_connector import (
    run_github_connector
)
from connectors.huggingface_connector import (
    run_huggingface_connector
)
from connectors.servicenow_connector import (
    is_configured as servicenow_configured,
    fetch_incidents_batch as fetch_servicenow_batch,
)
from normalize import normalize_all_sources
from error_handling import safe_error_type
from config import config
from index_contract import (
    COLLECTION_NAME, COLLECTION_CONFIGURATION, COLLECTION_METADATA,
    l2_relevance_score, validate_collection, validate_embedding_model,
)

CHROMA_DIR = config.CHROMA_DB_PATH

def run_multi_source_ingest():
    validate_embedding_model(config.EMBEDDING_MODEL)
    logger.info(
        "Starting multi-source ingestion pipeline"
    )
    logger.info("=" * 60)

    logger.info(
        f"Source 1: {len(SYNTHETIC_TICKETS)} "
        f"synthetic tickets loaded"
    )

    logger.info("Source 2: Fetching Kaggle data...")
    try:
        kaggle_tickets = run_kaggle_connector()
        logger.info(
            f"Kaggle: {len(kaggle_tickets)} tickets"
        )
    except Exception as e:
        logger.error("Kaggle connector failed: %s", safe_error_type(e))
        kaggle_tickets = []

    logger.info(
        "Source 3: Fetching GitHub Issues..."
    )
    try:
        github_tickets = run_github_connector()
        logger.info(
            f"GitHub: {len(github_tickets)} tickets"
        )
    except Exception as e:
        logger.error(
            f"GitHub connector failed: {safe_error_type(e)}"
        )
        github_tickets = []

    logger.info(
        "Source 4: Fetching HuggingFace "
        "ServiceNow data..."
    )
    try:
        hf_tickets = run_huggingface_connector()
        logger.info(
            f"HuggingFace: {len(hf_tickets)} tickets"
        )
    except Exception as e:
        logger.error(
            f"HuggingFace connector failed: {safe_error_type(e)}"
        )
        hf_tickets = []

    logger.info(
        "Source 5: Fetching live ServiceNow data..."
    )
    servicenow_tickets = []
    if servicenow_configured():
        try:
            servicenow_tickets = fetch_servicenow_batch(limit=200)
            logger.info(
                f"ServiceNow: {len(servicenow_tickets)} tickets"
            )
        except Exception as e:
            logger.error("ServiceNow connector failed: %s", safe_error_type(e))
            servicenow_tickets = []
    else:
        logger.info(
            "ServiceNow not configured - skipping "
            "(this is fine, other sources continue)"
        )

    logger.info("Normalizing all sources...")
    all_tickets = normalize_all_sources(
        SYNTHETIC_TICKETS,
        kaggle_tickets,
        github_tickets,
        hf_tickets,
        servicenow_tickets,
    )

    logger.info(
        f"Total after normalization: "
        f"{len(all_tickets)} tickets"
    )

    logger.info("Clearing existing ChromaDB collection...")
    import chromadb
    from chromadb.errors import NotFoundError

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        existing = client.get_collection(COLLECTION_NAME)
    except NotFoundError:
        logger.info("No existing collection to delete")
    else:
        validate_collection(existing, config.EMBEDDING_MODEL)
        client.delete_collection(COLLECTION_NAME)
        logger.info("Existing collection deleted")

    logger.info(
        "Embedding and storing in ChromaDB..."
    )

    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document

    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

    documents = []
    for ticket in all_tickets:
        text = (
            f"Ticket ID: {ticket['ticket_id']}"
            f" | Category: {ticket['category']}"
            f" | Issue: {ticket['issue']}"
            f" | Resolution: {ticket['resolution']}"
            f" | Resolved in: "
            f"{ticket['resolved_in_minutes']} minutes"
            f" | Source: {ticket['source']}"
        )
        doc = Document(
            page_content=text,
            metadata={
                "ticket_id": ticket["ticket_id"],
                "category": ticket["category"],
                "source": ticket["source"],
                "resolved_in_minutes":
                    ticket["resolved_in_minutes"]
            }
        )
        documents.append(doc)

    batch_size = 100
    total_batches = (
        len(documents) + batch_size - 1
    ) // batch_size

    vectorstore = None

    for i in tqdm(
        range(0, len(documents), batch_size),
        desc="Embedding batches",
        total=total_batches
    ):
        batch = documents[i:i + batch_size]

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=CHROMA_DIR,
                collection_configuration=COLLECTION_CONFIGURATION,
                collection_metadata=COLLECTION_METADATA,
                relevance_score_fn=l2_relevance_score,
            )
            validate_collection(vectorstore._collection, config.EMBEDDING_MODEL)
        else:
            vectorstore.add_documents(batch)

    final_count = vectorstore._collection.count()

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"Synthetic tickets: "
                f"{len(SYNTHETIC_TICKETS)}")
    logger.info(f"Kaggle tickets: "
                f"{len(kaggle_tickets)}")
    logger.info(f"GitHub tickets: "
                f"{len(github_tickets)}")
    logger.info(f"HuggingFace tickets: "
                f"{len(hf_tickets)}")
    logger.info(f"ServiceNow tickets: "
                f"{len(servicenow_tickets)}")
    logger.info(f"Total in ChromaDB: {final_count}")
    logger.info("=" * 60)

    summary = {
        "synthetic": len(SYNTHETIC_TICKETS),
        "kaggle": len(kaggle_tickets),
        "github": len(github_tickets),
        "huggingface": len(hf_tickets),
        "servicenow": len(servicenow_tickets),
        "total": final_count
    }

    import json
    summary_path = os.path.join(
        os.path.dirname(__file__),
        "ingestion_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Summary saved to ingestion_summary.json"
    )

    return summary

if __name__ == "__main__":
    run_multi_source_ingest()
