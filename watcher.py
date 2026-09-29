"""
File watcher for continuous ingestion into the IT Support RAG Agent.

Watches new_tickets/ for new JSON ticket files and automatically embeds
and ingests them into the existing ChromaDB collection. Sanitized copies
are written to new_tickets/processed/ and the raw drops are removed.

Run with:
    python3 watcher.py
"""

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json
import time
import os
import logging
import re
import uuid
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from normalize import normalize_ticket
from error_handling import safe_error_type
from config import config
from index_contract import (
    COLLECTION_NAME, l2_relevance_score, validate_collection,
    validate_embedding_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHER] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

WATCH_FOLDER = os.path.join(
    os.path.dirname(__file__), "new_tickets"
)
PROCESSED_FOLDER = os.path.join(
    os.path.dirname(__file__), "new_tickets",
    "processed"
)

# Match the IDs produced by data/tickets.py and the existing connectors.
# Free-form strings are not safe identifiers for embedding or persistence.
TICKET_ID_PATTERN = re.compile(
    r"(?:TKT-[0-9]{3}|KGL-[0-9]{5}|HF-[0-9]{5}|FB-[0-9]{5}"
    r"|GH-[A-Z]{3}-[0-9]{1,9}|(?:SNOW-)?INC[0-9]{7})"
)

class TicketIngestionHandler(FileSystemEventHandler):

    def __init__(self):
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain_core.documents import Document

        validate_embedding_model(config.EMBEDDING_MODEL)
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL
        )
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=config.CHROMA_DB_PATH,
            relevance_score_fn=l2_relevance_score,
            create_collection_if_not_exists=False,
        )
        validate_collection(self.vectorstore._collection, config.EMBEDDING_MODEL)
        logger.info(
            "Watcher initialized. "
            f"Monitoring: {WATCH_FOLDER}"
        )
        os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    def on_created(self, event):
        if event.is_directory:
            return

        filepath = event.src_path
        filename = os.path.basename(filepath)

        if not filename.endswith(".json"):
            return

        if "processed" in filepath:
            return

        logger.info("New ticket file detected")

        time.sleep(0.5)
        self._ingest_file(filepath, filename)

    def _ingest_file(self, filepath: str,
                     filename: str) -> None:
        from langchain_core.documents import Document

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                tickets = [data]
            elif isinstance(data, list):
                tickets = data
            else:
                logger.error("Invalid ticket file format")
                return

            normalized_tickets = []
            documents = []
            for idx, ticket in enumerate(tickets):
                required = [
                    "ticket_id", "category",
                    "issue", "resolution",
                    "resolved_in_minutes"
                ]
                if not isinstance(ticket, dict) or not all(
                    k in ticket for k in required
                ):
                    logger.warning("Ticket file contains a malformed record")
                    return

                ticket_id = str(ticket["ticket_id"])
                if not TICKET_ID_PATTERN.fullmatch(ticket_id):
                    logger.warning("Ticket file contains an invalid identifier")
                    return

                normalized = normalize_ticket(
                    {
                        "ticket_id": ticket_id,
                        "category": ticket["category"],
                        "issue": ticket["issue"],
                        "resolution": ticket["resolution"],
                        "resolved_in_minutes": ticket["resolved_in_minutes"],
                    },
                    idx,
                    "watcher",
                )
                if normalized is None:
                    logger.warning("Ticket file contains a record that failed normalization")
                    return
                normalized_tickets.append(normalized)

                text = (
                    f"Ticket ID: {normalized['ticket_id']}"
                    f" | Category: {normalized['category']}"
                    f" | Issue: {normalized['issue']}"
                    f" | Resolution: "
                    f"{normalized['resolution']}"
                    f" | Resolved in: "
                    f"{normalized['resolved_in_minutes']}"
                    f" minutes"
                )

                doc = Document(
                    page_content=text,
                    metadata={
                        "ticket_id": normalized["ticket_id"],
                        "category": normalized["category"],
                        "resolved_in_minutes":
                            normalized["resolved_in_minutes"]
                    }
                )
                documents.append(doc)

            if documents:
                self.vectorstore.add_documents(documents)
                logger.info(
                    f"Successfully ingested "
                    f"{len(documents)} ticket(s) "
                    "from ticket file"
                )

            if not documents:
                logger.warning("Ticket file contains no valid records")
                return

            processed_path = os.path.join(
                PROCESSED_FOLDER,
                f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}_"
                f"{uuid.uuid4().hex[:8]}.json"
            )
            # Retain only normalized content; never move the raw drop into
            # the processed directory.
            with open(processed_path, "x") as f:
                json.dump(normalized_tickets[0] if isinstance(data, dict)
                          else normalized_tickets, f)
            os.remove(filepath)
            logger.info("Stored sanitized ticket file in processed/")

        except json.JSONDecodeError:
            logger.error("Invalid ticket JSON — skipping")
        except Exception as e:
            logger.error("Ticket ingestion failed: %s", safe_error_type(e))

def run_watcher():
    handler = TicketIngestionHandler()
    observer = Observer()
    observer.schedule(
        handler, WATCH_FOLDER, recursive=False
    )
    observer.start()
    logger.info(
        "File watcher running. "
        "Press Ctrl+C to stop."
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Watcher stopped.")

    observer.join()

if __name__ == "__main__":
    run_watcher()
