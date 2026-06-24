"""
File watcher for continuous ingestion into the IT Support RAG Agent.

Watches new_tickets/ for new JSON ticket files and automatically embeds
and ingests them into the existing ChromaDB collection. Processed files
are moved into new_tickets/processed/ so they are not re-ingested.

Run with:
    python3 watcher.py
"""

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json
import time
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

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

class TicketIngestionHandler(FileSystemEventHandler):

    def __init__(self):
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain_core.documents import Document

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        self.vectorstore = Chroma(
            collection_name="it_support_tickets",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(
                os.path.dirname(__file__), "chroma_db"
            )
        )
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

        logger.info(
            f"New ticket file detected: {filename}"
        )

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
                logger.error(
                    f"Invalid format in {filename}"
                )
                return

            documents = []
            for ticket in tickets:
                required = [
                    "ticket_id", "category",
                    "issue", "resolution",
                    "resolved_in_minutes"
                ]
                if not all(
                    k in ticket for k in required
                ):
                    logger.warning(
                        f"Skipping malformed ticket "
                        f"in {filename}"
                    )
                    continue

                text = (
                    f"Ticket ID: {ticket['ticket_id']}"
                    f" | Category: {ticket['category']}"
                    f" | Issue: {ticket['issue']}"
                    f" | Resolution: "
                    f"{ticket['resolution']}"
                    f" | Resolved in: "
                    f"{ticket['resolved_in_minutes']}"
                    f" minutes"
                )

                doc = Document(
                    page_content=text,
                    metadata={
                        "ticket_id": ticket["ticket_id"],
                        "category": ticket["category"],
                        "resolved_in_minutes":
                            ticket["resolved_in_minutes"]
                    }
                )
                documents.append(doc)

            if documents:
                self.vectorstore.add_documents(documents)
                logger.info(
                    f"Successfully ingested "
                    f"{len(documents)} ticket(s) "
                    f"from {filename}"
                )

            processed_path = os.path.join(
                PROCESSED_FOLDER,
                f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}"
            )
            os.rename(filepath, processed_path)
            logger.info(
                f"Moved {filename} to processed/"
            )

        except json.JSONDecodeError:
            logger.error(
                f"Invalid JSON in {filename} "
                f"— skipping"
            )
        except Exception as e:
            logger.error(
                f"Error ingesting {filename}: {e}"
            )

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
