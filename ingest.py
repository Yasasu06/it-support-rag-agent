"""
Ingestion script for the IT Support RAG Agent.

Pipeline:
    1. Load environment variables (OPENAI_API_KEY) from a local .env file.
    2. Import the TICKETS list from data.tickets.
    3. Convert each ticket dictionary into a single readable string.
    4. Embed each string with OpenAI's text-embedding-3-small model.
    5. Store the embedded tickets in a local ChromaDB collection.

Run with:
    python3 ingest.py
"""

import shutil
from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from data.tickets import TICKETS

# Configuration -------------------------------------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "it_support_tickets"
EMBEDDING_MODEL = "text-embedding-3-small"


def ticket_to_string(ticket: dict) -> str:
    """Combine all fields of a ticket into one readable line."""
    return (
        f"Ticket ID: {ticket['ticket_id']} | "
        f"Category: {ticket['category']} | "
        f"Issue: {ticket['issue']} | "
        f"Resolution: {ticket['resolution']} | "
        f"Resolved in: {ticket['resolved_in_minutes']} minutes"
    )


def main() -> None:
    # Step 1: load environment variables (expects OPENAI_API_KEY in .env)
    load_dotenv()

    # Start from a clean collection so re-running doesn't duplicate or leave
    # stale tickets from a previous ingest.
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)

    # Step 2 + 3: load tickets and convert each into a readable string.
    print("Loading tickets...")

    documents = []
    for ticket in TICKETS:
        print(f"Converting ticket {ticket['ticket_id']} to string...")
        text = ticket_to_string(ticket)
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "ticket_id": ticket["ticket_id"],
                    "category": ticket["category"],
                    "resolved_in_minutes": ticket["resolved_in_minutes"],
                },
            )
        )

    # Step 4 + 5: embed each string and store everything in ChromaDB.
    print("Embedding and storing in ChromaDB...")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    # Step 6: final confirmation.
    print(f"Done. {len(documents)} tickets stored in ChromaDB.")


if __name__ == "__main__":
    main()
