#!/bin/bash
set -e

echo "Starting IT Support RAG Agent..."

# Persistent volume must be mounted at this path in the Railway dashboard
# (Settings -> Volumes) so ChromaDB survives across deploys/restarts.
export CHROMA_DB_PATH=/app/chroma_db

# Download spaCy model if not present
python3 -m spacy download en_core_web_sm

# Check if ChromaDB already has data
CHROMA_COUNT=$(/opt/venv/bin/python3 -c "
import os, sys
sys.path.insert(0, '/app')
os.chdir('/app')
try:
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    db = Chroma(
        collection_name='it_support_tickets',
        embedding_function=OpenAIEmbeddings(
            model='text-embedding-3-small'
        ),
        persist_directory=os.environ['CHROMA_DB_PATH']
    )
    count = db._collection.count()
    print(count)
except Exception as e:
    print('0')
" 2>/dev/null || echo "0")

echo "ChromaDB currently has $CHROMA_COUNT documents"

if [ "$CHROMA_COUNT" -lt "100" ]; then
    echo "ChromaDB is empty or low. Running ingestion with 8 minute timeout..."
    timeout 480 /opt/venv/bin/python3 ingest_all.py || echo "Ingestion timed out or failed - starting app anyway with available data"
    echo "Ingestion complete."
else
    echo "ChromaDB already populated. Skipping ingestion."
fi

echo "Starting Streamlit..."
exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
