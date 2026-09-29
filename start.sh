#!/bin/bash
set -e

cd /app
echo "Starting IT Support RAG Agent..."

# Use the tracked Chroma snapshot by default. CHROMA_DB_PATH may select an
# existing, prepopulated index; a Railway volume is optional, not assumed.
/opt/venv/bin/python3 startup_check.py

echo "Starting Streamlit..."
exec /opt/venv/bin/python3 -m streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
