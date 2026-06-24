# Enterprise IT Support AI Assistant

A production-grade AI system that answers enterprise IT support questions by retrieving and citing real historical incidents — built with Retrieval-Augmented Generation (RAG), a three-agent LangGraph pipeline, and a knowledge base assembled from four independent real-world and synthetic data sources.

Built to demonstrate Forward Deployed Engineer and Solutions Engineer capabilities for enterprise AI deployment.

---

## Live Demo

[Link to deployed app on Railway]

> Try asking: "My printer is not responding" or "How many VPN tickets exist in the system?"

---

## The Problem

Enterprise IT support teams spend significant time manually searching through knowledge bases and past incident tickets scattered across disconnected systems — ServiceNow, Confluence, email threads, and tribal knowledge.

## The Solution

An AI assistant grounded exclusively in enterprise IT incident data that:
- Retrieves the 3 most relevant past incidents using semantic similarity search
- Generates cited, grounded answers using only retrieved incident context
- Refuses out-of-scope questions rather than hallucinating
- Routes low-confidence queries to Tier 2 escalation automatically

---

## Architecture

### Multi-Agent Pipeline (LangGraph)

Every query flows through a three-agent StateGraph:

1. **Retrieval Agent** — semantic search across ChromaDB vector store, returns top 3 matching incidents with similarity scores
2. **Answer Agent** — GPT-4o-mini generates a grounded response using only retrieved context, cites Ticket IDs
3. **Triage Agent** — evaluates the confidence score, routes to Tier 1 or Tier 2 escalation

Analytical queries ("how many VPN tickets?") route to a separate Tool Agent with three ChromaDB tools: category search, ticket counting, and resolution time statistics.

### RAG Pipeline

- Embedding model: OpenAI `text-embedding-3-small`
- Vector store: ChromaDB with persistent Railway Volume storage
- Retrieval: top-3 semantic similarity search, with a separate category-metadata-filtered tool path for analytical queries
- Query reformulation: GPT-4o-mini rewrites vague queries before retrieval for improved recall
- Conversation memory: last 5 exchanges retained in a rolling in-memory buffer

### Data Sources (748 incidents after normalization)

| Source | Tickets Fetched | Description |
|--------|-------|-------------|
| Internal synthetic | 150 | Curated IT incident patterns across 8 categories |
| Kaggle customer support dataset | 797 | Real enterprise support tickets |
| GitHub Issues (closed, labeled) | 500 | Real open-source issue threads |
| HuggingFace ServiceNow dataset | 494 | Synthetic ServiceNow ITSM incidents |

*Figures are from the most recent full ingestion run. Per-source counts are pre-dedup fetch volumes; the pipeline normalizes, masks, and deduplicates across all four sources down to 748 final tickets.*

All data passes through a normalization pipeline: text cleaning/truncation, PII masking (Presidio), category inference, validity filtering (minimum length, placeholder/test-ticket rejection, duplicate issue/resolution rejection), and cross-source deduplication on issue text.

### Production Features

| Feature | Implementation |
|---------|---------------|
| PII Detection | Microsoft Presidio with `en_core_web_sm` |
| Query Audit Logging | JSONL audit trail per query |
| Evaluation System | LangSmith tracing, 85% accuracy on a 20-question test suite |
| Real-time Ingestion | Watchdog file monitor — drops a JSON file in `new_tickets/`, it's embedded and ingested automatically |
| Nightly Refresh | Schedule-based automatic re-ingestion at 02:00 UTC |
| Feedback Collection | Per-answer thumbs up/down, aggregated into a satisfaction percentage |
| Category Filtering | ChromaDB metadata filter by incident category |
| Response Streaming | Token-by-token streaming via LangChain's `.stream()` generator |
| Fine-tuning Pipeline | 120-example train / 30-example validation dataset prepared; job not yet submitted |

---

## Evaluation Results

Tested against 20 questions spanning all 8 incident categories:

- Overall accuracy: 85% (17/20)
- In-scope questions answered correctly: 13/16
- Out-of-scope questions refused correctly: 4/4
- All 3 failures were correctly-grounded refusals on weak/borderline matches, not fabricated answers — no hallucinated ticket IDs observed in the suite

---

## Tech Stack

**AI & Orchestration**
- GPT-4o-mini (OpenAI) — answer generation
- text-embedding-3-small (OpenAI) — semantic embeddings
- LangChain 1.3+ — RAG pipeline orchestration
- LangGraph 1.2+ — multi-agent StateGraph
- LangSmith — production tracing and evaluation

**Data & Storage**
- ChromaDB — vector store with persistent volume
- Kaggle API — real incident dataset ingestion
- GitHub Issues API — open source IT issues
- HuggingFace Datasets — ServiceNow ITSM data
- JSONL — audit and feedback logging

**Security & Compliance**
- Microsoft Presidio — PII detection and masking
- spaCy `en_core_web_sm` — NER model for PII
- Environment-based secrets management
- Full query audit trail

**Deployment**
- Railway — containerized deployment with persistent volume storage
- Nixpacks — automated build configuration
- Streamlit — chat interface

---

## Project Structure

```
it-support-rag-agent/
├── app.py                    # Streamlit chat interface
├── rag.py                    # RAG pipeline + streaming
├── agent_pipeline.py         # LangGraph multi-agent system
├── security.py               # PII detection + audit logging
├── normalize.py              # Multi-source data normalization
├── ingest_all.py             # Multi-source ingestion pipeline
├── watcher.py                # Real-time file ingestion
├── scheduler.py              # Nightly refresh scheduler
├── eval.py                   # Evaluation test suite
├── start.sh                  # Railway startup script
├── connectors/
│   ├── kaggle_connector.py   # Kaggle dataset ingestion
│   ├── github_connector.py   # GitHub Issues ingestion
│   └── huggingface_connector.py  # HF dataset ingestion
├── data/
│   └── tickets.py            # 150 synthetic IT incidents
└── finetune/
    ├── prepare_dataset.py    # Fine-tuning dataset prep
    └── run_finetune.py       # OpenAI fine-tuning pipeline
```

---

## Local Setup

```bash
# Clone and setup
git clone https://github.com/Yasasu06/it-support-rag-agent
cd it-support-rag-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create a .env file in the project root with:
# OPENAI_API_KEY=sk-...
# LANGSMITH_API_KEY=ls-...
# LANGCHAIN_PROJECT=...
# KAGGLE_USERNAME=...
# KAGGLE_KEY=...
# GITHUB_TOKEN=ghp_...
# GITHUB_USERNAME=...

# Download spaCy model
python3 -m spacy download en_core_web_sm

# Build knowledge base
python3 ingest_all.py

# Run the app
streamlit run app.py
```

---

## Key Design Decisions

**Why RAG over fine-tuning?**
RAG grounds answers in the actual incident database at query time. Fine-tuning bakes knowledge into model weights, making updates expensive. For a knowledge base that updates daily, RAG is the correct architectural choice.

**Why LangGraph over a single chain?**
The three-agent separation allows independent optimization of retrieval, generation, and triage. It also enables the analytical tool agent path for aggregate queries without affecting the RAG path.

**Why ChromaDB over Pinecone?**
ChromaDB runs locally and on Railway with zero infrastructure overhead. For a portfolio deployment, operational simplicity matters. Production migration to Pinecone or Azure AI Search would be a configuration change, not an architectural one.

**Confidence calibration**
Similarity scores from this ChromaDB configuration top out around 0.32 for correct matches. Thresholds were calibrated empirically against observed score distributions: High (≥0.60), Medium (≥0.20), Low (<0.20) — the Low tier triggers automatic Tier 2 escalation.

---

## Built By

**Yasaswi**
MS Technology Management, University of Illinois Urbana-Champaign
Targeting: Forward Deployed Engineer, Solutions Engineer, Customer Engineer roles at Microsoft, IBM, Salesforce, Glean, Databricks
