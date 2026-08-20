# Enterprise IT Support AI Assistant

![Tests](https://github.com/Yasasu06/it-support-rag-agent/actions/workflows/tests.yml/badge.svg)

A production-grade AI system that answers enterprise IT support questions by retrieving and citing real historical incidents — built with Retrieval-Augmented Generation (RAG), a four-agent LangGraph pipeline with adaptive retry, and a knowledge base assembled from multiple real-world and synthetic data sources plus a live ServiceNow API connection.

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

Every query flows through a four-agent StateGraph:

1. **Retrieval Agent** — semantic search across ChromaDB vector store, returns top 3 matching incidents with similarity scores; broadens the query on a retry
2. **Answer Agent** — GPT-4o-mini generates a grounded response using only retrieved context, cites Ticket IDs
3. **Verification Agent** — an independent GPT-4o-mini judge fact-checks the answer against the retrieved context and flags ungrounded claims (feature-flagged via `ENABLE_VERIFICATION_AGENT`)
4. **Triage Agent** — evaluates confidence and groundedness, routes to Tier 1 or Tier 2 escalation

**Adaptive retry:** when the answer fails groundedness verification or confidence is low, the pipeline automatically broadens the query and re-runs retrieval — up to `MAX_RETRIEVAL_RETRIES` (default 2) additional passes — before returning. This retry is surfaced in the UI (a "broadened search" badge on the answer and a session retry-rate stat in the System tab).

Analytical queries ("how many VPN tickets?") route to a separate Tool Agent with four ChromaDB/ServiceNow tools: category search, ticket counting, resolution-time statistics, and live ServiceNow status.

### RAG Pipeline

- Embedding model: OpenAI `text-embedding-3-small`
- Vector store: ChromaDB with persistent Railway Volume storage
- Retrieval: top-3 semantic similarity search, with a separate category-metadata-filtered tool path for analytical queries
- Query reformulation: GPT-4o-mini rewrites vague queries before retrieval for improved recall
- Conversation memory: last 5 exchanges retained in a rolling in-memory buffer

### Data Sources (750 incidents currently embedded)

Counts below are the **actual number of tickets embedded in the production ChromaDB right now**, verified by querying the collection metadata (not pre-dedup fetch volumes):

| Source | Embedded tickets | Description |
|--------|-------|-------------|
| GitHub Issues (closed, labeled) | 496 | Real open-source IT/dev issue threads |
| Internal synthetic | 152 | Curated IT incident patterns across 8 categories |
| HuggingFace ServiceNow dataset | 97 | Synthetic ServiceNow ITSM incidents |
| Kaggle support dataset | 5 | Real enterprise support tickets (after cross-source dedup) |
| **Live ServiceNow API** | 0 embedded | Real-time REST connection (`/api/now/table/incident`); batch-ingest supported, but the connected instance is currently empty |
| **Total** | **750** | 8 categories |

*The distribution is intentionally GitHub-weighted; Kaggle contributes only a handful of tickets after deduplication. ServiceNow is wired as a genuine live API connection (queried in real time via the `check_live_servicenow_status` tool), but the connected instance has no incidents, so it contributes 0 embedded tickets.*

All data passes through a normalization pipeline: text cleaning/truncation, PII masking (Presidio), category inference, validity filtering (minimum length, placeholder/test-ticket rejection, duplicate issue/resolution rejection), and cross-source deduplication on issue text.

### Production Features

| Feature | Implementation |
|---------|---------------|
| PII Detection | Microsoft Presidio with `en_core_web_sm` |
| Query Audit Logging | JSONL audit trail per query |
| Evaluation System | LangSmith tracing, 90% accuracy on a 20-question test suite |
| Real-time Ingestion | Watchdog file monitor — drops a JSON file in `new_tickets/`, it's embedded and ingested automatically |
| Nightly Refresh | Schedule-based automatic re-ingestion at 02:00 UTC |
| Feedback Collection | Per-answer thumbs up/down, aggregated into a satisfaction percentage |
| Category Filtering | ChromaDB metadata filter by incident category |
| Response Streaming | Token-by-token streaming via LangChain's `.stream()` generator |
| Fine-tuning Pipeline | 120-example train / 30-example validation dataset prepared; job not yet submitted |
| Multi-modal Input | Upload an error screenshot with (or instead of) your question — GPT-4o-mini vision extracts the issue, which flows through the same RAG pipeline (PII masking, verification, confidence scoring). Toggle with `ENABLE_IMAGE_UPLOAD`. |
| Rate Limiting | Per-session sliding-window limit (default 15 requests / 60s) applied before any OpenAI call on every submission path (typed, screenshot, and suggestion chips), so abuse is blocked before it can incur cost. Isolated per session — one user never affects another. |
| Answer Verification | Independent GPT-4o-mini judge fact-checks every answer against the retrieved context before delivery; ungrounded answers are flagged and force Tier 2 escalation. Toggle with `ENABLE_VERIFICATION_AGENT`. |
| Adaptive Retry | On failed verification or low confidence, the query is automatically broadened and the pipeline re-runs (up to 2 retries). Surfaced in the UI via a "broadened search" badge and a System-tab retry-rate stat. Toggle with `ENABLE_ADAPTIVE_RETRY`. |
| Failure Handling | Retry-with-backoff on transient API failures, graceful degradation to a safe escalated response on outages, fail-closed PII masking, and input validation — so a backend failure never surfaces a raw traceback to the user. |
| Live Evaluation | Every production query is scored in real time (relevance, groundedness, latency) via `live_eval.py`; aggregates shown in the System tab (separate from the offline eval harness). |
| Deployment Configurability | ~18 environment variables centralized in `config.py` (models, thresholds, retry limits, feature flags) with defaults that reproduce prior behavior exactly; active config shown in the System tab. |
| Visit Analytics | Anonymous per-session visit logging (timestamp + random UUID only, no PII) with a private, password-protected admin view at `?admin=<secret>` (gated by `ADMIN_ANALYTICS_KEY`). |

---

## Evaluation Results

Tested against 20 questions spanning all 8 incident categories:

- Overall accuracy: 90% (18/20)
- In-scope questions answered correctly: 14/16
- Out-of-scope questions refused correctly: 4/4
- Both remaining failures were refusals on in-scope questions with weak/borderline retrieval matches, not fabricated answers — no hallucinated ticket IDs observed in the suite

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
├── app.py                    # Streamlit UI (chat + System tab)
├── rag.py                    # RAG query layer + streaming
├── agent_pipeline.py         # LangGraph 4-agent pipeline + tools + adaptive retry
├── config.py                 # Centralized env-var config + feature flags
├── security.py               # Presidio PII masking (fail-closed) + audit logging
├── normalize.py              # Multi-source data normalization + dedup
├── error_handling.py         # Retry/backoff, safe-fallback, input validation
├── rate_limiter.py           # Per-session sliding-window rate limiting
├── image_processing.py       # Multi-modal: GPT-4o-mini vision screenshot extraction
├── live_eval.py              # Real-time per-query scoring
├── analytics.py              # Anonymous visit logging (admin-gated view)
├── ingest_all.py             # Multi-source ingestion pipeline
├── watcher.py                # Real-time file ingestion
├── scheduler.py              # Nightly refresh scheduler
├── eval.py                   # 20-question behavioral eval suite
├── start.sh / railway.toml   # Railway startup + deploy config
├── DECISIONS.md              # Documented architecture tradeoffs
├── connectors/
│   ├── kaggle_connector.py
│   ├── github_connector.py
│   ├── huggingface_connector.py
│   └── servicenow_connector.py   # Live ServiceNow REST connector
├── eval_harness/             # Offline 5-dimension evaluation
│   ├── metrics.py            # groundedness/relevance/citation/refusal/latency
│   ├── runner.py             # runs the harness over the test set
│   ├── analyze.py            # honest averages (excl. correct refusals)
│   └── compare_runs.py       # before/after diff
├── finetune/
│   ├── prepare_dataset.py    # Fine-tuning dataset prep (120/30 JSONL)
│   ├── run_finetune.py       # OpenAI fine-tuning job script (not executed)
│   ├── compare_models.py     # base-vs-finetuned comparison harness
│   └── FINETUNE_STATUS.md    # decision record + baseline
├── tests/                    # 54 pytest tests (51 fast + 3 API-gated)
│   ├── test_normalize.py  test_security.py  test_routing_logic.py
│   ├── test_retry_decision.py  test_error_handling.py  test_rate_limiter.py
│   ├── test_image_processing.py  test_finetune_config.py  test_integration.py
│   └── conftest.py
├── .github/workflows/tests.yml   # CI: runs fast tests on every push
└── data/
    └── tickets.py            # 152 synthetic IT incidents
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

## Configuration

All deployment-tunable behavior is centralized in [config.py](config.py) and
overridable via environment variables. Every value defaults to exactly what the
system used before the config layer existed, so **the defaults reproduce the
current behavior identically** — you only change anything by explicitly setting
a variable. The System tab shows the active configuration at runtime.

| Env var | Default | Controls |
|---------|---------|----------|
| `CHAT_MODEL` | `gpt-4o-mini` | LLM used for answers, verification, reformulation, and scoring |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for retrieval (must match the model the index was built with) |
| `RETRIEVAL_K` | `3` | Number of tickets retrieved per query |
| `CONFIDENCE_HIGH_THRESHOLD` | `0.60` | Similarity score at/above which confidence is labeled High |
| `CONFIDENCE_MEDIUM_THRESHOLD` | `0.20` | Score at/above which confidence is Medium; below triggers Tier 2 escalation |
| `MAX_RETRIEVAL_RETRIES` | `2` | Max adaptive-retry passes when an answer fails grounding/confidence |
| `RETRY_DELAY_SECONDS` | `1.0` | Delay between transient-failure retries in the error-handling layer |
| `LANGGRAPH_RECURSION_LIMIT` | `20` | LangGraph step budget for the multi-agent pipeline |
| `MAX_QUESTION_LENGTH` | `2000` | Max characters accepted for a user question before rejection |
| `ENABLE_LIVE_EVAL` | `true` | Real-time per-query scoring/logging (`live_eval`) |
| `ENABLE_VERIFICATION_AGENT` | `true` | Groundedness verification node in the pipeline; when false, answer routes straight to triage |
| `ENABLE_ADAPTIVE_RETRY` | `true` | Query-broadening retry loop; when false, the pipeline ends after the first pass |
| `ENABLE_SERVICENOW_LIVE` | `true` | Live ServiceNow status tool; when false it reports the connection is disabled |
| `ENABLE_IMAGE_UPLOAD` | `true` | Screenshot upload + vision extraction; when false the uploader is hidden |
| `ENABLE_RATE_LIMITING` | `true` | Per-session rate limiting on question submission; when false, no limit is applied |
| `RATE_LIMIT_MAX_REQUESTS` | `15` | Max requests allowed per session within the window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding-window length (seconds) for the rate limit |
| `APP_ENVIRONMENT` | `production` | Free-text environment label shown in the System tab |

Example — run a staging instance with a bigger model, verification off, and no
live eval:

```bash
CHAT_MODEL=gpt-4o ENABLE_VERIFICATION_AGENT=false ENABLE_LIVE_EVAL=false \
APP_ENVIRONMENT=staging streamlit run app.py
```

---

## Testing

The project has a pytest suite split into fast unit tests (no API key, no
network) and slower integration tests (real end-to-end pipeline).

```bash
# Run the full suite (fast unit tests + integration tests).
# Integration tests run only if OPENAI_API_KEY is available (from your .env);
# otherwise they skip automatically.
python -m pytest tests/ -v

# Run only the fast unit tests — no API key needed, completes in ~2 seconds.
python -m pytest tests/ -v --ignore=tests/test_integration.py
```

**54 tests total** (`pytest tests/ --collect-only`): **51 fast unit tests** plus
3 API-gated integration tests. The fast tests span 8 files — `test_normalize.py`,
`test_security.py`, `test_routing_logic.py`, `test_retry_decision.py`,
`test_error_handling.py`, `test_rate_limiter.py`, `test_image_processing.py`,
`test_finetune_config.py` — covering validity filtering, category inference,
deduplication, PII masking, audit logging, query routing, retry-decision logic,
error-handling/retry decorators, rate-limiting, image validation, and
fine-tuned-model selection — all with zero API cost.

**Integration tests** (`test_integration.py`) exercise the real RAG and
multi-agent pipelines against OpenAI and auto-skip when no API key is present.

### Continuous Integration

A [GitHub Actions workflow](.github/workflows/tests.yml) runs the fast unit test
suite (`pytest tests/ --ignore=tests/test_integration.py`) automatically on
every push and pull request to `main` (Ubuntu, Python 3.12). Integration tests
are intentionally excluded from CI to avoid exposing an API key as a secret and
incurring per-push API cost — a deliberate tradeoff documented in the workflow
file. Current status is shown by the badge at the top of this README.

---

## Evaluation Harness

Beyond pass/fail accuracy, the project ships a structured evaluation harness
(`eval_harness/`) that scores every answer on **five independent dimensions**,
so a run reveals *where* the system is weak rather than just whether it passed:

| Dimension | Type | What it measures |
|-----------|------|------------------|
| **Groundedness** | LLM judge | Is every claim in the answer supported by retrieved context? |
| **Relevance** | LLM judge | Does the answer actually address the question asked? |
| **Citation accuracy** | Pure logic | Does every cited Ticket ID appear in the retrieved sources? |
| **Refusal correctness** | Pure logic | Did it answer in-scope questions and refuse out-of-scope ones? |
| **Latency** | Pure logic | Response time against a chat-interface threshold |

Scoring each quality separately catches failure modes a single score hides —
e.g. an answer can be perfectly grounded yet irrelevant (answering a different
question with real ticket data), and citation accuracy independently guards
against citing a Ticket ID that was never retrieved.

```bash
# Run the full harness (multiple LLM calls per question — takes a few minutes)
python3 eval_harness/runner.py

# Compare two runs (e.g. before/after a change) to see improvements/regressions
python3 eval_harness/compare_runs.py old.json new.json
```

Each run writes a timestamped `eval_harness/harness_results.json` with
per-dimension averages plus full per-question detail.

**Reading the groundedness/relevance numbers honestly** (representative run —
LLM-judged scores vary somewhat run-to-run): the raw averages (groundedness
**~0.65–0.70**, relevance **~0.64–0.66**) are pulled down by the four
correctly-refused out-of-scope questions, which score 0 by construction — there
is no context to be grounded in and refusing is the right behaviour. Excluding
only those correct refusals (never false refusals, which are real misses),
the averages rise to roughly **0.81 groundedness** and **0.79 relevance**. Run
`python3 eval_harness/analyze.py` to see both numbers side by side plus the
genuine in-scope weak spots, which are kept in both figures rather than hidden.

This is **separate from the always-on live evaluation** (`live_eval.py`), which
scores real production queries continuously as they happen. The harness is for
deliberate, structured test runs against a fixed question set — the kind you run
before and after a change to measure impact.

---

## Key Design Decisions

**Why RAG over fine-tuning?**
RAG grounds answers in the actual incident database at query time. Fine-tuning bakes knowledge into model weights, making updates expensive. For a knowledge base that updates daily, RAG is the correct architectural choice.

**Why LangGraph over a single chain?**
The four-agent separation (retrieval, answer, verification, triage) allows independent optimization of each stage and makes the adaptive-retry loop a clean conditional edge. It also enables the analytical tool-agent path for aggregate queries without affecting the RAG path.

**Why ChromaDB over Pinecone?**
ChromaDB runs locally and on Railway with zero infrastructure overhead. For a portfolio deployment, operational simplicity matters. Production migration to Pinecone or Azure AI Search would be a configuration change, not an architectural one.

**Confidence calibration**
Similarity scores from this ChromaDB configuration top out around 0.32 for correct matches. Thresholds were calibrated empirically against observed score distributions: High (≥0.60), Medium (≥0.20), Low (<0.20) — the Low tier triggers automatic Tier 2 escalation.

---

## Known Tradeoffs & Production Considerations

Deliberate engineering decisions and prioritized gaps — documented for
transparency rather than hidden. Full rationale for the vector-store decision
lives in [DECISIONS.md](DECISIONS.md).

- **Vector store is pre-built and committed to git** (see [DECISIONS.md](DECISIONS.md))
  rather than rebuilt from a Railway persistent volume. This guarantees the
  deployed app always has data within Railway's cold-start window, at the cost
  of shipping an ~11MB binary in version control. Migrating to a persistent
  volume with conditional re-ingestion is the planned proper fix.
- **The connected ServiceNow instance is empty.** The connector is a genuine
  live REST integration, but the instance it points at has no incidents, so it
  contributes 0 embedded tickets and the live-status tool reports an empty
  instance. It demonstrates the integration, not a populated production feed.
- **Uneven source mix.** After cross-source deduplication the knowledge base is
  heavily GitHub-weighted (496 of 750); Kaggle contributes only 5 tickets.
- **Fine-tuning was scoped but never executed.** The dataset-prep, job script,
  and model-comparison harness are all built, but the OpenAI fine-tuning job was
  never submitted (self-serve access gated). The system can hot-swap a
  fine-tuned model via config the moment one exists — see
  [finetune/FINETUNE_STATUS.md](finetune/FINETUNE_STATUS.md).
- **Small eval set.** Accuracy is measured on a fixed 20-question suite; the
  LLM-judged metrics vary run-to-run, so the ~90% figure is a reproducible
  central value rather than a fixed guarantee.

---

## Built By

**Yasaswi**
MS Technology Management, University of Illinois Urbana-Champaign
Targeting: Forward Deployed Engineer, Solutions Engineer, Customer Engineer roles at Microsoft, IBM, Salesforce, Glean, Databricks
