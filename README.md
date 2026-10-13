# Enterprise IT Support AI Assistant

![Tests](https://github.com/Yasasu06/it-support-rag-agent/actions/workflows/tests.yml/badge.svg)

A RAG assistant for IT support questions, built with a LangGraph support workflow and a separate analytical tool path. Its tracked 750-record index combines transformed public GitHub issues, fictional internal-style support records, synthetic ServiceNow-format incidents, and a small public support dataset. An optional, read-only ServiceNow connector can retrieve live incident information when configured.

Built to demonstrate Forward Deployed Engineer and Solutions Engineer capabilities for enterprise AI deployment.

---

## Hosted Demo

**[it-support-rag-agent-production.up.railway.app](https://it-support-rag-agent-production.up.railway.app/)**

> Try asking: "My printer is not responding" or "How many VPN tickets exist in the system?"

---

## The Problem

Enterprise IT support teams spend significant time manually searching through knowledge bases and past incident tickets scattered across disconnected systems — ServiceNow, Confluence, email threads, and tribal knowledge.

## The Solution

An IT support assistant that:
- Retrieves up to three relevant records from the tracked knowledge base
- Generates answers from retrieved context and cites source ticket IDs
- Returns a no-evidence response when retrieval finds no support; it can decline out-of-scope questions
- Recommends Tier 2 review when retrieval or grounding warrants it

---

## Architecture

### Multi-Agent Pipeline (LangGraph)

Support-answer queries use LangGraph with retrieval, answer, and triage nodes;
the verification node is added when enabled:

1. **Retrieval Agent** — semantic search across ChromaDB, returning up to three matching records and a retrieval relevance signal; broadens the query on a retry
2. **Answer Agent** — the configured chat model (GPT-4o-mini by default) drafts an answer from retrieved context and cites Ticket IDs; without evidence, it returns a no-evidence response instead of calling the answer model
3. **Verification Agent** — when enabled, a separate model call checks whether the draft is supported by retrieved context
4. **Triage Agent** — uses the retrieval relevance signal and, when available, the grounding check to label Tier 1 responses or recommend Tier 2 review

Tier 2 is a review recommendation; no ServiceNow ticket is created automatically.

**Adaptive retry:** when enabled, a failed grounding check or retrieval signal below the routing cutoff broadens the query and re-runs retrieval — up to `MAX_RETRIEVAL_RETRIES` (default 2) additional passes — before returning. This retry is surfaced in the UI (a "broadened search" badge on the answer and a session retry-rate stat in the System tab). Optional live evaluation scores completed support responses after the graph finishes.

Analytical queries ("how many VPN tickets?") route to a separate Tool Agent with four ChromaDB/ServiceNow tools: category search, ticket counting, resolution-time statistics, and live ServiceNow status.

### RAG Pipeline

- Embedding model: OpenAI `text-embedding-3-small`
- Vector store: tracked ChromaDB snapshot by default; an optional prepopulated volume can be selected with `CHROMA_DB_PATH`
- Retrieval: top-3 semantic similarity search, with an optional category metadata filter for support answers
- Query reformulation: the configured chat model can rewrite vague queries before retrieval
- Conversation context: the chat UI supplies up to four recent messages to reformulation; a separate RAG helper retains five exchanges in memory

### Data Sources (750 records in the tracked snapshot)

Counts below describe the **tracked Chroma snapshot** in this repository, reconciled with its manifest and collection records (not a live Railway query):

| Source | Embedded records | Description |
|--------|-------|-------------|
| GitHub Issues (closed, labeled) | 496 | Transformed public software issue threads |
| Internal synthetic | 152 | Fictional internal-style IT support records across 8 categories |
| HuggingFace ServiceNow dataset | 97 | Synthetic ServiceNow-format incidents |
| Kaggle support dataset | 5 | Public customer-support records after deduplication |
| **Live ServiceNow API** | 0 embedded | Optional REST connector (`/api/now/table/incident`); requires credentials and a reachable instance |
| **Total** | **750** | 8 categories |

*The tracked snapshot is GitHub-weighted; Kaggle contributes only a handful of records after deduplication. The ServiceNow connector can make live read requests when configured, but the repository cannot establish whether a Railway instance is currently connected. No live ServiceNow API records are embedded in the tracked snapshot.*

Newly ingested records pass through text cleaning, Presidio-based PII masking, category inference, validity filtering, and cross-source deduplication. The tracked snapshot predates some current safeguards; retrieved content is sanitized again before model use.

### Current Capabilities

| Capability | Status | What it does |
|------------|--------|--------------|
| Support answers and analytical questions | Implemented | Support answers use retrieval, answer, and triage nodes; analytical questions use a separate tool-agent flow. |
| Retrieval and category filtering | Implemented | Searches the configured Chroma collection; a selected support category applies a metadata filter. |
| Privacy and audit | Implemented | Presidio masks detected text before model calls; local query logs store metadata rather than raw questions or answers. Masking does not guarantee detection of every sensitive detail. |
| Feedback and visit analytics | Implemented | Stores local JSONL feedback metadata and random per-session visit IDs; the admin view requires `ADMIN_ANALYTICS_KEY`. |
| Reliability controls | Implemented | Validates input, limits requests per session, retries transient failures, and returns a review recommendation on degraded responses. |
| Grounding verification and adaptive retry | Optional | Feature flags control the judge and query-broadening retry loop for support answers. |
| Live evaluation and LangSmith tracing | Optional | Completed support answers can be scored locally; tracing runs only when externally configured and may include generated, retrieved, or error content in child traces. |
| Screenshot input and ServiceNow reads | Optional | Vision extracts an issue from a supplied image; configured ServiceNow requests read incident data. No ticket is created. |
| Watcher and scheduler | Manual utility | Started explicitly; Railway startup does not launch them. The scheduler runs at 02:00 host-local time. |
| Fine-tuning preparation and model comparison | Manual utility | The current 152-record source produces a 121/31 training/validation split and supports a separate comparison harness. No fine-tuned model is active by default. |
| Response delivery | Implemented | The Streamlit chat displays a completed graph response; a separate RAG helper supports token streaming. |

---

## Evaluation Results

The saved [`eval_results.json`](eval_results.json) run from **2026-09-16** uses a fixed set of 20 questions spanning eight support categories plus out-of-scope prompts. It checks expected answer-versus-refusal behavior with text markers; it does not grade resolution quality or deployed-user outcomes:

- Expected behavior matched: **18/20** questions
- In-scope prompts answered: **14/16**; out-of-scope prompts refused: **4/4**
- The other two in-scope prompts were refused. This result is a small, fixed benchmark, not a general accuracy guarantee.

---

## Tech Stack

**AI & Orchestration**
- GPT-4o-mini (OpenAI) — answer generation
- text-embedding-3-small (OpenAI) — semantic embeddings
- LangChain 1.3+ — RAG pipeline orchestration
- LangGraph 1.2+ — multi-agent StateGraph
- LangSmith — optional tracing when configured

**Data & Storage**
- ChromaDB — tracked vector-store snapshot; optional external persistence
- Kaggle API — public support dataset ingestion
- GitHub Issues API — public software issues
- HuggingFace Datasets — ServiceNow ITSM data
- JSONL — audit and feedback logging

**Privacy & Safety**
- Microsoft Presidio — PII detection and masking
- spaCy `en_core_web_sm` — NER model for PII
- Environment-based secrets management
- Local query metadata audit log

**Deployment**
- Railway — Nixpacks deployment using the tracked snapshot by default
- Nixpacks — automated build configuration
- Streamlit — chat interface

---

## Project Structure

```
it-support-rag-agent/
├── app.py                    # Streamlit UI (chat + System tab)
├── rag.py                    # Separate RAG helper, including streaming
├── agent_pipeline.py         # Four-node support graph + analytical tools
├── config.py                 # Centralized env-var config + feature flags
├── security.py               # Presidio PII masking (fail-closed) + audit logging
├── normalize.py              # Multi-source data normalization + dedup
├── error_handling.py         # Retry/backoff, safe-fallback, input validation
├── rate_limiter.py           # Per-session sliding-window rate limiting
├── image_processing.py       # Multi-modal: GPT-4o-mini vision screenshot extraction
├── live_eval.py              # Optional completed support-run scoring
├── analytics.py              # Per-session visit logging (admin-gated view)
├── ingest_all.py             # Multi-source ingestion pipeline
├── watcher.py                # Optional manually started file ingestion
├── scheduler.py              # Optional manually started refresh scheduler
├── eval.py                   # 20-question behavioral eval suite
├── start.sh / railway.toml   # Railway startup + deploy config
├── DECISIONS.md              # Documented architecture tradeoffs
├── connectors/
│   ├── kaggle_connector.py
│   ├── github_connector.py
│   ├── huggingface_connector.py
│   └── servicenow_connector.py   # Optional read-only REST connector
├── eval_harness/             # Offline 5-dimension evaluation
│   ├── metrics.py            # grounding/relevance/citation-ID/refusal/latency
│   ├── runner.py             # runs the harness over the test set
│   ├── analyze.py            # honest averages (excl. correct refusals)
│   └── compare_runs.py       # before/after diff
├── finetune/
│   ├── prepare_dataset.py    # Manual fine-tuning JSONL preparation
│   ├── run_finetune.py       # Manual OpenAI fine-tuning upload/job script
│   ├── compare_models.py     # base-vs-finetuned comparison harness
│   └── FINETUNE_STATUS.md    # decision record + baseline
├── tests/                    # Fast regression tests + API-gated integration tests
├── .github/workflows/tests.yml   # CI: runs fast tests on every push
└── data/
    └── tickets.py            # 152 fictional internal-style support records
```

---

## Local Setup

Use Python **3.12.x**, matching CI and Railway. `requirements.txt` pins direct
packages and applies the generated `constraints-py312.txt` transitive pins.
The spaCy `en_core_web_sm` 3.8.0 model is installed with those requirements.

```bash
# Clone and setup
git clone https://github.com/Yasasu06/it-support-rag-agent
cd it-support-rag-agent
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

# Create a .env file in the project root with:
# OPENAI_API_KEY=sk-...
# Optional LangSmith tracing:
# LANGSMITH_API_KEY=ls-...
# LANGCHAIN_PROJECT=...
# KAGGLE_USERNAME=...
# KAGGLE_KEY=...
# GITHUB_TOKEN=ghp_...
# GITHUB_USERNAME=...

# Run the app
streamlit run app.py
```

The repository includes a tracked Chroma snapshot. See
[`index_manifest.json`](index_manifest.json) for its model, metric, record count,
and known source provenance. Railway startup validates this existing index,
requires `OPENAI_API_KEY`, and checks the installed spaCy PII model; it does not
fetch datasets or rebuild Chroma. The default `CHROMA_DB_PATH` is the
repository's `chroma_db/` directory, resolved
from the project root even when launched elsewhere. A Railway volume is optional:
set `CHROMA_DB_PATH` to its mount path only after placing a compatible,
non-empty index there. An empty mount hides the tracked snapshot at that path
and causes startup to fail. To rebuild deliberately, run `python3.12 ingest_all.py`
as a separate maintenance action; it calls external sources and the embedding
API and replaces the configured collection. The watcher and scheduler are also
manual utilities and are not launched by Railway. Query audit, feedback, live
evaluation, and visit logs are local JSONL files; their history is not retained
across redeploys unless storage for those paths is configured separately. To
intentionally refresh transitive pins, review direct versions in
`requirements.txt`, run `python3.12 refresh_constraints.py`, review the diff,
and validate the result in CI before deployment.

---

## Configuration

Deployment settings are centralized in [config.py](config.py) and overridable
via environment variables. The defaults select the documented chat model,
tracked index, and enabled feature flags. The System tab shows the active
configuration at runtime.

| Env var | Default | Controls |
|---------|---------|----------|
| `CHAT_MODEL` | `gpt-4o-mini` | Base LLM for answers, verification, reformulation, and scoring |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for retrieval (must match the model the index was built with) |
| `CHROMA_DB_PATH` | Repository `chroma_db/` | Existing index used by startup, retrieval, ingestion, and optional utilities; relative paths resolve from the repository root |
| `RETRIEVAL_K` | `3` | Number of tickets retrieved per query |
| `CONFIDENCE_HIGH_THRESHOLD` | `0.60` | Legacy configuration value; the chat UI does not display a High band |
| `CONFIDENCE_MEDIUM_THRESHOLD` | `0.20` | Raw retrieval relevance cutoff for Tier 2 review recommendation; not a probability |
| `MAX_RETRIEVAL_RETRIES` | `2` | Max adaptive-retry passes when an answer fails grounding or falls below the retrieval review cutoff |
| `RETRY_DELAY_SECONDS` | `1.0` | Delay between transient-failure retries in the error-handling layer |
| `LANGGRAPH_RECURSION_LIMIT` | `20` | LangGraph step budget for the multi-agent pipeline |
| `MAX_QUESTION_LENGTH` | `2000` | Max characters accepted for a user question before rejection |
| `ENABLE_LIVE_EVAL` | `true` | Scores and logs completed support-answer runs when enabled |
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

### ServiceNow connection checks

Configure `SERVICENOW_INSTANCE_URL` with the HTTPS instance address (without
a login or API path), plus `SERVICENOW_USERNAME` and `SERVICENOW_PASSWORD`
in the application environment. For the hosted demo, these settings belong
in Railway; the ignored local `.env` is not deployed from Git.

The System tab's connection button verifies incident API JSON and displays
the failure reason. A developer instance can return an HTTP 200 hibernation
page, which does not establish API access. If it is hibernating, sign in to
the [ServiceNow Developer portal](https://developer.servicenow.com/), open
**Manage my Instance**, select **Wake Instance**, and wait until the status
is **Online** before retrying. Compare the current instance URL and credentials
with Railway's settings if the instance has been replaced. HTTP 401 indicates
an authentication failure; HTTP 403 indicates denied incident read access.

---

## Testing

The project has fast tests that do not call external APIs and separate
API-gated integration tests for the live pipeline.

```bash
# Run the full suite (fast unit tests + integration tests).
# Integration tests run only if OPENAI_API_KEY is available (from your .env);
# otherwise they skip automatically.
python -m pytest tests/ -v

# Run only the fast tests — no API key needed.
python -m pytest tests/ -v --ignore=tests/test_integration.py
```

The fast tests cover normalization, privacy boundaries, index compatibility,
deployment preflight, routing, retries, rate limiting, image validation, and
fine-tuned-model selection without external API cost. Run collection locally
for the current test count; the suite continues to grow.

**Integration tests** (`test_integration.py`) exercise the real RAG and
multi-agent pipelines against OpenAI and auto-skip when no API key is present.

### Continuous Integration

A [GitHub Actions workflow](.github/workflows/tests.yml) runs the fast unit test
suite (`pytest tests/ --ignore=tests/test_integration.py`) automatically on
every push and pull request to `main` (Ubuntu, Python 3.12). Integration tests
are intentionally excluded from CI because they make external API calls and
incur cost. The badge at the top links to the workflow's current status.

---

## Evaluation Harness

Beyond the fixed answer/refusal benchmark, the optional manual evaluation
harness (`eval_harness/`) records five dimensions for each test case. LLM-judged
scores are estimates and can vary between runs:

| Dimension | Type | What it measures |
|-----------|------|------------------|
| **Groundedness** | LLM judge | How well is the answer supported by retrieved context? Skipped when no context exists. |
| **Relevance** | LLM judge | How well does the answer address the question? |
| **Citation-ID validity** | Pure logic | Do cited Ticket IDs appear in the retrieved source-ID list? This does not establish semantic citation correctness. |
| **Refusal correctness** | Pure logic | Did it answer in-scope questions and refuse out-of-scope ones? |
| **Latency** | Pure logic | Response time against a chat-interface threshold |

Separate dimensions make it easier to inspect grounding, relevance, and
unsupported source IDs without treating any one score as overall accuracy.

```bash
# Run the full harness (multiple LLM calls per question — takes a few minutes)
python3 eval_harness/runner.py

# Compare two runs (e.g. before/after a change) to see improvements/regressions
python3 eval_harness/compare_runs.py old.json new.json
```

Each run writes `eval_harness/harness_results.json` with a timestamp,
per-dimension averages, and per-question detail. This local artifact is not a
live production dashboard.

The harness output is a local generated file and is not included in the
repository. Citation-ID validity only checks IDs against retrieved sources;
it does not establish semantic support. `python3 eval_harness/analyze.py`
shows overall judge-score averages alongside averages excluding correct
out-of-scope refusals, while retaining false refusals in both views.

This is **separate from optional live evaluation** (`live_eval.py`), which
scores completed support-answer runs when enabled. The harness is for
deliberate, structured test runs against a fixed question set — the kind you run
before and after a change to measure impact.

---

## Key Design Decisions

**Why RAG over fine-tuning?**
RAG grounds answers in the incident database at query time. Fine-tuning bakes knowledge into model weights, making updates expensive. The repository provides a manual path to refresh the knowledge base.

**Why LangGraph over a single chain?**
The four support nodes (retrieval, answer, optional verification, triage) make the retry loop an explicit conditional edge. Analytical questions take a separate tool-agent path.

**Why ChromaDB?**
ChromaDB lets this deployment use a tracked local snapshot without operating a separate vector-search service. Moving to another search backend would require migration and compatibility checks.

**Retrieval routing**
The raw Chroma retrieval relevance signal is not a calibrated probability. The existing 0.20 cutoff guides Tier 2 review recommendations; the chat UI does not display High/Medium/Low confidence bands.

---

## Deployment Model & Future Extensions

Deliberate engineering decisions and prioritized gaps — documented for
transparency rather than hidden. Full rationale for the vector-store decision
lives in [DECISIONS.md](DECISIONS.md).

- **Vector store is pre-built and committed to git** (see [DECISIONS.md](DECISIONS.md)).
  Startup uses and validates this snapshot by default. The repository does not
  prove that a Railway volume is attached; an optional volume must contain a
  compatible index before `CHROMA_DB_PATH` is pointed at it. Refresh is manual.
- **ServiceNow contributes no records to the tracked snapshot.** The optional
  read connector requires external credentials and a reachable instance; its
  status in a deployed environment must be checked there.
- **Uneven source mix.** After cross-source deduplication the knowledge base is
  heavily GitHub-weighted (496 of 750); Kaggle contributes only 5 tickets.
- **Fine-tuning remains an extension.** Dataset preparation, upload safeguards,
  and a separate base-versus-candidate comparison harness are available. No
  fine-tuning job or fine-tuned production model is evidenced here; see
  [finetune/FINETUNE_STATUS.md](finetune/FINETUNE_STATUS.md).
- **Evaluation scope.** The saved 18/20 result measures expected answer/refusal
  behavior on one fixed set. LLM-judged grounding and relevance are separate
  estimates and vary by run.

---

## Built By

**Yasaswi**
