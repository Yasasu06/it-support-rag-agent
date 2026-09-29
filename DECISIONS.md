# Architecture Decisions

## Vector Store Deployment Decision

chroma_db/ is intentionally committed to version control, which is not standard
practice for a generated data artifact. This is a documented tradeoff, not an
oversight.

**Why:** The default deployment cannot assume a persistent volume. Rebuilding
the 750-record mixed-source snapshot on every cold start would require dataset
access and embedding API calls, increasing startup time and external costs.

**Tradeoff accepted:** Shipping a pre-built vector store in git gives startup
an index without fetching datasets or calling the embedding API. Startup checks
that the configured collection is present and compatible; it fails clearly if
the index is missing or invalid. Updating the tracked snapshot requires a
deliberate manual re-ingest and commit.

**Optional future storage:** A Railway volume can be configured outside this
repository and selected with `CHROMA_DB_PATH` once it contains a compatible,
non-empty index. The repository does not prove that such a volume is attached.
Startup never rebuilds an empty volume automatically.
