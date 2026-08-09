# Architecture Decisions

## Vector Store Deployment Decision

chroma_db/ is intentionally committed to version control, which is not standard
practice for a generated data artifact. This is a documented tradeoff, not an
oversight.

**Why:** Railway's ephemeral filesystem does not persist data across deploys by
default. Setting up a reliable persistent volume with conditional re-ingestion
was attempted but proved unreliable within Railway's startup time window, given
the embedding API calls required to rebuild ~750 tickets from 5 live/batch
sources on every cold start.

**Tradeoff accepted:** Shipping a pre-built vector store in git guarantees the
deployed app always has data, at the cost of repo cleanliness (an ~11MB binary
in git history) and requiring a manual re-ingest + re-commit whenever the
knowledge base is updated.

**Planned proper fix:** Migrate to a Railway persistent volume with start.sh's
existing data-presence check (already implemented) correctly skipping
re-ingestion when the volume has data. This requires dedicated testing time to
avoid breaking the live deployment during the transition, and is intentionally
deprioritized behind feature work.
