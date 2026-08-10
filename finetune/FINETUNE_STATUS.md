# Fine-Tuning Status

## What is built

- **`prepare_dataset.py`** — converts the 150-ticket dataset into OpenAI
  fine-tuning JSONL, split 80/20 into `training_data.jsonl` (120 examples) and
  `validation_data.jsonl` (30 examples). Each example pairs a reported issue
  with a cited, resolution-grounded assistant answer.
- **`run_finetune.py`** — uploads the train/validation files and starts an
  OpenAI fine-tuning job (base `gpt-4o-mini-2024-07-18`, suffix
  `it-support-rag`, 3 epochs), then polls job status and records the resulting
  model ID.

## Why it is not deployed

Self-serve fine-tuning for the relevant model tier has not been available on the
account (OpenAI paused/gated self-serve access). The dataset and job scripts are
complete and validated; the blocker is **platform access, not system
readiness**.

## Loop Closure (Comparison Infrastructure)

While OpenAI's self-serve fine-tuning remains unavailable, the comparison and
deployment infrastructure is fully built and tested:

- **`finetune/compare_models.py`** — runs the full 20-question eval harness
  against any two models side by side, scoring groundedness, relevance,
  citation accuracy, refusal correctness, and latency independently. It reuses
  the **exact production answer prompt and retrieval** (only the model is
  swapped), so any measured delta is attributable to the model alone and the
  numbers are directly comparable to the production eval harness.
- **Current baseline (base model `gpt-4o-mini`, 2026-08-09):**

  | Dimension | Score |
  |-----------|-------|
  | groundedness | 0.720 |
  | relevance | 0.685 |
  | citation accuracy | 1.000 |
  | refusal correctness | 1.000 |
  | latency | 1.000 |

- **Production is ready to serve a fine-tuned model via two environment
  variables** — `USE_FINETUNED_MODEL=true` and `FINETUNED_MODEL_ID=<id>` — with
  zero code changes required. `config.active_chat_model()` selects the
  fine-tuned model when both are set, and falls back to the base model
  otherwise.
- **Workflow when access is restored** (OpenAI RFT, or a different provider):
  train → get model ID → run
  `python3 finetune/compare_models.py --finetuned <id>` → review the side-by-side
  comparison → set the two env vars **only if** the fine-tuned model wins on the
  metrics that matter. The comparison is the gate; the swap is a config flip.
