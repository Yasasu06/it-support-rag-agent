# Fine-Tuning Status

## Available workflow

- **`prepare_dataset.py`** manually converts the internal-style support records
  into sanitized OpenAI JSONL examples. The current 152-record source yields
  121 training and 31 validation examples with its 80/20 split. Older local
  generated files may have different counts; the JSONL outputs are not tracked.
- **`run_finetune.py`** validates every JSONL record before an optional upload,
  then can submit and poll an OpenAI fine-tuning job when deliberately run with
  the required account access and credentials.
- **`compare_models.py`** can score a base model and a supplied candidate model
  on the fixed 20-question set. It uses the configured Chroma collection, the
  answer prompt, and the evaluation metrics for grounding, relevance,
  citation-ID validity, refusal behavior, and latency. This is a separate RAG
  comparison path, not the full deployed LangGraph workflow; its results do not
  isolate every possible runtime difference to model choice.

## Current deployment status

The repository contains no submitted fine-tuning job record or active
fine-tuned model ID. The default application uses `CHAT_MODEL`. Configuration
can select a candidate model when both `USE_FINETUNED_MODEL=true` and
`FINETUNED_MODEL_ID` are set, but that path should be evaluated before use.
Current account eligibility or platform access cannot be established from this
repository.

To explore this extension, prepare the dataset, submit a job only when access
is available, compare the resulting model with the base model, and review the
results before changing serving configuration. The comparison harness is an
evaluation aid, not evidence that a fine-tuned model has been deployed.
