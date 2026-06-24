"""
Uploads the fine-tuning dataset and starts an OpenAI fine-tuning job.

Fine-tuning runs asynchronously on OpenAI's servers (10-20 minutes).
Run without arguments to start a new job, or with "status" to check
the most recently started job.

Run with:
    python3 finetune/run_finetune.py
    python3 finetune/run_finetune.py status
"""

import openai
import os
import json
import time
from dotenv import load_dotenv

load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        ".env"
    )
)

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

FINETUNE_DIR = os.path.dirname(__file__)

def upload_training_file(filepath: str) -> str:
    print(f"Uploading {filepath}...")
    with open(filepath, "rb") as f:
        response = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    file_id = response.id
    print(f"Uploaded. File ID: {file_id}")
    return file_id

def start_finetune_job(
    train_file_id: str,
    val_file_id: str
) -> str:
    print("Starting fine-tune job...")
    job = client.fine_tuning.jobs.create(
        training_file=train_file_id,
        validation_file=val_file_id,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={
            "n_epochs": 3
        },
        suffix="it-support-rag"
    )
    job_id = job.id
    print(f"Job started. Job ID: {job_id}")
    print(
        f"Status: {job.status}"
    )
    print(
        "Fine-tuning runs asynchronously. "
        "Check status at: "
        "platform.openai.com/finetune"
    )

    status_path = os.path.join(
        FINETUNE_DIR, "finetune_job.json"
    )
    with open(status_path, "w") as f:
        json.dump({
            "job_id": job_id,
            "train_file_id": train_file_id,
            "val_file_id": val_file_id,
            "status": job.status,
            "model": "gpt-4o-mini-2024-07-18",
            "suffix": "it-support-rag"
        }, f, indent=2)

    print(
        f"Job details saved to finetune_job.json"
    )
    return job_id

def check_job_status(job_id: str) -> None:
    job = client.fine_tuning.jobs.retrieve(job_id)
    print(f"Job ID: {job_id}")
    print(f"Status: {job.status}")
    if job.fine_tuned_model:
        print(
            f"Model ready: {job.fine_tuned_model}"
        )
        model_path = os.path.join(
            FINETUNE_DIR, "finetuned_model.txt"
        )
        with open(model_path, "w") as f:
            f.write(job.fine_tuned_model)
        print(f"Model ID saved to finetuned_model.txt")
    else:
        print(
            "Model not ready yet. "
            "Check again in 10-15 minutes."
        )

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        status_path = os.path.join(
            FINETUNE_DIR, "finetune_job.json"
        )
        if os.path.exists(status_path):
            with open(status_path) as f:
                data = json.load(f)
            check_job_status(data["job_id"])
        else:
            print(
                "No job found. "
                "Run without arguments first."
            )
    else:
        train_path = os.path.join(
            FINETUNE_DIR, "training_data.jsonl"
        )
        val_path = os.path.join(
            FINETUNE_DIR, "validation_data.jsonl"
        )

        if not os.path.exists(train_path):
            print(
                "Training data not found. "
                "Run prepare_dataset.py first."
            )
            sys.exit(1)

        train_id = upload_training_file(train_path)
        val_id = upload_training_file(val_path)
        start_finetune_job(train_id, val_id)
