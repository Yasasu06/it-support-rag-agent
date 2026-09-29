"""
Converts the current internal-style ticket dataset into OpenAI JSONL examples.

Splits the resulting examples 80/20 into training_data.jsonl and
validation_data.jsonl for use with run_finetune.py.

Run with:
    python3 finetune/prepare_dataset.py
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(
    os.path.dirname(__file__)
))
from data.tickets import TICKETS
from security import sanitize_external_text

SYSTEM_PROMPT = (
    "You are an enterprise IT support assistant. "
    "Answer questions based on known IT incidents. "
    "Always cite the Ticket ID your answer is "
    "based on. Be specific and actionable."
)

def ticket_to_training_example(ticket: dict) -> dict:
    issue = sanitize_external_text(ticket["issue"])
    resolution = sanitize_external_text(ticket["resolution"])
    ticket_id = sanitize_external_text(str(ticket["ticket_id"]))
    user_message = (
        f"An employee reported this IT issue: "
        f"{issue} "
        f"What is the resolution?"
    )
    user_message = sanitize_external_text(user_message)

    assistant_message = (
        f"Based on Ticket ID {ticket_id}: "
        f"{resolution} "
        f"This issue was resolved in "
        f"{ticket['resolved_in_minutes']} minutes."
    )
    assistant_message = sanitize_external_text(assistant_message)

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "assistant",
                "content": assistant_message
            }
        ]
    }

def prepare_dataset() -> tuple[int, int]:
    """Write sanitized examples only when explicitly run, never on import."""
    examples = [ticket_to_training_example(ticket) for ticket in TICKETS]
    split = int(len(examples) * 0.8)
    train_examples, val_examples = examples[:split], examples[split:]

    output_path = os.path.join(os.path.dirname(__file__), "training_data.jsonl")
    validation_path = os.path.join(os.path.dirname(__file__), "validation_data.jsonl")
    with open(output_path, "w") as f:
        for example in train_examples:
            f.write(json.dumps(example) + "\n")
    with open(validation_path, "w") as f:
        for example in val_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    return len(train_examples), len(val_examples)


if __name__ == "__main__":
    prepare_dataset()
