"""
Converts the 150-ticket dataset into OpenAI fine-tuning JSONL format.

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

SYSTEM_PROMPT = (
    "You are an enterprise IT support assistant. "
    "Answer questions based on known IT incidents. "
    "Always cite the Ticket ID your answer is "
    "based on. Be specific and actionable."
)

def ticket_to_training_example(ticket: dict) -> dict:
    user_message = (
        f"An employee reported this IT issue: "
        f"{ticket['issue']} "
        f"What is the resolution?"
    )

    assistant_message = (
        f"Based on Ticket ID {ticket['ticket_id']}: "
        f"{ticket['resolution']} "
        f"This issue was resolved in "
        f"{ticket['resolved_in_minutes']} minutes."
    )

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

os.makedirs(
    os.path.dirname(__file__), exist_ok=True
)

output_path = os.path.join(
    os.path.dirname(__file__),
    "training_data.jsonl"
)

examples = [
    ticket_to_training_example(t)
    for t in TICKETS
]

with open(output_path, "w") as f:
    for example in examples:
        f.write(json.dumps(example) + "\n")

print(f"Created {len(examples)} training examples")
print(f"Saved to: {output_path}")

validation_path = os.path.join(
    os.path.dirname(__file__),
    "validation_data.jsonl"
)

split = int(len(examples) * 0.8)
train_examples = examples[:split]
val_examples = examples[split:]

with open(output_path, "w") as f:
    for ex in train_examples:
        f.write(json.dumps(ex) + "\n")

with open(validation_path, "w") as f:
    for ex in val_examples:
        f.write(json.dumps(ex) + "\n")

print(f"Training examples: {len(train_examples)}")
print(
    f"Validation examples: {len(val_examples)}"
)
