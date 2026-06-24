"""
Downloads and processes the real ServiceNow IT helpdesk dataset from Kaggle.
"""

import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import logging

load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        ".env"
    )
)

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "real_data"
)

DATASET_ID = "suraj520/customer-support-ticket-dataset"
OUTPUT_FILE = os.path.join(
    DATA_DIR, "kaggle_tickets.json"
)

CATEGORY_MAP = {
    "Technical issue": "Hardware",
    "Network problem": "Network",
    "Software bug": "Software Access",
    "Account access": "Password",
    "Hardware failure": "Hardware",
    "Billing issue": "ERP Access",
    "Product inquiry": "Software Access",
    "Other": "Software Access"
}

def download_kaggle_dataset() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_dir = os.path.join(DATA_DIR, "kaggle_raw")
    os.makedirs(raw_dir, exist_ok=True)

    logger.info("Downloading Kaggle dataset...")

    import subprocess
    result = subprocess.run([
        "python3", "-m", "kaggle", "datasets",
        "download",
        "-d", DATASET_ID,
        "-p", raw_dir,
        "--unzip"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed: {result.stderr}"
        )

    logger.info("Download complete")

    csv_files = list(Path(raw_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found after download"
        )

    return str(csv_files[0])

def process_kaggle_data(csv_path: str) -> list:
    logger.info(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path)

    logger.info(
        f"Columns found: {list(df.columns)}"
    )
    logger.info(f"Total rows: {len(df)}")

    text_col = None
    resolution_col = None
    category_col = None

    for col in df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in
               ['description', 'subject', 'body',
                'message', 'text', 'issue']):
            if text_col is None:
                text_col = col
        if any(x in col_lower for x in
               ['resolution', 'response', 'solution',
                'answer', 'reply']):
            if resolution_col is None:
                resolution_col = col
        if any(x in col_lower for x in
               ['category', 'type', 'topic',
                'department', 'product']):
            if category_col is None:
                category_col = col

    if not text_col:
        text_col = df.columns[1] if len(
            df.columns) > 1 else df.columns[0]
    if not resolution_col:
        resolution_col = df.columns[-1]

    logger.info(
        f"Using: issue={text_col}, "
        f"resolution={resolution_col}, "
        f"category={category_col}"
    )

    tickets = []
    skipped = 0

    for idx, row in df.iterrows():
        issue = str(row.get(text_col, "")).strip()
        resolution = str(
            row.get(resolution_col, "")
        ).strip()

        if (not issue or not resolution or
            issue == "nan" or
            resolution == "nan" or
            len(issue) < 20 or
            len(resolution) < 20):
            skipped += 1
            continue

        raw_category = ""
        if category_col:
            raw_category = str(
                row.get(category_col, "")
            ).strip()

        mapped_category = CATEGORY_MAP.get(
            raw_category, "Software Access"
        )

        ticket_id = f"KGL-{idx+1:05d}"

        issue_clean = issue[:500].replace(
            '\n', ' '
        ).replace('\r', ' ')
        resolution_clean = resolution[:800].replace(
            '\n', ' '
        ).replace('\r', ' ')

        tickets.append({
            "ticket_id": ticket_id,
            "category": mapped_category,
            "issue": issue_clean,
            "resolution": resolution_clean,
            "resolved_in_minutes": 30,
            "source": "kaggle"
        })

        if len(tickets) >= 3000:
            break

    logger.info(
        f"Processed {len(tickets)} tickets "
        f"(skipped {skipped} incomplete)"
    )

    return tickets

def save_kaggle_tickets(tickets: list) -> str:
    with open(OUTPUT_FILE, "w") as f:
        json.dump(tickets, f, indent=2)
    logger.info(f"Saved to {OUTPUT_FILE}")
    return OUTPUT_FILE

def run_kaggle_connector() -> list:
    if os.path.exists(OUTPUT_FILE):
        logger.info(
            "Kaggle data already downloaded. "
            "Loading from cache..."
        )
        with open(OUTPUT_FILE) as f:
            tickets = json.load(f)
        logger.info(
            f"Loaded {len(tickets)} cached tickets"
        )
        return tickets

    csv_path = download_kaggle_dataset()
    tickets = process_kaggle_data(csv_path)
    save_kaggle_tickets(tickets)
    return tickets

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tickets = run_kaggle_connector()
    print(f"\nTotal Kaggle tickets: {len(tickets)}")
    print("Sample ticket:")
    print(json.dumps(tickets[0], indent=2))
