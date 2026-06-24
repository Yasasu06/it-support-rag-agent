"""
Downloads and processes the synthetic ServiceNow IT support dataset from
Hugging Face, which has properly structured resolution fields.
"""

import os
import json
import logging
from dotenv import load_dotenv

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
OUTPUT_FILE = os.path.join(
    DATA_DIR, "huggingface_tickets.json"
)

CATEGORY_MAP = {
    "network": "Network",
    "hardware": "Hardware",
    "software": "Software Access",
    "access": "Password",
    "email": "Email",
    "printer": "Printer",
    "vpn": "VPN",
    "erp": "ERP Access",
    "database": "ERP Access",
    "security": "Password",
    "infrastructure": "Network",
    "application": "Software Access"
}

def map_category(raw_category: str) -> str:
    if not raw_category:
        return "Software Access"
    raw_lower = raw_category.lower().strip()
    for key, value in CATEGORY_MAP.items():
        if key in raw_lower:
            return value
    return "Software Access"

def run_huggingface_connector() -> list:
    if os.path.exists(OUTPUT_FILE):
        logger.info(
            "HuggingFace data already downloaded. "
            "Loading from cache..."
        )
        with open(OUTPUT_FILE) as f:
            tickets = json.load(f)
        logger.info(
            f"Loaded {len(tickets)} cached tickets"
        )
        return tickets

    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info(
        "Downloading ServiceNow dataset "
        "from Hugging Face..."
    )

    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "6StringNinja/synthetic-servicenow-incidents",
            split="train"
        )

        logger.info(
            f"Downloaded {len(dataset)} records"
        )
        logger.info(
            f"Columns: {dataset.column_names}"
        )

        tickets = []
        skipped = 0

        for idx, row in enumerate(dataset):
            issue = ""
            resolution = ""
            category_raw = ""

            for field in [
                'short_description',
                'description',
                'issue',
                'problem',
                'title'
            ]:
                if row.get(field) and \
                        str(row[field]).strip() and \
                        str(row[field]) != "nan":
                    issue = str(row[field]).strip()
                    break

            for field in [
                'resolution',
                'close_notes',
                'solution',
                'resolution_notes',
                'close_code'
            ]:
                if row.get(field) and \
                        str(row[field]).strip() and \
                        str(row[field]) != "nan":
                    resolution = str(
                        row[field]
                    ).strip()
                    break

            for field in [
                'category',
                'subcategory',
                'type',
                'incident_type'
            ]:
                if row.get(field) and \
                        str(row[field]).strip() and \
                        str(row[field]) != "nan":
                    category_raw = str(
                        row[field]
                    ).strip()
                    break

            if (not issue or not resolution or
                    len(issue) < 20 or
                    len(resolution) < 20):
                skipped += 1
                continue

            mapped_category = map_category(
                category_raw
            )

            resolved_minutes = 30
            for field in [
                'time_to_resolve',
                'resolution_time',
                'resolved_in_minutes'
            ]:
                if row.get(field):
                    try:
                        val = float(row[field])
                        if 5 <= val <= 480:
                            resolved_minutes = int(val)
                        break
                    except (ValueError, TypeError):
                        pass

            tickets.append({
                "ticket_id": f"HF-{idx+1:05d}",
                "category": mapped_category,
                "issue": issue[:600],
                "resolution": resolution[:800],
                "resolved_in_minutes": resolved_minutes,
                "source": "huggingface_servicenow"
            })

            if len(tickets) >= 500:
                break

        logger.info(
            f"Processed {len(tickets)} tickets "
            f"(skipped {skipped} incomplete)"
        )

        with open(OUTPUT_FILE, "w") as f:
            json.dump(tickets, f, indent=2)

        logger.info(f"Saved to {OUTPUT_FILE}")
        return tickets

    except Exception as e:
        logger.error(
            f"HuggingFace download failed: {e}"
        )
        logger.info(
            "Falling back to IT incident log dataset..."
        )
        return _fallback_kaggle_download()

def _fallback_kaggle_download() -> list:
    import subprocess
    from pathlib import Path
    import pandas as pd

    raw_dir = os.path.join(
        DATA_DIR, "hf_fallback_raw"
    )
    os.makedirs(raw_dir, exist_ok=True)

    datasets_to_try = [
        "shamiulislamshifat/it-incident-log-dataset",
        "macimottin/it-incidents"
    ]

    for dataset_id in datasets_to_try:
        try:
            result = subprocess.run([
                "python3", "-m", "kaggle",
                "datasets", "download",
                "-d", dataset_id,
                "-p", raw_dir,
                "--unzip"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                csv_files = list(
                    Path(raw_dir).glob("*.csv")
                )
                if csv_files:
                    df = pd.read_csv(
                        str(csv_files[0])
                    )
                    logger.info(
                        f"Fallback dataset columns: "
                        f"{list(df.columns)}"
                    )
                    tickets = []
                    for idx, row in df.iterrows():
                        issue = ""
                        resolution = ""
                        for col in df.columns:
                            cl = col.lower()
                            if any(x in cl for x in [
                                'description',
                                'short_desc',
                                'issue', 'title'
                            ]):
                                val = str(
                                    row.get(col, "")
                                ).strip()
                                if len(val) > 20:
                                    issue = val[:600]
                                    break
                        for col in df.columns:
                            cl = col.lower()
                            if any(x in cl for x in [
                                'resolution',
                                'close_note',
                                'solution',
                                'fix'
                            ]):
                                val = str(
                                    row.get(col, "")
                                ).strip()
                                if len(val) > 20:
                                    resolution = val[:800]
                                    break
                        if issue and resolution:
                            tickets.append({
                                "ticket_id": f"FB-{idx+1:05d}",
                                "category": "Software Access",
                                "issue": issue,
                                "resolution": resolution,
                                "resolved_in_minutes": 30,
                                "source": "kaggle_it_incidents"
                            })
                        if len(tickets) >= 2000:
                            break
                    if tickets:
                        with open(OUTPUT_FILE, "w") as f:
                            json.dump(tickets, f)
                        return tickets
        except Exception as ex:
            logger.error(
                f"Fallback {dataset_id} failed: {ex}"
            )

    logger.warning(
        "All downloads failed. "
        "Returning empty list."
    )
    return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tickets = run_huggingface_connector()
    print(f"\nTotal HuggingFace tickets: {len(tickets)}")
    if tickets:
        print("Sample ticket:")
        import json
        print(json.dumps(tickets[0], indent=2))
