"""
Optional knowledge base refresh scheduler for the IT Support RAG Agent.

When started manually, runs ingest_all.py at 02:00 in the host's local time
and logs each run's outcome to scheduler_log.jsonl.

Run directly to start the scheduler loop:
    python3 scheduler.py
"""

import schedule
import time
import subprocess
import logging
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from error_handling import safe_error_type

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SCHEDULER] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(__file__)
PYTHON = sys.executable
INGEST_SCRIPT = os.path.join(
    PROJECT_DIR, "ingest_all.py"
)
LOG_FILE = os.path.join(
    PROJECT_DIR, "scheduler_log.jsonl"
)


def run_ingestion():
    logger.info(
        "Starting scheduled knowledge base refresh..."
    )
    start_time = datetime.utcnow()

    try:
        result = subprocess.run(
            [PYTHON, INGEST_SCRIPT],
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=PROJECT_DIR
        )

        success = result.returncode == 0
        duration = (
            datetime.utcnow() - start_time
        ).seconds

        log_entry = {
            "timestamp": start_time.isoformat(),
            "success": success,
            "duration_seconds": duration,
            "exit_code": result.returncode,
        }

        import json
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if success:
            logger.info(
                f"Refresh complete in {duration}s"
            )
        else:
            logger.error("Refresh failed with exit code %s", result.returncode)

    except subprocess.TimeoutExpired:
        logger.error("Ingestion timed out after 1 hour")
    except Exception as e:
        logger.error("Scheduler error: %s", safe_error_type(e))


schedule.every().day.at("02:00").do(run_ingestion)


def run_scheduler():
    logger.info(
        "Scheduler started. "
        "Knowledge base refresh scheduled at 02:00 host local time"
    )
    logger.info(
        f"Next run: "
        f"{schedule.next_run()}"
    )

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    run_scheduler()
