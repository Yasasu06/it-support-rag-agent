"""
Nightly knowledge base refresh scheduler for the IT Support RAG Agent.

Runs ingest_all.py automatically every night at 02:00 UTC and logs each
run's outcome to scheduler_log.jsonl.

Run directly to start the scheduler loop:
    python3 scheduler.py
"""

import schedule
import time
import subprocess
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SCHEDULER] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(__file__)
PYTHON = os.path.join(
    PROJECT_DIR, "venv", "bin", "python3"
)
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
            "stdout_tail": result.stdout[-500:],
            "stderr_tail": result.stderr[-200:]
                if result.stderr else ""
        }

        import json
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if success:
            logger.info(
                f"Refresh complete in {duration}s"
            )
        else:
            logger.error(
                f"Refresh failed: {result.stderr[-200:]}"
            )

    except subprocess.TimeoutExpired:
        logger.error("Ingestion timed out after 1 hour")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")


schedule.every().day.at("02:00").do(run_ingestion)


def run_scheduler():
    logger.info(
        "Scheduler started. "
        "Knowledge base refresh scheduled at 02:00 UTC"
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
