"""
Private, privacy-preserving visit analytics for the IT Support RAG Agent.

Logs only a UTC timestamp and a random per-session id — no IP, name, email, or
any personally identifying information. The session id is an anonymous UUID
generated per browser session, not tied to any real identity. Failures never
break the app.
"""

import json
import os
from datetime import datetime, timezone

ANALYTICS_LOG = os.path.join(
    os.path.dirname(__file__), "visit_log.jsonl"
)


def log_visit(session_id: str) -> None:
    """
    Logs a single visit with timestamp only - no IP,
    no name, no email, no personally identifying
    information. session_id is a random anonymous
    identifier generated per browser session, not tied
    to any real identity.
    """
    entry = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "session_id": session_id
    }
    try:
        with open(ANALYTICS_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # analytics failure must never break the app


def get_analytics_summary() -> dict:
    """
    Returns aggregate visit stats. Safe defaults if no
    log exists yet.
    """
    if not os.path.exists(ANALYTICS_LOG):
        return {
            "total_visits": 0,
            "unique_sessions": 0,
            "visits_today": 0,
            "recent_visits": []
        }

    entries = []
    try:
        with open(ANALYTICS_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception:
        return {
            "total_visits": 0,
            "unique_sessions": 0,
            "visits_today": 0,
            "recent_visits": []
        }

    if not entries:
        return {
            "total_visits": 0,
            "unique_sessions": 0,
            "visits_today": 0,
            "recent_visits": []
        }

    today = datetime.now(timezone.utc).date()
    visits_today = sum(
        1 for e in entries
        if datetime.fromisoformat(
            e["timestamp"]
        ).date() == today
    )

    unique_sessions = len(set(
        e["session_id"] for e in entries
    ))

    recent = sorted(
        entries, key=lambda e: e["timestamp"],
        reverse=True
    )[:20]

    return {
        "total_visits": len(entries),
        "unique_sessions": unique_sessions,
        "visits_today": visits_today,
        "recent_visits": recent
    }
