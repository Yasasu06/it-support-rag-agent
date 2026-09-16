"""
Live ServiceNow REST API connector for the IT Support RAG Agent.

Supports both batch ingestion (building ChromaDB) and real-time lookup
(tool agent queries during a conversation). Gracefully no-ops when
credentials are not configured so the rest of the app is never broken
by an absent ServiceNow instance.
"""

import logging
import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
)

logger = logging.getLogger(__name__)

SERVICENOW_URL = os.getenv("SERVICENOW_INSTANCE_URL")
SERVICENOW_USER = os.getenv("SERVICENOW_USERNAME")
SERVICENOW_PASS = os.getenv("SERVICENOW_PASSWORD")

CATEGORY_MAP = {
    "network": "Network",
    "hardware": "Hardware",
    "software": "Software Access",
    "inquiry / help": "Software Access",
    "database": "ERP Access",
    "email": "Email",
}


def is_configured() -> bool:
    """
    Returns True only if all three ServiceNow credentials are present.
    Never crashes if missing — callers check this before making any requests.
    """
    return bool(SERVICENOW_URL and SERVICENOW_USER and SERVICENOW_PASS)


def _get_auth():
    return (SERVICENOW_USER, SERVICENOW_PASS)


def _extract_error_detail(response: "requests.Response") -> str:
    """
    Pulls ServiceNow's own error detail out of the response body when present.
    ServiceNow error responses are normally JSON like:
        {"error": {"message": "...", "detail": "..."}, "status": "failure"}
    but a WAF/proxy block page can return HTML instead - fall back to a
    truncated raw-text snippet in that case so the real cause is never hidden
    behind a bare status code. Never raises.
    """
    try:
        body = response.json()
        err = body.get("error", {})
        message = err.get("message") or body.get("status")
        detail = err.get("detail")
        if message and detail:
            return f"{message}: {detail}"
        if message:
            return str(message)
    except Exception:
        pass
    snippet = (response.text or "").strip().replace("\n", " ")[:200]
    return snippet or "no error body returned"


def test_connection() -> dict:
    """
    Tests the live ServiceNow connection with a minimal 1-record fetch.
    Returns a dict with 'success' bool and 'message' string.
    Never raises — always returns a safe dict even on total failure.
    """
    if not is_configured():
        return {
            "success": False,
            "message": "ServiceNow credentials not configured in .env",
        }

    # Defensively strip a trailing slash so we never send a double-slash path
    # like https://instance.service-now.com//api/now/... if the env var was
    # copied with one. (Verified this doesn't actually break ServiceNow's
    # gateway, but it's still wrong and worth normalizing.)
    base_url = SERVICENOW_URL.rstrip("/")

    try:
        url = f"{base_url}/api/now/table/incident?sysparm_limit=1"
        response = requests.get(
            url,
            auth=_get_auth(),
            headers={"Accept": "application/json"},
            # Bumped from 10s: ServiceNow Personal Developer Instances
            # hibernate after inactivity and can take well over 10s to wake on
            # the first request after being asleep.
            timeout=25,
        )

        if response.status_code == 200:
            return {
                "success": True,
                "message": "Connected successfully to ServiceNow instance",
            }
        elif response.status_code == 401:
            return {
                "success": False,
                "message": (
                    "Authentication failed (401) - check SERVICENOW_USERNAME/"
                    f"SERVICENOW_PASSWORD. ServiceNow said: {_extract_error_detail(response)}"
                ),
            }
        elif response.status_code == 403:
            return {
                "success": False,
                "message": (
                    "Forbidden (403) - credentials are valid but this user "
                    "lacks permission to read the incident table (check the "
                    "user has the 'itil' role or an ACL granting incident "
                    f"read access on this instance). ServiceNow said: {_extract_error_detail(response)}"
                ),
            }
        else:
            return {
                "success": False,
                "message": (
                    f"Unexpected status code: {response.status_code}. "
                    f"ServiceNow said: {_extract_error_detail(response)}"
                ),
            }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "message": (
                "Connection timed out after 25s - if this is a fresh/idle "
                "ServiceNow dev instance it may be hibernating; open it in a "
                "browser once to wake it up, then retry."
            ),
        }
    except requests.exceptions.ConnectionError as e:
        return {
            "success": False,
            "message": f"Could not reach ServiceNow instance - check SERVICENOW_INSTANCE_URL ({e})",
        }
    except Exception as e:
        return {"success": False, "message": f"Unexpected error: {str(e)}"}


def fetch_incidents_batch(limit: int = 100) -> list:
    """
    Batch fetch of closed incidents for ingestion into ChromaDB.
    Returns a list of ticket dicts in the standard schema, or [] on any failure.
    """
    if not is_configured():
        logger.warning("ServiceNow not configured - skipping batch fetch")
        return []

    try:
        url = f"{SERVICENOW_URL.rstrip('/')}/api/now/table/incident"
        params = {
            "sysparm_limit": limit,
            "sysparm_query": "active=false",
            "sysparm_fields": (
                "number,short_description,description,"
                "close_notes,category,sys_created_on"
            ),
        }

        response = requests.get(
            url,
            auth=_get_auth(),
            params=params,
            headers={"Accept": "application/json"},
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(
                f"ServiceNow batch fetch failed: {response.status_code}"
            )
            return []

        records = response.json().get("result", [])
        tickets = []

        for record in records:
            short_desc = record.get("short_description", "").strip()
            description = record.get("description", "").strip()
            close_notes = record.get("close_notes", "").strip()

            issue = short_desc or description
            resolution = close_notes

            if not issue or not resolution or len(issue) < 15 or len(resolution) < 15:
                continue

            raw_category = record.get("category", "").lower()
            category = CATEGORY_MAP.get(raw_category, "Software Access")

            ticket_number = record.get("number", "UNKNOWN")
            tickets.append({
                "ticket_id": f"SNOW-{ticket_number}",
                "category": category,
                "issue": issue[:600],
                "resolution": resolution[:800],
                "resolved_in_minutes": 30,
                "source": "servicenow_live",
            })

        logger.info(
            f"Fetched {len(tickets)} valid tickets from ServiceNow "
            f"(of {len(records)} raw records)"
        )
        return tickets

    except Exception as e:
        logger.error(f"ServiceNow batch fetch error: {e}")
        return []


def live_lookup_incident(incident_number: str) -> Optional[dict]:
    """
    Real-time lookup of a specific incident by number from the live instance.
    Returns None if not found or on any error — never raises.
    """
    if not is_configured():
        return None

    try:
        url = f"{SERVICENOW_URL.rstrip('/')}/api/now/table/incident"
        params = {
            "sysparm_query": f"number={incident_number}",
            "sysparm_limit": 1,
        }

        response = requests.get(
            url,
            auth=_get_auth(),
            params=params,
            headers={"Accept": "application/json"},
            timeout=10,
        )

        if response.status_code != 200:
            return None

        records = response.json().get("result", [])
        if not records:
            return None

        record = records[0]
        return {
            "number": record.get("number"),
            "short_description": record.get("short_description"),
            "state": record.get("state"),
            "category": record.get("category"),
            "close_notes": record.get("close_notes"),
        }

    except Exception as e:
        logger.error(f"Live lookup error: {e}")
        return None


def live_incident_count_by_state() -> dict:
    """
    Real-time count of incidents grouped by state from the live instance —
    not from ChromaDB. Returns empty dict on any failure.
    """
    if not is_configured():
        return {}

    try:
        url = f"{SERVICENOW_URL.rstrip('/')}/api/now/table/incident"
        params = {
            "sysparm_limit": 1000,
            "sysparm_fields": "state",
        }

        response = requests.get(
            url,
            auth=_get_auth(),
            params=params,
            headers={"Accept": "application/json"},
            timeout=15,
        )

        if response.status_code != 200:
            return {}

        records = response.json().get("result", [])
        state_counts: dict = {}
        for record in records:
            state = record.get("state", "unknown")
            state_counts[state] = state_counts.get(state, 0) + 1

        return state_counts

    except Exception as e:
        logger.error(f"Live count error: {e}")
        return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing ServiceNow connection...")
    result = test_connection()
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")

    if result["success"]:
        print("\nFetching sample batch (5 records)...")
        tickets = fetch_incidents_batch(limit=5)
        print(f"Got {len(tickets)} valid tickets")
        if tickets:
            print(f"Sample: {tickets[0]}")

        print("\nTesting live incident count...")
        counts = live_incident_count_by_state()
        print(f"State counts: {counts}")
