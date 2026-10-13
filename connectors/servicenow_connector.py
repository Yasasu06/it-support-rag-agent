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
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv
from error_handling import safe_error_type
from security import sanitize_external_text

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
    return bool(
        (SERVICENOW_URL or "").strip()
        and (SERVICENOW_USER or "").strip()
        and SERVICENOW_PASS
    )


def _get_base_url() -> str:
    """Accept an HTTPS instance origin without leaking configuration values."""
    base_url = (SERVICENOW_URL or "").strip().rstrip("/")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in parsed.netloc)
    ):
        raise ValueError("Invalid ServiceNow instance URL")
    # Accessing port also rejects malformed port values before a request.
    parsed.port
    return base_url


def _get_auth():
    return ((SERVICENOW_USER or "").strip(), SERVICENOW_PASS)


def test_connection() -> dict:
    """
    Tests the live ServiceNow connection with a minimal 1-record fetch.
    Returns a dict with 'success' bool and 'message' string.
    Never raises — always returns a safe dict even on total failure.
    """
    if not is_configured():
        return {
            "success": False,
            "message": (
                "ServiceNow is not configured. Set SERVICENOW_INSTANCE_URL, "
                "SERVICENOW_USERNAME, and SERVICENOW_PASSWORD in the application environment."
            ),
        }

    try:
        base_url = _get_base_url()
    except ValueError:
        return {
            "success": False,
            "message": (
                "SERVICENOW_INSTANCE_URL must be the HTTPS instance address "
                "without a login/API path, embedded credentials, query, or fragment."
            ),
        }

    try:
        url = f"{base_url}/api/now/table/incident"
        response = requests.get(
            url,
            auth=_get_auth(),
            params={"sysparm_limit": 1, "sysparm_fields": "sys_id"},
            headers={"Accept": "application/json"},
            timeout=25,
            allow_redirects=False,
        )

        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("result"), list):
                return {
                    "success": True,
                    "message": "Connected successfully to ServiceNow instance",
                }
            # Developer instances can return an HTTP 200 hibernation page.
            # Report the condition using fixed text, never the provider body.
            if "hibernating" in response.text.lower():
                return {
                    "success": False,
                    "message": (
                        "The ServiceNow instance is hibernating. Sign in to "
                        "developer.servicenow.com, open Manage my Instance, select "
                        "Wake Instance, and retry once its status is Online."
                    ),
                }
            return {
                "success": False,
                "message": (
                    "ServiceNow returned an unexpected response instead of incident API JSON. "
                    "Check the instance URL and whether the instance is Online."
                ),
            }
        elif 300 <= response.status_code < 400:
            return {
                "success": False,
                "message": (
                    "ServiceNow redirected the incident API request. Check the instance "
                    "URL and status; a login or redirect page does not confirm API access."
                ),
            }
        elif response.status_code == 401:
            return {
                "success": False,
                "message": (
                    "Authentication failed (HTTP 401). Check ServiceNow credentials."
                ),
            }
        elif response.status_code == 403:
            return {
                "success": False,
                "message": (
                    "ServiceNow denied incident read access (HTTP 403). "
                    "Check the account role and access rules."
                ),
            }
        else:
            return {
                "success": False,
                "message": (
                    f"ServiceNow request failed (HTTP {response.status_code})."
                ),
            }

    except requests.exceptions.SSLError:
        return {
            "success": False,
            "message": (
                "Could not verify the ServiceNow HTTPS certificate (TLS error). Check the instance "
                "URL and certificate configuration."
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "message": (
                "The ServiceNow connection timed out. Check the instance status "
                "in developer.servicenow.com and wake it if it is Hibernating, then retry."
            ),
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "message": "Could not reach ServiceNow. Check the instance URL and network.",
        }
    except Exception as e:
        logger.error("ServiceNow connection check failed: %s", safe_error_type(e))
        return {"success": False, "message": "ServiceNow connection check failed."}


def fetch_incidents_batch(limit: int = 100) -> list:
    """
    Batch fetch of closed incidents for ingestion into ChromaDB.
    Returns a list of ticket dicts in the standard schema, or [] on any failure.
    """
    if not is_configured():
        logger.warning("ServiceNow not configured - skipping batch fetch")
        return []

    try:
        url = f"{_get_base_url()}/api/now/table/incident"
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
            allow_redirects=False,
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
        logger.error("ServiceNow batch fetch error: %s", safe_error_type(e))
        return []


def live_lookup_incident(incident_number: str) -> Optional[dict]:
    """
    Real-time lookup of a specific incident by number from the live instance.
    Returns None if not found or on any error — never raises.
    """
    if not is_configured():
        return None

    try:
        url = f"{_get_base_url()}/api/now/table/incident"
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
            allow_redirects=False,
        )

        if response.status_code != 200:
            return None

        records = response.json().get("result", [])
        if not records:
            return None

        record = records[0]
        return {
            "number": sanitize_external_text(str(record.get("number") or "")),
            "short_description": sanitize_external_text(str(record.get("short_description") or "")),
            "state": sanitize_external_text(str(record.get("state") or "")),
            "category": sanitize_external_text(str(record.get("category") or "")),
            "close_notes": sanitize_external_text(str(record.get("close_notes") or "")),
        }

    except Exception as e:
        logger.error("Live lookup error: %s", safe_error_type(e))
        return None


def live_incident_count_by_state() -> dict:
    """
    Real-time count of incidents grouped by state from the live instance —
    not from ChromaDB. Returns empty dict on any failure.
    """
    if not is_configured():
        return {}

    try:
        url = f"{_get_base_url()}/api/now/table/incident"
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
            allow_redirects=False,
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
        logger.error("Live count error: %s", safe_error_type(e))
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

        print("\nTesting live incident count...")
        counts = live_incident_count_by_state()
        print(f"State counts: {counts}")
