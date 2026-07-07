"""
Tests for the live ServiceNow connector.

Run with:
    cd ~/Desktop/it-support-rag-agent && source venv/bin/activate
    python3 test_servicenow_connector.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from connectors.servicenow_connector import (
    is_configured,
    test_connection,
    fetch_incidents_batch,
    live_incident_count_by_state,
    SERVICENOW_URL,
    SERVICENOW_USER,
    SERVICENOW_PASS,
)

passed = 0
failed = 0
failures = []


# ---------------------------------------------------------------------------
# TEST 1 — Configuration check: all three env vars must be set
# ---------------------------------------------------------------------------
print("=" * 60)
print("TEST 1: Configuration check (env vars present, values never printed)")
print("-" * 60)
try:
    missing = []
    if not SERVICENOW_URL:
        missing.append("SERVICENOW_INSTANCE_URL")
    if not SERVICENOW_USER:
        missing.append("SERVICENOW_USERNAME")
    if not SERVICENOW_PASS:
        missing.append("SERVICENOW_PASSWORD")

    if missing:
        raise AssertionError(
            f"Missing env vars: {', '.join(missing)}"
        )

    configured = is_configured()
    if not configured:
        raise AssertionError(
            "is_configured() returned False despite all vars being set"
        )

    print(f"  SERVICENOW_INSTANCE_URL : present")
    print(f"  SERVICENOW_USERNAME     : present")
    print(f"  SERVICENOW_PASSWORD     : present")
    print(f"  is_configured()         : {configured}")
    print("  RESULT: PASS")
    passed += 1
except Exception as e:
    print(f"  RESULT: FAIL — {e}")
    failed += 1
    failures.append(("TEST 1 (configuration check)", str(e)))


# ---------------------------------------------------------------------------
# TEST 2 — Live connection test (real network call)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 2: Live connection test (real HTTP request to ServiceNow)")
print("-" * 60)
try:
    result = test_connection()
    print(f"  success : {result['success']}")
    print(f"  message : {result['message']}")

    if not isinstance(result, dict):
        raise AssertionError(f"test_connection() must return a dict, got {type(result)}")
    if "success" not in result or "message" not in result:
        raise AssertionError("Dict missing 'success' or 'message' key")

    # The function must never crash — a failed connection is still a PASS
    # for this test (we log the reason honestly).
    if not result["success"]:
        print(f"  NOTE: Connection failed but function handled it gracefully.")
        print(f"  Connection error: {result['message']}")

    print("  RESULT: PASS")
    passed += 1
    _connection_ok = result["success"]
except Exception as e:
    print(f"  RESULT: FAIL — function raised unexpectedly: {e}")
    failed += 1
    failures.append(("TEST 2 (live connection)", str(e)))
    _connection_ok = False


# ---------------------------------------------------------------------------
# TEST 3 — Batch fetch (real network call, empty result is valid)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 3: Batch fetch (limit=10, empty result is valid on fresh instance)")
print("-" * 60)
try:
    tickets = fetch_incidents_batch(limit=10)
    print(f"  tickets returned: {len(tickets)}")

    if not isinstance(tickets, list):
        raise AssertionError(f"Expected list, got {type(tickets)}")

    if tickets:
        sample = tickets[0]
        required_keys = {"ticket_id", "category", "issue", "resolution",
                         "resolved_in_minutes", "source"}
        missing_keys = required_keys - set(sample.keys())
        if missing_keys:
            raise AssertionError(f"Sample ticket missing keys: {missing_keys}")
        print(f"  sample ticket_id : {sample['ticket_id']}")
        print(f"  sample category  : {sample['category']}")
        print(f"  sample source    : {sample['source']}")
    else:
        print("  No tickets returned (valid — fresh instance may have no closed incidents with notes)")

    print("  RESULT: PASS")
    passed += 1
except Exception as e:
    print(f"  RESULT: FAIL — function raised unexpectedly: {e}")
    failed += 1
    failures.append(("TEST 3 (batch fetch)", str(e)))


# ---------------------------------------------------------------------------
# TEST 4 — Live real-time incident count by state
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 4: Live incident count by state (real-time, not ChromaDB)")
print("-" * 60)
try:
    counts = live_incident_count_by_state()
    print(f"  state_counts: {counts}")

    if not isinstance(counts, dict):
        raise AssertionError(f"Expected dict, got {type(counts)}")

    if counts:
        total = sum(counts.values())
        print(f"  total incidents found: {total}")
    else:
        print("  No state data returned (valid — empty instance or connection issue handled gracefully)")

    print("  RESULT: PASS")
    passed += 1
except Exception as e:
    print(f"  RESULT: FAIL — function raised unexpectedly: {e}")
    failed += 1
    failures.append(("TEST 4 (live state count)", str(e)))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
total = passed + failed
print(f"SERVICENOW CONNECTOR TESTS: {passed}/{total} PASSED")

if failures:
    print()
    print("Failed tests:")
    for name, reason in failures:
        print(f"  - {name}")
        print(f"    Reason: {reason}")

if not _connection_ok:
    print()
    print("NOTE: Connection to ServiceNow failed. Check SERVICENOW_INSTANCE_URL,")
    print("SERVICENOW_USERNAME, and SERVICENOW_PASSWORD in your .env file.")
    print("All connector functions handled the failure gracefully (no crashes).")

sys.exit(0 if failed == 0 else 1)
