"""
Fast unit tests for normalize.py — pure logic, no API calls, no network.

mask_pii (Presidio, local but slow-ish) is mocked with a pass-through so these
tests are deterministic and run instantly. Everything else is exercised for
real: validity filtering, category inference, and cross-source deduplication.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import normalize
from normalize import (
    is_valid_ticket,
    infer_category,
    normalize_all_sources,
)


@pytest.fixture(autouse=True)
def passthrough_mask_pii(monkeypatch):
    """Replace mask_pii with a no-op so normalization needs zero PII work."""
    monkeypatch.setattr(normalize, "mask_pii", lambda text: (text, []))


# --- is_valid_ticket ---------------------------------------------------------
def test_ticket_with_short_issue_is_rejected():
    ticket = {"issue": "too short", "resolution": "a properly long resolution text"}
    assert is_valid_ticket(ticket) is False


def test_ticket_with_short_resolution_is_rejected():
    ticket = {"issue": "a properly long issue description here", "resolution": "nope"}
    assert is_valid_ticket(ticket) is False


def test_valid_ticket_passes_validation():
    ticket = {
        "issue": "User cannot connect to the corporate VPN from home",
        "resolution": "Reset the VPN profile and reissued fresh credentials",
    }
    assert is_valid_ticket(ticket) is True


def test_identical_issue_and_resolution_is_rejected():
    text = "the exact same sentence repeated as both fields here"
    assert is_valid_ticket({"issue": text, "resolution": text}) is False


# --- infer_category ----------------------------------------------------------
def test_category_inference_vpn_keyword():
    assert infer_category("Cannot start the VPN tunnel", "Reset anyconnect") == "VPN"


def test_category_inference_printer_keyword():
    assert infer_category("The office printer is jammed", "Cleared the toner") == "Printer"


def test_category_inference_defaults_to_software_access():
    # No mapped keyword anywhere -> documented default.
    assert infer_category("zzz qqq", "wxyz") == "Software Access"


def test_existing_valid_category_is_preserved():
    # A ticket already tagged with a valid category keeps it, ignoring text.
    assert infer_category("printer jam", "toner", existing_category="VPN") == "VPN"


# --- deduplication via normalize_all_sources ---------------------------------
def test_duplicate_issue_text_is_deduplicated():
    dup = {
        "ticket_id": "T-1",
        "category": "VPN",
        "issue": "User cannot connect to the corporate VPN from home office",
        "resolution": "Reset the VPN profile and reissued fresh credentials",
        "resolved_in_minutes": 20,
        "source": "synthetic",
    }
    # Same issue text submitted twice — only one should survive.
    result = normalize_all_sources([dup, dict(dup, ticket_id="T-2")], [], [])
    assert len(result) == 1


def test_distinct_tickets_are_both_kept():
    t1 = {
        "ticket_id": "T-1",
        "category": "VPN",
        "issue": "User cannot connect to the corporate VPN from home office",
        "resolution": "Reset the VPN profile and reissued fresh credentials",
        "resolved_in_minutes": 20,
        "source": "synthetic",
    }
    t2 = {
        "ticket_id": "T-2",
        "category": "Printer",
        "issue": "The third floor printer will not pick up paper from tray two",
        "resolution": "Cleared the jam and replaced the worn pickup roller",
        "resolved_in_minutes": 15,
        "source": "synthetic",
    }
    result = normalize_all_sources([t1, t2], [], [])
    assert len(result) == 2
