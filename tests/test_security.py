"""
Fast unit tests for security.py.

These exercise the REAL Presidio/spaCy pipeline (local, no API cost) plus the
JSONL audit logging. The log-file path is redirected to a temp file via
monkeypatch so tests never touch the real query_audit_log.jsonl.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import security
from security import mask_pii, log_query, get_audit_summary


# --- mask_pii ----------------------------------------------------------------
def test_email_is_masked():
    original = "Please email john.doe@example.com about the outage"
    masked, types = mask_pii(original)
    assert masked != original
    assert "EMAIL_ADDRESS" in types


def test_phone_number_is_masked():
    original = "Call me at 616-555-0123 about my VPN"
    masked, types = mask_pii(original)
    # Presidio recognizers vary in the exact entity label; assert the text was
    # actually changed and at least one PII type was detected.
    assert masked != original
    assert len(types) > 0


def test_text_with_no_pii_is_unchanged():
    original = "The third floor printer is out of toner"
    masked, types = mask_pii(original)
    assert masked == original
    assert types == []


# --- get_audit_summary defaults ----------------------------------------------
def test_audit_summary_defaults_when_no_log_file(monkeypatch, tmp_path):
    missing = tmp_path / "does_not_exist.jsonl"
    monkeypatch.setattr(security, "LOG_FILE", str(missing))
    summary = get_audit_summary()
    assert summary["total_queries"] == 0
    assert summary["pii_queries"] == 0
    assert summary["escalated_queries"] == 0
    assert summary["avg_confidence"] == 0
    assert summary["common_pii_types"] == []


# --- log_query writes a valid JSON line --------------------------------------
def test_log_query_writes_valid_json_line(monkeypatch, tmp_path):
    log_path = tmp_path / "test_audit.jsonl"
    monkeypatch.setattr(security, "LOG_FILE", str(log_path))

    log_query(
        question="john.doe@example.com cannot print",
        masked_question="<EMAIL_ADDRESS> cannot print",
        answer="Restarted the print spooler (TKT-031)",
        confidence_score=0.256789,
        escalation=False,
        tier="Tier 1",
        pii_detected=["EMAIL_ADDRESS"],
        sources=[{"ticket_id": "TKT-031"}],
    )

    assert log_path.exists()
    lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1

    entry = json.loads(lines[0])  # must be valid JSON
    assert entry["question_masked"] == "<EMAIL_ADDRESS> cannot print"
    assert entry["pii_found"] is True
    assert entry["tier"] == "Tier 1"
    assert entry["confidence_score"] == 0.257  # rounded to 3 dp
    assert entry["source_ticket_ids"] == ["TKT-031"]


def test_get_audit_summary_reads_back_logged_entries(monkeypatch, tmp_path):
    log_path = tmp_path / "roundtrip_audit.jsonl"
    monkeypatch.setattr(security, "LOG_FILE", str(log_path))

    log_query("q1", "q1", "a1", 0.5, False, "Tier 1", [], [])
    log_query("q2", "q2", "a2", 0.1, True, "Tier 2", ["EMAIL_ADDRESS"], [])

    summary = get_audit_summary()
    assert summary["total_queries"] == 2
    assert summary["pii_queries"] == 1
    assert summary["escalated_queries"] == 1
