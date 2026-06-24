"""
PII detection, masking, and query audit logging for the IT Support RAG Agent.

This module handles two jobs:
    1. PII detection/masking — uses Microsoft Presidio to detect and mask
       personal information before any text enters the LLM.
    2. Query audit logging — appends a compliance log entry (JSONL) for
       every query, and summarizes that log for the app's sidebar.

Run directly to execute a small built-in test:
    python3 security.py
"""

import json
import os
from collections import Counter
from datetime import datetime

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

try:
    import streamlit as st
    if hasattr(st, 'secrets'):
        for key, value in st.secrets.items():
            if key not in os.environ:
                os.environ[key] = str(value)
except Exception:
    pass

# Presidio's AnalyzerEngine() defaults to the large spaCy model if no
# nlp_engine is given. Pin to the small model explicitly so it matches what's
# actually downloaded at startup (railway.toml), keeping memory usage down.
_nlp_engine = NlpEngineProvider(
    nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }
).create_engine()

analyzer = AnalyzerEngine(nlp_engine=_nlp_engine)
anonymizer = AnonymizerEngine()

LOG_FILE = "query_audit_log.jsonl"

PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "PERSON",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
]


# Job 1: PII detection and masking -------------------------------------------
def mask_pii(text: str) -> tuple[str, list]:
    """
    Detects and masks PII in text.
    Returns: (masked_text, list of detected PII types)
    """
    results = analyzer.analyze(
        text=text,
        entities=PII_ENTITIES,
        language="en",
    )

    if not results:
        return text, []

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
    )

    detected_types = list(set([r.entity_type for r in results]))
    return anonymized.text, detected_types


def is_safe_query(text: str) -> tuple[bool, list]:
    """
    Returns True if no PII detected.
    Returns False + list of PII types if PII found.
    """
    _, detected = mask_pii(text)
    return len(detected) == 0, detected


# Job 2: query audit logging --------------------------------------------------
def log_query(
    question: str,
    masked_question: str,
    answer: str,
    confidence_score: float,
    escalation: bool,
    tier: str,
    pii_detected: list,
    sources: list,
) -> None:
    """
    Appends one line per query to query_audit_log.jsonl
    JSONL format: one JSON object per line
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question_masked": masked_question,
        "pii_detected": pii_detected,
        "pii_found": len(pii_detected) > 0,
        "answer_length": len(answer),
        "confidence_score": round(confidence_score, 3),
        "escalation_required": escalation,
        "tier": tier,
        "source_ticket_ids": [s.get("ticket_id", "unknown") for s in sources],
        "session_id": os.getenv("STREAMLIT_SESSION_ID", "local"),
    }

    log_path = os.path.join(os.path.dirname(__file__), LOG_FILE)
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def get_audit_summary() -> dict:
    """
    Reads query_audit_log.jsonl and returns summary stats:
    - total queries
    - queries with PII detected
    - queries escalated to Tier 2
    - average confidence score
    - most common PII types detected
    """
    log_path = os.path.join(os.path.dirname(__file__), LOG_FILE)

    if not os.path.exists(log_path):
        return {
            "total_queries": 0,
            "pii_queries": 0,
            "escalated_queries": 0,
            "avg_confidence": 0,
            "common_pii_types": [],
        }

    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        return {
            "total_queries": 0,
            "pii_queries": 0,
            "escalated_queries": 0,
            "avg_confidence": 0,
            "common_pii_types": [],
        }

    pii_entries = [e for e in entries if e.get("pii_found")]
    escalated = [e for e in entries if e.get("escalation_required")]

    all_pii_types = []
    for e in pii_entries:
        all_pii_types.extend(e.get("pii_detected", []))

    pii_counts = Counter(all_pii_types)

    avg_conf = sum(e.get("confidence_score", 0) for e in entries) / len(entries)

    return {
        "total_queries": len(entries),
        "pii_queries": len(pii_entries),
        "escalated_queries": len(escalated),
        "avg_confidence": round(avg_conf, 3),
        "common_pii_types": pii_counts.most_common(3),
    }


# Simple built-in test ---------------------------------------------------------
if __name__ == "__main__":
    text1 = "My printer is not working"
    masked1, types1 = mask_pii(text1)
    print(f"No PII test: '{masked1}' | Types: {types1}")

    text2 = "john.smith@steelcase.com cannot print"
    masked2, types2 = mask_pii(text2)
    print(f"Email test: '{masked2}' | Types: {types2}")

    text3 = "Call me at 616-555-0123 about my VPN"
    masked3, types3 = mask_pii(text3)
    print(f"Phone test: '{masked3}' | Types: {types3}")

    # Exercise Job 2 (audit logging) so this test also produces a log entry.
    log_query(
        question=text2,
        masked_question=masked2,
        answer="Resolved by restarting the print spooler. (TKT-031)",
        confidence_score=0.25,
        escalation=False,
        tier="Tier 1",
        pii_detected=types2,
        sources=[{"ticket_id": "TKT-031"}],
    )
    print(f"Audit log test: wrote 1 entry to {LOG_FILE}")
    print(f"Audit summary: {get_audit_summary()}")
