"""
Takes tickets from all sources and converts them to a consistent schema.
"""

import json
import os
import re
import logging
from typing import Optional
from security import mask_pii

logger = logging.getLogger(__name__)

VALID_CATEGORIES = [
    "VPN", "Password", "Software Access",
    "Hardware", "Email", "Printer",
    "Network", "ERP Access"
]

KEYWORD_CATEGORY_MAP = {
    "vpn": "VPN",
    "anyconnect": "VPN",
    "cisco": "VPN",
    "tunnel": "VPN",
    "password": "Password",
    "locked": "Password",
    "reset": "Password",
    "login": "Password",
    "credentials": "Password",
    "printer": "Printer",
    "print": "Printer",
    "toner": "Printer",
    "network": "Network",
    "wifi": "Network",
    "ethernet": "Network",
    "dns": "Network",
    "dhcp": "Network",
    "outlook": "Email",
    "email": "Email",
    "exchange": "Email",
    "teams": "Email",
    "sap": "ERP Access",
    "erp": "ERP Access",
    "oracle": "ERP Access",
    "laptop": "Hardware",
    "screen": "Hardware",
    "keyboard": "Hardware",
    "mouse": "Hardware",
    "crash": "Hardware",
    "install": "Software Access",
    "software": "Software Access",
    "application": "Software Access",
    "license": "Software Access"
}

def infer_category(
    issue: str,
    resolution: str,
    existing_category: str = ""
) -> str:
    if existing_category in VALID_CATEGORIES:
        return existing_category

    combined = (issue + " " + resolution).lower()

    for keyword, category in \
            KEYWORD_CATEGORY_MAP.items():
        if keyword in combined:
            return category

    return "Software Access"

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r'http\S+', '[URL]', text)
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    text = re.sub(r'#{1,6}\s', '', text)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}',
                  r'\1', text)
    text = re.sub(r'`{1,3}[^`]*`{1,3}',
                  '[CODE]', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

def is_valid_ticket(ticket: dict) -> bool:
    issue = ticket.get("issue", "")
    resolution = ticket.get("resolution", "")

    if len(issue) < 15:
        return False
    if len(resolution) < 15:
        return False
    if issue.lower() == resolution.lower():
        return False
    if "lorem ipsum" in issue.lower():
        return False
    if "test ticket" in issue.lower():
        return False

    return True

def normalize_ticket(
    ticket: dict,
    idx: int,
    source: str = "unknown"
) -> Optional[dict]:
    try:
        issue_raw = str(
            ticket.get("issue", "")
        ).strip()
        resolution_raw = str(
            ticket.get("resolution", "")
        ).strip()

        issue_clean = clean_text(issue_raw)[:600]
        resolution_clean = clean_text(
            resolution_raw
        )[:800]

        issue_masked, _ = mask_pii(issue_clean)
        resolution_masked, _ = mask_pii(
            resolution_clean
        )

        existing_category = ticket.get(
            "category", ""
        )
        category = infer_category(
            issue_masked,
            resolution_masked,
            existing_category
        )

        ticket_id = ticket.get(
            "ticket_id",
            f"{source.upper()[:3]}-{idx:05d}"
        )

        try:
            minutes = int(
                ticket.get("resolved_in_minutes", 30)
            )
            minutes = max(5, min(minutes, 480))
        except (ValueError, TypeError):
            minutes = 30

        normalized = {
            "ticket_id": ticket_id,
            "category": category,
            "issue": issue_masked,
            "resolution": resolution_masked,
            "resolved_in_minutes": minutes,
            "source": ticket.get("source", source)
        }

        if not is_valid_ticket(normalized):
            return None

        return normalized

    except Exception as e:
        logger.warning(
            f"Failed to normalize ticket "
            f"{idx}: {e}"
        )
        return None

def normalize_all_sources(
    synthetic_tickets: list,
    kaggle_tickets: list,
    github_tickets: list,
    huggingface_tickets: list = []
) -> list:
    all_normalized = []
    seen_issues = set()

    logger.info(
        f"Normalizing: {len(synthetic_tickets)} "
        f"synthetic + {len(kaggle_tickets)} Kaggle "
        f"+ {len(github_tickets)} GitHub "
        f"+ {len(huggingface_tickets)} HuggingFace"
    )

    for idx, ticket in enumerate(synthetic_tickets):
        normalized = normalize_ticket(
            ticket, idx, "synthetic"
        )
        if normalized:
            issue_key = normalized["issue"][:100]
            if issue_key not in seen_issues:
                seen_issues.add(issue_key)
                all_normalized.append(normalized)

    for idx, ticket in enumerate(kaggle_tickets):
        normalized = normalize_ticket(
            ticket, idx, "kaggle"
        )
        if normalized:
            issue_key = normalized["issue"][:100]
            if issue_key not in seen_issues:
                seen_issues.add(issue_key)
                all_normalized.append(normalized)
            if len(all_normalized) >= 3500:
                break

    for idx, ticket in enumerate(github_tickets):
        normalized = normalize_ticket(
            ticket, idx, "github"
        )
        if normalized:
            issue_key = normalized["issue"][:100]
            if issue_key not in seen_issues:
                seen_issues.add(issue_key)
                all_normalized.append(normalized)

    for idx, ticket in enumerate(huggingface_tickets):
        normalized = normalize_ticket(
            ticket, idx, "huggingface"
        )
        if normalized:
            issue_key = normalized["issue"][:100]
            if issue_key not in seen_issues:
                seen_issues.add(issue_key)
                all_normalized.append(normalized)

    logger.info(
        f"Total normalized tickets: "
        f"{len(all_normalized)}"
    )

    return all_normalized

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample = [
        {
            "ticket_id": "TEST-001",
            "category": "VPN",
            "issue": "User john.smith@company.com "
                     "cannot connect to VPN",
            "resolution": "Reset VPN credentials "
                         "and updated profile",
            "resolved_in_minutes": 25,
            "source": "test"
        }
    ]

    result = normalize_ticket(sample[0], 0, "test")
    print("Normalization test:")
    print(json.dumps(result, indent=2))
