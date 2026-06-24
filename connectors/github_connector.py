"""
Pulls real closed issues from major open source infrastructure repositories.
"""

import os
import json
import time
import requests
from dotenv import load_dotenv
import logging

load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        ".env"
    )
)

logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv(
    "GITHUB_USERNAME", "Yasasu06"
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "real_data"
)
OUTPUT_FILE = os.path.join(
    DATA_DIR, "github_tickets.json"
)

REPOS = [
    {
        "owner": "microsoft",
        "repo": "vscode",
        "category": "Software Access",
        "label": "bug"
    },
    {
        "owner": "kubernetes",
        "repo": "kubernetes",
        "category": "Network",
        "label": "kind/bug"
    }
]

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def fetch_issues_for_repo(
    owner: str,
    repo: str,
    category: str,
    label: str,
    max_issues: int = 100
) -> list:
    tickets = []
    page = 1

    while len(tickets) < max_issues:
        url = (
            f"https://api.github.com/repos/"
            f"{owner}/{repo}/issues"
        )
        params = {
            "state": "closed",
            "labels": label,
            "per_page": 50,
            "page": page,
            "sort": "updated",
            "direction": "desc"
        }

        try:
            response = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=10
            )

            if response.status_code == 403:
                logger.warning(
                    "GitHub rate limit hit. "
                    "Waiting 60 seconds..."
                )
                time.sleep(60)
                continue

            if response.status_code != 200:
                logger.error(
                    f"GitHub API error: "
                    f"{response.status_code} "
                    f"for {owner}/{repo}"
                )
                break

            issues = response.json()

            if not issues:
                break

            for issue in issues:
                if issue.get("pull_request"):
                    continue

                title = issue.get("title", "").strip()
                body = issue.get("body", "") or ""
                body = body.strip()

                if (not title or
                    len(title) < 10 or
                    len(body) < 30):
                    continue

                comments_url = issue.get(
                    "comments_url", ""
                )
                resolution = ""

                if issue.get("comments", 0) > 0:
                    try:
                        comments_resp = requests.get(
                            comments_url,
                            headers=HEADERS,
                            timeout=10
                        )
                        if comments_resp.status_code == 200:
                            comments = comments_resp.json()
                            last_comments = comments[-3:]
                            resolution = " | ".join([
                                c.get("body", "")[:300]
                                for c in last_comments
                                if c.get("body")
                            ])
                    except Exception:
                        pass

                if not resolution:
                    resolution = (
                        f"Issue closed in "
                        f"{owner}/{repo} repository. "
                        f"See GitHub for full resolution."
                    )

                issue_text = (
                    f"{title}. {body[:400]}"
                ).replace('\n', ' ').replace(
                    '\r', ' '
                )
                resolution_clean = resolution[:600].replace(
                    '\n', ' '
                ).replace('\r', ' ')

                ticket_id = (
                    f"GH-{owner[:3].upper()}"
                    f"-{issue['number']}"
                )

                tickets.append({
                    "ticket_id": ticket_id,
                    "category": category,
                    "issue": issue_text,
                    "resolution": resolution_clean,
                    "resolved_in_minutes": 45,
                    "source": "github",
                    "repo": f"{owner}/{repo}"
                })

                if len(tickets) >= max_issues:
                    break

            page += 1
            time.sleep(1)

        except requests.exceptions.Timeout:
            logger.error(
                f"Timeout fetching {owner}/{repo}"
            )
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            break

    return tickets

def run_github_connector() -> list:
    if os.path.exists(OUTPUT_FILE):
        logger.info(
            "GitHub data already fetched. "
            "Loading from cache..."
        )
        with open(OUTPUT_FILE) as f:
            tickets = json.load(f)
        logger.info(
            f"Loaded {len(tickets)} cached tickets"
        )
        return tickets

    os.makedirs(DATA_DIR, exist_ok=True)
    all_tickets = []

    for repo_config in REPOS:
        logger.info(
            f"Fetching issues from "
            f"{repo_config['owner']}/"
            f"{repo_config['repo']}..."
        )
        tickets = fetch_issues_for_repo(
            owner=repo_config["owner"],
            repo=repo_config["repo"],
            category=repo_config["category"],
            label=repo_config["label"],
            max_issues=20
        )
        logger.info(
            f"Got {len(tickets)} issues from "
            f"{repo_config['owner']}/"
            f"{repo_config['repo']}"
        )
        all_tickets.extend(tickets)
        time.sleep(2)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_tickets, f, indent=2)

    logger.info(
        f"Total GitHub tickets: {len(all_tickets)}"
    )
    return all_tickets

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tickets = run_github_connector()
    print(f"\nTotal GitHub tickets: {len(tickets)}")
    if tickets:
        print("Sample ticket:")
        print(json.dumps(tickets[0], indent=2))
