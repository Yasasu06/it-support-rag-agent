"""
Streamlit chat interface for the Enterprise IT Support RAG Agent.

Provides a chat UI with suggestion chips, retrieval and review status,
and an expandable view of the source tickets behind support answers.

Run with:
    streamlit run app.py
"""

import html
import hmac
import os
import json as _json
import re
import uuid

from dotenv import load_dotenv

# Load environment variables before anything else touches OpenAI/Chroma.
load_dotenv()

import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from data.tickets import TICKETS
from rag import (
    clear_conversation_memory,
    reformulate_query,
)
from agent_pipeline import run_agent_pipeline, run_tool_agent
from security import get_audit_summary
from live_eval import get_live_eval_summary
from error_handling import validate_user_input
from config import config
from index_contract import (
    COLLECTION_NAME, MANIFEST, l2_relevance_score, validate_collection,
    validate_embedding_model,
)
from image_processing import (
    validate_image,
    extract_text_from_image,
    build_combined_query,
)
from rate_limiter import check_rate_limit, get_rate_limit_status
from analytics import log_visit, get_analytics_summary

import logging

logger = logging.getLogger(__name__)
from connectors.servicenow_connector import (
    is_configured as servicenow_configured,
    test_connection as servicenow_test_connection,
)

ANALYTICAL_KEYWORDS = [
    "how many", "count", "average",
    "statistics", "stats", "distribution",
    "most common", "breakdown", "summary",
    "how long", "resolution time", "sla",
    "trend", "compare", "analysis",
    "servicenow", "live status", "current status",
    "connection status", "is servicenow",
    "servicenow working", "servicenow connected",
]


# Word-boundary matching, not plain substring containment. A naive `kw in
# q_lower` check let short keywords match inside unrelated words - e.g. "count"
# is a substring of "account", so "I cannot access SAP after my account was
# locked" was misrouted to the analytical tool-agent path (which always
# renders with sources=[], since it doesn't do RAG citation) instead of the
# RAG path, silently dropping the "Source tickets" expander. "sla" has the
# same risk against "Slack"/"translate". \b anchors each keyword (including
# multi-word phrases) to real word boundaries so only genuine matches count.
_ANALYTICAL_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in ANALYTICAL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def is_analytical_query(question: str) -> bool:
    return bool(_ANALYTICAL_KEYWORD_PATTERN.search(question))

FEEDBACK_FILE = os.path.join(
    os.path.dirname(__file__),
    "feedback_log.jsonl"
)


def log_feedback(
    rating: str,
    *,
    tier: str | None = None,
    confidence_score: float | None = None,
    source_count: int = 0,
    timestamp: str | None = None,
) -> bool:
    from datetime import datetime

    if rating not in {"positive", "negative"}:
        raise ValueError("Invalid feedback rating")

    entry = {
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        "rating": rating,
        "tier": tier if tier in {"Tier 1", "Tier 2"} else None,
        "confidence_score": round(confidence_score, 3)
        if isinstance(confidence_score, (int, float)) else None,
        "source_count": max(0, int(source_count)),
    }
    try:
        with open(FEEDBACK_FILE, "a") as f:
            f.write(_json.dumps(entry) + "\n")
    except OSError:
        logger.warning("Feedback storage is unavailable")
        return False
    return True

# Configuration --------------------------------------------------------------
CHROMA_DIR = config.CHROMA_DB_PATH
# Sourced from config so the UI vectorstore stays in lockstep with the model
# used to build the index (default unchanged: text-embedding-3-small).
EMBEDDING_MODEL = config.EMBEDDING_MODEL
TOP_K = 3

CATEGORIES = [
    "All Categories",
    "VPN",
    "Password",
    "Software Access",
    "Hardware",
    "Email",
    "Printer",
    "Network",
    "ERP Access",
]

SUGGESTIONS = [
    "VPN won't connect after password reset",
    "Printer not responding",
    "Can't access SAP system",
    "Outlook not syncing emails",
]

# Bold per-category color coding (WCAG AA text contrast; avoids pure red/green).
CATEGORY_COLORS = {
    "VPN": {"bg": "#E8EDFF", "text": "#2453FF", "solid": "#2453FF"},
    "Password": {"bg": "#F3E8FE", "text": "#7C3AED", "solid": "#7C3AED"},
    "Software Access": {"bg": "#E3F6EE", "text": "#0F9D6C", "solid": "#0F9D6C"},
    "Hardware": {"bg": "#FDECE0", "text": "#E8590C", "solid": "#E8590C"},
    "Email": {"bg": "#E0F7FB", "text": "#0891B2", "solid": "#0891B2"},
    "Printer": {"bg": "#FBE5FA", "text": "#C026D3", "solid": "#C026D3"},
    "Network": {"bg": "#FDF3D9", "text": "#CA8A04", "solid": "#CA8A04"},
    "ERP Access": {"bg": "#FCE4EF", "text": "#DB2777", "solid": "#DB2777"},
}


def get_category_color(category: str) -> dict:
    """Return color dict for a category, default to VPN blue if not found."""
    return CATEGORY_COLORS.get(
        category,
        {"bg": "#E8EDFF", "text": "#2453FF", "solid": "#2453FF"},
    )


TICKETS_BY_ID = {t["ticket_id"]: t for t in TICKETS}

# Exact warning blocks injected in place of st.info()/st.warning().
PII_HTML = (
    '<div style="background:#E8EDFF;border-left:3px solid #2453FF;border-radius:8px;'
    'padding:0.75rem 1rem;margin-top:0.5rem;">'
    '<div style="color:#3730A3;font-weight:600;font-size:0.85rem;">Text Masking Applied</div>'
    '<div style="color:#3730A3;font-size:0.8rem;margin-top:0.25rem;">'
    "Detected personal information in your text was masked before support "
    "model calls. Masking may not identify every sensitive detail."
    "</div></div>"
)

ESCALATION_HTML = (
    '<div style="background:#FFFBEB;border-left:3px solid #F59E0B;border-radius:8px;'
    'padding:0.75rem 1rem;margin-top:0.5rem;">'
    '<div style="color:#92400E;font-weight:600;font-size:0.85rem;">Tier 2 Review Recommended</div>'
    '<div style="color:#92400E;font-size:0.8rem;margin-top:0.25rem;">'
    "This response was flagged for further review. "
    "No ServiceNow ticket was created automatically."
    "</div></div>"
)

ERROR_MESSAGE = (
    "The knowledge base or model service is unavailable. "
    "Please try again later."
)

_ticket_count = str(MANIFEST["tracked_record_count"])

st.set_page_config(
    page_title="IT Support AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global styling ---------------------------------------------------------------
st.markdown(
    """
    <style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #F7F7F4;
  --surface: #FFFFFF;
  --surface-alt: #F0F0EC;
  --ink: #12141C;
  --ink-muted: #5B6072;
  --ink-faint: #9599A8;
  --sidebar-bg: #0A0E1A;
  --sidebar-ink: #C9D1D9;
  --sidebar-muted: #6E7681;
  --sidebar-border: #1E2433;
  --accent: #2453FF;
  --accent-soft: #EEF1FF;
  --success: #0F9D6C;
  --success-soft: #E3F6EE;
  --warning: #C97A1A;
  --warning-soft: #FBEEDD;
  --danger: #D64545;
  --danger-soft: #FBE7E7;
  --border: #E4E4DF;
}

* { font-family: 'Inter', sans-serif !important; }

h1, h2, h3, .display-text {
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: -0.02em !important;
  color: var(--ink) !important;
}

.mono, .mono * {
  font-family: 'JetBrains Mono', monospace !important;
}

/* Restore Material icon font Streamlit needs - do
   NOT let the wildcard reset break native icons */
[data-testid="stIconMaterial"],
.material-icons, .material-symbols-rounded {
  font-family: 'Material Symbols Rounded' !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
  display: none !important;
}

.stApp {
  background:
    radial-gradient(
      ellipse 1400px 700px at 15% -5%,
      rgba(36, 83, 255, 0.04) 0%,
      transparent 55%
    ),
    #F7F7F4 !important;
}

.main .block-container {
  max-width: 820px !important;
  padding: 2rem 2rem 6rem 2rem !important;
}

/* Sidebar - terminal panel */
section[data-testid="stSidebar"] {
  background-color: var(--sidebar-bg) !important;
  border-right: 1px solid var(--sidebar-border) !important;
}
/* Force the sidebar to stay open and on-screen regardless of Streamlit's
   collapse state, but ONLY at desktop/tablet widths. A persisted/collapsed
   state (remembered across reloads) otherwise slides it off-screen via
   transform / zero width. The pre-redesign CSS pinned it open this way; the
   redesign dropped it, which is why the sidebar "stopped rendering". */
@media (min-width: 769px) {
  section[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    transform: none !important;
    width: 300px !important;
    min-width: 300px !important;
    max-width: 300px !important;
  }
  section[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] {
    width: 300px !important;
    min-width: 300px !important;
  }
}
/* Mobile (<=768px): app.py sets initial_sidebar_state="expanded", which is
   Streamlit's default at EVERY viewport width, not just desktop, and
   aria-expanded is "true" from the very first mount regardless of screen
   size - there's no built-in "start collapsed on mobile" signal to key off.
   Simply not forcing it open here still left it permanently laid out at
   300px, squeezing the ~90px-wide remainder of a phone screen unusable.
   Move it off-canvas (position: fixed, off-screen) unconditionally on
   mobile instead, so the main content is always fully usable on a phone.
   Known tradeoff: Streamlit's own collapse/expand toggle button has no
   visible effect at this width, since nothing here reads its state - a
   separate mobile-drawer polish item, not the overlap bug this fixes. */
@media (max-width: 768px) {
  section[data-testid="stSidebar"] {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    height: 100vh !important;
    width: 300px !important;
    min-width: 300px !important;
    max-width: 85vw !important;
    transform: translateX(-100%) !important;
  }
  section[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] {
    width: 300px !important;
    min-width: 300px !important;
  }
}
section[data-testid="stSidebar"] * {
  color: var(--sidebar-ink) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2 {
  font-family: 'Space Grotesk', sans-serif !important;
  color: #FFFFFF !important;
}
section[data-testid="stSidebar"] p {
  color: var(--sidebar-muted) !important;
  font-size: 0.8rem !important;
  line-height: 1.5 !important;
}
section[data-testid="stSidebar"] hr {
  border-color: var(--sidebar-border) !important;
}

/* Sidebar structural depth gradient (new rule; does NOT modify the force-open
   block above) — a slightly lighter navy at the top settling into the sidebar
   background, so it reads as depth rather than a colored glow. */
section[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] {
  background: linear-gradient(
    180deg,
    #111827 0px,
    var(--sidebar-bg) 120px
  );
}

/* Sidebar metric cards - instrument readout style */
section[data-testid="stSidebar"] [data-testid="stMetric"] {
  background: #111827 !important;
  border: 1px solid var(--sidebar-border) !important;
  border-left: 2px solid #2453FF !important;
  border-radius: 6px !important;
  padding: 0.7rem 0.9rem !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.65rem !important;
  color: var(--sidebar-muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', monospace !important;
  color: #FFFFFF !important;
  font-weight: 600 !important;
}

/* Tabs */
[data-testid="stTabs"] button {
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  color: var(--ink-muted) !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}

/* Header bar */
div.app-header, .app-header {
  display: flex !important;
  align-items: center !important;
  gap: 0.9rem !important;
  padding: 1.1rem 1.3rem !important;
  margin-bottom: 1.5rem !important;
  border-radius: 14px !important;
  background: linear-gradient(120deg, #1A2E6E 0%, #2D1B69 55%, #3D1A5C 100%) !important;
  box-shadow: 0 4px 16px rgba(10, 14, 26, 0.3) !important;
}
.app-header .badge {
  width: 42px;
  height: 42px;
  border-radius: 10px;
  background: rgba(255,255,255,0.2) !important;
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  color: #FFFFFF !important;
  font-size: 0.95rem;
}
.app-header .title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.05rem;
  color: #FFFFFF !important;
}
.app-header .subtitle {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  color: rgba(255,255,255,0.85) !important;
}

/* Chat bubbles */
[data-testid="stChatMessage"]:has(
  [data-testid="stChatMessageAvatarUser"]) {
  background: var(--accent) !important;
  border-radius: 14px 14px 3px 14px !important;
  padding: 0.85rem 1.15rem !important;
  margin-left: 18% !important;
}
[data-testid="stChatMessage"]:has(
  [data-testid="stChatMessageAvatarUser"]) * {
  color: #FFFFFF !important;
}
[data-testid="stChatMessage"]:has(
  [data-testid="stChatMessageAvatarAssistant"]) {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-left: 3px solid var(--accent) !important;
  border-radius: 3px 14px 14px 14px !important;
  padding: 1rem 1.15rem !important;
  margin-right: 12% !important;
  box-shadow: 0 1px 3px rgba(18,20,28,0.04) !important;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  display: none !important;
}

/* Status readout for retrieval, grounding, and review */
.signal-readout {
  display: inline-flex; align-items: center; gap: 0.6rem;
  background: var(--surface-alt);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.45rem 0.75rem;
  margin-top: 0.6rem;
}
.signal-readout .signal-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--ink-faint);
}
.signal-readout .signal-value {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600; font-size: 0.8rem;
}

/* Source ticket cards */
.ticket-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.9rem 1.05rem;
  margin-bottom: 0.6rem;
}
.ticket-card .ticket-id {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600; color: var(--ink); font-size: 0.8rem;
}
.ticket-card .ticket-category {
  background: var(--accent-soft); color: var(--accent);
  font-size: 0.65rem; font-weight: 600;
  padding: 0.15rem 0.5rem; border-radius: 100px;
  font-family: 'JetBrains Mono', monospace;
  text-transform: uppercase; letter-spacing: 0.04em;
}
.ticket-card .ticket-time {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem; color: var(--ink-faint);
}

/* Buttons */
.stButton > button {
  font-family: 'Inter', sans-serif !important;
  border-radius: 8px !important;
}

/* Suggestion chips on welcome screen */
.stButton > button {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--ink) !important;
  font-weight: 500 !important;
  box-shadow: 0 1px 2px rgba(18,20,28,0.04) !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  box-shadow: 0 3px 10px rgba(36,83,255,0.12) !important;
}

/* Chat input */
[data-testid="stChatInput"] {
  border: 1.5px solid var(--border) !important;
  border-radius: 12px !important;
  background: var(--surface) !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

/* System tab cards */
.sys-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.1rem 1.2rem;
}
.sys-card h4 {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.9rem !important; margin-bottom: 0.4rem !important;
}
.sys-card p {
  color: var(--ink-muted) !important;
  font-size: 0.82rem !important; line-height: 1.5 !important;
}

/* Alerts */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* ---- App-specific widget fixes (make the design system above work with
   this app's existing widgets; visual only, no new design patterns) ---- */

/* Sidebar widgets need dark-panel treatment, not the light chip style */
section[data-testid="stSidebar"] .stButton > button {
  background: #161B22 !important;
  color: var(--sidebar-ink) !important;
  border: 1px solid var(--sidebar-border) !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  border-color: var(--accent) !important;
  color: #FFFFFF !important;
}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div {
  background: #161B22 !important;
  border: 1px solid var(--sidebar-border) !important;
  border-radius: 6px !important;
}

/* Back button ("New chat") - quiet text link, not a full chip */
.st-key-back_btn button {
  background: transparent !important;
  border: none !important;
  color: var(--ink-muted) !important;
  box-shadow: none !important;
  width: auto !important;
}
.st-key-back_btn button:hover {
  color: var(--accent) !important;
  box-shadow: none !important;
}

/* Source-tickets toggle button - subtle instrument control */
[class*="st-key-toggle_sources_"] button {
  background: var(--surface-alt) !important;
  border: 1px solid var(--border) !important;
  color: var(--ink-muted) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.72rem !important;
  box-shadow: none !important;
}
[class*="st-key-toggle_sources_"] button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}

/* Chat input inner textarea */
[data-testid="stChatInput"] textarea {
  color: var(--ink) !important;
  font-family: 'Inter', sans-serif !important;
}

/* Spinner + tabs content spacing */
[data-testid="stSpinner"] svg { color: var(--accent) !important; }
[data-testid="stTabs"] button:hover { color: var(--accent) !important; }

/* Slide-up entrance for chat messages */
@keyframes slideUp {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
[data-testid="stChatMessage"] { animation: slideUp 0.25s ease-out; }

/* Colorful welcome-screen suggestion chips. Targeted by Streamlit's per-widget
   key class (.st-key-chip_N) — the same reliable pattern this app already uses
   for .st-key-back_btn / st-key-toggle_sources_. Placed last so it wins over
   the generic .stButton > button rules. (Wrapping buttons in markdown <div>s is
   unreliable in Streamlit — the tags don't actually enclose the widget DOM.) */
div[class*="st-key-chip_1"] button, div[class*="st-key-chip_1"] > div > button {
  border-left: 4px solid #2453FF !important; background: #FFFFFF !important; background-color: #FFFFFF !important; color: #2453FF !important; font-weight: 600 !important; text-align: left !important;
}
div[class*="st-key-chip_2"] button, div[class*="st-key-chip_2"] > div > button {
  border-left: 4px solid #C026D3 !important; background: #FFFFFF !important; background-color: #FFFFFF !important; color: #C026D3 !important; font-weight: 600 !important; text-align: left !important;
}
div[class*="st-key-chip_3"] button, div[class*="st-key-chip_3"] > div > button {
  border-left: 4px solid #DB2777 !important; background: #FFFFFF !important; background-color: #FFFFFF !important; color: #DB2777 !important; font-weight: 600 !important; text-align: left !important;
}
div[class*="st-key-chip_4"] button, div[class*="st-key-chip_4"] > div > button {
  border-left: 4px solid #0891B2 !important; background: #FFFFFF !important; background-color: #FFFFFF !important; color: #0891B2 !important; font-weight: 600 !important; text-align: left !important;
}
div[class*="st-key-chip_"] button:hover, div[class*="st-key-chip_"] > div > button:hover {
  filter: brightness(0.97) !important; box-shadow: 0 3px 10px rgba(18,20,28,0.10) !important;
}
</style>
    """,
    unsafe_allow_html=True,
)


# Cached resources -------------------------------------------------------------
@st.cache_resource
def get_app_vectorstore():
    validate_embedding_model(EMBEDDING_MODEL)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        relevance_score_fn=l2_relevance_score,
        create_collection_if_not_exists=False,
    )
    validate_collection(vectorstore._collection, EMBEDDING_MODEL)
    return vectorstore


# Helpers ------------------------------------------------------------------------
def escape_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def response_status_labels(msg: dict) -> list[tuple[str, str]]:
    """Describe observed processing without treating relevance as probability."""
    if msg.get("response_kind") == "analytical":
        status = (
            "Analytical tool completed"
            if msg.get("tool_status") == "completed"
            else "Analytical tool unavailable"
        )
        return [("Tool status", status)]
    if msg.get("response_kind") != "rag":
        return []

    source_count = len(msg.get("sources") or [])
    source_label = f"{source_count} source ticket{'s' if source_count != 1 else ''} retrieved"
    grounded = msg.get("grounded")
    grounding_label = (
        "Check passed" if grounded is True
        else "Review needed" if grounded is False
        else "Not run"
    )
    review_label = (
        "Tier 2 recommended" if msg.get("escalation")
        else "Tier 1 response"
    )
    return [
        ("Retrieval", source_label),
        ("Grounding", grounding_label),
        ("Review", review_label),
    ]


def session_retry_summary(messages: list[dict]) -> tuple[int, int, int]:
    """Count broadened answers from completed support graph runs only."""
    support_answers = [
        msg for msg in messages
        if msg.get("role") == "assistant"
        and msg.get("response_kind") == "rag"
        and msg.get("graph_completed") is True
    ]
    retried = sum(1 for msg in support_answers if (msg.get("retry_count") or 0) > 0)
    total = len(support_answers)
    rate = int(round(100 * retried / total)) if total else 0
    return retried, total, rate


def format_grounded_rate(rate: float | None) -> str:
    """Show unchecked grounding as unavailable, not as a failed check."""
    return f"{rate:.0%}" if rate is not None else "N/A"


def check_servicenow_connection() -> dict:
    """Retain the safe diagnostic so it remains available after a UI rerun."""
    result = servicenow_test_connection()
    st.session_state.servicenow_connection_result = result
    return result


def render_servicenow_connection_result(result: dict) -> None:
    if result["success"]:
        st.success(result["message"])
    else:
        st.error(result["message"])


def render_response_status(msg: dict) -> None:
    labels = response_status_labels(msg)
    if labels:
        st.markdown(
            "".join(
                '<div class="signal-readout">'
                f'<span class="signal-label">{html.escape(label)}</span>'
                f'<span class="signal-value">{html.escape(value)}</span>'
                '</div>'
                for label, value in labels
            ),
            unsafe_allow_html=True,
        )


def render_retry_badge(retry_count) -> None:
    # Surfaces the adaptive-retry system when retrieval or grounding caused
    # another pass. This just
    # reads the retry_count the pipeline already returns — no logic change. Shown
    # only when it actually happened; nothing rendered when retry_count is 0.
    count = retry_count or 0
    if count <= 0:
        return
    label = f"{count} retr{'y' if count == 1 else 'ies'}"
    st.markdown(
        f'<div style="margin-top:0.4rem;margin-left:0.5rem;">'
        f'<span style="display:inline-flex;align-items:center;gap:0.35rem;'
        f'background:var(--surface-alt);border:1px solid var(--border);'
        f'border-radius:100px;padding:0.2rem 0.6rem;'
        f"font-family:'JetBrains Mono',monospace;font-size:0.62rem;"
        f'text-transform:uppercase;letter-spacing:0.04em;color:var(--ink-faint);">'
        f'<span style="width:5px;height:5px;border-radius:50%;'
        f'background:#2453FF;flex-shrink:0;"></span>'
        f'Broadened search · {label}'
        f'</span></div>',
        unsafe_allow_html=True,
    )


def render_sources_expander(sources: list, msg_index) -> None:
    state_key = f"show_sources_{msg_index}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    arrow = "▾" if st.session_state[state_key] else "▸"
    label = f"{arrow} Source tickets ({len(sources)} retrieved)"
    if st.button(label, key=f"toggle_sources_{msg_index}"):
        st.session_state[state_key] = not st.session_state[state_key]

    if st.session_state[state_key]:
        for src in sources:
            colors = get_category_color(src.get("category", "VPN"))
            st.markdown(
                f"""
                <div class="ticket-card" style="
                    border-left: 4px solid {colors['solid']};
                    background: linear-gradient(135deg, {colors['bg']}59 0%, #FFFFFF 35%);
                ">
                  <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
                    <span class="ticket-id">{escape_html(src["ticket_id"])}</span>
                    <span style="
                        background:{colors['solid']};color:#FFFFFF;font-size:0.65rem;font-weight:600;
                        padding:0.2rem 0.6rem;border-radius:100px;font-family:'JetBrains Mono',monospace;
                        text-transform:uppercase;letter-spacing:0.04em;">{escape_html(src["category"].upper())}</span>
                  </div>
                  <div style="color:var(--ink-muted);font-size:0.82rem;margin-bottom:0.35rem;">
                    <b style="color:var(--ink);">Issue:</b> {escape_html(src["issue"])}
                  </div>
                  <div style="color:var(--ink-muted);font-size:0.82rem;margin-bottom:0.5rem;">
                    <b style="color:var(--ink);">Resolution:</b> {escape_html(src["resolution"])}
                  </div>
                  <div class="ticket-time">
                    Resolved in {src["resolved_in_minutes"]} min
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_user_message(content: str, image_extract: str = "") -> None:
    with st.chat_message("user"):
        st.markdown(content)
        # When the turn included a screenshot, let the user verify what the
        # vision model read from it. Absent an image this is a no-op, so the
        # text-only path renders exactly as before.
        if image_extract:
            with st.expander("What I read from your screenshot"):
                st.write(image_extract)


def render_feedback_buttons(msg: dict, msg_key: str) -> None:
    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 8])
    metadata = {
        "tier": msg.get("tier"),
        "confidence_score": msg.get("confidence_score"),
        "source_count": len(msg.get("sources") or []),
    }
    with fb_col1:
        if st.button("Helpful", key=f"up_{msg_key}", help="This answer was helpful"):
            if log_feedback("positive", **metadata):
                st.toast("Thanks for the feedback!")
            else:
                st.warning("Feedback could not be saved right now.")
    with fb_col2:
        if st.button("Not helpful", key=f"down_{msg_key}", help="This answer wasn't helpful"):
            if log_feedback("negative", **metadata):
                st.toast("Feedback recorded — we'll work on improving this.")
            else:
                st.warning("Feedback could not be saved right now.")


def render_assistant_message(msg: dict, idx: int) -> None:
    with st.chat_message("assistant"):
        st.markdown(msg["content"])
        if msg.get("pii_detected"):
            st.markdown(PII_HTML, unsafe_allow_html=True)
        render_response_status(msg)
        render_retry_badge(msg.get("retry_count", 0))
        if msg.get("escalation"):
            st.markdown(ESCALATION_HTML, unsafe_allow_html=True)
        if msg.get("sources"):
            render_sources_expander(msg["sources"], idx)
        if msg["content"] != ERROR_MESSAGE:
            render_feedback_buttons(msg, str(idx))


def get_sources_and_top_score(query: str, category: str = "All Categories"):
    vectorstore = get_app_vectorstore()
    if category != "All Categories":
        results = vectorstore.similarity_search_with_relevance_scores(
            query, k=TOP_K, filter={"category": category}
        )
    else:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=TOP_K)

    sources = []
    for doc, score in results:
        ticket_id = doc.metadata.get("ticket_id")
        ticket = TICKETS_BY_ID.get(ticket_id, {})
        sources.append(
            {
                "ticket_id": ticket_id,
                "category": ticket.get("category", doc.metadata.get("category", "Unknown")),
                "issue": ticket.get("issue", ""),
                "resolution": ticket.get("resolution", ""),
                "resolved_in_minutes": ticket.get(
                    "resolved_in_minutes", doc.metadata.get("resolved_in_minutes")
                ),
                "score": score,
            }
        )

    top_score = results[0][1] if results else 0.0
    return sources, top_score


def process_question(question: str, category: str, image_extract: str = "") -> None:
    # When a screenshot was read, the pipeline sees the typed text + extracted
    # screenshot content combined, while the user sees their typed question (or
    # a placeholder if they typed nothing) plus a "what I read" expander. With
    # image_extract == "" this is byte-identical to the prior text-only path.
    if image_extract:
        pipeline_question = build_combined_query(question, image_extract)
        display_text = question.strip() or "[Screenshot uploaded]"
    else:
        pipeline_question = question
        display_text = question

    # Reject empty or oversized input before it enters the pipeline, so a bad
    # submission gets a clear message instead of running the full agent chain.
    is_valid, validation_error = validate_user_input(
        pipeline_question, max_length=config.MAX_QUESTION_LENGTH
    )
    if not is_valid:
        st.session_state.messages.append(
            {"role": "user", "content": display_text, "image_extract": image_extract}
        )
        render_user_message(display_text, image_extract)
        with st.chat_message("assistant"):
            st.warning(validation_error)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": validation_error,
                "question": display_text,
                "confidence": None,
                "confidence_score": None,
                "sources": [],
                "escalation": None,
                "tier": None,
                "pii_detected": [],
            }
        )
        return

    analytical = is_analytical_query(pipeline_question)

    # Analytical questions use the separate tool agent. The category selector
    # filters support-answer retrieval only; tool calls choose their own inputs.
    augmented_question = pipeline_question
    if not analytical:
        # Give follow-up questions ("what about that?") access to recent
        # conversation context, then let reformulate_query clean up the result
        # before it hits retrieval/run_agent_pipeline.
        prior_messages = st.session_state.messages[-4:]
        if prior_messages:
            history_text = "\n\nPrevious conversation:\n"
            for prior_msg in prior_messages:
                role = "User" if prior_msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {str(prior_msg['content'])[:200]}\n"
            augmented_question = f"{augmented_question}{history_text}"
        try:
            augmented_question = reformulate_query(augmented_question)
        except Exception:
            st.error(ERROR_MESSAGE)
            return

    st.session_state.messages.append(
        {"role": "user", "content": display_text, "image_extract": image_extract}
    )
    render_user_message(display_text, image_extract)

    with st.chat_message("assistant"):
        try:
            with st.spinner(":shimmer[Searching knowledge base...]"):
                if analytical:
                    result = run_tool_agent(pipeline_question)
                    sources = []
                else:
                    selected_category = None if category == "All Categories" else category
                    result = run_agent_pipeline(
                        augmented_question, category=selected_category
                    )
                    sources = result.get("sources", [])

                answer = result["answer"]
                escalation = result["escalation"]
                pii_detected = result["pii_detected"]

            assistant_msg = {
                "role": "assistant",
                "content": answer,
                "question": display_text,
                "response_kind": "analytical" if analytical else "rag",
                "graph_completed": result.get("graph_completed", False),
                "tool_status": result.get("tool_status"),
                "confidence_score": result.get("confidence_score"),
                "sources": sources,
                "grounded": result.get("grounded"),
                "escalation": escalation,
                "tier": result.get("tier"),
                "pii_detected": pii_detected,
                "retry_count": result.get("retry_count", 0),
            }

            # The graph completes verification, retry, and triage before display.
            st.markdown(answer)
            if pii_detected:
                st.markdown(PII_HTML, unsafe_allow_html=True)
            render_response_status(assistant_msg)
            render_retry_badge(result.get("retry_count", 0))
            if escalation:
                st.markdown(ESCALATION_HTML, unsafe_allow_html=True)
            if sources:
                render_sources_expander(sources, len(st.session_state.messages))

            st.session_state.messages.append(assistant_msg)
        except Exception:
            st.error(ERROR_MESSAGE)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": ERROR_MESSAGE,
                    "question": display_text,
                    "confidence": None,
                    "confidence_score": None,
                    "sources": [],
                    "escalation": None,
                    "tier": None,
                    "pii_detected": [],
                }
            )


def render_welcome_screen() -> None:
    st.markdown(
        "<div style='text-align:center;padding-top:1.5rem;'>"
        "<h2 style='margin-top:0.5rem;margin-bottom:0.5rem;'>How can I help you today?</h2>"
        "<p style='color:#64748B;font-size:0.95rem;max-width:520px;margin:0 auto;'>"
        "Ask me about any IT issue — I'll search our knowledge base and "
        "give you a grounded answer with sources."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    chips = [
        (label, label)
        for label in SUGGESTIONS
    ]

    row1 = st.columns(2, gap="small")
    row2 = st.columns(2, gap="small")
    grid = [row1[0], row1[1], row2[0], row2[1]]

    for i, (col, (label, prefill_text)) in enumerate(zip(grid, chips), start=1):
        with col:
            if st.button(label, key=f"chip_{i}", width="stretch"):
                st.session_state.prefill = prefill_text
                st.rerun()


# Session state init -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state.prefill = None


# Per-session visit logging -----------------------------------------------------
# Logs a timestamp + random session id once per browser session; no IP, name,
# or email field is collected here.
# Guarded by "session_id not in session_state" so it fires once per session, not
# on every question or rerun.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    log_visit(st.session_state.session_id)

# Private, session-authenticated analytics view ------------------------------------
def admin_key_matches(candidate: str, configured_key: str) -> bool:
    return bool(configured_key) and hmac.compare_digest(candidate, configured_key)


ADMIN_KEY = os.getenv("ADMIN_ANALYTICS_KEY", "")
if ADMIN_KEY:
    with st.sidebar.expander("Admin analytics"):
        if not st.session_state.get("admin_authenticated", False):
            with st.form("admin_login", clear_on_submit=True):
                supplied_key = st.text_input("Admin key", type="password")
                submitted = st.form_submit_button("Unlock")
            if submitted:
                if admin_key_matches(supplied_key, ADMIN_KEY):
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid admin key.")

if ADMIN_KEY and st.session_state.get("admin_authenticated", False):
    st.markdown("## Private Analytics (Admin Only)")
    summary = get_analytics_summary()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Visits", summary["total_visits"])
    with col2:
        st.metric("Unique Sessions", summary["unique_sessions"])
    with col3:
        st.metric("Visits Today", summary["visits_today"])

    st.markdown("### Recent Visit Log")
    if summary["recent_visits"]:
        for visit in summary["recent_visits"]:
            st.text(
                f"{visit['timestamp']} - "
                f"session: {visit['session_id'][:8]}..."
            )
    else:
        st.text("No visits logged yet.")

    st.markdown("---")
    if st.button("Return to app"):
        st.session_state.admin_authenticated = False
        st.rerun()
    st.stop()


# Sidebar -------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:0.6rem;'>"
        "<div>"
        "<div style='display:flex;align-items:center;gap:0.5rem;'>"
        "<div style='width:7px;height:7px;border-radius:50%;background:#10B981;"
        "box-shadow:0 0 6px rgba(16,185,129,0.6);flex-shrink:0;'></div>"
        "<div style='font-size:1.1rem;font-weight:700;color:#F1F5F9;'>IT Support AI</div>"
        "</div>"
        "<div style='font-size:0.75rem;color:#94A3B8;margin-top:0.15rem;'>"
        "Powered by RAG · GPT-4o-mini</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        "<p style='color:#64748B;font-size:0.72rem;"
        "line-height:1.5;word-wrap:break-word;"
        "overflow-wrap:break-word;max-width:240px'>"
        "IT support answers draw on incident data, cite sources, "
        "and show retrieval and review status.</p>",
        unsafe_allow_html=True
    )

    category = st.selectbox(
        "Support answer category", CATEGORIES, key="category_filter",
        help="Filters retrieved tickets for support answers. Analytical tools use their own inputs.",
    )
    st.divider()

    st.markdown(
        "<div style='border-left:2px solid #2453FF;padding-left:0.6rem;"
        "margin-bottom:0.5rem;'>"
        "<span style='font-size:0.7rem;font-weight:700;letter-spacing:0.08em;"
        "text-transform:uppercase;color:#94A3B8;'>Knowledge Base</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Tracked Snapshot Tickets", _ticket_count)
        st.metric("Model", "GPT-4o")
    with col_b:
        st.metric("Cats", "8")
        st.metric("Saved", "~12 min")
    st.divider()

    if st.button("Clear conversation", key="clear_btn"):
        st.session_state.messages = []
        st.session_state.prefill = None
        clear_conversation_memory()
        st.rerun()

    summary = get_audit_summary()
    if summary["total_queries"] > 0:
        st.divider()
        st.markdown(
            "<div style='border-left:2px solid #10B981;padding-left:0.6rem;"
            "margin-bottom:0.5rem;'>"
            "<span style='font-size:0.7rem;font-weight:700;letter-spacing:0.08em;"
            "text-transform:uppercase;color:#94A3B8;'>System Health</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        col_c, col_d = st.columns(2)
        with col_c:
            st.metric("Queries", summary["total_queries"])
            st.metric("Escalated", summary["escalated_queries"])
        with col_d:
            st.metric("PII Caught", summary["pii_queries"])

    # Rate-limit usage indicator (transparency, not enforcement) — helps a user
    # see they're approaching the limit rather than being surprised by a block.
    if config.ENABLE_RATE_LIMITING:
        _rl = get_rate_limit_status(
            max_requests=config.RATE_LIMIT_MAX_REQUESTS,
            window_seconds=config.RATE_LIMIT_WINDOW_SECONDS,
        )
        st.markdown(
            f"<p style='color:#64748B;font-size:0.72rem;margin-top:0.75rem'>"
            f"Requests: {_rl['used']}/{_rl['limit']} in the last "
            f"{_rl['window_seconds']}s</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#475569;"
        "font-size:0.72rem;margin-top:1rem'>"
        "Built by <strong style='color:#94A3B8'>"
        "Yasaswi</strong></p>",
        unsafe_allow_html=True
    )


# Tabs: Chat / System Architecture ------------------------------------------------
tab1, tab2 = st.tabs(["Chat", "System"])

with tab1:
    # Main header — always shown, before any conditional logic ----------------------
    st.markdown(
        f'''
<div class="app-header">
  <div class="badge">IT</div>
  <div>
    <div class="title">IT Support AI Assistant</div>
  <div class="subtitle">TRACKED SNAPSHOT: {_ticket_count} RECORDS · {len(CATEGORIES) - 1} CATEGORIES · CONFIGURABLE MODEL</div>
  </div>
</div>
''',
        unsafe_allow_html=True,
    )

    if st.session_state.messages:
        col_back, _ = st.columns([1, 7])
        with col_back:
            if st.button("← New chat", key="back_btn"):
                st.session_state.messages = []
                st.session_state.prefill = None
                st.rerun()
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # Optional screenshot upload (feature-flagged). Rendered above the pinned
    # chat input; when a file is present at submit time its content is extracted
    # and merged into the question. With no upload, the flow below is unchanged.
    uploaded_image = None
    analyze_clicked = False
    if config.ENABLE_IMAGE_UPLOAD:
        st.session_state.setdefault("uploader_nonce", 0)
        uploaded_image = st.file_uploader(
            "Or upload a screenshot of your error (optional)",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"screenshot_upload_{st.session_state.uploader_nonce}",
        )
        st.caption(
            "Uploaded screenshots are sent to an external model provider for analysis. "
            "Remove sensitive details before uploading."
        )
        if uploaded_image is not None:
            analyze_clicked = st.button(
                "Analyze screenshot", key="analyze_screenshot_btn"
            )

    # Pinned chat input at the bottom of the page ------------------------------------
    with st.bottom:
        prompt = st.chat_input("Describe your IT issue...")

    # Process a suggestion-chip click before normal input.
    if "prefill" in st.session_state and st.session_state.prefill:
        pending = st.session_state.prefill
        st.session_state.prefill = None
        # Suggestion chips also trigger a full pipeline call — rate limit them too.
        if config.ENABLE_RATE_LIMITING:
            allowed, rl_error = check_rate_limit(
                max_requests=config.RATE_LIMIT_MAX_REQUESTS,
                window_seconds=config.RATE_LIMIT_WINDOW_SECONDS,
            )
            if not allowed:
                st.warning(rl_error)
                st.stop()
        process_question(pending, category)
        st.rerun()

    # A submission is either typed text (chat_input) or an image-only "Analyze
    # screenshot" click. submit_text is None when nothing was submitted.
    has_image = config.ENABLE_IMAGE_UPLOAD and uploaded_image is not None
    if prompt:
        submit_text = prompt
    elif has_image and analyze_clicked:
        submit_text = ""
    else:
        submit_text = None

    if submit_text is not None:
        # Rate limit FIRST — before ANY OpenAI usage (image vision extraction or
        # the RAG pipeline), so abuse is blocked before it can incur cost. This
        # single gate covers both the typed and image-upload submission paths.
        if config.ENABLE_RATE_LIMITING:
            allowed, rl_error = check_rate_limit(
                max_requests=config.RATE_LIMIT_MAX_REQUESTS,
                window_seconds=config.RATE_LIMIT_WINDOW_SECONDS,
            )
            if not allowed:
                st.warning(rl_error)
                st.stop()

        image_extract = ""
        proceed = True
        if has_image:
            file_bytes = uploaded_image.getvalue()
            valid, img_error = validate_image(file_bytes, uploaded_image.type)
            if not valid:
                # Invalid image blocks the pipeline (do not proceed).
                st.error(img_error)
                proceed = False
            else:
                with st.spinner("Analyzing screenshot..."):
                    extraction = extract_text_from_image(
                        file_bytes, uploaded_image.type, submit_text
                    )
                if extraction["success"]:
                    image_extract = extraction["extracted_text"]
                else:
                    # Graceful degradation: log the detail internally, show the
                    # user a generic message (never expose raw exception text),
                    # and fall back to the typed question only.
                    logger.error(
                        "Image extraction failed: %s",
                        extraction.get("error_type", "ProviderError"),
                    )
                    st.error(
                        "I couldn't read that screenshot. Please describe "
                        "your issue in text instead."
                    )
                    image_extract = ""
                # Clear the uploader so this image isn't reused on the next question.
                st.session_state.uploader_nonce += 1
        if proceed:
            process_question(submit_text, category, image_extract=image_extract)
            st.rerun()

    # Main content: welcome screen or chat history -----------------------------------
    if not st.session_state.messages:
        render_welcome_screen()
    else:
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                render_user_message(msg["content"], msg.get("image_extract", ""))
            else:
                render_assistant_message(msg, idx)

with tab2:
    # Section 1: Data sources -----------------------------------------------------
    st.markdown("### Tracked Snapshot Data Sources")
    st.caption("These counts describe the repository snapshot; an index selected with CHROMA_DB_PATH may differ.")

    _snapshot_sources = {item["name"]: item["records"] for item in MANIFEST["sources"]}
    _ing = {
        "synthetic": _snapshot_sources["synthetic"],
        "kaggle": _snapshot_sources["kaggle"],
        "github": _snapshot_sources["github"],
        "huggingface": _snapshot_sources["huggingface_servicenow"],
    }

    _sn_configured = servicenow_configured()
    _sn_status = "Not configured"
    _sn_color = "#94A3B8"
    if not config.ENABLE_SERVICENOW_LIVE:
        _sn_status = "Disabled"
    elif _sn_configured:
        if st.button("Check ServiceNow connection", key="servicenow_connection_check"):
            with st.spinner("Checking ServiceNow connection..."):
                check_servicenow_connection()
        connection_result = st.session_state.get("servicenow_connection_result")
        connection_ok = connection_result["success"] if connection_result else None
        if connection_result:
            render_servicenow_connection_result(connection_result)
        if connection_ok is True:
            _sn_status = "Live — Connected"
            _sn_color = "#10B981"
        elif connection_ok is False:
            _sn_status = "Configured — Connection Failed"
            _sn_color = "#EF4444"
        else:
            _sn_status = "Configured — Not checked"

    ds_col1, ds_col2, ds_col3, ds_col4, ds_col5 = st.columns(5)
    with ds_col1:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid #2453FF;border-radius:10px;padding:1.25rem">
        <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;font-weight:600">Internal Synthetic</div>
        <div style="font-size:2rem;font-weight:700;color:#0F172A;margin:0.5rem 0">{_ing['synthetic']}</div>
        <div style="font-size:0.75rem;color:#64748B">Fictional internal-style support records</div>
        <div style="margin-top:0.75rem;font-size:0.72rem;color:#2453FF;font-weight:500">8 categories · Structured format</div>
        </div>
        """, unsafe_allow_html=True)

    with ds_col2:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid #10B981;border-radius:10px;padding:1.25rem">
        <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;font-weight:600">Kaggle Dataset</div>
        <div style="font-size:2rem;font-weight:700;color:#0F172A;margin:0.5rem 0">{_ing['kaggle']}</div>
        <div style="font-size:0.75rem;color:#64748B">Public customer-support dataset</div>
        <div style="margin-top:0.75rem;font-size:0.72rem;color:#10B981;font-weight:500">Small contribution after deduplication</div>
        </div>
        """, unsafe_allow_html=True)

    with ds_col3:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid #F59E0B;border-radius:10px;padding:1.25rem">
        <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;font-weight:600">GitHub Issues</div>
        <div style="font-size:2rem;font-weight:700;color:#0F172A;margin:0.5rem 0">{_ing['github']}</div>
        <div style="font-size:0.75rem;color:#64748B">Transformed public software issues</div>
        <div style="margin-top:0.75rem;font-size:0.72rem;color:#F59E0B;font-weight:500">VS Code · Kubernetes · Terraform</div>
        </div>
        """, unsafe_allow_html=True)

    with ds_col4:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid #8B5CF6;border-radius:10px;padding:1.25rem">
        <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;font-weight:600">HuggingFace ServiceNow</div>
        <div style="font-size:2rem;font-weight:700;color:#0F172A;margin:0.5rem 0">{_ing['huggingface']}</div>
        <div style="font-size:0.75rem;color:#64748B">Synthetic ServiceNow incidents</div>
        <div style="margin-top:0.75rem;font-size:0.72rem;color:#8B5CF6;font-weight:500">ITSM schema · Structured fields</div>
        </div>
        """, unsafe_allow_html=True)

    with ds_col5:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid #0EA5E9;border-radius:10px;padding:1.25rem">
        <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;font-weight:600">ServiceNow Live</div>
        <div style="font-size:1rem;font-weight:700;color:{_sn_color};margin:0.5rem 0">{_sn_status}</div>
        <div style="font-size:0.75rem;color:#64748B">Optional read connection; no ticket creation</div>
        <div style="margin-top:0.75rem;font-size:0.72rem;color:#0EA5E9;font-weight:500">REST API · Live incidents</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:1.5rem 0'></div>", unsafe_allow_html=True)

    # Section 2: Agent pipeline ----------------------------------------------------
    st.markdown("### Multi-Agent Pipeline")
    st.markdown(
        "<p style='color:#64748B;font-size:0.85rem;margin-bottom:1rem'>Support questions "
        "use the retrieval, answer, and triage graph. Verification and adaptive retry "
        "run when enabled; analytical questions use a separate tool path.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div style="display:flex;gap:0;align-items:stretch;margin-bottom:0.75rem">

      <div style="flex:1;background:#E8EDFF;border:1px solid #C7D2FE;border-radius:10px 0 0 10px;padding:1rem">
        <div style="font-weight:700;font-size:0.82rem;color:#3730A3;margin-bottom:0.25rem">Retrieval Agent</div>
        <div style="font-size:0.72rem;color:#4338CA;line-height:1.5">
        Searches ChromaDB for up to 3 tickets, applying the selected category filter. Broadens the query on retry.</div>
        <div style="margin-top:0.6rem;font-size:0.68rem;color:#6366F1;font-weight:500">ChromaDB · OpenAI Embeddings</div>
      </div>

      <div style="display:flex;align-items:center;padding:0 0.35rem;color:#CBD5E1;font-size:1.3rem">→</div>

      <div style="flex:1;background:#F0FDF4;border:1px solid #BBF7D0;border-radius:0;padding:1rem">
        <div style="font-weight:700;font-size:0.82rem;color:#166534;margin-bottom:0.25rem">Answer Agent</div>
        <div style="font-size:0.72rem;color:#15803D;line-height:1.5">
        Drafts an answer from retrieved ticket context and cites Ticket IDs.</div>
        <div style="margin-top:0.6rem;font-size:0.68rem;color:#16A34A;font-weight:500">GPT-4o-mini default · RAG grounding</div>
      </div>

      <div style="display:flex;align-items:center;padding:0 0.35rem;color:#CBD5E1;font-size:1.3rem">→</div>

      <div style="flex:1;background:#FFF1F2;border:1px solid #FECDD3;border-radius:0;padding:1rem">
        <div style="font-weight:700;font-size:0.82rem;color:#9F1239;margin-bottom:0.25rem">Verification Agent</div>
        <div style="font-size:0.72rem;color:#BE123C;line-height:1.5">
        When enabled, an independent LLM checks support answers against retrieved tickets and flags unsupported claims.</div>
        <div style="margin-top:0.6rem;font-size:0.68rem;color:#E11D48;font-weight:500">GPT-4o-mini judge · Groundedness</div>
      </div>

      <div style="display:flex;align-items:center;padding:0 0.35rem;color:#CBD5E1;font-size:1.3rem">→</div>

      <div style="flex:1;background:#FFFBEB;border:1px solid #FDE68A;border-radius:0 10px 10px 0;padding:1rem">
        <div style="font-weight:700;font-size:0.82rem;color:#92400E;margin-bottom:0.25rem">Triage Agent</div>
        <div style="font-size:0.72rem;color:#B45309;line-height:1.5">
        Uses the retrieval routing cutoff and grounding result to label Tier 1 responses or recommend Tier 2 review.</div>
        <div style="margin-top:0.6rem;font-size:0.68rem;color:#D97706;font-weight:500">LangGraph · Auto-routing</div>
      </div>

    </div>
    <div style="font-size:0.72rem;color:#94A3B8;margin-bottom:1.5rem">
    ↩ When enabled, adaptive retry broadens a support query after a failed grounding check or a retrieval score below the routing cutoff (max 2 retries).
    </div>
    """, unsafe_allow_html=True)

    # Section 3: Current capabilities ------------------------------------------------
    st.markdown("### Current Capabilities")

    pf_row1_col1, pf_row1_col2, pf_row1_col3 = st.columns(3)
    with pf_row1_col1:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">PII Detection</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        Microsoft Presidio masks detected typed text and retrieved content before support model calls. The local audit log stores query metadata.</div>
        </div>
        """, unsafe_allow_html=True)

    with pf_row1_col2:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Live Evaluation</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        When enabled, completed support-answer runs record relevance, grounding status, and latency. Analytical tool answers are excluded.</div>
        </div>
        """, unsafe_allow_html=True)

    with pf_row1_col3:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">File Ingestion Utility</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        An optional file watcher can ingest new tickets from new_tickets/ when started manually.</div>
        </div>
        """, unsafe_allow_html=True)

    pf_row2_col1, pf_row2_col2, pf_row2_col3 = st.columns(3)
    with pf_row2_col1:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Multi-source Pipeline</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        Connectors for Kaggle, GitHub Issues, HuggingFace, and live ServiceNow REST API. Normalization handles schema differences.</div>
        </div>
        """, unsafe_allow_html=True)

    with pf_row2_col2:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Retrieval Review</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        A raw retrieval relevance signal below the configured routing cutoff recommends Tier 2 review. It is not a probability.</div>
        </div>
        """, unsafe_allow_html=True)

    with pf_row2_col3:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Fine-tuning Pipeline</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        Manual preparation splits the current internal-style records into training and validation examples. A separate harness can compare a candidate model; none is active by default.</div>
        </div>
        """, unsafe_allow_html=True)

    pf_row3_col1, pf_row3_col2, pf_row3_col3 = st.columns(3)
    with pf_row3_col1:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Answer Verification</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        When enabled, an independent model checks support answers against retrieved tickets. Analytical tool answers bypass this check.</div>
        </div>
        """, unsafe_allow_html=True)

    with pf_row3_col2:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Adaptive Retry</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        When enabled, failed grounding or a low retrieval signal triggers a broader support-query search (up to 2 retries).</div>
        </div>
        """, unsafe_allow_html=True)

    with pf_row3_col3:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">ServiceNow Live</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        Use the System tab button to check the optional connection; the analytical status tool can request live incident counts when configured.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;margin-top:1rem">
    <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Resilience, Security &amp; Rate Limiting</div>
    <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
    Input validation, per-session rate limits, transient-error retries, and fail-closed text masking protect the request path. Degraded responses recommend human review.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;margin-top:1rem">
    <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Multi-modal Input</div>
    <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
    When enabled, users can upload error screenshots alongside their question. Vision extracts the technical issue; the resulting text follows the support-answer path with text masking and optional verification.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;margin-top:1rem">
    <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Testing, CI/CD &amp; Evaluation Harness</div>
    <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
    GitHub Actions runs fast tests on pushes and pull requests to main; API-gated integration tests are separate. A manual evaluation harness records grounding, relevance, citation-ID validity, refusal behavior, and latency on a fixed question set.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin:1rem 0'></div>", unsafe_allow_html=True)

    # Section 4: Live Evaluation Metrics --------------------------------------------
    st.markdown("### Live Evaluation Metrics")
    st.markdown(
        "<p style='color:#64748B;font-size:0.85rem'>"
        "When enabled, live evaluation records completed support-answer runs. "
        "Analytical tool answers are not evaluated here. Grounded Rate uses "
        "only checked runs; relevance is an LLM judge score, not answer accuracy.</p>",
        unsafe_allow_html=True,
    )

    live_stats = get_live_eval_summary()
    if live_stats["total_scored"] > 0:
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            st.metric("Queries Scored", live_stats["total_scored"])
        with lc2:
            st.metric("Grounded Rate", format_grounded_rate(live_stats["grounded_rate"]))
        with lc3:
            st.metric("Avg Judge Relevance", f"{live_stats['avg_relevance']:.2f}/1")
        with lc4:
            st.metric("Avg Latency", f"{live_stats['avg_latency']:.1f}s")
    else:
        st.markdown(
            "<p style='color:#94A3B8;font-size:0.8rem'>"
            "No support answers scored yet. Live evaluation must be enabled "
            "for metrics to appear here.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin:1rem 0'></div>", unsafe_allow_html=True)

    # Section 5: Tech stack ---------------------------------------------------------
    st.markdown("### Tech Stack")

    st.markdown("""
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:1.5rem;
    display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.25rem">

      <div>
        <div style="font-size:0.7rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;margin-bottom:0.75rem">
        AI & Orchestration</div>
        <div style="font-size:0.8rem;color:#374151;line-height:2">
        GPT-4o-mini default (OpenAI)<br>
        LangChain + LangGraph<br>
        LangSmith Tracing (if configured)<br>
        text-embedding-3-small<br>
        Multi-agent StateGraph
        </div>
      </div>

      <div>
        <div style="font-size:0.7rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;margin-bottom:0.75rem">
        Data & Storage</div>
        <div style="font-size:0.8rem;color:#374151;line-height:2">
        ChromaDB (vector store)<br>
        Kaggle Support Dataset<br>
        GitHub Issues API<br>
        JSONL audit logging
        </div>
      </div>

      <div>
        <div style="font-size:0.7rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;margin-bottom:0.75rem">
        Security & Deployment</div>
        <div style="font-size:0.8rem;color:#374151;line-height:2">
        Microsoft Presidio (PII)<br>
        Watchdog file monitor (manual)<br>
        Streamlit frontend<br>
        Python 3.12<br>
        Railway deployment
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin:1rem 0'></div>", unsafe_allow_html=True)

    # Section 4b: Effective configuration -------------------------------------------
    st.markdown("### Effective Configuration")
    st.markdown(
        "<p style='color:#64748B;font-size:0.85rem'>"
        "Every value below is environment-overridable via <code>config.py</code> "
        "— models, thresholds, retry behavior, and feature flags — with no code "
        "changes. Shown here as the currently active configuration.</p>",
        unsafe_allow_html=True,
    )

    _cfg = config.summary()

    def _flag_pill(label: str, on: bool) -> str:
        color = "#10B981" if on else "#94A3B8"
        bg = "#ECFDF5" if on else "#F1F5F9"
        state = "on" if on else "off"
        return (
            f"<span style='display:inline-block;margin:0.15rem 0.35rem 0.15rem 0;"
            f"background:{bg};color:{color};font-size:0.72rem;font-weight:600;"
            f"border-radius:50px;padding:0.2rem 0.65rem'>{label}: {state}</span>"
        )

    st.markdown(f"""
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:1.5rem;
    display:grid;grid-template-columns:1fr 1fr;gap:1.25rem">
      <div>
        <div style="font-size:0.7rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;margin-bottom:0.75rem">
        Settings</div>
        <div style="font-size:0.8rem;color:#374151;line-height:2">
        Environment: <strong>{escape_html(str(_cfg['environment']))}</strong><br>
        Chat model: <strong>{escape_html(str(_cfg['chat_model']))}</strong><br>
        Embedding model: <strong>{escape_html(str(_cfg['embedding_model']))}</strong><br>
        Retrieval k: <strong>{_cfg['retrieval_k']}</strong><br>
        Retrieval review cutoff (raw score): <strong>{_cfg['confidence_medium']}</strong><br>
        Max retries: <strong>{_cfg['max_retries']}</strong><br>
        Max question length: <strong>{_cfg['max_question_length']}</strong>
        </div>
      </div>
      <div>
        <div style="font-size:0.7rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;margin-bottom:0.75rem">
        Feature Flags</div>
        <div>
        {_flag_pill("Live eval", _cfg['live_eval_enabled'])}
        {_flag_pill("Verification agent", _cfg['verification_enabled'])}
        {_flag_pill("Adaptive retry", _cfg['adaptive_retry_enabled'])}
        {_flag_pill("ServiceNow live", _cfg['servicenow_live_enabled'])}
        {_flag_pill("Image upload", _cfg['image_upload_enabled'])}
        {_flag_pill("Rate limiting", _cfg['rate_limiting_enabled'])}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin:1rem 0'></div>", unsafe_allow_html=True)

    # Section 4c: Adaptive retry rate for support answers in this session -----
    st.markdown("### Adaptive Retry")
    _retried, _support_count, _retry_rate = session_retry_summary(
        st.session_state.get("messages", [])
    )
    if _support_count:
        rr_col1, rr_col2, rr_col3 = st.columns(3)
        with rr_col1:
            st.metric("Retry Rate", f"{_retry_rate}%")
        with rr_col2:
            st.metric("Broadened", _retried)
        with rr_col3:
            st.metric("Support answers", _support_count)
        st.markdown(
            f"<p style='color:#94A3B8;font-size:0.78rem'>"
            f"{_retry_rate}% of support answers this session needed a broadened search — "
            "a grounding check or retrieval signal triggered another search pass.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<p style='color:#94A3B8;font-size:0.8rem'>"
            "No support answers yet this session — ask a support question in the Chat tab to see the "
            "adaptive-retry rate.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin:1rem 0'></div>", unsafe_allow_html=True)

    # Section 5: Feedback summary ---------------------------------------------------
    st.markdown("### Feedback Summary")

    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE) as _fb_f:
                _feedback_entries = [_json.loads(line) for line in _fb_f if line.strip()]
        except (OSError, _json.JSONDecodeError):
            logger.warning("Feedback history is unavailable")
            _feedback_entries = []

        if _feedback_entries:
            _positive = sum(1 for e in _feedback_entries if e.get("rating") == "positive")
            _total = len(_feedback_entries)
            _satisfaction = int((_positive / _total) * 100) if _total > 0 else 0

            fb_sum_col1, fb_sum_col2, fb_sum_col3 = st.columns(3)
            with fb_sum_col1:
                st.metric("Total Feedback", _total)
            with fb_sum_col2:
                st.metric("Helpful", _positive)
            with fb_sum_col3:
                st.metric("Satisfaction", f"{_satisfaction}%")
        else:
            st.markdown(
                "<p style='color:#94A3B8;font-size:0.8rem'>"
                "No feedback collected yet.</p>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<p style='color:#94A3B8;font-size:0.8rem'>"
            "No feedback collected yet.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin:1rem 0'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:2rem;padding-top:1rem;
    border-top:1px solid #E2E8F0;text-align:center">
      <span style="color:#94A3B8;font-size:0.75rem">
      Built by <strong style="color:#0F172A">
      Yasaswi</strong></span>
    </div>
    """, unsafe_allow_html=True)
