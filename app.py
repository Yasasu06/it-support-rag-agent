"""
Streamlit chat interface for the Enterprise IT Support RAG Agent.

Wraps the grounded answer chain in rag.py with a clean, light-themed chat
UI: suggestion chips, a confidence badge derived from retrieval similarity,
and an expandable view of the source tickets behind every answer.

Run with:
    streamlit run app.py
"""

import html
import os
import json as _json
import uuid

from dotenv import load_dotenv

# Load environment variables before anything else touches OpenAI/Chroma.
load_dotenv()

import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from data.tickets import TICKETS
from rag import (
    get_answer,
    get_answer_with_memory,
    clear_conversation_memory,
    reformulate_query,
    get_answer_streaming,
)
from agent_pipeline import run_agent_pipeline, run_tool_agent
from security import get_audit_summary
from live_eval import get_live_eval_summary
from error_handling import validate_user_input
from config import config
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


def is_analytical_query(question: str) -> bool:
    q_lower = question.lower()
    return any(kw in q_lower for kw in ANALYTICAL_KEYWORDS)

FEEDBACK_FILE = os.path.join(
    os.path.dirname(__file__),
    "feedback_log.jsonl"
)


def log_feedback(
    question: str,
    answer: str,
    rating: str,
    timestamp: str = None,
) -> None:
    from datetime import datetime

    entry = {
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        "question": question[:200],
        "answer": answer[:300],
        "rating": rating,
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(_json.dumps(entry) + "\n")

# Configuration --------------------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_DB_PATH", "chroma_db")
COLLECTION_NAME = "it_support_tickets"
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

# level -> exact badge HTML
CONFIDENCE_HTML = {
    "High": (
        '<div style="display:inline-flex;align-items:center;gap:0.35rem;'
        'background:#DCFCE7;color:#166534;border-radius:50px;'
        'padding:0.3rem 0.85rem;font-size:0.78rem;font-weight:600;'
        'margin-top:0.5rem;">High Confidence</div>'
    ),
    "Medium": (
        '<div style="display:inline-flex;align-items:center;gap:0.35rem;'
        'background:#FEF9C3;color:#854D0E;border-radius:50px;'
        'padding:0.3rem 0.85rem;font-size:0.78rem;font-weight:600;'
        'margin-top:0.5rem;">Medium Confidence — verify with IT team</div>'
    ),
    "Low": (
        '<div style="display:inline-flex;align-items:center;gap:0.35rem;'
        'background:#FEE2E2;color:#991B1B;border-radius:50px;'
        'padding:0.3rem 0.85rem;font-size:0.78rem;font-weight:600;'
        'margin-top:0.5rem;">Low Confidence — escalate to Tier 2</div>'
    ),
}

# Exact warning blocks injected in place of st.info()/st.warning().
PII_HTML = (
    '<div style="background:#E8EDFF;border-left:3px solid #2453FF;border-radius:8px;'
    'padding:0.75rem 1rem;margin-top:0.5rem;">'
    '<div style="color:#3730A3;font-weight:600;font-size:0.85rem;">Privacy Protection Active</div>'
    '<div style="color:#3730A3;font-size:0.8rem;margin-top:0.25rem;">'
    "Personal information detected in your message was automatically masked "
    "before processing to protect your privacy."
    "</div></div>"
)

ESCALATION_HTML = (
    '<div style="background:#FFFBEB;border-left:3px solid #F59E0B;border-radius:8px;'
    'padding:0.75rem 1rem;margin-top:0.5rem;">'
    '<div style="color:#92400E;font-weight:600;font-size:0.85rem;">Tier 2 Escalation Required</div>'
    '<div style="color:#92400E;font-size:0.8rem;margin-top:0.25rem;">'
    "This issue could not be resolved with available knowledge base matches. "
    "A ticket has been queued in ServiceNow for Tier 2 review."
    "</div></div>"
)

ERROR_MESSAGE = (
    "Could not connect to knowledge base. "
    "Ensure ingest.py has been run and your API key is valid."
)

_summary_path = os.path.join(
    os.path.dirname(__file__),
    "ingestion_summary.json"
)
if os.path.exists(_summary_path):
    with open(_summary_path) as _f:
        _summary = _json.load(_f)
    _ticket_count = str(_summary.get("total", 150))
    _source_text = (
        f"Kaggle + GitHub + Internal"
    )
else:
    _ticket_count = "150"
    _source_text = "Internal only"

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
  --sidebar-bg: #0D1117;
  --sidebar-ink: #C9D1D9;
  --sidebar-muted: #6E7681;
  --sidebar-border: #21262D;
  --accent: #2453FF;
  --accent-soft: #E8EDFF;
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
      ellipse 1200px 800px at 20% -10%,
      rgba(36,83,255,0.06) 0%,
      transparent 50%
    ),
    radial-gradient(
      ellipse 1000px 600px at 100% 10%,
      rgba(192,38,212,0.05) 0%,
      transparent 50%
    ),
    #F7F7F4 !important;
}

.main .block-container {
  max-width: 820px !important;
  padding: 2rem 2rem 6rem 2rem !important;
}

/* Sidebar - terminal panel */
/* Force the sidebar to stay open and on-screen regardless of Streamlit's
   collapse state. A persisted/collapsed state (remembered across reloads) or a
   narrow-viewport auto-collapse otherwise slides it off-screen via transform /
   zero width. The pre-redesign CSS pinned it open this way; the redesign
   dropped it, which is why the sidebar "stopped rendering". */
section[data-testid="stSidebar"] {
  background-color: var(--sidebar-bg) !important;
  border-right: 1px solid var(--sidebar-border) !important;
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

/* Sidebar accent glow (new rule; does NOT modify the force-open block above) */
section[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] {
  background: linear-gradient(
    180deg,
    rgba(36,83,255,0.15) 0%,
    transparent 200px
  ), var(--sidebar-bg);
}

/* Sidebar metric cards - instrument readout style */
section[data-testid="stSidebar"] [data-testid="stMetric"] {
  background: #161B22 !important;
  border: 1px solid var(--sidebar-border) !important;
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
.app-header {
  display: flex;
  align-items: center;
  gap: 0.9rem;
  padding: 1.1rem 1.3rem;
  margin-bottom: 1.5rem;
  border-radius: 14px;
  background: linear-gradient(120deg, #2453FF 0%, #7C3AED 55%, #C026D3 100%);
  box-shadow: 0 8px 24px rgba(36,83,255,0.18);
}
.app-header .badge {
  width: 42px;
  height: 42px;
  border-radius: 10px;
  background: rgba(255,255,255,0.2);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  color: #FFFFFF;
  font-size: 0.95rem;
}
.app-header .title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.05rem;
  color: #FFFFFF;
}
.app-header .subtitle {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  color: rgba(255,255,255,0.85);
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

/* Signal readout - the signature element, replaces
   confidence badge + match quality bar */
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
.signal-readout .ticks {
  display: flex; gap: 2px; align-items: flex-end;
}
.signal-readout .tick {
  width: 3px; background: var(--border); border-radius: 1px;
}
.signal-readout.grounded .signal-value { color: #059669 !important; }
.signal-readout.grounded .tick.on { background: #059669; }
.signal-readout.warning .signal-value { color: #D97706 !important; }
.signal-readout.warning .tick.on { background: #D97706; }
.signal-readout.danger .signal-value { color: #DC2626 !important; }
.signal-readout.danger .tick.on { background: #DC2626; }

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
.st-key-chip_1 button {
  border-left: 4px solid #2453FF !important; background: #E8EDFF !important; color: #2453FF !important;
}
.st-key-chip_2 button {
  border-left: 4px solid #C026D3 !important; background: #FBE5FA !important; color: #C026D3 !important;
}
.st-key-chip_3 button {
  border-left: 4px solid #DB2777 !important; background: #FCE4EF !important; color: #DB2777 !important;
}
.st-key-chip_4 button {
  border-left: 4px solid #0891B2 !important; background: #E0F7FB !important; color: #0891B2 !important;
}
[class*="st-key-chip_"] button { font-weight: 600 !important; text-align: left !important; }
[class*="st-key-chip_"] button:hover {
  filter: brightness(0.97);
  box-shadow: 0 3px 10px rgba(18,20,28,0.10) !important;
}
</style>
    """,
    unsafe_allow_html=True,
)


# Cached resources -------------------------------------------------------------
@st.cache_resource
def get_app_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


# Helpers ------------------------------------------------------------------------
def escape_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def confidence_from_score(score: float) -> str:
    if score >= config.CONFIDENCE_HIGH_THRESHOLD:
        return "High"
    if score >= config.CONFIDENCE_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def render_signal_readout(level: str, top_score: float) -> None:
    # The signature "signal readout" — one component replacing the old
    # confidence badge + match-quality bar. The score/threshold LOGIC is
    # unchanged (same similarity_pct formula and level bands); only the visual
    # form changes: a mono percentage plus five rising tick marks.
    similarity_pct = min(int(top_score * 400), 99) if top_score > 0 else 0
    tick_count = min(5, max(1, round(similarity_pct / 20)))
    ticks_html = "".join(
        f'<div class="tick {"on" if i < tick_count else ""}" '
        f'style="height:{8 + i * 3}px"></div>'
        for i in range(5)
    )
    level_class = (
        "grounded" if level == "High"
        else "warning" if level == "Medium"
        else "danger"
    )
    st.markdown(
        f'<div class="signal-readout {level_class}">'
        f'<span class="signal-label">Confidence</span>'
        f'<span class="signal-value">{similarity_pct}%</span>'
        f'<div class="ticks">{ticks_html}</div>'
        f'<span class="signal-label">{level.upper()}</span>'
        f'</div>',
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
                    background: linear-gradient(135deg, {colors['bg']} 0%, #FFFFFF 40%);
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


def stream_text(text: str):
    """Yields a string word-by-word so it can be fed to st.write_stream."""
    import re

    for chunk in re.split(r"(\s+)", text):
        if chunk:
            yield chunk


def render_feedback_buttons(question: str, answer: str, msg_key: str) -> None:
    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 8])
    with fb_col1:
        if st.button("Helpful", key=f"up_{msg_key}", help="This answer was helpful"):
            log_feedback(question, answer, "positive")
            st.toast("Thanks for the feedback!")
    with fb_col2:
        if st.button("Not helpful", key=f"down_{msg_key}", help="This answer wasn't helpful"):
            log_feedback(question, answer, "negative")
            st.toast("Feedback recorded — we'll work on improving this.")


def render_assistant_message(msg: dict, idx: int) -> None:
    with st.chat_message("assistant"):
        st.markdown(msg["content"])
        if msg.get("pii_detected"):
            st.markdown(PII_HTML, unsafe_allow_html=True)
        if msg.get("confidence"):
            render_signal_readout(msg["confidence"], msg.get("confidence_score") or 0.0)
        if msg.get("escalation"):
            st.markdown(ESCALATION_HTML, unsafe_allow_html=True)
        if msg.get("sources"):
            render_sources_expander(msg["sources"], idx)
        if msg["content"] != ERROR_MESSAGE:
            render_feedback_buttons(msg.get("question", ""), msg["content"], str(idx))


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

    # Analytical questions ("how many", "average resolution time", ...) go
    # to the tool-calling agent as-is — the category-focus suffix, recent
    # conversation history, and reformulate_query rewrite below are tuned
    # for semantic retrieval/grounding and would just add noise to a
    # tool-selection prompt, so we skip them for this path.
    augmented_question = pipeline_question
    if not analytical:
        if category != "All Categories":
            augmented_question = f"{pipeline_question} Focus on {category} related tickets if possible."

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
        augmented_question = reformulate_query(augmented_question)

    st.session_state.messages.append(
        {"role": "user", "content": display_text, "image_extract": image_extract}
    )
    render_user_message(display_text, image_extract)

    with st.chat_message("assistant"):
        try:
            with st.spinner(":shimmer[Searching knowledge base...]"):
                if analytical:
                    result = run_tool_agent(pipeline_question)
                    sources, top_score = [], result["confidence_score"]
                else:
                    result = run_agent_pipeline(augmented_question)
                    sources = result.get("sources", [])
                    top_score = result.get("confidence_score") or 0.0

                answer = result["answer"]
                escalation = result["escalation"]
                tier = result["tier"]
                pii_detected = result["pii_detected"]

            level = confidence_from_score(top_score)

            st.write_stream(stream_text(answer))
            if pii_detected:
                st.markdown(PII_HTML, unsafe_allow_html=True)
            render_signal_readout(level, top_score)
            if escalation:
                st.markdown(ESCALATION_HTML, unsafe_allow_html=True)
            if sources:
                render_sources_expander(sources, len(st.session_state.messages))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "question": display_text,
                    "confidence": level,
                    "confidence_score": top_score,
                    "sources": sources,
                    "escalation": escalation,
                    "tier": tier,
                    "pii_detected": pii_detected,
                }
            )
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


# Anonymous per-session visit logging ---------------------------------------------
# Logs a timestamp + random session id once per browser session (no PII, no IP).
# Guarded by "session_id not in session_state" so it fires once per session, not
# on every question or rerun.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    log_visit(st.session_state.session_id)

# Private, password-protected analytics view ---------------------------------------
# Only appears when the URL carries ?admin=<secret> matching the ADMIN_ANALYTICS_KEY
# env var. Regular users never see it and cannot reach it without the secret; if
# the env var is unset, the feature is entirely inert.
ADMIN_KEY = os.getenv("ADMIN_ANALYTICS_KEY", "")
if ADMIN_KEY and st.query_params.get("admin") == ADMIN_KEY:
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
    st.markdown(
        "*This is a private admin view. "
        "Remove ?admin=... from the URL to return to "
        "the normal app.*"
    )
    st.stop()


# Sidebar -------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:0.6rem;'>"
        "<div>"
        "<div style='font-size:1.1rem;font-weight:700;color:#F1F5F9;'>IT Support AI</div>"
        "<div style='font-size:0.75rem;color:#94A3B8;'>Powered by RAG · GPT-4o-mini</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        "<p style='color:#64748B;font-size:0.72rem;"
        "line-height:1.5;word-wrap:break-word;"
        "overflow-wrap:break-word;max-width:240px'>"
        "AI assistant grounded in enterprise IT "
        "incident data. Cites sources and flags "
        "confidence on every response.</p>",
        unsafe_allow_html=True
    )

    category = st.selectbox("Filter by category", CATEGORIES, key="category_filter")
    st.divider()

    st.markdown("**Knowledge Base**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Tickets", _ticket_count)
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
        st.markdown("**System Health**")
        col_c, col_d = st.columns(2)
        with col_c:
            st.metric("Queries", summary["total_queries"])
            st.metric("Escalated", summary["escalated_queries"])
        with col_d:
            st.metric("PII Caught", summary["pii_queries"])
            st.metric("Avg Conf", f"{summary['avg_confidence']:.0%}")

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
    <div class="subtitle">{_ticket_count} INCIDENTS · {len(CATEGORIES) - 1} CATEGORIES · GPT-4O-MINI</div>
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
                    logger.error("Image extraction failed: %s", extraction["error"])
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
    st.markdown("### Data Sources")

    _ing_path = os.path.join(os.path.dirname(__file__), "ingestion_summary.json")
    if os.path.exists(_ing_path):
        with open(_ing_path) as _ing_f:
            _ing = _json.load(_ing_f)
    else:
        _ing = {"synthetic": 150, "kaggle": 0, "github": 0, "huggingface": 0, "total": 150}

    _sn_configured = servicenow_configured()
    _sn_status = "Not configured"
    _sn_color = "#94A3B8"
    if _sn_configured:
        _sn_result = servicenow_test_connection()
        if _sn_result["success"]:
            _sn_status = "Live — Connected"
            _sn_color = "#10B981"
        else:
            _sn_status = "Configured — Connection Failed"
            _sn_color = "#EF4444"

    ds_col1, ds_col2, ds_col3, ds_col4, ds_col5 = st.columns(5)
    with ds_col1:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid #2453FF;border-radius:10px;padding:1.25rem">
        <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;font-weight:600">Internal Synthetic</div>
        <div style="font-size:2rem;font-weight:700;color:#0F172A;margin:0.5rem 0">{_ing.get('synthetic', 150)}</div>
        <div style="font-size:0.75rem;color:#64748B">Enterprise IT Support incident patterns</div>
        <div style="margin-top:0.75rem;font-size:0.72rem;color:#2453FF;font-weight:500">8 categories · Structured format</div>
        </div>
        """, unsafe_allow_html=True)

    with ds_col2:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid #10B981;border-radius:10px;padding:1.25rem">
        <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;font-weight:600">Kaggle Dataset</div>
        <div style="font-size:2rem;font-weight:700;color:#0F172A;margin:0.5rem 0">{_ing.get('kaggle', 0)}</div>
        <div style="font-size:0.75rem;color:#64748B">Real enterprise support tickets</div>
        <div style="margin-top:0.75rem;font-size:0.72rem;color:#10B981;font-weight:500">Real-world · Multi-industry</div>
        </div>
        """, unsafe_allow_html=True)

    with ds_col3:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid #F59E0B;border-radius:10px;padding:1.25rem">
        <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;font-weight:600">GitHub Issues</div>
        <div style="font-size:2rem;font-weight:700;color:#0F172A;margin:0.5rem 0">{_ing.get('github', 0)}</div>
        <div style="font-size:0.75rem;color:#64748B">Real open source IT issues</div>
        <div style="margin-top:0.75rem;font-size:0.72rem;color:#F59E0B;font-weight:500">VS Code · Kubernetes · Terraform</div>
        </div>
        """, unsafe_allow_html=True)

    with ds_col4:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid #8B5CF6;border-radius:10px;padding:1.25rem">
        <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;font-weight:600">HuggingFace ServiceNow</div>
        <div style="font-size:2rem;font-weight:700;color:#0F172A;margin:0.5rem 0">{_ing.get('huggingface', 0)}</div>
        <div style="font-size:0.75rem;color:#64748B">Synthetic ServiceNow incidents</div>
        <div style="margin-top:0.75rem;font-size:0.72rem;color:#8B5CF6;font-weight:500">ITSM schema · Structured fields</div>
        </div>
        """, unsafe_allow_html=True)

    with ds_col5:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid #0EA5E9;border-radius:10px;padding:1.25rem">
        <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;font-weight:600">ServiceNow Live</div>
        <div style="font-size:1rem;font-weight:700;color:{_sn_color};margin:0.5rem 0">{_sn_status}</div>
        <div style="font-size:0.75rem;color:#64748B">Real-time connection, not batch</div>
        <div style="margin-top:0.75rem;font-size:0.72rem;color:#0EA5E9;font-weight:500">REST API · Live incidents</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:1.5rem 0'></div>", unsafe_allow_html=True)

    # Section 2: Agent pipeline ----------------------------------------------------
    st.markdown("### Multi-Agent Pipeline")
    st.markdown(
        "<p style='color:#64748B;font-size:0.85rem;margin-bottom:1rem'>Every query flows through "
        "a LangGraph orchestrated four-agent pipeline before returning an answer. "
        "Adaptive retry broadens the query and reruns the full pipeline up to 2×.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div style="display:flex;gap:0;align-items:stretch;margin-bottom:0.75rem">

      <div style="flex:1;background:#E8EDFF;border:1px solid #C7D2FE;border-radius:10px 0 0 10px;padding:1rem">
        <div style="font-weight:700;font-size:0.82rem;color:#3730A3;margin-bottom:0.25rem">Retrieval Agent</div>
        <div style="font-size:0.72rem;color:#4338CA;line-height:1.5">
        Searches ChromaDB via semantic similarity. Returns top 3 tickets with scores. Broadens query on retry.</div>
        <div style="margin-top:0.6rem;font-size:0.68rem;color:#6366F1;font-weight:500">ChromaDB · OpenAI Embeddings</div>
      </div>

      <div style="display:flex;align-items:center;padding:0 0.35rem;color:#CBD5E1;font-size:1.3rem">→</div>

      <div style="flex:1;background:#F0FDF4;border:1px solid #BBF7D0;border-radius:0;padding:1rem">
        <div style="font-weight:700;font-size:0.82rem;color:#166534;margin-bottom:0.25rem">Answer Agent</div>
        <div style="font-size:0.72rem;color:#15803D;line-height:1.5">
        Generates grounded answer using retrieved context only. Cites Ticket IDs. Refuses out-of-scope questions.</div>
        <div style="margin-top:0.6rem;font-size:0.68rem;color:#16A34A;font-weight:500">GPT-4o-mini · RAG grounding</div>
      </div>

      <div style="display:flex;align-items:center;padding:0 0.35rem;color:#CBD5E1;font-size:1.3rem">→</div>

      <div style="flex:1;background:#FFF1F2;border:1px solid #FECDD3;border-radius:0;padding:1rem">
        <div style="font-weight:700;font-size:0.82rem;color:#9F1239;margin-bottom:0.25rem">Verification Agent</div>
        <div style="font-size:0.72rem;color:#BE123C;line-height:1.5">
        Independent LLM fact-checks every answer against source tickets. Flags hallucinated claims for retry.</div>
        <div style="margin-top:0.6rem;font-size:0.68rem;color:#E11D48;font-weight:500">GPT-4o-mini judge · Groundedness</div>
      </div>

      <div style="display:flex;align-items:center;padding:0 0.35rem;color:#CBD5E1;font-size:1.3rem">→</div>

      <div style="flex:1;background:#FFFBEB;border:1px solid #FDE68A;border-radius:0 10px 10px 0;padding:1rem">
        <div style="font-weight:700;font-size:0.82rem;color:#92400E;margin-bottom:0.25rem">Triage Agent</div>
        <div style="font-size:0.72rem;color:#B45309;line-height:1.5">
        Evaluates confidence score and groundedness. Routes to Tier 1 or Tier 2 escalation automatically.</div>
        <div style="margin-top:0.6rem;font-size:0.68rem;color:#D97706;font-weight:500">LangGraph · Auto-routing</div>
      </div>

    </div>
    <div style="font-size:0.72rem;color:#94A3B8;margin-bottom:1.5rem">
    ↩ Adaptive retry: if verification fails or confidence is low, a broadened query reruns the pipeline (max 2 retries).
    </div>
    """, unsafe_allow_html=True)

    # Section 3: Production features ------------------------------------------------
    st.markdown("### Production Features")

    pf_row1_col1, pf_row1_col2, pf_row1_col3 = st.columns(3)
    with pf_row1_col1:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">PII Detection</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        Microsoft Presidio masks emails, phones, and SSNs before any data reaches the LLM. Full audit trail logged.</div>
        </div>
        """, unsafe_allow_html=True)

    with pf_row1_col2:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Live Evaluation</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        Every production query scored in real time: relevance, groundedness, and latency. Metrics visible in System tab.</div>
        </div>
        """, unsafe_allow_html=True)

    with pf_row1_col3:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Real-time Ingestion</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        Watchdog file monitor auto-ingests new tickets dropped into new_tickets/ folder. ChromaDB updates in under 4 seconds.</div>
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
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Confidence Scoring</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        Calibrated similarity thresholds route low-confidence queries to Tier 2 escalation automatically. No silent failures.</div>
        </div>
        """, unsafe_allow_html=True)

    with pf_row2_col3:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Fine-tuning Pipeline</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        120-example dataset prepared and model comparison harness built and validated. System can serve a fine-tuned model via config the moment one exists — blocked on OpenAI platform access, not system readiness.</div>
        </div>
        """, unsafe_allow_html=True)

    pf_row3_col1, pf_row3_col2, pf_row3_col3 = st.columns(3)
    with pf_row3_col1:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Answer Verification</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        Independent GPT-4o-mini judge verifies every answer against source tickets before delivery. Flags hallucinated claims.</div>
        </div>
        """, unsafe_allow_html=True)

    with pf_row3_col2:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Adaptive Retry</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        On failed verification or low confidence, query is automatically broadened and the full pipeline reruns (up to 2 retries).</div>
        </div>
        """, unsafe_allow_html=True)

    with pf_row3_col3:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;height:140px">
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">ServiceNow Live</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        Real-time REST API connection to ServiceNow. Live incident counts and connection status checked on every query.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;margin-top:1rem">
    <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Resilience, Security &amp; Rate Limiting</div>
    <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
    Retry logic on transient API failures, graceful degradation to a safe escalated response on service outages, fail-closed PII masking, input validation, and per-session sliding-window rate limiting (blocks abuse before it can incur API cost) all keep the public demo stable and safe.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;margin-top:1rem">
    <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Multi-modal Input</div>
    <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
    Users can upload error screenshots alongside their question. GPT-4o-mini vision extracts the technical issue, which flows through the same RAG pipeline as typed questions — including PII masking, verification, and confidence scoring.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:1rem;margin-top:1rem">
    <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">Testing, CI/CD &amp; Evaluation Harness</div>
    <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
    A pytest suite of fast unit tests (zero API cost) runs automatically on every push via GitHub Actions, alongside integration tests for the live pipeline. Beyond pass/fail, an offline evaluation harness scores answers on five independent dimensions — groundedness, relevance, citation accuracy, refusal correctness, and latency — for structured before/after comparison across changes.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin:1rem 0'></div>", unsafe_allow_html=True)

    # Section 4: Live Evaluation Metrics --------------------------------------------
    st.markdown("### Live Evaluation Metrics")
    st.markdown(
        "<p style='color:#64748B;font-size:0.85rem'>"
        "Every production query is automatically scored in real time — "
        "not just a fixed test set.</p>",
        unsafe_allow_html=True,
    )

    live_stats = get_live_eval_summary()
    if live_stats["total_scored"] > 0:
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            st.metric("Queries Scored", live_stats["total_scored"])
        with lc2:
            st.metric("Grounded Rate", f"{live_stats['grounded_rate']:.0%}")
        with lc3:
            st.metric("Avg Relevance", f"{live_stats['avg_relevance']:.0%}")
        with lc4:
            st.metric("Avg Latency", f"{live_stats['avg_latency']:.1f}s")
    else:
        st.markdown(
            "<p style='color:#94A3B8;font-size:0.8rem'>"
            "No queries scored yet. Ask a question in the Chat tab to see "
            "live metrics appear here.</p>",
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
        GPT-4o-mini (OpenAI)<br>
        LangChain + LangGraph<br>
        LangSmith Tracing<br>
        text-embedding-3-small<br>
        Multi-agent StateGraph
        </div>
      </div>

      <div>
        <div style="font-size:0.7rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;margin-bottom:0.75rem">
        Data & Storage</div>
        <div style="font-size:0.8rem;color:#374151;line-height:2">
        ChromaDB (vector store)<br>
        Kaggle IT Dataset<br>
        GitHub Issues API<br>
        JSONL audit logging
        </div>
      </div>

      <div>
        <div style="font-size:0.7rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;margin-bottom:0.75rem">
        Security & Deployment</div>
        <div style="font-size:0.8rem;color:#374151;line-height:2">
        Microsoft Presidio (PII)<br>
        Watchdog file monitor<br>
        Streamlit frontend<br>
        Python 3.14<br>
        Railway / Streamlit Cloud
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
        Confidence High / Medium: <strong>{_cfg['confidence_high']} / {_cfg['confidence_medium']}</strong><br>
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

    # Section 5: Feedback summary ---------------------------------------------------
    st.markdown("### Feedback Summary")

    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE) as _fb_f:
            _feedback_entries = [_json.loads(line) for line in _fb_f if line.strip()]

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
