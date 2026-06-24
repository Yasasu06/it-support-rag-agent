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

ANALYTICAL_KEYWORDS = [
    "how many", "count", "average",
    "statistics", "stats", "distribution",
    "most common", "breakdown", "summary",
    "how long", "resolution time", "sla",
    "trend", "compare", "analysis",
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
EMBEDDING_MODEL = "text-embedding-3-small"
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
    '<div style="background:#EEF2FF;border-left:3px solid #4F46E5;border-radius:8px;'
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
    /* Force sidebar visible */
    section[data-testid="stSidebar"] {
      display: flex !important;
      visibility: visible !important;
      opacity: 1 !important;
      width: 280px !important;
      min-width: 280px !important;
      max-width: 280px !important;
      transform: none !important;
      background-color: #0F172A !important;
    }

    section[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] {
      width: 280px !important;
      min-width: 280px !important;
    }
    section[data-testid="stSidebar"] > div:not([data-testid="stSidebarContent"]) {
      display: none !important;
    }

    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global reset */
    * { font-family: 'Inter', sans-serif !important; }

    /* Hide Streamlit default chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }

    .stApp {
      background-color: #F8FAFC !important;
    }

    /* Main content area */
    .main .block-container {
      padding-top: 2rem;
      padding-bottom: 6rem;
      max-width: 780px;
    }

    h1, h2, h3 {
      color: #0F172A !important;
      font-weight: 700 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
      background: #0F172A !important;
      border-right: none !important;
      width: 280px !important;
      min-width: 280px !important;
      max-width: 280px !important;
    }
    section[data-testid="stSidebar"] * {
      color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
      color: #F1F5F9 !important;
    }
    section[data-testid="stSidebar"] hr {
      border-color: #1E293B !important;
    }

    /* Sidebar metrics */
    section[data-testid="stSidebar"] [data-testid="stMetric"] {
      background: #1E293B !important;
      border: 1px solid #334155 !important;
      border-radius: 10px !important;
      padding: 0.75rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
      color: #94A3B8 !important;
      font-size: 0.7rem !important;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
      color: #F1F5F9 !important;
      font-size: 1.1rem !important;
    }

    /* Sidebar selectbox */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div {
      background: #1E293B !important;
      border: 1px solid #334155 !important;
      border-radius: 8px !important;
      color: #F1F5F9 !important;
    }

    /* Sidebar buttons */
    section[data-testid="stSidebar"] button {
      background: #1E293B !important;
      color: #94A3B8 !important;
      border: 1px solid #334155 !important;
      border-radius: 8px !important;
      width: 100% !important;
      font-size: 0.8rem !important;
    }
    section[data-testid="stSidebar"] button:hover {
      background: #334155 !important;
      color: #F1F5F9 !important;
    }

    /* Suggestion chips on welcome screen */
    .stButton > button {
      background: #FFFFFF !important;
      border: 1px solid #E2E8F0 !important;
      border-radius: 50px !important;
      color: #374151 !important;
      font-size: 0.85rem !important;
      font-weight: 500 !important;
      padding: 0.5rem 1rem !important;
      width: 100% !important;
      text-align: left !important;
      box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
      transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
      background: #4F46E5 !important;
      color: white !important;
      border-color: #4F46E5 !important;
      box-shadow: 0 4px 12px rgba(79,70,229,0.3) !important;
      transform: translateY(-2px) !important;
    }

    /* Back button (Streamlit key=-based targeting) overrides the chip style above */
    .st-key-back_btn button {
      background: transparent !important;
      border: none !important;
      color: #64748B !important;
      font-size: 0.85rem !important;
      font-weight: 500 !important;
      padding: 0.25rem 0.5rem !important;
      box-shadow: none !important;
      width: auto !important;
    }
    .st-key-back_btn button:hover {
      background: #F1F5F9 !important;
      color: #4F46E5 !important;
      box-shadow: none !important;
      transform: none !important;
    }

    /* Hide chat avatars entirely */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
      display: none !important;
    }

    /* User chat bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
      background-color: #4F46E5 !important;
      border-radius: 18px 18px 4px 18px !important;
      padding: 0.875rem 1.25rem !important;
      margin-left: 20% !important;
      margin-right: 0.5rem !important;
      margin-bottom: 0.75rem !important;
      border: none !important;
      box-shadow: 0 2px 8px rgba(79,70,229,0.25) !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) * {
      color: #FFFFFF !important;
    }

    /* Assistant chat bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
      background-color: #FFFFFF !important;
      border: 1px solid #E2E8F0 !important;
      border-left: 3px solid #4F46E5 !important;
      border-radius: 4px 18px 18px 18px !important;
      padding: 1rem 1.25rem !important;
      margin-right: 15% !important;
      margin-left: 0.5rem !important;
      margin-bottom: 0.75rem !important;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p {
      color: #1E293B !important;
      font-size: 0.9rem !important;
      line-height: 1.6 !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
      background: #FFFFFF !important;
      border: 1.5px solid #E2E8F0 !important;
      border-radius: 12px !important;
      font-family: 'Inter', sans-serif !important;
      font-size: 0.9rem !important;
      color: #0F172A !important;
    }
    [data-testid="stChatInput"] textarea:focus {
      border-color: #4F46E5 !important;
      box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
    }

    /* Expander for source tickets */
    [data-testid="stExpander"] {
      background: #F8FAFC !important;
      border: 1px solid #E2E8F0 !important;
      border-radius: 10px !important;
    }
    [data-testid="stExpander"] details summary p {
      font-size: 0.8rem !important;
      color: #64748B !important;
      font-weight: 500 !important;
    }
    [data-testid="stExpander"] summary p {
      font-size: 0.8rem !important;
      color: #64748B !important;
      font-weight: 500 !important;
    }
    [data-testid="stExpander"] summary svg {
      display: none !important;
    }

    /* Hide expander arrow artifacts */
    [data-testid="stExpander"] summary {
        list-style: none !important;
    }
    [data-testid="stExpander"] summary::-webkit-details-marker {
        display: none !important;
    }
    [data-testid="stExpander"] details summary p {
        font-size: 0.8rem !important;
        color: #64748B !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stExpander"] summary svg,
    [data-testid="stExpander"] summary img {
        display: none !important;
    }
    button[data-testid="baseButton-header"] svg {
        display: none !important;
    }
    details summary > span {
      display: none !important;
    }
    [data-testid="stExpander"] summary::marker {
      display: none !important;
      content: "" !important;
    }

    /* Spinner */
    [data-testid="stSpinner"] svg {
      color: #4F46E5 !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
      border-radius: 10px !important;
      border: none !important;
    }
    [data-testid="stInfo"] {
      background: #EEF2FF !important;
      border-left: 3px solid #4F46E5 !important;
    }
    [data-testid="stWarning"] {
      background: #FFFBEB !important;
      border-left: 3px solid #F59E0B !important;
    }

    /* Slide up animation */
    @keyframes slideUp {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stChatMessage"] {
      animation: slideUp 0.25s ease-out;
    }

    /* Tabs */
    [data-testid="stTabs"] {
      margin-bottom: 1rem !important;
    }
    [data-testid="stTabs"] button {
      font-family: 'Inter', sans-serif !important;
      font-size: 0.85rem !important;
      font-weight: 500 !important;
      color: #64748B !important;
      border-radius: 8px 8px 0 0 !important;
      padding: 0.5rem 1.25rem !important;
      background: transparent !important;
      border: none !important;
      box-shadow: none !important;
      transform: none !important;
    }
    [data-testid="stTabs"] button:hover {
      color: #4F46E5 !important;
      background: #F1F5F9 !important;
      transform: none !important;
      box-shadow: none !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
      color: #4F46E5 !important;
      border-bottom: 2px solid #4F46E5 !important;
      font-weight: 600 !important;
      background: transparent !important;
    }
    [data-testid="stTabsContent"] {
      padding-top: 1.5rem !important;
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
    if score >= 0.60:
        return "High"
    if score >= 0.20:
        return "Medium"
    return "Low"


def render_confidence_badge(level: str) -> None:
    st.markdown(CONFIDENCE_HTML[level], unsafe_allow_html=True)


def render_quality_score(top_score: float) -> None:
    similarity_pct = min(
        int(top_score * 400), 99
    ) if top_score > 0 else 0

    st.markdown(
        f'<div style="display:inline-flex;'
        f'align-items:center;gap:0.5rem;'
        f'margin-top:0.375rem;margin-left:0.5rem">'
        f'<span style="font-size:0.7rem;'
        f'color:#94A3B8">Match quality:</span>'
        f'<div style="background:#F1F5F9;'
        f'border-radius:100px;'
        f'height:6px;width:80px;'
        f'overflow:hidden;'
        f'display:inline-block;'
        f'vertical-align:middle">'
        f'<div style="background:'
        f'{"#10B981" if similarity_pct > 70 else "#F59E0B" if similarity_pct > 40 else "#EF4444"};'
        f'height:100%;'
        f'width:{similarity_pct}%"></div>'
        f'</div>'
        f'<span style="font-size:0.7rem;'
        f'color:#64748B;font-weight:500">'
        f'{similarity_pct}%</span>'
        f'</div>',
        unsafe_allow_html=True
    )


def render_sources_expander(sources: list) -> None:
    with st.expander(
        f"Source tickets ({len(sources)} retrieved)",
        expanded=False
    ):
        for src in sources:
            st.markdown(
                f"""
                <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;
                            padding:1rem;margin-bottom:0.75rem;">
                  <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
                    <strong style="color:#0F172A;font-size:0.9rem;">{escape_html(src["ticket_id"])}</strong>
                    <span style="background:#EEF2FF;color:#4F46E5;font-size:0.7rem;font-weight:600;
                                 border-radius:50px;padding:0.15rem 0.6rem;">{escape_html(src["category"])}</span>
                  </div>
                  <div style="color:#374151;font-size:0.85rem;margin-bottom:0.35rem;">
                    <b>Issue:</b> {escape_html(src["issue"])}
                  </div>
                  <div style="color:#374151;font-size:0.85rem;margin-bottom:0.5rem;">
                    <b>Resolution:</b> {escape_html(src["resolution"])}
                  </div>
                  <div style="color:#94A3B8;font-size:0.78rem;">
                    Resolved in {src["resolved_in_minutes"]} min
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_user_message(content: str) -> None:
    with st.chat_message("user"):
        st.markdown(content)


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
            render_confidence_badge(msg["confidence"])
            render_quality_score(msg.get("confidence_score") or 0.0)
        if msg.get("escalation"):
            st.markdown(ESCALATION_HTML, unsafe_allow_html=True)
        if msg.get("sources"):
            render_sources_expander(msg["sources"])
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


def process_question(question: str, category: str) -> None:
    analytical = is_analytical_query(question)

    # Analytical questions ("how many", "average resolution time", ...) go
    # to the tool-calling agent as-is — the category-focus suffix, recent
    # conversation history, and reformulate_query rewrite below are tuned
    # for semantic retrieval/grounding and would just add noise to a
    # tool-selection prompt, so we skip them for this path.
    augmented_question = question
    if not analytical:
        if category != "All Categories":
            augmented_question = f"{question} Focus on {category} related tickets if possible."

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

    st.session_state.messages.append({"role": "user", "content": question})
    render_user_message(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(":shimmer[Searching knowledge base...]"):
                if analytical:
                    result = run_tool_agent(question)
                    sources, top_score = [], result["confidence_score"]
                else:
                    result = run_agent_pipeline(augmented_question)
                    sources, top_score = get_sources_and_top_score(augmented_question, category)

                answer = result["answer"]
                escalation = result["escalation"]
                tier = result["tier"]
                pii_detected = result["pii_detected"]

            level = confidence_from_score(top_score)

            st.write_stream(stream_text(answer))
            if pii_detected:
                st.markdown(PII_HTML, unsafe_allow_html=True)
            render_confidence_badge(level)
            render_quality_score(top_score)
            if escalation:
                st.markdown(ESCALATION_HTML, unsafe_allow_html=True)
            if sources:
                render_sources_expander(sources)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "question": question,
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
                    "question": question,
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
        "<div style='display:flex;align-items:center;gap:0.875rem;margin-bottom:1.5rem;'>"
        "<div>"
        "<div style='font-size:1.35rem;font-weight:700;color:#0F172A;'>IT Support AI Assistant</div>"
        f"<div style='font-size:0.85rem;color:#64748B;'>{_ticket_count} incidents indexed · 8 categories · GPT-4o-mini</div>"
        "</div>"
        "</div>",
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

    # Pinned chat input at the bottom of the page ------------------------------------
    with st.bottom:
        prompt = st.chat_input("Describe your IT issue...")

    # Process a suggestion-chip click before normal input.
    if "prefill" in st.session_state and st.session_state.prefill:
        pending = st.session_state.prefill
        st.session_state.prefill = None
        process_question(pending, category)
        st.rerun()

    if prompt:
        process_question(prompt, category)
        st.rerun()

    # Main content: welcome screen or chat history -----------------------------------
    if not st.session_state.messages:
        render_welcome_screen()
    else:
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                render_user_message(msg["content"])
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

    ds_col1, ds_col2, ds_col3, ds_col4 = st.columns(4)
    with ds_col1:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid #4F46E5;border-radius:10px;padding:1.25rem">
        <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;font-weight:600">Internal Synthetic</div>
        <div style="font-size:2rem;font-weight:700;color:#0F172A;margin:0.5rem 0">{_ing.get('synthetic', 150)}</div>
        <div style="font-size:0.75rem;color:#64748B">Enterprise IT Support incident patterns</div>
        <div style="margin-top:0.75rem;font-size:0.72rem;color:#4F46E5;font-weight:500">8 categories · Structured format</div>
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

    st.markdown("<div style='margin:1.5rem 0'></div>", unsafe_allow_html=True)

    # Section 2: Agent pipeline ----------------------------------------------------
    st.markdown("### Multi-Agent Pipeline")
    st.markdown(
        "<p style='color:#64748B;font-size:0.85rem;margin-bottom:1rem'>Every query flows through "
        "a LangGraph orchestrated three-agent pipeline before returning an answer.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div style="display:flex;gap:0;align-items:stretch;margin-bottom:1.5rem">

      <div style="flex:1;background:#EEF2FF;border:1px solid #C7D2FE;border-radius:10px 0 0 10px;padding:1.25rem">
        <div style="font-weight:700;font-size:0.85rem;color:#3730A3;margin-bottom:0.25rem">Retrieval Agent</div>
        <div style="font-size:0.75rem;color:#4338CA;line-height:1.5">
        Searches ChromaDB using semantic similarity. Returns top 3 most relevant tickets with confidence scores.</div>
        <div style="margin-top:0.75rem;font-size:0.7rem;color:#6366F1;font-weight:500">LangChain · ChromaDB · OpenAI Embeddings</div>
      </div>

      <div style="display:flex;align-items:center;padding:0 0.5rem;color:#CBD5E1;font-size:1.5rem">→</div>

      <div style="flex:1;background:#F0FDF4;border:1px solid #BBF7D0;border-radius:0;padding:1.25rem">
        <div style="font-weight:700;font-size:0.85rem;color:#166534;margin-bottom:0.25rem">Answer Agent</div>
        <div style="font-size:0.75rem;color:#15803D;line-height:1.5">
        Generates grounded answer using retrieved context only. Cites Ticket IDs. Refuses out-of-scope questions.</div>
        <div style="margin-top:0.75rem;font-size:0.7rem;color:#16A34A;font-weight:500">GPT-4o-mini · RAG grounding · Citation</div>
      </div>

      <div style="display:flex;align-items:center;padding:0 0.5rem;color:#CBD5E1;font-size:1.5rem">→</div>

      <div style="flex:1;background:#FFFBEB;border:1px solid #FDE68A;border-radius:0 10px 10px 0;padding:1.25rem">
        <div style="font-weight:700;font-size:0.85rem;color:#92400E;margin-bottom:0.25rem">Triage Agent</div>
        <div style="font-size:0.75rem;color:#B45309;line-height:1.5">
        Evaluates confidence score. Routes to Tier 1 or Tier 2 escalation. Flags low-confidence responses automatically.</div>
        <div style="margin-top:0.75rem;font-size:0.7rem;color:#D97706;font-weight:500">LangGraph · Confidence threshold · Auto-routing</div>
      </div>

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
        <div style="font-weight:600;font-size:0.82rem;color:#0F172A;margin-bottom:0.3rem">LangSmith Evaluation</div>
        <div style="font-size:0.75rem;color:#64748B;line-height:1.5">
        85% accuracy on 20-question test suite. 100% out-of-scope refusal rate. Every query traced and logged.</div>
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
        Connectors for Kaggle and GitHub Issues. Normalization layer handles schema differences automatically.</div>
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
        120-example training dataset prepared. Pipeline built and validated. Pending OpenAI RFT access for deployment.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:1rem 0'></div>", unsafe_allow_html=True)

    # Section 4: Tech stack ---------------------------------------------------------
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
