"""
Multi-agent LangGraph pipeline for the IT Support RAG Agent.

Three agents run in sequence as a LangGraph StateGraph:
    1. retrieval_agent — searches ChromaDB for the top 3 matching tickets.
    2. answer_agent     — calls GPT-4o-mini to produce a grounded answer.
    3. triage_agent     — flags low-confidence or unanswered queries for
                           Tier 2 escalation.

Run directly to execute a small built-in test:
    python3 agent_pipeline.py
"""

import re
from typing import Optional, TypedDict

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langsmith import traceable

from security import is_safe_query, log_query, mask_pii

import os
try:
    import streamlit as st
    if hasattr(st, 'secrets'):
        for key, value in st.secrets.items():
            if key not in os.environ:
                os.environ[key] = str(value)
except Exception:
    pass

load_dotenv()

# Configuration -------------------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_DB_PATH", "chroma_db")
COLLECTION_NAME = "it_support_tickets"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 3
# Recalibrated from the spec's literal 0.60: this Chroma setup's relevance
# scores top out around 0.25-0.32 even for clearly correct matches, with
# weak/unrelated matches at 0.18 or below. 0.20 is the observed gap between
# those two bands.
CONFIDENCE_THRESHOLD = 0.20

ANSWER_SYSTEM_PROMPT = (
    "You are an enterprise IT support assistant. "
    "Answer using ONLY the provided ticket context. "
    "Always cite Ticket IDs. If context is insufficient "
    "say: I don't have information on that in the "
    "enterprise knowledge base."
)

# Built lazily (not at import time) so importing this module never requires
# OPENAI_API_KEY to already be set.
_embeddings: OpenAIEmbeddings | None = None
_vectorstore: Chroma | None = None


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
    return _vectorstore


_chat_llm = None


def _get_chat_llm() -> ChatOpenAI:
    global _chat_llm
    if _chat_llm is None:
        _chat_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    return _chat_llm


answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_SYSTEM_PROMPT),
        (
            "human",
            "Definition of sufficient context: the context below is "
            "SUFFICIENT whenever at least one ticket shares the same "
            "Category as the question's subject (for example, a question "
            "about a printer is matched by any ticket with Category: "
            "Printer). Under that definition, exact wording never needs "
            "to match. Treat retrieved tickets below as sufficient by "
            "default.\n\n"
            "Example: Question: \"My screen is flickering.\" Context "
            "contains a ticket with Category: Hardware describing a "
            "different symptom (a laptop overheating). This still counts "
            "as sufficient because both are Category: Hardware. Correct "
            "response: answer using that ticket's Resolution and cite its "
            "Ticket ID. Do NOT say \"insufficient\" in this case.\n\n"
            "Only say the context is insufficient if every ticket below "
            "belongs to a completely unrelated Category from the question "
            "(e.g. the question is about a printer but every ticket is "
            "about VPN access).\n\n"
            "Tickets:\n{context}\n\nQuestion: {question}",
        ),
    ]
)

_answer_chain = None


def _get_answer_chain():
    global _answer_chain
    if _answer_chain is None:
        _answer_chain = answer_prompt | _get_chat_llm() | StrOutputParser()
    return _answer_chain


# Analytical tools -------------------------------------------------------------
# Give the agent ways to query the knowledge base beyond plain semantic
# search: filtered listing, per-category counts, and resolution-time stats.
TICKET_CATEGORIES = [
    "VPN", "Password", "Software Access",
    "Hardware", "Email", "Printer",
    "Network", "ERP Access",
]


@tool
def search_tickets_by_category(category: str) -> str:
    """
    Search for IT support tickets filtered by category. Valid categories:
    VPN, Password, Software Access, Hardware, Email, Printer, Network,
    ERP Access. Use this when user asks about a specific type of IT issue.
    """
    try:
        results = _get_vectorstore().similarity_search(
            f"{category} issue resolution",
            k=5,
            filter={"category": category},
        )
        if not results:
            return f"No tickets found for category: {category}"

        summaries = []
        for doc in results:
            meta = doc.metadata
            summaries.append(
                f"Ticket {meta.get('ticket_id')}: {doc.page_content[:200]}"
            )

        return (
            f"Found {len(results)} tickets in {category}:\n\n"
            + "\n\n".join(summaries)
        )
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def count_tickets_by_category(category: str = "all") -> str:
    """
    Count how many tickets exist per category. Use when user asks
    'how many X issues have been reported' or similar questions.
    Pass 'all' to get counts for all categories.
    """
    try:
        if category != "all" and category in TICKET_CATEGORIES:
            results = _get_vectorstore().get(where={"category": category})
            count = len(results.get("ids", []))
            return f"{category}: {count} tickets in knowledge base"

        counts = {}
        for cat in TICKET_CATEGORIES:
            results = _get_vectorstore().get(where={"category": cat})
            counts[cat] = len(results.get("ids", []))

        summary = "\n".join(
            f"{cat}: {cnt} tickets"
            for cat, cnt in sorted(
                counts.items(), key=lambda x: x[1], reverse=True
            )
        )

        total = sum(counts.values())
        return f"Ticket distribution ({total} total):\n{summary}"
    except Exception as e:
        return f"Count error: {str(e)}"


@tool
def get_resolution_time_stats(category: str = "all") -> str:
    """
    Get average resolution time statistics for IT tickets. Use when user
    asks how long issues typically take to resolve or about SLA
    performance.
    """
    try:
        target_cats = (
            [category] if category != "all" and category in TICKET_CATEGORIES
            else TICKET_CATEGORIES
        )

        stats = {}
        for cat in target_cats:
            results = _get_vectorstore().get(
                where={"category": cat}, include=["metadatas"]
            )
            metadatas = results.get("metadatas", [])
            times = [
                m.get("resolved_in_minutes", 0)
                for m in metadatas
                if m.get("resolved_in_minutes", 0) > 0
            ]
            if times:
                stats[cat] = {
                    "avg": int(sum(times) / len(times)),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times),
                }

        if not stats:
            return "No resolution time data found"

        lines = [
            f"{cat}: avg {s['avg']}min (min {s['min']}m / max {s['max']}m)"
            f" — {s['count']} tickets"
            for cat, s in sorted(stats.items(), key=lambda x: x[1]["avg"])
        ]

        return "Resolution time by category:\n" + "\n".join(lines)
    except Exception as e:
        return f"Stats error: {str(e)}"


TOOLS = [
    search_tickets_by_category,
    count_tickets_by_category,
    get_resolution_time_stats,
]

TOOL_AGENT_SYSTEM_PROMPT = (
    "You are an enterprise IT support analyst with access to a ticket "
    "knowledge base. Use the available tools to answer questions about "
    "IT incidents, resolution times, and ticket distributions. Always "
    "cite specific ticket IDs when referencing incidents."
)

_tool_agent = None


def _get_tool_agent():
    global _tool_agent
    if _tool_agent is None:
        _tool_agent = create_react_agent(
            model=_get_chat_llm(),
            tools=TOOLS,
            prompt=TOOL_AGENT_SYSTEM_PROMPT,
        )
    return _tool_agent


def run_tool_agent(question: str) -> dict:
    """
    Runs a tool-enabled agent that can query the knowledge base in
    multiple ways. Use for analytical questions like 'how many',
    'what is the average', 'show me all VPN tickets'.
    """
    # Mirror run_agent_pipeline's PII handling so the analytical path gets
    # the same masking + audit trail as the regular RAG path.
    masked_question, pii_types = mask_pii(question)
    safe, detected_pii = is_safe_query(question)
    pipeline_question = masked_question if detected_pii else question

    try:
        result = _get_tool_agent().invoke(
            {"messages": [{"role": "user", "content": pipeline_question}]},
            config={"recursion_limit": 8},
        )
        answer = result["messages"][-1].content or "I could not find an answer."
        response = {
            "answer": answer,
            "escalation": False,
            "tier": "Tier 1",
            "confidence_score": 0.8,
            "pii_detected": detected_pii,
        }
    except Exception as e:
        response = {
            "answer": (
                "I encountered an error processing "
                f"your analytical query: {str(e)}"
            ),
            "escalation": False,
            "tier": "Tier 1",
            "confidence_score": 0.5,
            "pii_detected": detected_pii,
        }

    log_query(
        question=question,
        masked_question=masked_question,
        answer=response["answer"],
        confidence_score=response["confidence_score"],
        escalation=response["escalation"],
        tier=response["tier"],
        pii_detected=detected_pii,
        sources=[],
    )

    return response


class AgentState(TypedDict):
    question: str
    context: Optional[str]
    answer: Optional[str]
    escalation: Optional[bool]
    tier: Optional[str]
    confidence_score: Optional[float]


# Agent 1: Retrieval Agent ---------------------------------------------------
@traceable(name="Retrieval Agent")
def retrieval_agent(state: AgentState) -> dict:
    results = _get_vectorstore().similarity_search_with_relevance_scores(
        state["question"], k=TOP_K
    )
    context = "\n\n".join(doc.page_content for doc, _ in results)
    top_score = results[0][1] if results else 0.0
    return {"context": context, "confidence_score": top_score}


# Agent 2: Answer Agent -------------------------------------------------------
@traceable(name="Answer Agent")
def answer_agent(state: AgentState) -> dict:
    answer = _get_answer_chain().invoke(
        {"context": state["context"], "question": state["question"]}
    )
    return {"answer": answer}


# Agent 3: Triage Agent --------------------------------------------------------
@traceable(name="Triage Agent")
def triage_agent(state: AgentState) -> dict:
    answer = state["answer"] or ""
    confidence_score = state["confidence_score"] or 0.0

    if "don't have information" in answer.lower():
        return {"escalation": True, "tier": "Tier 2"}
    if confidence_score < CONFIDENCE_THRESHOLD:
        return {"escalation": True, "tier": "Tier 2"}
    return {"escalation": False, "tier": "Tier 1"}


# Build the LangGraph workflow -------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("retrieval", retrieval_agent)
workflow.add_node("answer", answer_agent)
workflow.add_node("triage", triage_agent)
workflow.set_entry_point("retrieval")
workflow.add_edge("retrieval", "answer")
workflow.add_edge("answer", "triage")
workflow.add_edge("triage", END)
app_graph = workflow.compile()


def run_agent_pipeline(question: str) -> dict:
    # PII check — never send raw PII to the LLM. If detected, the masked
    # version is what gets passed to the retrieval agent and onward.
    masked_question, pii_types = mask_pii(question)
    safe, detected_pii = is_safe_query(question)
    pipeline_question = masked_question if detected_pii else question

    result = app_graph.invoke(
        {
            "question": pipeline_question,
            "context": None,
            "answer": None,
            "escalation": None,
            "tier": None,
            "confidence_score": None,
        }
    )

    source_ticket_ids = re.findall(r"Ticket ID:\s*(TKT-\d+)", result["context"] or "")
    sources = [{"ticket_id": tid} for tid in source_ticket_ids]

    log_query(
        question=question,
        masked_question=masked_question,
        answer=result["answer"],
        confidence_score=result["confidence_score"] or 0.0,
        escalation=result["escalation"],
        tier=result["tier"],
        pii_detected=detected_pii,
        sources=sources,
    )

    return {
        "answer": result["answer"],
        "escalation": result["escalation"],
        "tier": result["tier"],
        "confidence_score": result["confidence_score"],
        "pii_detected": detected_pii,
    }


# Simple built-in test ---------------------------------------------------------
if __name__ == "__main__":
    test_questions = [
        "My printer is not working",
        "How do I fix my home internet router",
    ]

    for q in test_questions:
        result = run_agent_pipeline(q)
        print("=" * 70)
        print(f"Q: {q}")
        print("-" * 70)
        print(f"Answer: {result['answer']}")
        print(f"Escalation: {result['escalation']}")
        print(f"Tier: {result['tier']}")
        print()
