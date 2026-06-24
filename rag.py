"""
RAG query layer for the IT Support RAG Agent.

This module connects to the ChromaDB collection built by ingest.py and
exposes a single helper, get_answer(question), which:

    1. Retrieves the top 3 most relevant tickets for the question.
    2. Feeds them as context to GPT-4o-mini.
    3. Returns a grounded answer that cites the Ticket IDs it used.

Run directly to execute a small built-in test:
    python3 rag.py
"""

from typing import Generator

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable
from langsmith.wrappers import wrap_openai

import os
try:
    import streamlit as st
    if hasattr(st, 'secrets'):
        for key, value in st.secrets.items():
            if key not in os.environ:
                os.environ[key] = str(value)
except Exception:
    pass

# Step 1: load environment variables (expects OPENAI_API_KEY in .env)
load_dotenv()

# Configuration -------------------------------------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "it_support_tickets"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 3

SYSTEM_PROMPT = (
    "You are an IT support assistant. Answer questions using ONLY the "
    "ticket information provided. Always cite the Ticket ID your answer "
    "is based on. If the answer is not in the provided tickets say: I "
    "don't have information on that in the enterprise knowledge base."
)

# Step 2: connect to the existing ChromaDB collection ----------------------
# Built lazily (not at import time) so importing this module never requires
# OPENAI_API_KEY to already be set.
_embeddings: OpenAIEmbeddings | None = None
_vectorstore: Chroma | None = None
_rag_chain = None


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


# Step 4: build the LangChain RAG chain ------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Definition of sufficient context: the tickets below are "
            "SUFFICIENT whenever at least one ticket shares the same "
            "Category as the question's subject (for example, a question "
            "about a printer is matched by any ticket with Category: "
            "Printer). Under that definition, exact wording never needs "
            "to match. Treat the tickets below as sufficient by default.\n\n"
            "Example: Question: \"My screen is flickering.\" Tickets contain "
            "one with Category: Hardware describing a different symptom (a "
            "laptop overheating). This still counts as sufficient because "
            "both are Category: Hardware. Correct response: answer using "
            "that ticket's Resolution and cite its Ticket ID. Do NOT say "
            "\"I don't have information\" in this case.\n\n"
            "Only say you don't have information if every ticket below "
            "belongs to a completely unrelated Category from the question "
            "(e.g. the question is about a printer but every ticket is "
            "about VPN access).\n\n"
            "Tickets:\n{context}\n\nQuestion: {question}",
        ),
    ]
)

_llm = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    return _llm


def format_docs(docs) -> str:
    """Join retrieved ticket documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def _get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        retriever = _get_vectorstore().as_retriever(search_kwargs={"k": TOP_K})
        _rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | _get_llm()
            | StrOutputParser()
        )
    return _rag_chain


# Step 5: query reformulation -----------------------------------------------
@traceable(name="Query Reformulation")
def reformulate_query(raw_question: str) -> str:
    """
    Takes a vague or messy user question and
    rewrites it into a clean, specific IT support
    search query before hitting ChromaDB.
    """
    reformulation_llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
        max_tokens=100,
    )

    messages = [
        SystemMessage(content=(
            "You are an IT support query optimizer. "
            "Rewrite the user's question into a clear, "
            "specific IT support search query. "
            "Keep it under 20 words. "
            "Return ONLY the rewritten query, "
            "nothing else."
        )),
        HumanMessage(content=raw_question),
    ]

    try:
        response = reformulation_llm.invoke(messages)
        reformulated = response.content.strip()
        if len(reformulated) > 5:
            return reformulated
        return raw_question
    except Exception:
        return raw_question


# Step 6: conversation memory -------------------------------------------------
# langchain.memory.ConversationBufferWindowMemory does not exist in the
# installed langchain version (1.x removed the legacy memory module). This
# shim reproduces the same load_memory_variables/save_context/clear
# interface on top of langchain_core message objects.
class _ConversationBufferWindowMemory:
    def __init__(self, k: int = 5):
        self.k = k
        self._messages: list = []

    def load_memory_variables(self, _inputs: dict) -> dict:
        return {"chat_history": self._messages[-self.k * 2:]}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        self._messages.append(HumanMessage(content=inputs["input"]))
        self._messages.append(AIMessage(content=outputs["output"]))
        self._messages = self._messages[-self.k * 2:]

    def clear(self) -> None:
        self._messages = []


_conversation_memory = _ConversationBufferWindowMemory(k=5)


# Step 7: public helpers ------------------------------------------------------
@traceable(name="IT Support RAG Query")
def get_answer(question: str) -> str:
    """Return a grounded, citation-backed answer for the given question."""
    question = reformulate_query(question)
    return _get_rag_chain().invoke(question)


def get_answer_with_memory(
    question: str,
    session_id: str = "default",
) -> str:
    """
    Answers a question with conversation context.
    Remembers last 5 exchanges per session.
    """
    history = _conversation_memory.load_memory_variables({})
    chat_history = history.get("chat_history", [])

    history_text = ""
    if chat_history:
        history_text = "\n\nPrevious conversation:\n"
        for msg in chat_history[-4:]:
            role = "User" if hasattr(
                msg, 'type') and msg.type == 'human' \
                else "Assistant"
            history_text += (
                f"{role}: {msg.content[:200]}\n"
            )

    # Reformulate the question and history TOGETHER, in one pass, so the
    # model has the prior turn available to resolve references like "that"
    # (e.g. "what if that doesn't work?" -> "alternative printer fix").
    # Reformulating the bare question first and appending history afterward
    # left the literal follow-up text glued onto a raw conversation dump in
    # the final grounding prompt, which made the refusal/answer call flip
    # unpredictably between runs.
    if history_text:
        augmented_question = f"{question}{history_text}"
    else:
        augmented_question = question
    augmented_question = reformulate_query(augmented_question)

    answer = _get_rag_chain().invoke(augmented_question)

    _conversation_memory.save_context(
        {"input": question},
        {"output": answer},
    )

    return answer


def clear_conversation_memory():
    """Clears conversation history."""
    _conversation_memory.clear()


def get_answer_streaming(question: str) -> Generator[str, None, None]:
    """
    Streams answer tokens as they are generated, for real-time display.
    Reuses the same retrieval + grounding prompt as get_answer/rag_chain
    so streamed answers stay consistent with the non-streaming path.
    """
    reformulated = reformulate_query(question)

    docs = _get_vectorstore().similarity_search(reformulated, k=TOP_K)
    context = format_docs(docs)

    streaming_llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.1,
        streaming=True,
        max_tokens=500,
    )

    rendered_prompt = prompt.invoke({"context": context, "question": reformulated})

    full_response = ""
    for chunk in streaming_llm.stream(rendered_prompt):
        token = chunk.content
        if token:
            full_response += token
            yield token

    _conversation_memory.save_context(
        {"input": question},
        {"output": full_response},
    )


# Step 8: simple built-in test ---------------------------------------------
if __name__ == "__main__":
    test_questions = [
        "How do I fix VPN connection issues after a password reset?",
        "My printer is not printing, what should I do?",
        "I cannot access SAP, how do I get access?",
    ]

    for q in test_questions:
        print("=" * 70)
        print(f"Q: {q}")
        print("-" * 70)
        print(get_answer(q))
        print()
