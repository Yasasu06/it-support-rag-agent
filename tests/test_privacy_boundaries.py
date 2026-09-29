"""Focused checks for data leaving the UI, pipeline, and manual utilities."""

import inspect
import importlib
import json
from types import SimpleNamespace

import pytest

import agent_pipeline
import app
import image_processing
import live_eval
import rag
import security
import watcher
from connectors import servicenow_connector
from eval_harness import metrics


def test_reformulation_masks_question_and_prior_history_before_model(monkeypatch):
    captured = {}

    def fake_reformulator(text):
        captured["input"] = text
        return text

    monkeypatch.setattr(rag, "_reformulate_query_safe", fake_reformulator)
    raw = (
        "Email jane.doe@example.com about VPN.\n"
        "Previous conversation:\nUser: Call 616-555-0123 about the same issue."
    )
    rag.reformulate_query(raw)

    assert "jane.doe@example.com" not in captured["input"]
    assert "616-555-0123" not in captured["input"]
    assert "Previous conversation:" in captured["input"]


def test_trace_redaction_fails_closed(monkeypatch):
    def broken_mask(_text):
        raise RuntimeError("private provider detail")

    monkeypatch.setattr(security, "mask_pii", broken_mask)
    assert security.sanitize_trace_payload({"question": "raw secret"}) == {
        "[REDACTED]": "[REDACTED]"
    }


def test_vision_masks_typed_context_before_provider(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="VPN error"))]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **_kwargs: fake_client)

    result = image_processing.extract_text_from_image(
        b"x" * 150, "image/png", "Please email jane.doe@example.com about this"
    )

    assert result["success"] is True
    prompt = captured["messages"][0]["content"][0]["text"]
    assert "jane.doe@example.com" not in prompt
    assert captured["messages"][0]["content"][1]["type"] == "image_url"


def test_feedback_log_contains_metadata_not_question_or_answer(monkeypatch, tmp_path):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(app, "FEEDBACK_FILE", str(path))

    app.log_feedback(
        "positive", tier="Tier 2", confidence_score=0.23456,
        source_count=2, timestamp="2026-09-28T00:00:00",
    )
    entry = json.loads(path.read_text().strip())

    assert entry == {
        "timestamp": "2026-09-28T00:00:00",
        "rating": "positive",
        "tier": "Tier 2",
        "confidence_score": 0.235,
        "source_count": 2,
    }
    assert "question" not in entry and "answer" not in entry


def test_admin_key_uses_session_form_not_url():
    assert app.admin_key_matches("right", "right") is True
    assert app.admin_key_matches("wrong", "right") is False
    assert app.admin_key_matches("anything", "") is False
    assert "query_params" not in inspect.getsource(app)


@pytest.mark.parametrize("ticket_id", [
    "TKT-151", "KGL-00001", "HF-00001", "FB-00001",
    "GH-MIC-12", "GH-KUB-12",
    "INC0012345", "SNOW-INC0012345",
])
def test_watcher_normalizes_before_embedding_and_processed_copy(monkeypatch, tmp_path, ticket_id):
    processed = tmp_path / "processed"
    processed.mkdir()
    monkeypatch.setattr(watcher, "PROCESSED_FOLDER", str(processed))
    incoming = tmp_path / "incoming.json"
    incoming.write_text(json.dumps({
        "ticket_id": ticket_id,
        "category": "VPN",
        "issue": "jane.doe@example.com cannot connect to the VPN from home",
        "resolution": "Reset the VPN profile and called 616-555-0123 to confirm access",
        "resolved_in_minutes": 20,
    }))

    class FakeStore:
        documents = None

        def add_documents(self, documents):
            self.documents = documents

    handler = object.__new__(watcher.TicketIngestionHandler)
    handler.vectorstore = FakeStore()
    handler._ingest_file(str(incoming), incoming.name)

    assert handler.vectorstore.documents is not None
    embedded = handler.vectorstore.documents[0].page_content
    saved = json.loads(next(processed.glob("*.json")).read_text())
    assert "jane.doe@example.com" not in embedded
    assert "616-555-0123" not in embedded
    assert "jane.doe@example.com" not in json.dumps(saved)
    assert "616-555-0123" not in json.dumps(saved)
    assert handler.vectorstore.documents[0].metadata["ticket_id"] == ticket_id
    assert saved["ticket_id"] == ticket_id
    assert saved["source"] == "watcher"
    assert not incoming.exists()


@pytest.mark.parametrize("ticket_id", [
    "sk_test_abcdefghijklmnopqrstuvwxyz",
    "A1" * 40,
])
def test_watcher_rejects_token_like_ids_before_embedding_or_persistence(
    monkeypatch, tmp_path, ticket_id,
):
    processed = tmp_path / "processed"
    processed.mkdir()
    monkeypatch.setattr(watcher, "PROCESSED_FOLDER", str(processed))
    incoming = tmp_path / "incoming.json"
    incoming.write_text(json.dumps({
        "ticket_id": ticket_id,
        "category": "VPN",
        "issue": "VPN connection fails after a software update",
        "resolution": "Reset the VPN profile and confirmed the connection works",
        "resolved_in_minutes": 20,
    }))

    class FakeStore:
        calls = 0

        def add_documents(self, _documents):
            self.calls += 1

    handler = object.__new__(watcher.TicketIngestionHandler)
    handler.vectorstore = FakeStore()
    handler._ingest_file(str(incoming), incoming.name)

    assert handler.vectorstore.calls == 0
    assert list(processed.glob("*.json")) == []


def test_provider_error_body_is_not_returned_to_user(monkeypatch):
    monkeypatch.setattr(servicenow_connector, "SERVICENOW_URL", "https://example.invalid")
    monkeypatch.setattr(servicenow_connector, "SERVICENOW_USER", "test-user")
    monkeypatch.setattr(servicenow_connector, "SERVICENOW_PASS", "test-password")

    response = SimpleNamespace(
        status_code=500,
        text="provider token=private-value internal request detail",
    )
    monkeypatch.setattr(servicenow_connector.requests, "get", lambda *_a, **_k: response)
    result = servicenow_connector.test_connection()

    assert result["success"] is False
    assert "HTTP 500" in result["message"]
    assert "private-value" not in result["message"]
    assert "internal request detail" not in result["message"]


def test_analytical_error_does_not_render_exception(monkeypatch):
    class BrokenAgent:
        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("provider token=private-value")

    monkeypatch.setattr(agent_pipeline, "_get_tool_agent", lambda: BrokenAgent())
    monkeypatch.setattr(agent_pipeline, "mask_pii", lambda text: (text, []))
    monkeypatch.setattr(agent_pipeline, "is_safe_query", lambda _text: (True, []))
    monkeypatch.setattr(agent_pipeline, "log_query", lambda **_kwargs: None)

    result = agent_pipeline.run_tool_agent("how many VPN tickets")
    assert "private-value" not in result["answer"]
    assert "couldn't process" in result["answer"]


def test_live_evaluation_gets_masked_question_and_answer(monkeypatch):
    address = "jane.doe@example.com"
    replacement = "<EMAIL_ADDRESS>"
    captured = {}

    monkeypatch.setattr(
        agent_pipeline, "mask_pii",
        lambda text: (text.replace(address, replacement), ["EMAIL_ADDRESS"]),
    )
    monkeypatch.setattr(agent_pipeline, "is_safe_query", lambda _text: (False, ["EMAIL_ADDRESS"]))
    monkeypatch.setattr(
        agent_pipeline, "sanitize_external_text",
        lambda text: text.replace(address, replacement),
    )
    monkeypatch.setattr(agent_pipeline, "log_query", lambda **_kwargs: None)
    monkeypatch.setattr(agent_pipeline.config, "ENABLE_LIVE_EVAL", True)
    monkeypatch.setattr(
        agent_pipeline, "app_graph",
        SimpleNamespace(invoke=lambda *_a, **_k: {
            "context": "Ticket ID: TKT-900",
            "answer": f"Contact {address}",
            "confidence_score": 0.3,
            "escalation": False,
            "tier": "Tier 1",
            "grounded": True,
            "verification_notes": "Grounded",
            "retry_count": 0,
            "retry_history": [],
            "retrieved_sources": [],
        }),
    )
    monkeypatch.setattr(
        agent_pipeline, "score_response_relevance",
        lambda question, answer, _llm: captured.update(
            {"question": question, "answer": answer}
        ) or 0.8,
    )
    monkeypatch.setattr(
        agent_pipeline, "log_live_eval",
        lambda **kwargs: captured.update({"logged": kwargs}),
    )

    agent_pipeline._run_agent_pipeline_impl(f"Help {address} with VPN")

    assert address not in captured["question"]
    assert address not in captured["answer"]
    assert address not in captured["logged"]["question"]
    assert address not in captured["logged"]["answer"]


def test_retrieved_context_is_sanitized_before_answer_model(monkeypatch):
    document = SimpleNamespace(
        page_content=(
            "Ticket ID: TKT-901 | Category: VPN | "
            "Issue: jane.doe@example.com cannot connect to VPN | "
            "Resolution: Reset the profile for jane.doe@example.com | "
            "Resolved in: 20 minutes"
        ),
        metadata={"ticket_id": "TKT-901", "category": "VPN"},
    )
    monkeypatch.setattr(
        agent_pipeline, "_get_vectorstore",
        lambda: SimpleNamespace(
            similarity_search_with_relevance_scores=lambda *_a, **_k: [(document, 0.3)]
        ),
    )

    retrieval = getattr(agent_pipeline.retrieval_agent, "__wrapped__", agent_pipeline.retrieval_agent)
    result = retrieval({"question": "VPN issue", "retry_count": 0, "retry_history": []})

    assert "jane.doe@example.com" not in result["context"]
    assert "jane.doe@example.com" not in json.dumps(result["retrieved_sources"])


def test_answer_agent_sanitizes_direct_inputs_and_generated_answer(monkeypatch):
    captured = {}

    class FakeChain:
        def invoke(self, payload):
            captured.update(payload)
            return "Contact jane.doe@example.com about the fix"

    monkeypatch.setattr(agent_pipeline, "_get_answer_chain", lambda: FakeChain())
    answer = getattr(agent_pipeline.answer_agent, "__wrapped__", agent_pipeline.answer_agent)
    result = answer({
        "question": "Help jane.doe@example.com",
        "context": "Ticket TKT-900 belongs to jane.doe@example.com",
    })

    assert "jane.doe@example.com" not in captured["question"]
    assert "jane.doe@example.com" not in captured["context"]
    assert "jane.doe@example.com" not in result["answer"]


def _valid_training_record():
    return {"messages": [
        {"role": "system", "content": "You are an IT support assistant."},
        {"role": "user", "content": "The VPN connection fails after an update."},
        {"role": "assistant", "content": "Based on Ticket ID TKT-151: Reset the VPN profile."},
    ]}


def test_finetune_upload_guard_accepts_valid_data_without_network(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    finetune = importlib.import_module("finetune.run_finetune")
    path = tmp_path / "training.jsonl"
    path.write_text(json.dumps(_valid_training_record()) + "\n")

    class FakeFiles:
        calls = 0

        def create(self, *, file, purpose):
            self.calls += 1
            assert purpose == "fine-tune"
            assert file.read() == path.read_bytes()
            return SimpleNamespace(id="file-test")

    files = FakeFiles()
    monkeypatch.setattr(finetune, "client", SimpleNamespace(files=files))

    assert finetune.upload_training_file(str(path)) == "file-test"
    assert files.calls == 1


@pytest.mark.parametrize("ticket_index", [40, 55, 121])
def test_prepared_example_passes_upload_validation(monkeypatch, tmp_path, ticket_index):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    finetune = importlib.import_module("finetune.run_finetune")
    from finetune.prepare_dataset import ticket_to_training_example
    from data.tickets import TICKETS

    path = tmp_path / "training.jsonl"
    path.write_text(json.dumps(ticket_to_training_example(TICKETS[ticket_index])) + "\n")
    finetune.validate_training_file(str(path))


@pytest.mark.parametrize("case", [
    "pii_content", "extra_record_field", "extra_message_field",
    "duplicate_field", "malformed_json", "missing_messages", "empty",
    "invalid_second_record",
])
def test_finetune_upload_guard_rejects_invalid_data_before_network(monkeypatch, tmp_path, case):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    finetune = importlib.import_module("finetune.run_finetune")
    path = tmp_path / "training.jsonl"
    record = _valid_training_record()
    if case == "pii_content":
        record["messages"][1]["content"] = "Email jane.doe@example.com about VPN"
    elif case == "extra_record_field":
        record["private_note"] = "jane.doe@example.com"
    elif case == "extra_message_field":
        record["messages"][1]["private_note"] = "jane.doe@example.com"
    elif case == "missing_messages":
        record = {"unexpected": "data"}

    if case == "empty":
        path.write_text("")
    elif case == "malformed_json":
        path.write_text('{"messages": [\n')
    elif case == "duplicate_field":
        path.write_text(
            '{"messages":[],"messages":' + json.dumps(record["messages"]) + '}\n'
        )
    elif case == "invalid_second_record":
        path.write_text(json.dumps(record) + "\n" + '{"messages": [\n')
    else:
        path.write_text(json.dumps(record) + "\n")

    class FakeFiles:
        calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(id="unexpected-upload")

    files = FakeFiles()
    monkeypatch.setattr(finetune, "client", SimpleNamespace(files=files))

    with pytest.raises(ValueError):
        finetune.upload_training_file(str(path))
    assert files.calls == 0


def test_manual_evaluation_judge_receives_sanitized_text():
    captured = {}

    class FakeLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return SimpleNamespace(content="8")

    result = metrics.score_relevance(
        "Help jane.doe@example.com",
        "Email jane.doe@example.com",
        lambda: FakeLLM(),
    )

    assert result["score"] == 0.8
    assert "jane.doe@example.com" not in captured["prompt"]


def test_live_relevance_helper_sanitizes_direct_call():
    captured = {}

    class FakeLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return SimpleNamespace(content="7")

    score = live_eval.score_response_relevance(
        "Help jane.doe@example.com", "Contact jane.doe@example.com", lambda: FakeLLM()
    )

    assert score == 0.7
    assert "jane.doe@example.com" not in captured["prompt"]
