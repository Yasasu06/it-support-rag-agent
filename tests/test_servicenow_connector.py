"""Offline checks for ServiceNow API diagnostics and their user-facing paths."""

import json

import pytest
import requests

from connectors import servicenow_connector


_INSTANCE = "https://connection-test.example.invalid"
_USERNAME = "connector-test-user"
_PASSWORD = "connector-test-password"
_PRIVATE_DETAIL = "private-provider-detail"


@pytest.fixture(autouse=True)
def configured_connector_without_network(monkeypatch):
    monkeypatch.setattr(servicenow_connector, "SERVICENOW_URL", _INSTANCE)
    monkeypatch.setattr(servicenow_connector, "SERVICENOW_USER", _USERNAME)
    monkeypatch.setattr(servicenow_connector, "SERVICENOW_PASS", _PASSWORD)

    def unexpected_request(*_args, **_kwargs):
        pytest.fail("ServiceNow tests must not make external requests")

    monkeypatch.setattr(servicenow_connector.requests, "get", unexpected_request)


def _response(body, *, status=200, content_type="application/json"):
    response = requests.Response()
    response.status_code = status
    response.headers["Content-Type"] = content_type
    response._content = body.encode("utf-8")
    response.url = f"{_INSTANCE}/api/now/table/incident"
    return response


def _assert_safe_failure(result):
    assert result["success"] is False
    assert isinstance(result["message"], str) and result["message"]
    for private_value in (_INSTANCE, _USERNAME, _PASSWORD, _PRIVATE_DETAIL):
        assert private_value not in result["message"]


@pytest.mark.parametrize("records", [[], [{"sys_id": "example-record"}]])
def test_connection_accepts_only_a_valid_incident_api_response(monkeypatch, records):
    captured = {}
    monkeypatch.setattr(servicenow_connector, "SERVICENOW_URL", f"  {_INSTANCE}/\n")
    monkeypatch.setattr(servicenow_connector, "SERVICENOW_USER", f" {_USERNAME}\n")
    monkeypatch.setattr(servicenow_connector, "SERVICENOW_PASS", f" {_PASSWORD} ")

    def fake_get(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return _response(json.dumps({"result": records}))

    monkeypatch.setattr(servicenow_connector.requests, "get", fake_get)
    result = servicenow_connector.test_connection()

    assert result["success"] is True
    assert captured["url"] == f"{_INSTANCE}/api/now/table/incident"
    assert captured["params"] == {"sysparm_limit": 1, "sysparm_fields": "sys_id"}
    assert captured["auth"] == (_USERNAME, f" {_PASSWORD} ")
    assert captured["headers"]["Accept"] == "application/json"
    assert captured["allow_redirects"] is False
    assert captured["timeout"] == 25


@pytest.mark.parametrize("body, content_type", [
    (f"<html>Sign in {_PRIVATE_DETAIL}</html>", "text/html"),
    (_PRIVATE_DETAIL, "application/json"),
    (json.dumps({"result": {"private_note": _PRIVATE_DETAIL}}), "application/json"),
    (json.dumps({"error": _PRIVATE_DETAIL}), "application/json"),
    (json.dumps([{"result": []}]), "application/json"),
])
def test_http_200_login_or_invalid_payload_is_not_a_connection_success(
    monkeypatch, body, content_type,
):
    response = _response(body, content_type=content_type)
    monkeypatch.setattr(servicenow_connector.requests, "get", lambda *_a, **_k: response)

    result = servicenow_connector.test_connection()

    _assert_safe_failure(result)
    assert "unexpected" in result["message"].lower()
    assert "api" in result["message"].lower()


def test_http_200_hibernation_page_tells_user_to_wake_instance(monkeypatch):
    response = _response(
        f"<html>Instance is HIBERNATING. {_PRIVATE_DETAIL}</html>",
        content_type="text/html; charset=utf-8",
    )
    monkeypatch.setattr(servicenow_connector.requests, "get", lambda *_a, **_k: response)

    result = servicenow_connector.test_connection()

    _assert_safe_failure(result)
    assert "hibernat" in result["message"].lower()
    assert "wake instance" in result["message"].lower()


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_auth_redirect_is_reported_without_following_it(monkeypatch, status):
    response = _response(_PRIVATE_DETAIL, status=status, content_type="text/html")
    response.headers["Location"] = f"{_INSTANCE}/login.do?token={_PRIVATE_DETAIL}"
    captured = {}

    def fake_get(_url, **kwargs):
        captured.update(kwargs)
        return response

    monkeypatch.setattr(servicenow_connector.requests, "get", fake_get)
    result = servicenow_connector.test_connection()

    _assert_safe_failure(result)
    assert "redirect" in result["message"].lower()
    assert captured["allow_redirects"] is False


@pytest.mark.parametrize("status, reason", [
    (401, "Authentication failed"),
    (403, "denied incident read access"),
    (500, "HTTP 500"),
    (503, "HTTP 503"),
])
def test_http_failure_preserves_reason_without_provider_body(monkeypatch, status, reason):
    response = _response(f"token={_PRIVATE_DETAIL}", status=status)
    monkeypatch.setattr(servicenow_connector.requests, "get", lambda *_a, **_k: response)

    result = servicenow_connector.test_connection()

    _assert_safe_failure(result)
    assert reason.lower() in result["message"].lower()


def test_http_401_with_invalid_mfa_header_identifies_mfa_requirement(monkeypatch):
    response = _response(_PRIVATE_DETAIL, status=401)
    response.headers["X-MFA_TOKEN"] = "invalid"
    monkeypatch.setattr(servicenow_connector.requests, "get", lambda *_a, **_k: response)

    result = servicenow_connector.test_connection()

    _assert_safe_failure(result)
    assert "requires mfa" in result["message"].lower()
    assert "username and password" in result["message"].lower()


@pytest.mark.parametrize("error, reason", [
    (requests.exceptions.SSLError(_PRIVATE_DETAIL), "TLS"),
    (requests.exceptions.Timeout(_PRIVATE_DETAIL), "timed out"),
    (requests.exceptions.ConnectionError(_PRIVATE_DETAIL), "reach"),
    (RuntimeError(_PRIVATE_DETAIL), "connection check failed"),
])
def test_request_failures_return_curated_diagnostics(monkeypatch, error, reason):
    def fake_get(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(servicenow_connector.requests, "get", fake_get)
    result = servicenow_connector.test_connection()

    _assert_safe_failure(result)
    assert reason.lower() in result["message"].lower()


@pytest.mark.parametrize("setting", ["SERVICENOW_URL", "SERVICENOW_USER", "SERVICENOW_PASS"])
def test_missing_settings_are_reported_before_any_request(monkeypatch, setting):
    monkeypatch.setattr(servicenow_connector, setting, None)

    result = servicenow_connector.test_connection()

    _assert_safe_failure(result)
    assert "configured" in result["message"].lower()
    assert ".env" not in result["message"]


@pytest.mark.parametrize("url", [
    "connection-test.example.invalid",
    "http://connection-test.example.invalid",
    "https://",
    f"{_INSTANCE}/api/now/table/incident",
    f"https://{_USERNAME}:{_PASSWORD}@connection-test.example.invalid",
    f"{_INSTANCE}?token={_PRIVATE_DETAIL}",
])
def test_invalid_instance_urls_are_rejected_before_any_request(monkeypatch, url):
    monkeypatch.setattr(servicenow_connector, "SERVICENOW_URL", url)

    result = servicenow_connector.test_connection()

    _assert_safe_failure(result)
    assert "url" in result["message"].lower()


@pytest.mark.parametrize("success", [True, False])
def test_app_retains_and_renders_the_full_connection_result(monkeypatch, success):
    import app

    class SessionState(dict):
        def __setattr__(self, name, value):
            self[name] = value

    session = SessionState()
    rendered = []
    result = {
        "success": success,
        "message": "Connected successfully." if success else "Authentication failed (HTTP 401).",
    }
    monkeypatch.setattr(app.st, "session_state", session)
    monkeypatch.setattr(app, "servicenow_test_connection", lambda: result)
    monkeypatch.setattr(app.st, "success", lambda message: rendered.append((True, message)))
    monkeypatch.setattr(app.st, "error", lambda message: rendered.append((False, message)))

    checked = app.check_servicenow_connection()
    app.render_servicenow_connection_result(session["servicenow_connection_result"])

    assert checked == result
    assert session["servicenow_connection_result"] == result
    assert rendered == [(success, result["message"])]


def test_analytical_tool_keeps_connection_failure_reason_and_skips_counts(monkeypatch):
    import agent_pipeline

    reason = "ServiceNow denied incident read access (HTTP 403). Check the account role."
    monkeypatch.setattr(agent_pipeline.config, "ENABLE_SERVICENOW_LIVE", True)
    monkeypatch.setattr(agent_pipeline, "servicenow_configured", lambda: True)
    monkeypatch.setattr(
        agent_pipeline, "servicenow_test_connection",
        lambda: {"success": False, "message": reason},
    )
    monkeypatch.setattr(
        agent_pipeline, "live_incident_count_by_state",
        lambda: pytest.fail("incident counts must not run after a failed connection check"),
    )

    assert agent_pipeline.check_live_servicenow_status.invoke({}) == reason
