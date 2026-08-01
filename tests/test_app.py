from datetime import datetime, timedelta, timezone

import httpx
from fastapi.testclient import TestClient

from vertex_proxy.app import Settings, create_app


class AsyncBytes(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


class FakeCredentials:
    def __init__(self) -> None:
        self.token = None
        self.expiry = None
        self.refresh_count = 0

    def refresh(self, request) -> None:
        self.refresh_count += 1
        self.token = f"adc-token-{self.refresh_count}"
        self.expiry = datetime.now(timezone.utc) + timedelta(hours=1)


def test_settings_and_global_host() -> None:
    settings = Settings.from_env(
        {
            "GOOGLE_CLOUD_PROJECT": "sample-project",
            "VERTEX_LOCATION": "global",
            "VERTEX_MODELS": "google/gemini-2.5-flash, google/gemini-2.5-pro",
        }
    )
    assert settings.upstream_host == "aiplatform.googleapis.com"
    assert settings.models == ("google/gemini-2.5-flash", "google/gemini-2.5-pro")


def test_openai_proxy_refreshes_adc_and_streams_response() -> None:
    credentials = FakeCredentials()
    seen_requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=AsyncBytes(b'data: {"ok":true}\n\n', b"data: [DONE]\n\n"),
        )

    settings = Settings(
        project="sample-project",
        location="us-central1",
        proxy_api_key="local-secret",
    )
    app = create_app(
        settings,
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions?alt=sse",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "google/gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.text.endswith("data: [DONE]\n\n")
    assert credentials.refresh_count == 1
    assert len(seen_requests) == 1
    upstream = seen_requests[0]
    assert str(upstream.url) == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/sample-project/"
        "locations/us-central1/endpoints/openapi/chat/completions?alt=sse"
    )
    assert upstream.headers["authorization"] == "Bearer adc-token-1"
    assert upstream.headers.get("x-api-key") is None
    
    import json
    upstream_body_bytes = upstream.read()
    assert "content-length" in upstream.headers
    assert int(upstream.headers["content-length"]) == len(upstream_body_bytes)

    upstream_body = json.loads(upstream_body_bytes)
    assert upstream_body["model"] == "google/gemini-2.5-flash"


def test_rejects_bad_proxy_key_before_adc_refresh() -> None:
    credentials = FakeCredentials()
    settings = Settings(
        project="sample-project",
        location="us-central1",
        proxy_api_key="local-secret",
    )
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(lambda r: None))

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={})

    assert response.status_code == 401
    assert credentials.refresh_count == 0


def test_retries_once_with_fresh_token_after_upstream_401() -> None:
    credentials = FakeCredentials()
    authorizations: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers["authorization"])
        if len(authorizations) == 1:
            return httpx.Response(401, stream=AsyncBytes(b'{"error":"expired"}'))
        return httpx.Response(200, stream=AsyncBytes(b'{"ok":true}'))

    app = create_app(
        Settings(project="sample-project", location="us-central1"),
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"messages": []})

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert authorizations == ["Bearer adc-token-1", "Bearer adc-token-2"]
    assert credentials.refresh_count == 2


def test_native_proxy_builds_project_scoped_url() -> None:
    credentials = FakeCredentials()
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=AsyncBytes(b'{"candidates":[]}'),
        )

    settings = Settings(project="sample-project", location="asia-east1")
    app = create_app(
        settings,
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/vertex/v1/publishers/google/models/gemini-2.5-flash:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
        )

    assert response.status_code == 200
    assert seen == [
        "https://asia-east1-aiplatform.googleapis.com/v1/projects/sample-project/"
        "locations/asia-east1/publishers/google/models/"
        "gemini-2.5-flash:generateContent"
    ]


def test_request_response_logging(caplog) -> None:
    import logging
    import os
    credentials = FakeCredentials()
    
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=AsyncBytes(b'{"response_ok":true}'),
        )

    settings = Settings(
        project="sample-project",
        location="us-central1",
        proxy_api_key="local-secret",
    )
    app = create_app(
        settings,
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )

    # 1. Test "full" log mode (default)
    os.environ["VERTEX_PROXY_LOG_MODE"] = "full"
    with caplog.at_level(logging.INFO, logger="vertex_proxy"):
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer local-secret", "x-custom-header": "test-val"},
                json={"model": "gemini-2.5-flash", "test_key": "test_val"},
            )

    assert response.status_code == 200
    log_records = [rec.message for rec in caplog.records if rec.name == "vertex_proxy"]
    received_logs = [log for log in log_records if "Received request" in log]
    completed_logs = [log for log in log_records if "Completed response" in log]
    
    assert len(received_logs) == 1
    assert len(completed_logs) == 1
    
    # Verify signature "authorization" is excluded from output
    assert "authorization" not in received_logs[0]
    assert "Bearer" not in received_logs[0]
    assert "test-val" in received_logs[0]
    assert "google/gemini-2.5-flash" in received_logs[0]  # The preprocessed model and formatted JSON
    assert "test_key" in received_logs[0]
    # Check that it contains formatted JSON with newlines
    assert "\n  \"response_ok\": true" in completed_logs[0]

    # 2. Test "none" log mode
    caplog.clear()
    os.environ["VERTEX_PROXY_LOG_MODE"] = "none"
    with caplog.at_level(logging.INFO, logger="vertex_proxy"):
        with TestClient(app) as client:
            client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer local-secret"},
                json={"model": "gemini-2.5-flash"},
            )
    log_records_none = [rec.message for rec in caplog.records if rec.name == "vertex_proxy"]
    assert not any("Received request" in log or "Completed response" in log for log in log_records_none)

    # 3. Test "messages" log mode
    caplog.clear()
    os.environ["VERTEX_PROXY_LOG_MODE"] = "messages"
    with caplog.at_level(logging.INFO, logger="vertex_proxy"):
        with TestClient(app) as client:
            client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer local-secret"},
                json={
                    "model": "gemini-2.5-flash",
                    "messages": [{"role": "user", "content": "hello_test_message"}]
                },
            )
    log_records_msg = [rec.message for rec in caplog.records if rec.name == "vertex_proxy"]
    msg_logs = [log for log in log_records_msg if "Request messages" in log]
    completed_logs_msg = [log for log in log_records_msg if "Completed response" in log]
    
    assert len(msg_logs) == 1
    assert "hello_test_message" in msg_logs[0]
    assert len(completed_logs_msg) == 0  # No response logged in messages mode

    # 4. Test "errors" log mode
    # 4.1 A successful response without content_filter
    caplog.clear()
    os.environ["VERTEX_PROXY_LOG_MODE"] = "errors"
    with caplog.at_level(logging.INFO, logger="vertex_proxy"):
        with TestClient(app) as client:
            client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer local-secret"},
                json={"model": "gemini-2.5-flash"},
            )
    log_records_err1 = [rec.message for rec in caplog.records if rec.name == "vertex_proxy"]
    assert not any("Completed response" in log for log in log_records_err1)

    # 4.2 A response with content_filter
    # Create a new app instance with a mock handler that returns content_filter error
    async def filter_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=AsyncBytes(b'{"choices":[{"delta":{"refusal":"sensitive"},"finish_reason":"content_filter"}]}'),
        )
    app_filter = create_app(
        settings,
        credentials=credentials,
        upstream_transport=httpx.MockTransport(filter_handler),
    )
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="vertex_proxy"):
        with TestClient(app_filter) as client:
            client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer local-secret"},
                json={"model": "gemini-2.5-flash"},
            )
    log_records_err2 = [rec.message for rec in caplog.records if rec.name == "vertex_proxy"]
    err2_completed = [log for log in log_records_err2 if "Completed response" in log]
    assert len(err2_completed) == 1
    assert "content_filter" in err2_completed[0]
