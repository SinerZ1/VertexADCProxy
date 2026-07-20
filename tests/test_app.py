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
    upstream_body = json.loads(upstream.read())
    assert upstream_body["model"] == "gemini-2.5-flash"


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
