import json
from datetime import datetime, timedelta, timezone

import httpx
from fastapi.testclient import TestClient

from vertex_proxy.app import Settings, create_app
from vertex_proxy.anthropic_gemini import resolve_gemini_model


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

    multi_region = Settings(project="sample-project", location="us")
    assert multi_region.upstream_host == "aiplatform.us.rep.googleapis.com"


def test_resolve_gemini_model_passthrough_and_prefix() -> None:
    settings = Settings(
        project="sample-project",
        location="global",
        models=("google/gemini-2.5-flash", "google/gemini-2.5-pro"),
    )
    # Direct Gemini model ID
    assert resolve_gemini_model("gemini-2.5-flash", settings) == "gemini-2.5-flash"
    assert resolve_gemini_model("google/gemini-2.5-pro", settings) == "gemini-2.5-pro"

    # Claude Desktop prefix conversion (claude- -> gemini-)
    assert resolve_gemini_model("claude-2.5-flash", settings) == "gemini-2.5-flash"
    assert resolve_gemini_model("claude-2.5-pro", settings) == "gemini-2.5-pro"


def test_resolve_gemini_model_validation_error() -> None:
    settings = Settings(
        project="sample-project",
        location="global",
        models=("google/gemini-2.5-flash",),
    )
    try:
        resolve_gemini_model("claude-2.5-pro", settings)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "未在已启用的模型列表中" in str(exc)


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


def test_anthropic_gemini_converts_messages_tools_and_response() -> None:
    credentials = FakeCredentials()
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "request-id": "gemini-request"},
            stream=AsyncBytes(
                json.dumps(
                    {
                        "id": "chatcmpl-1",
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "I will inspect it.",
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": '{"path":"README.md"}',
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {"prompt_tokens": 31, "completion_tokens": 9},
                    }
                ).encode("utf-8")
            ),
        )

    app = create_app(
        Settings(
            project="sample-project",
            location="global",
            models=("gemini-2.5-pro",),
        ),
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages?beta=true",
            json={
                "model": "claude-2.5-pro",
                "system": [
                    {"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral"}}
                ],
                "max_tokens": 2048,
                "messages": [
                    {"role": "user", "content": "Inspect the project."},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "old_call",
                                "name": "read_file",
                                "input": {"path": "pyproject.toml"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "old_call",
                                "content": "project metadata",
                            },
                            {"type": "text", "text": "Now inspect the README."},
                        ],
                    },
                ],
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a local file",
                        "input_schema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            },
        )

    assert response.status_code == 200
    assert response.headers["request-id"] == "gemini-request"
    result = response.json()
    assert result["model"] == "claude-2.5-pro"
    assert result["stop_reason"] == "tool_use"
    assert result["usage"] == {"input_tokens": 31, "output_tokens": 9}
    assert result["content"] == [
        {"type": "text", "text": "I will inspect it."},
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "read_file",
            "input": {"path": "README.md"},
        },
    ]

    assert len(seen) == 1
    assert str(seen[0].url) == (
        "https://aiplatform.googleapis.com/v1/projects/sample-project/locations/global/"
        "endpoints/openapi/chat/completions"
    )
    upstream = json.loads(seen[0].content)
    assert upstream["model"] == "google/gemini-2.5-pro"
    assert upstream["messages"][0] == {
        "role": "system",
        "content": "You are a coding agent.",
    }
    assert upstream["messages"][2]["tool_calls"][0]["id"] == "old_call"
    assert upstream["messages"][3] == {
        "role": "tool",
        "tool_call_id": "old_call",
        "content": "project metadata",
    }
    assert upstream["messages"][4] == {
        "role": "user",
        "content": "Now inspect the README.",
    }
    assert upstream["tools"][0]["function"]["name"] == "read_file"
    assert upstream["tool_choice"] == "auto"
    assert upstream["parallel_tool_calls"] is False


def test_anthropic_gemini_rejects_unenabled_model() -> None:
    credentials = FakeCredentials()
    app = create_app(
        Settings(
            project="sample-project",
            location="global",
            models=("gemini-2.5-flash",),
        ),
        credentials=credentials,
    )
    with TestClient(app) as client:
        messages = client.post(
            "/v1/messages",
            json={
                "model": "claude-2.5-pro",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert messages.status_code == 400
    assert "未在已启用的模型列表中" in messages.json()["error"]["message"]
    assert credentials.refresh_count == 0


def test_anthropic_gemini_count_tokens_uses_native_api() -> None:
    credentials = FakeCredentials()
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=AsyncBytes(b'{"totalTokens":42,"totalBillableCharacters":120}'),
        )

    app = create_app(
        Settings(
            project="sample-project",
            location="asia-east1",
            models=("google/gemini-2.5-pro",),
        ),
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "claude-2.5-pro",
                "system": "You are a coding agent.",
                "messages": [{"role": "user", "content": "Inspect main.py"}],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 42}
    assert str(seen[0].url) == (
        "https://asia-east1-aiplatform.googleapis.com/v1/projects/sample-project/"
        "locations/asia-east1/publishers/google/models/gemini-2.5-pro:countTokens"
    )


def test_anthropic_gemini_headers_cleanup() -> None:
    import gzip
    credentials = FakeCredentials()
    raw_data = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
    compressed_data = gzip.compress(raw_data)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "Content-Length": str(len(compressed_data)),
            },
            stream=AsyncBytes(compressed_data),
        )

    app = create_app(
        Settings(
            project="sample-project",
            location="global",
            models=("gemini-2.5-flash",),
        ),
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        res = client.post(
            "/v1/messages",
            json={
                "model": "claude-2.5-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert res.status_code == 200
    assert "content-encoding" not in res.headers
    assert res.headers.get("content-length") != str(len(compressed_data))
    assert res.headers["content-type"] == "text/event-stream; charset=utf-8"



def test_list_models_includes_gemini_and_claude_aliases() -> None:
    credentials = FakeCredentials()
    settings = Settings(
        project="sample-project",
        location="us-central1",
        models=("google/gemini-2.5-flash", "google/gemini-2.5-pro"),
    )
    app = create_app(settings, credentials=credentials)

    with TestClient(app) as client:
        res = client.get("/v1/models")
        assert res.status_code == 200
        data = res.json()["data"]
        model_ids = [item["id"] for item in data]
        assert "gemini-2.5-flash" in model_ids
        assert "claude-2.5-flash" in model_ids
        assert "gemini-2.5-pro" in model_ids
        assert "claude-2.5-pro" in model_ids

        # Check display_name
        claude_item = next(item for item in data if item["id"] == "claude-2.5-flash")
        assert claude_item["display_name"] == "gemini-2.5-flash"

        # Check single model endpoint
        res2 = client.get("/v1/models/claude-2.5-flash")
        assert res2.status_code == 200
        assert res2.json()["display_name"] == "gemini-2.5-flash"

        res3 = client.get("/v1/models/unknown-model")
        assert res3.status_code == 404
