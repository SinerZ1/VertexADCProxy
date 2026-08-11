import json
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

    multi_region = Settings(project="sample-project", location="us")
    assert multi_region.upstream_host == "aiplatform.us.rep.googleapis.com"


def test_settings_parses_anthropic_model_map() -> None:
    settings = Settings.from_env(
        {
            "GOOGLE_CLOUD_PROJECT": "sample-project",
            "VERTEX_LOCATION": "global",
            "VERTEX_ANTHROPIC_MODEL_MAP": (
                "team-sonnet=claude-sonnet-4-6, team-haiku=claude-haiku-4-5"
            ),
        }
    )
    assert settings.resolve_anthropic_model("team-sonnet") == "claude-sonnet-4-6"
    assert settings.resolve_anthropic_model("team-haiku") == "claude-haiku-4-5"


def test_settings_parses_anthropic_gemini_backend() -> None:
    settings = Settings.from_env(
        {
            "GOOGLE_CLOUD_PROJECT": "sample-project",
            "VERTEX_LOCATION": "global",
            "VERTEX_ANTHROPIC_BACKEND": "gemini",
            "VERTEX_ANTHROPIC_GEMINI_MODEL": "gemini-2.5-pro",
        }
    )
    assert settings.anthropic_backend == "gemini"
    assert settings.anthropic_gemini_model == "gemini-2.5-pro"


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


def test_anthropic_messages_maps_model_and_preserves_open_body_fields() -> None:
    credentials = FakeCredentials()
    seen: list[httpx.Request] = []
    long_prompt = "x" * 250
    long_description = "tool description " * 20

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "request-id": "vertex-request"},
            stream=AsyncBytes(b'{"id":"msg_123","type":"message"}'),
        )

    app = create_app(
        Settings(
            project="sample-project",
            location="global",
            proxy_api_key="local-secret",
        ),
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )
    request_body = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": long_prompt}],
        "tools": [
            {
                "name": "demo_tool",
                "description": long_description,
                "input_schema": {"type": "object", "properties": {}},
                "defer_loading": True,
            }
        ],
        "output_config": {"effort": "high"},
        "stream": False,
    }

    with TestClient(app) as client:
        response = client.post(
            "/v1/messages?beta=true",
            headers={
                "authorization": "Bearer local-secret",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "example-beta-2026-01-01",
            },
            json=request_body,
        )

    assert response.status_code == 200
    assert response.json() == {"id": "msg_123", "type": "message"}
    assert response.headers["request-id"] == "vertex-request"
    assert len(seen) == 1
    upstream = seen[0]
    assert str(upstream.url) == (
        "https://aiplatform.googleapis.com/v1/projects/sample-project/locations/global/"
        "publishers/anthropic/models/claude-sonnet-4-5:rawPredict"
    )
    assert upstream.headers["authorization"] == "Bearer adc-token-1"
    assert upstream.headers["anthropic-version"] == "2023-06-01"
    assert upstream.headers["anthropic-beta"] == "example-beta-2026-01-01"
    assert upstream.headers.get("x-api-key") is None

    upstream_body = json.loads(upstream.content)
    assert "model" not in upstream_body
    assert upstream_body["anthropic_version"] == "vertex-2023-10-16"
    assert upstream_body["messages"][0]["content"] == long_prompt
    assert upstream_body["tools"][0]["description"] == long_description
    assert upstream_body["tools"][0]["defer_loading"] is True
    assert upstream_body["output_config"] == {"effort": "high"}


def test_anthropic_messages_streams_sse_without_filtering_or_done_marker() -> None:
    credentials = FakeCredentials()
    seen_urls: list[str] = []
    chunks = (
        b"event: message_start\n",
        b'data: {"type":"message_start"}\n\n',
        b"event: ping\n",
        b'data: {"type":"ping"}\n\n',
        b"event: message_stop\n",
        b'data: {"type":"message_stop"}\n\n',
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=AsyncBytes(*chunks),
        )

    app = create_app(
        Settings(project="sample-project", location="us-east5"),
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages?beta=true",
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.content == b"".join(chunks)
    assert b"data: [DONE]" not in response.content
    assert seen_urls == [
        "https://us-east5-aiplatform.googleapis.com/v1/projects/sample-project/"
        "locations/us-east5/publishers/anthropic/models/"
        "claude-sonnet-4-6:streamRawPredict"
    ]


def test_anthropic_count_tokens_uses_configured_model_mapping() -> None:
    credentials = FakeCredentials()
    seen_body: list[dict] = []
    seen_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_url.append(str(request.url))
        seen_body.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=AsyncBytes(b'{"input_tokens":14}'),
        )

    settings = Settings(
        project="sample-project",
        location="asia-southeast1",
        anthropic_model_map=(("company-sonnet", "claude-sonnet-4-6"),),
    )
    app = create_app(
        settings,
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "company-sonnet",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"name": "demo", "input_schema": {"type": "object"}}],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 14}
    assert seen_url == [
        "https://asia-southeast1-aiplatform.googleapis.com/v1/projects/sample-project/"
        "locations/asia-southeast1/publishers/anthropic/models/count-tokens:rawPredict"
    ]
    assert seen_body == [
        {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"name": "demo", "input_schema": {"type": "object"}}],
        }
    ]


def test_anthropic_invalid_request_is_400_before_adc_refresh() -> None:
    credentials = FakeCredentials()
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    app = create_app(
        Settings(project="sample-project", location="global"),
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={"model": "../../unsafe", "messages": []},
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert credentials.refresh_count == 0
    assert calls == 0


def test_anthropic_upstream_error_is_forwarded_unmodified() -> None:
    credentials = FakeCredentials()
    error_body = b'{"type":"error","error":{"type":"invalid_request_error","message":"thinking rejected"}}'

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            headers={"content-type": "application/json", "request-id": "error-request"},
            stream=AsyncBytes(error_body),
        )

    app = create_app(
        Settings(project="sample-project", location="global"),
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 400
    assert response.content == error_body
    assert response.headers["request-id"] == "error-request"


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
            anthropic_backend="gemini",
            anthropic_gemini_model="gemini-2.5-pro",
        ),
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages?beta=true",
            json={
                "model": "claude-code-gemini",
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
    assert result["model"] == "claude-code-gemini"
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


def test_anthropic_gemini_accepts_gateway_roles_and_sanitizes_tool_schema() -> None:
    credentials = FakeCredentials()
    seen_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_body.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=AsyncBytes(
                b'{"choices":[{"message":{"role":"assistant","content":"done"},'
                b'"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1}}'
            ),
        )

    app = create_app(
        Settings(
            project="sample-project",
            location="global",
            anthropic_backend="gemini",
            anthropic_gemini_model="gemini-2.5-pro",
        ),
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-code-gemini",
                "messages": [
                    {"role": "system", "content": "Gateway system message"},
                    {"role": "user", "content": "Run the tool"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call_1",
                                "name": "demo",
                                "input": {"mode": "fast"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "tool output",
                    },
                ],
                "tools": [
                    {
                        "name": "demo",
                        "input_schema": {
                            "$schema": "https://json-schema.org/draft/2020-12/schema",
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "mode": {
                                    "oneOf": [
                                        {"const": "fast"},
                                        {"type": "string"},
                                    ]
                                }
                            },
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    upstream = seen_body[0]
    assert upstream["messages"][0] == {
        "role": "system",
        "content": "Gateway system message",
    }
    assert upstream["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "tool output",
    }
    schema = upstream["tools"][0]["function"]["parameters"]
    assert "$schema" not in schema
    assert "additionalProperties" not in schema
    assert schema["properties"]["mode"]["anyOf"][0] == {"enum": ["fast"]}


def test_anthropic_gemini_converts_streaming_text_and_tool_call() -> None:
    credentials = FakeCredentials()
    upstream_chunks = (
        b'data: {"choices":[{"delta":{"content":"Checking "}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"now."}}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_9","function":{"name":"read_","arguments":"{\\"path\\":"}}]}}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"file","arguments":"\\"main.py\\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":7}}\n\n',
        b"data: [DONE]\n\n",
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=AsyncBytes(*upstream_chunks),
        )

    app = create_app(
        Settings(
            project="sample-project",
            location="us-central1",
            anthropic_backend="gemini",
            anthropic_gemini_model="google/gemini-2.5-flash",
        ),
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-code-gemini",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Check main.py"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: message_start" in response.text
    assert '"type":"text_delta","text":"Checking "' in response.text
    assert '"type":"text_delta","text":"now."' in response.text
    assert '"type":"tool_use","id":"call_9","name":"read_file"' in response.text
    assert '\"partial_json\":\"{\\\"path\\\":\\\"main.py\\\"}\"' in response.text
    assert '"stop_reason":"tool_use"' in response.text
    assert '"output_tokens":7' in response.text
    assert response.text.rstrip().endswith('data: {"type":"message_stop"}')


def test_anthropic_gemini_requires_configured_model() -> None:
    credentials = FakeCredentials()
    app = create_app(
        Settings(
            project="sample-project",
            location="global",
            anthropic_backend="gemini",
        ),
        credentials=credentials,
    )
    with TestClient(app) as client:
        messages = client.post(
            "/v1/messages",
            json={
                "model": "claude-code-gemini",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        count_tokens = client.post(
            "/v1/messages/count_tokens",
            json={"model": "claude-code-gemini", "messages": []},
        )

    assert messages.status_code == 400
    assert "VERTEX_ANTHROPIC_GEMINI_MODEL" in messages.json()["error"]["message"]
    assert count_tokens.status_code == 400
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
            anthropic_backend="gemini",
        ),
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "claude-code-gemini",
                "system": "You are a coding agent.",
                "messages": [{"role": "user", "content": "Inspect main.py"}],
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "input_schema": {
                            "$schema": "https://json-schema.org/draft/2020-12/schema",
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {"path": {"type": "string"}},
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 42}
    assert str(seen[0].url) == (
        "https://asia-east1-aiplatform.googleapis.com/v1/projects/sample-project/"
        "locations/asia-east1/publishers/google/models/gemini-2.5-pro:countTokens"
    )
    upstream = json.loads(seen[0].content)
    assert upstream["systemInstruction"] == {
        "parts": [{"text": "You are a coding agent."}]
    }
    assert upstream["contents"] == [
        {"role": "user", "parts": [{"text": "Inspect main.py"}]}
    ]
    assert upstream["tools"][0]["functionDeclarations"][0]["name"] == "read_file"
    count_schema = upstream["tools"][0]["functionDeclarations"][0]["parameters"]
    assert "$schema" not in count_schema
    assert "additionalProperties" not in count_schema


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

    # 4.3 A non-200 HTTP status response (e.g., 500 error)
    async def status_500_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            headers={"content-type": "application/json"},
            stream=AsyncBytes(b'{"error":{"message":"Internal error","type":"server_error"}}'),
        )
    app_500 = create_app(
        settings,
        credentials=credentials,
        upstream_transport=httpx.MockTransport(status_500_handler),
    )
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="vertex_proxy"):
        with TestClient(app_500) as client:
            client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer local-secret"},
                json={"model": "gemini-2.5-flash"},
            )
    log_records_err3 = [rec.message for rec in caplog.records if rec.name == "vertex_proxy"]
    err3_completed = [log for log in log_records_err3 if "Completed response" in log]
    assert len(err3_completed) == 1
    assert "Status 500" in err3_completed[0]


def test_get_single_model() -> None:
    credentials = FakeCredentials()
    settings = Settings(
        project="sample-project",
        location="us-central1",
        models=("google/gemini-3.6-flash", "google/gemini-2.5-pro"),
    )
    app = create_app(settings, credentials=credentials)

    with TestClient(app) as client:
        # Test getting configured model without google/ prefix
        res = client.get("/v1/models/gemini-3.6-flash")
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == "google/gemini-3.6-flash"
        assert data["object"] == "model"
        assert data["owned_by"] == "google"

        # Test getting configured model with google/ prefix
        res2 = client.get("/v1/models/google/gemini-3.6-flash")
        assert res2.status_code == 200
        assert res2.json()["id"] == "google/gemini-3.6-flash"

        # Test getting non-configured model when models setting is populated
        res3 = client.get("/v1/models/unknown-model")
        assert res3.status_code == 404
        assert "Model 'unknown-model' not found" in res3.json()["error"]["message"]

    # Test when models setting is empty (all model IDs allowed)
    settings_empty = Settings(project="sample-project", location="us-central1")
    app_empty = create_app(settings_empty, credentials=credentials)
    with TestClient(app_empty) as client:
        res4 = client.get("/v1/models/gemini-3.6-flash")
        assert res4.status_code == 200
        assert res4.json()["id"] == "gemini-3.6-flash"


def test_upstream_html_404_converted_to_json() -> None:
    credentials = FakeCredentials()

    async def handler(request: httpx.Request) -> httpx.Response:
        html_body = b"<!DOCTYPE html><html><body>Error 404</body></html>"
        return httpx.Response(
            404,
            headers={"content-type": "text/html; charset=UTF-8"},
            stream=AsyncBytes(html_body),
        )

    app = create_app(
        Settings(project="sample-project", location="us-central1"),
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )

    with TestClient(app) as client:
        res = client.get("/v1/props")
        assert res.status_code == 404
        assert res.headers["content-type"] == "application/json"
        data = res.json()
        assert "error" in data
        assert "The requested URL '/v1/props' was not found" in data["error"]["message"]


def test_health_endpoints() -> None:
    app = create_app(
        Settings(project="sample-project", location="us-central1"),
        credentials=FakeCredentials(),
    )
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/v1/healthz").json() == {"status": "ok"}
        assert client.head("/api/hello").status_code == 204


def test_thought_signature_auto_injection() -> None:
    credentials = FakeCredentials()
    seen_openai_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_openai_body.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=AsyncBytes(b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'),
        )

    app = create_app(
        Settings(project="sample-project", location="us-central1"),
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )

    with TestClient(app) as client:
        res = client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-3.6-flash",
                "messages": [
                    {"role": "user", "content": "Search the web"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "default_api:WebSearch",
                                    "arguments": '{"query":"test"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_123",
                        "content": "search result",
                    },
                ],
            },
        )

    assert res.status_code == 200
    assert len(seen_openai_body) == 1
    upstream_msg = seen_openai_body[0]["messages"][1]
    assert upstream_msg["role"] == "assistant"
    tc = upstream_msg["tool_calls"][0]
    assert tc["extra_content"]["google"]["thought_signature"] == "skip_thought_signature_validator"
    assert tc["thought_signature"] == "skip_thought_signature_validator"
    assert tc["thoughtSignature"] == "skip_thought_signature_validator"
    assert tc["function"]["thought_signature"] == "skip_thought_signature_validator"
