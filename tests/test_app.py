import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
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
    # Case-insensitive prefix conversion (Claude- -> gemini-)
    assert resolve_gemini_model("Claude-2.5-flash", settings) == "gemini-2.5-flash"
    assert resolve_gemini_model("CLAUDE-2.5-PRO", settings) == "gemini-2.5-pro"


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
        "https://asia-east1-aiplatform.googleapis.com/v1beta1/projects/sample-project/"
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


def test_anthropic_gemini_count_tokens_merges_consecutive_turns_and_handles_thinking() -> None:
    import json
    credentials = FakeCredentials()
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=AsyncBytes(b'{"totalTokens":100}'),
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
                "model": "Claude-2.5-pro",
                "messages": [
                    {"role": "user", "content": "Question 1"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Thinking about it..."},
                            {"type": "tool_use", "id": "tool_1", "name": "get_weather", "input": {"city": "Paris"}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "tool_1", "content": "22C sunny"},
                        ],
                    },
                    {"role": "tool", "tool_call_id": "tool_1", "content": "cached result"},
                    {"role": "user", "content": "Follow-up question"},
                ],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 100}

    body = json.loads(seen[0].content.decode("utf-8"))
    contents = body["contents"]
    # Verify contents have strictly alternating roles (user -> model -> user)
    roles = [c["role"] for c in contents]
    assert roles == ["user", "model", "user"]
    # The last user turn should have merged all 3 consecutive user/tool blocks
    assert len(contents[2]["parts"]) == 3


def test_anthropic_gemini_count_tokens_handles_gzipped_upstream_error() -> None:
    import gzip
    credentials = FakeCredentials()
    err_json = b'{"error":{"code":400,"message":"Invalid argument provided","status":"INVALID_ARGUMENT"}}'
    compressed_err = gzip.compress(err_json)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            headers={
                "content-type": "application/json; charset=UTF-8",
                "content-encoding": "gzip",
            },
            stream=AsyncBytes(compressed_err),
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
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 400
    # Client receives decompressed UTF-8 json, not raw binary gzip
    assert response.json()["error"]["message"] == "Invalid argument provided"


def test_adc_token_provider_reuses_unexpired_token_on_refresh_failure() -> None:
    from datetime import datetime, timedelta, timezone
    from google.auth.credentials import Credentials
    from vertex_proxy.app import AdcTokenProvider
    from google.auth.transport.requests import Request as GoogleAuthRequest

    class FlakyCredentials(Credentials):
        def __init__(self):
            super().__init__()
            self.token = "existing-valid-token"
            self.expiry = datetime.now(timezone.utc) + timedelta(seconds=60)
            self.refresh_attempted = False

        def refresh(self, request):
            self.refresh_attempted = True
            raise ConnectionResetError("Connection reset by peer during OAuth refresh")

    creds = FlakyCredentials()
    auth_req = GoogleAuthRequest()
    # skew is 300s, so needs_refresh is True (since expiry is in 60s < 300s)
    provider = AdcTokenProvider(creds, auth_req, refresh_skew=300.0)

    # In async loop, token() should try to refresh, encounter error, but reuse existing unexpired token
    import asyncio
    token = asyncio.run(provider.token())
    assert creds.refresh_attempted is True
    assert token == "existing-valid-token"


def test_anthropic_messages_supports_case_insensitive_claude_prefix() -> None:
    credentials = FakeCredentials()
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1234567,
                "model": "gemini-3.1-pro-preview",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello there!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    app = create_app(
        Settings(
            project="sample-project",
            location="global",
            models=("gemini-3.1-pro-preview",),
        ),
        credentials=credentials,
        upstream_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "Claude-3.1-pro-preview",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["role"] == "assistant"
    assert res_data["content"][0]["text"] == "Hello there!"


def test_settings_safety_settings_from_env_json() -> None:
    # Test list format
    env = {
        "GOOGLE_CLOUD_PROJECT": "proj",
        "VERTEX_LOCATION": "global",
        "VERTEX_SAFETY_SETTINGS": json.dumps([
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        ]),
    }
    s = Settings.from_env(env)
    assert s.safety_settings == (
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    )

    # Test dict format
    env_dict = {
        "GOOGLE_CLOUD_PROJECT": "proj",
        "VERTEX_LOCATION": "global",
        "VERTEX_SAFETY_SETTINGS": json.dumps({
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
        }),
    }
    s_dict = Settings.from_env(env_dict)
    assert s_dict.safety_settings == (
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
    )

    # Test unconfigured
    env_unconfigured = {
        "GOOGLE_CLOUD_PROJECT": "proj",
        "VERTEX_LOCATION": "global",
    }
    s_unconf = Settings.from_env(env_unconfigured)
    assert s_unconf.safety_settings is None


def test_settings_safety_settings_from_config_file(tmp_path: Any) -> None:
    cfg = tmp_path / "proxy_config.json"
    cfg.write_text(json.dumps({
        "safety_settings": [
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}
        ]
    }), encoding="utf-8")

    env = {
        "GOOGLE_CLOUD_PROJECT": "proj",
        "VERTEX_LOCATION": "global",
        "PROXY_CONFIG_FILE": str(cfg),
    }
    s = Settings.from_env(env)
    assert s.safety_settings == (
        {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
    )


def test_openai_proxy_chat_completions_injects_safety_settings_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    credentials = FakeCredentials()
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, stream=AsyncBytes(b'{"choices": [{"message": {"content": "Hello"}}]}'))

    settings = Settings(
        project="sample-project",
        location="us-central1",
        safety_settings=(
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        ),
    )
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with caplog.at_level("INFO"):
        with TestClient(app) as client:
            res = client.post(
                "/v1/chat/completions",
                json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}]}
            )
            assert res.status_code == 200

    assert len(seen) == 1
    req_body = json.loads(seen[0].read())
    assert "safetySettings" in req_body
    assert req_body["safetySettings"] == [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    ]
    assert any("safetySettings applied (2 categories)" in r.message for r in caplog.records)
    assert any("HARM_CATEGORY_HARASSMENT" in r.message for r in caplog.records)


def test_openai_proxy_chat_completions_unconfigured_preserves_behavior(caplog: pytest.LogCaptureFixture) -> None:
    credentials = FakeCredentials()
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, stream=AsyncBytes(b'{"choices": [{"message": {"content": "Hello"}}]}'))

    settings = Settings(
        project="sample-project",
        location="us-central1",
        safety_settings=None,
    )
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with caplog.at_level("INFO"):
        with TestClient(app) as client:
            res = client.post(
                "/v1/chat/completions",
                json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}]}
            )
            assert res.status_code == 200

    assert len(seen) == 1
    req_body = json.loads(seen[0].read())
    assert "safetySettings" not in req_body
    assert any("safetySettings: None (using upstream defaults)" in r.message for r in caplog.records)


def test_openai_proxy_diagnoses_prompt_blocked_and_prohibited_content(caplog: pytest.LogCaptureFixture) -> None:
    credentials = FakeCredentials()

    # Input-side blocked response
    async def handler_input_blocked(request: httpx.Request) -> httpx.Response:
        body = json.dumps({
            "promptFeedback": {
                "blockReason": "SAFETY",
                "blockReasonMessage": "Input blocked by safety",
                "safetyRatings": [{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "probability": "HIGH", "blocked": True}]
            },
            "candidates": []
        }).encode("utf-8")
        return httpx.Response(200, stream=AsyncBytes(body))

    app1 = create_app(Settings(project="sample-project", location="us-central1"), credentials=credentials, upstream_transport=httpx.MockTransport(handler_input_blocked))

    with caplog.at_level("WARNING"):
        with TestClient(app1) as client:
            res = client.post("/v1/chat/completions", json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "blocked input"}]})
            assert res.status_code == 200
            assert any("Gemini prompt blocked (input-side)" in r.message for r in caplog.records)

    # Output-side prohibited content response
    async def handler_output_blocked(request: httpx.Request) -> httpx.Response:
        body = json.dumps({
            "candidates": [{
                "finishReason": "PROHIBITED_CONTENT",
                "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH"}]
            }]
        }).encode("utf-8")
        return httpx.Response(200, stream=AsyncBytes(body))

    app2 = create_app(Settings(project="sample-project", location="us-central1"), credentials=credentials, upstream_transport=httpx.MockTransport(handler_output_blocked))

    with caplog.at_level("WARNING"):
        with TestClient(app2) as client:
            res = client.post("/v1/chat/completions", json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "test"}]})
            assert res.status_code == 200
            assert any("finishReason=PROHIBITED_CONTENT" in r.message for r in caplog.records)



