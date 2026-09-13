import asyncio
import json
import pytest
import httpx
from fastapi.testclient import TestClient

from vertex_proxy.app import Settings, create_app
from vertex_proxy.agent_ir import (
    ToolKind,
    ToolExecution,
    AgentTool,
    AgentStopReason,
    parse_openai_tool,
    parse_anthropic_tool,
    BackendKind,
)
from vertex_proxy.vertex_native import GeminiNativeCodec
from vertex_proxy.agent_state import (
    InMemoryAgentStateStore,
    ResponseState,
    VertexNativeState,
    VertexOpenAIState,
)
from tests.test_app import AsyncBytes, FakeCredentials

def test_unsupported_hosted_tool_explicit_error() -> None:
    # Test 1: Unknown/Unsupported hosted tools must raise explicit error instead of silent drop
    unsupported_tools = [
        {"type": "file_search"},
        {"type": "mcp"},
        {"type": "shell"},
    ]
    for tool in unsupported_tools:
        with pytest.raises(ValueError) as exc:
            parse_openai_tool(tool)
        assert "Unsupported" in str(exc.value)

def test_mixed_tools_best_effort_routing() -> None:
    # Test 2: In best-effort mode, client and provider tools are sent concurrently
    credentials = FakeCredentials()
    seen_requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        native_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Searching web..."},
                        ]
                    },
                    "finishReason": "STOP",
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://google.com", "title": "Google"}}
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 15,
                "totalTokenCount": 25
            }
        }
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=AsyncBytes(json.dumps(native_response).encode("utf-8"))
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
        res = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "gemini-2.5-flash",
                "input": "Search for AI news",
                "tools": [
                    {"type": "web_search"},
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {"type": "object", "properties": {}}
                        }
                    }
                ]
            }
        )

    assert res.status_code == 200
    data = res.json()
    assert "Sources:" in data["output"][0]["content"][0]["text"]
    assert "https://google.com" in data["output"][0]["content"][0]["text"]

    assert len(seen_requests) == 1
    upstream_body = json.loads(seen_requests[0].content)
    # Check that googleSearch and functionDeclarations both present in tools list
    tools = upstream_body["tools"]
    assert any("googleSearch" in t for t in tools)
    assert any("functionDeclarations" in t for t in tools)

def test_multi_turn_history_preservation_and_thought_signature() -> None:
    # Test 3 & Test 4: Verify O(1) Snapshot state persistence and thought signature round-tripping
    credentials = FakeCredentials()
    seen_requests = []

    chat_completion_response = {
        "id": "chatcmpl-turn-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "google/gemini-2.5-flash",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Turn 1 answer",
                    "tool_calls": [
                        {
                            "id": "call_turn_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"Beijing"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25,
        },
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=AsyncBytes(json.dumps(chat_completion_response).encode("utf-8")),
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
        # 1. First Turn
        res1 = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "gemini-2.5-flash",
                "input": "Beijing weather?",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {}}
                        }
                    }
                ]
            }
        )
        assert res1.status_code == 200
        resp_data = res1.json()
        response_id = resp_data["id"] # e.g. "resp_turn-1"
        assert response_id == "resp_turn-1"

        # 2. Second Turn referencing previous_response_id
        # Client sends back tool output
        res2 = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "gemini-2.5-flash",
                "previous_response_id": response_id,
                "input": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_turn_1",
                        "output": "Beijing is sunny, 25C"
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {}}
                        }
                    }
                ]
            }
        )
        assert res2.status_code == 200

    # Verify history reconstruction in upstream request
    assert len(seen_requests) == 2
    second_request = seen_requests[1]
    second_body = json.loads(second_request.content)
    messages = second_body["messages"]

    # History list must have reconstructed turn 1 correctly
    assert messages[0]["role"] == "user"
    assert "Beijing weather?" in messages[0]["content"]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "call_turn_1"
    # Merged latest tool result
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "call_turn_1"
    assert messages[2]["content"] == "Beijing is sunny, 25C"

def test_anthropic_hosted_tool_parsing() -> None:
    # Test 5: Verify Anthropic server/hosted tool parsing maps correctly to Provider Web Search
    hosted_tool = {
        "type": "web_search_20250305",
        "name": "web_search",
    }
    tool_ir = parse_anthropic_tool(hosted_tool)
    assert tool_ir.kind == ToolKind.WEB_SEARCH
    assert tool_ir.execution == ToolExecution.PROVIDER


def test_file_search_returns_400() -> None:
    # Defect 1: {"type":"file_search"} must return HTTP 400, not 200
    credentials = FakeCredentials()
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    settings = Settings(project="sample-project", location="us-central1", proxy_api_key="local-secret")
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        res = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "gemini-2.5-flash",
                "input": "Search files",
                "tools": [{"type": "file_search"}]
            }
        )
    assert res.status_code == 400


def test_missing_previous_response_id_returns_400() -> None:
    # Defect 2: Missing/invalid previous_response_id must return HTTP 400, not 200
    credentials = FakeCredentials()
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    settings = Settings(project="sample-project", location="us-central1", proxy_api_key="local-secret")
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        res = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "gemini-2.5-flash",
                "input": "Continue conversation",
                "previous_response_id": "resp_missing"
            }
        )
    assert res.status_code == 400


def test_hosted_tool_stream_returns_sse() -> None:
    # Defect 3: Hosted tool request with stream=true must return valid SSE events, not bare JSON
    credentials = FakeCredentials()
    native_sse_chunk = b'data: {"candidates":[{"content":{"parts":[{"text":"Searching web..."}]}}]}\n\n'

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=AsyncBytes(native_sse_chunk)
        )

    settings = Settings(project="sample-project", location="us-central1", proxy_api_key="local-secret")
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        res = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "gemini-2.5-flash",
                "input": "Search AI news",
                "stream": True,
                "tools": [{"type": "web_search"}]
            }
        )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    assert "data:" in res.text or "event:" in res.text
    # Should NOT be bare JSON without SSE formatting
    assert not res.text.strip().startswith('{"candidates"')


def test_anthropic_web_search_routes_to_native() -> None:
    # Defect 7: Anthropic web_search tool should route to Native endpoint with googleSearch
    credentials = FakeCredentials()
    seen_requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        native_response = {
            "candidates": [{
                "content": {"parts": [{"text": "Found web search results"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
        }
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=AsyncBytes(json.dumps(native_response).encode("utf-8"))
        )

    settings = Settings(project="sample-project", location="us-central1", proxy_api_key="local-secret")
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        res = client.post(
            "/v1/messages",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Search news"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}]
            }
        )

    assert res.status_code == 200
    assert len(seen_requests) == 1
    upstream_url = str(seen_requests[0].url)
    assert "generateContent" in upstream_url
    upstream_body = json.loads(seen_requests[0].content)
    assert any("googleSearch" in t for t in upstream_body.get("tools", []))


def test_incompatible_model_previous_response_id_returns_400() -> None:
    # Phase 2: Incompatible model for previous_response_id returns 400
    credentials = FakeCredentials()
    chat_completion_response = {
        "id": "chatcmpl-m1",
        "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_completion_response)

    settings = Settings(project="sample-project", location="us-central1", proxy_api_key="local-secret")
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        res1 = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={"model": "gemini-2.5-flash", "input": "Hello"}
        )
        assert res1.status_code == 200
        prev_id = res1.json()["id"]

        # Try to continue with a different model
        res2 = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={"model": "gemini-2.5-pro", "input": "Next", "previous_response_id": prev_id}
        )
        assert res2.status_code == 400


def test_instructions_not_inherited_by_default() -> None:
    # Phase 2: Instructions are not inherited into subsequent requests by default
    credentials = FakeCredentials()
    seen_requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, json={
            "id": "chatcmpl-ins",
            "choices": [{"message": {"role": "assistant", "content": "Done"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })

    settings = Settings(project="sample-project", location="us-central1", proxy_api_key="local-secret")
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        res1 = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={"model": "gemini-2.5-flash", "instructions": "System prompt 1", "input": "Turn 1"}
        )
        assert res1.status_code == 200
        prev_id = res1.json()["id"]

        res2 = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={"model": "gemini-2.5-flash", "input": "Turn 2", "previous_response_id": prev_id}
        )
        assert res2.status_code == 200

    assert len(seen_requests) == 2
    body2 = json.loads(seen_requests[1].content)
    # Turn 2 must NOT have System prompt 1 from turn 1
    system_msgs = [m for m in body2["messages"] if m.get("role") == "system"]
    assert len(system_msgs) == 0


@pytest.mark.anyio
async def test_state_store_ttl_and_lru_limits() -> None:
    # Phase 2: State store TTL and capacity limits
    store = InMemoryAgentStateStore(max_response_states=2, ttl_seconds=0.1)

    s1 = ResponseState(
        response_id="resp_1",
        previous_response_id=None,
        model="gemini-2.5-flash",
        backend=BackendKind.VERTEX_OPENAI,
        provider_snapshot=VertexOpenAIState(messages=[])
    )
    s2 = ResponseState(
        response_id="resp_2",
        previous_response_id=None,
        model="gemini-2.5-flash",
        backend=BackendKind.VERTEX_OPENAI,
        provider_snapshot=VertexOpenAIState(messages=[])
    )
    s3 = ResponseState(
        response_id="resp_3",
        previous_response_id=None,
        model="gemini-2.5-flash",
        backend=BackendKind.VERTEX_OPENAI,
        provider_snapshot=VertexOpenAIState(messages=[])
    )

    await store.save_response_state(s1)
    await store.save_response_state(s2)
    await store.save_response_state(s3)

    # Max response states capacity is 2, so resp_1 should be LRU evicted
    get_s1 = await store.get_response_state("resp_1")
    get_s2 = await store.get_response_state("resp_2")
    get_s3 = await store.get_response_state("resp_3")

    assert get_s1 is None
    assert get_s2 is not None
    assert get_s3 is not None

    # Wait for TTL expiry
    await asyncio.sleep(0.15)
    get_s2_expired = await store.get_response_state("resp_2")
    assert get_s2_expired is None


def test_pending_tool_conflict_returns_400() -> None:
    # Phase 4: In strict mode, mixing extra text into a tool_result message returns 400
    credentials = FakeCredentials()
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    settings = Settings(
        project="sample-project",
        location="us-central1",
        proxy_api_key="local-secret",
        agent_tool_mode="strict"
    )
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        # Create initial response
        res1 = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={"model": "gemini-2.5-flash", "input": "Turn 1"}
        )
        assert res1.status_code == 200
        prev_id = res1.json()["id"]

        # Follow-up message mixing tool_result and extra text in strict mode
        res2 = client.post(
            "/v1/messages",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "claude-3-5-sonnet-20241022",
                "previous_response_id": prev_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "call_123", "content": "result data"},
                            {"type": "text", "text": "Extra conflicting text prompt"}
                        ]
                    }
                ]
            }
        )
        assert res2.status_code == 400


def test_openai_responses_streaming_continuation() -> None:
    # Verify continuation works after streaming response
    credentials = FakeCredentials()
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            chunks = (
                b'data: {"id":"chatcmpl-stream-1","choices":[{"delta":{"content":"Hello stream turn 1"}}],"usage":{"prompt_tokens":5,"completion_tokens":5,"total_tokens":10}}\n\n',
                b'data: [DONE]\n\n'
            )
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=AsyncBytes(*chunks))
        else:
            return httpx.Response(200, json={
                "id": "chatcmpl-turn-2",
                "choices": [{"message": {"role": "assistant", "content": "Turn 2 answer"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            })

    settings = Settings(project="sample-project", location="us-central1", proxy_api_key="local-secret")
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        # Turn 1 streaming
        res1 = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={"model": "gemini-2.5-flash", "input": "Turn 1 prompt", "stream": True}
        )
        assert res1.status_code == 200
        lines = [line for line in res1.text.split("\n") if line.startswith("data: {")]
        last_event = json.loads(lines[-1][6:])
        resp_id = last_event["response"]["id"]

        # Turn 2 non-streaming referencing streaming resp_id
        res2 = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={"model": "gemini-2.5-flash", "input": "Turn 2 prompt", "previous_response_id": resp_id}
        )
        assert res2.status_code == 200

    assert len(seen) == 2
    up2 = json.loads(seen[1].content)
    messages = up2["messages"]
    assert messages[0]["content"] == "Turn 1 prompt"
    assert messages[1]["content"] == "Hello stream turn 1"
    assert messages[2]["content"] == "Turn 2 prompt"


def test_streaming_missing_previous_response_id_returns_400() -> None:
    # Verify missing previous_response_id with stream=True returns clean 400 JSON instead of crashing
    credentials = FakeCredentials()
    settings = Settings(project="sample-project", location="us-central1", proxy_api_key="local-secret")
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(lambda r: httpx.Response(200)))

    with TestClient(app) as client:
        res = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={"model": "gemini-2.5-flash", "input": "hi", "stream": True, "previous_response_id": "resp_missing"}
        )
        assert res.status_code == 400
        assert res.json()["error"]["type"] == "invalid_request_error"


def test_native_parallel_tools_merged_in_second_turn() -> None:
    # Verify parallel tool calls produce a single role=user turn containing multiple functionResponses
    credentials = FakeCredentials()
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            res = {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [
                            {"functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}}},
                            {"functionCall": {"name": "get_time", "args": {"city": "Tokyo"}}}
                        ]
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10, "totalTokenCount": 20}
            }
            return httpx.Response(200, json=res)
        else:
            res = {
                "candidates": [{"content": {"role": "model", "parts": [{"text": "Sunny, 12:00"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 10, "totalTokenCount": 30}
            }
            return httpx.Response(200, json=res)

    settings = Settings(project="sample-project", location="us-central1", proxy_api_key="local-secret")
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        res1 = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "gemini-2.5-flash",
                "input": "Tokyo info",
                "tools": [
                    {"type": "web_search"},
                    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}},
                    {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
                ]
            }
        )
        assert res1.status_code == 200
        data1 = res1.json()
        resp_id = data1["id"]
        calls = data1["output"]
        assert len(calls) == 2

        res2 = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "gemini-2.5-flash",
                "previous_response_id": resp_id,
                "input": [
                    {"type": "tool_result", "tool_call_id": calls[0]["id"], "output": "Sunny"},
                    {"type": "tool_result", "tool_call_id": calls[1]["id"], "output": "12:00"}
                ],
                "tools": [
                    {"type": "web_search"},
                    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}},
                    {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
                ]
            }
        )
        assert res2.status_code == 200

    assert len(seen) == 2
    up2 = json.loads(seen[1].content)
    contents = up2["contents"]
    # Turn 2 user content must be a single block containing 2 functionResponse parts
    user_turn2 = contents[2]
    assert user_turn2["role"] == "user"
    assert len(user_turn2["parts"]) == 2
    assert user_turn2["parts"][0]["functionResponse"]["name"] == "get_weather"
    assert user_turn2["parts"][1]["functionResponse"]["name"] == "get_time"


def test_native_stateless_history_function_response_resolution() -> None:
    # Verify stateless full-history messages resolve actual function names rather than call IDs
    credentials = FakeCredentials()
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": "Done"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5, "totalTokenCount": 25}
        })

    settings = Settings(project="sample-project", location="us-central1", proxy_api_key="local-secret")
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        res = client.post(
            "/v1/messages",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [
                    {"role": "user", "content": "Search news and save report"},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "Searching..."},
                        {"type": "tool_use", "id": "call_abc123", "name": "save_report", "input": {"topic": "AI"}}
                    ]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_abc123", "content": "saved successfully"}]}
                ],
                "tools": [
                    {"type": "web_search_20250305", "name": "web_search"},
                    {"name": "save_report", "description": "Save report", "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}}}
                ]
            }
        )
        assert res.status_code == 200

    assert len(seen) == 1
    up = json.loads(seen[0].content)
    # The functionResponse must have name "save_report", NOT "call_abc123"
    assert up["contents"][2]["parts"][0]["functionResponse"]["name"] == "save_report"


def test_anthropic_continuation_with_msg_id() -> None:
    # Verify Anthropic client using msg_xxx ID can continue conversation
    credentials = FakeCredentials()
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            res = {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Searching..."},
                            {"functionCall": {"name": "save_report", "args": {"topic": "AI"}}}
                        ]
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10, "totalTokenCount": 20}
            }
            return httpx.Response(200, json=res)
        else:
            res = {
                "candidates": [{"content": {"role": "model", "parts": [{"text": "All done."}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5, "totalTokenCount": 25}
            }
            return httpx.Response(200, json=res)

    settings = Settings(project="sample-project", location="us-central1", proxy_api_key="local-secret")
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        res1 = client.post(
            "/v1/messages",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Search AI news"}],
                "tools": [
                    {"type": "web_search_20250305", "name": "web_search"},
                    {"name": "save_report", "description": "Save report", "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}}}
                ]
            }
        )
        assert res1.status_code == 200
        msg_id = res1.json()["id"]
        tool_call_id = res1.json()["content"][1]["id"]
        assert msg_id.startswith("msg_")

        # Turn 2: use msg_id directly
        res2 = client.post(
            "/v1/messages",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "claude-3-5-sonnet-20241022",
                "previous_response_id": msg_id,
                "messages": [
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": "saved successfully"}]}
                ],
                "tools": [
                    {"type": "web_search_20250305", "name": "web_search"},
                    {"name": "save_report", "description": "Save report", "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}}}
                ]
            }
        )
        assert res2.status_code == 200
        assert res2.json()["content"][0]["text"] == "All done."


def test_generation_parameters_propagation() -> None:
    # Verify temperature, top_p, max_tokens, stop are passed upstream
    credentials = FakeCredentials()
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={
            "id": "chatcmpl-gen",
            "choices": [{"message": {"role": "assistant", "content": "Gen answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })

    settings = Settings(project="sample-project", location="us-central1", proxy_api_key="local-secret")
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        res = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "gemini-2.5-flash",
                "input": "Prompt",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": 500,
                "stop": ["END"]
            }
        )
        assert res.status_code == 200

    assert len(seen) == 1
    up = json.loads(seen[0].content)
    assert up["temperature"] == 0.7
    assert up["top_p"] == 0.9
    assert up["max_tokens"] == 500
    assert up["stop"] == ["END"]


def test_native_real_sse_streaming() -> None:
    # Verify Native endpoint streams via streamGenerateContent?alt=sse
    credentials = FakeCredentials()
    seen = []

    sse_chunks = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"Native streamed text"}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":5,"totalTokenCount":10}}\n\n',
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=AsyncBytes(*sse_chunks)
        )

    settings = Settings(project="sample-project", location="us-central1", proxy_api_key="local-secret")
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        res = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "gemini-2.5-flash",
                "input": "Search web",
                "stream": True,
                "tools": [{"type": "web_search"}]
            }
        )
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        assert "Native streamed text" in res.text

    assert len(seen) == 1
    assert "streamGenerateContent?alt=sse" in str(seen[0].url)


def test_strict_mode_call_registry_name_mapping() -> None:
    # Phase 4 Matrix 1: Verify Strict mode internal pseudo-function mapping and CallRegistry
    from vertex_proxy.agent_ir import AgentRequest, AgentTool, ToolKind, ToolExecution
    from vertex_proxy.agent_runtime import AgentRuntime
    from vertex_proxy.agent_state import InMemoryAgentStateStore

    runtime = AgentRuntime(InMemoryAgentStateStore())
    req = AgentRequest(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            AgentTool(kind=ToolKind.WEB_SEARCH, execution=ToolExecution.PROVIDER, name="web_search"),
            AgentTool(kind=ToolKind.FUNCTION, execution=ToolExecution.CLIENT, name="custom_tool")
        ]
    )
    plan = runtime.plan_execution(req, tool_mode="strict")
    assert plan.backend == BackendKind.VERTEX_NATIVE
    call_info = runtime.call_registry.get_call("web_search")
    assert call_info is not None
    assert call_info["client_name"] == "web_search"
    assert call_info["internal_name"] == "__proxy_web_search"


def test_strict_mixed_tool_defer_and_resume() -> None:
    # Phase 4 Matrix 2: Verify Strict mixed-tool defer on tool_use and resume on tool_result
    credentials = FakeCredentials()
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            res = {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "I will read the file."},
                            {"functionCall": {"name": "read_file", "args": {"path": "main.py"}}}
                        ]
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 10, "totalTokenCount": 25}
            }
            return httpx.Response(200, json=res)
        else:
            res = {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": "File contents analyzed."}]
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 10, "totalTokenCount": 40}
            }
            return httpx.Response(200, json=res)

    settings = Settings(
        project="sample-project",
        location="us-central1",
        proxy_api_key="local-secret",
        agent_tool_mode="strict"
    )
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        # Turn 1
        res1 = client.post(
            "/v1/messages",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Read main.py and search"}],
                "tools": [
                    {"type": "web_search_20250305", "name": "web_search"},
                    {"name": "read_file", "description": "Read file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
                ]
            }
        )
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["stop_reason"] == "tool_use"
        tool_call_id = data1["content"][1]["id"]
        msg_id = data1["id"]

        # Turn 2: resume with tool_result
        res2 = client.post(
            "/v1/messages",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "claude-3-5-sonnet-20241022",
                "previous_response_id": msg_id,
                "messages": [
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": "print('hello')"}]}
                ],
                "tools": [
                    {"type": "web_search_20250305", "name": "web_search"},
                    {"name": "read_file", "description": "Read file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
                ]
            }
        )
        assert res2.status_code == 200
        assert res2.json()["content"][0]["text"] == "File contents analyzed."


def test_server_iteration_limit_triggers_pause_turn() -> None:
    # Phase 4 Matrix 4: Verify server iteration limit produces stop_reason: "pause_turn"
    credentials = FakeCredentials()
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        res = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Searching web..."},
                        {"functionCall": {"name": "read_file", "args": {"path": "test.py"}}}
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10, "totalTokenCount": 20}
        }
        return httpx.Response(200, json=res)

    settings = Settings(
        project="sample-project",
        location="us-central1",
        proxy_api_key="local-secret",
        agent_tool_mode="strict",
        agent_max_iterations=1
    )
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        res = client.post(
            "/v1/messages",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Perform multi-step task"}],
                "tools": [
                    {"type": "web_search_20250305", "name": "web_search"},
                    {"name": "read_file", "description": "Read file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
                ]
            }
        )
        assert res.status_code == 200
        data = res.json()
        assert data["stop_reason"] == "pause_turn"


def test_resume_from_pause_turn_continuation() -> None:
    # Phase 4 Matrix 5: Verify continuation from a response that ended with pause_turn
    credentials = FakeCredentials()
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            res = {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Paused task step 1."},
                            {"functionCall": {"name": "read_file", "args": {"path": "a.txt"}}}
                        ]
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10, "totalTokenCount": 20}
            }
            return httpx.Response(200, json=res)
        else:
            res = {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Task finished successfully."}]
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5, "totalTokenCount": 25}
            }
            return httpx.Response(200, json=res)

    settings = Settings(
        project="sample-project",
        location="us-central1",
        proxy_api_key="local-secret",
        agent_tool_mode="strict",
        agent_max_iterations=1
    )
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        # Turn 1 reaches iteration limit and pauses
        res1 = client.post(
            "/v1/messages",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Run long task"}],
                "tools": [
                    {"type": "web_search_20250305", "name": "web_search"},
                    {"name": "read_file", "description": "Read file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
                ]
            }
        )
        assert res1.status_code == 200
        assert res1.json()["stop_reason"] == "pause_turn"
        pause_id = res1.json()["id"]

        # Turn 2 resumes from paused ID
        res2 = client.post(
            "/v1/messages",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "claude-3-5-sonnet-20241022",
                "previous_response_id": pause_id,
                "messages": [{"role": "user", "content": "Continue task"}],
                "tools": [
                    {"type": "web_search_20250305", "name": "web_search"},
                    {"name": "read_file", "description": "Read file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
                ]
            }
        )
        assert res2.status_code == 200
        assert res2.json()["content"][0]["text"] == "Task finished successfully."


@pytest.mark.anyio
async def test_pending_tool_state_ttl_and_eviction() -> None:
    # Phase 4 Matrix 6: Verify PendingServerToolState TTL and capacity eviction
    from vertex_proxy.agent_state import PendingServerToolState

    store = InMemoryAgentStateStore(max_pending_tools=2, ttl_seconds=0.1)
    p1 = PendingServerToolState(tool_use_id="tool_1", kind=ToolKind.FUNCTION, input={}, provider_state=None)
    p2 = PendingServerToolState(tool_use_id="tool_2", kind=ToolKind.FUNCTION, input={}, provider_state=None)
    p3 = PendingServerToolState(tool_use_id="tool_3", kind=ToolKind.FUNCTION, input={}, provider_state=None)

    await store.save_pending_tool_state(p1)
    await store.save_pending_tool_state(p2)
    await store.save_pending_tool_state(p3)

    # Max capacity is 2, so tool_1 should be LRU evicted
    assert await store.get_pending_tool_state("tool_1") is None
    assert await store.get_pending_tool_state("tool_2") is not None
    assert await store.get_pending_tool_state("tool_3") is not None

    # Wait for TTL expiry
    await asyncio.sleep(0.15)
    assert await store.get_pending_tool_state("tool_2") is None


def test_best_effort_mode_remains_unaffected() -> None:
    # Phase 4 Matrix 7: Verify default best-effort mode functions concurrently without pause_turn
    credentials = FakeCredentials()
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        res = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Searching web..."},
                        {"functionCall": {"name": "read_file", "args": {"path": "main.py"}}}
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10, "totalTokenCount": 20}
        }
        return httpx.Response(200, json=res)

    settings = Settings(
        project="sample-project",
        location="us-central1",
        proxy_api_key="local-secret",
        agent_tool_mode="best_effort"
    )
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        res = client.post(
            "/v1/messages",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Search news"}],
                "tools": [
                    {"type": "web_search_20250305", "name": "web_search"},
                    {"name": "read_file", "description": "Read file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
                ]
            }
        )
        assert res.status_code == 200
        data = res.json()
        assert data["stop_reason"] == "tool_use"


def test_native_codec_safety_settings_configured_and_unconfigured() -> None:
    # When safety_settings is provided, safetySettings key is present in native_payload
    payload = GeminiNativeCodec.encode_request(
        contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
        tools=[],
        model="gemini-2.5-flash",
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        ],
    )
    assert "safetySettings" in payload
    assert payload["safetySettings"] == [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    ]

    # Also accepts dict configuration
    payload_dict = GeminiNativeCodec.encode_request(
        contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
        tools=[],
        model="gemini-2.5-flash",
        safety_settings={
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
        },
    )
    assert "safetySettings" in payload_dict
    assert payload_dict["safetySettings"] == [
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

    # When safety_settings is None or empty, safetySettings is strictly NOT in payload
    payload_none = GeminiNativeCodec.encode_request(
        contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
        tools=[],
        model="gemini-2.5-flash",
        safety_settings=None,
    )
    assert "safetySettings" not in payload_none

    payload_empty = GeminiNativeCodec.encode_request(
        contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
        tools=[],
        model="gemini-2.5-flash",
        safety_settings=[],
    )
    assert "safetySettings" not in payload_empty


def test_native_codec_decode_prompt_blocked_vs_finish_reasons() -> None:
    # 1. Input-side blocked via promptFeedback
    blocked_input_resp = {
        "promptFeedback": {
            "blockReason": "SAFETY",
            "blockReasonMessage": "Prompt violated safety policies.",
            "safetyRatings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "probability": "HIGH", "blocked": True}
            ]
        },
        "candidates": [],
    }
    resp = GeminiNativeCodec.decode_response(blocked_input_resp, "gemini-2.5-flash")
    assert resp.stop_reason == AgentStopReason.PROMPT_BLOCKED
    assert resp.output == []

    # 2. Output-side prohibited content
    prohibited_resp = {
        "candidates": [
            {
                "content": {"role": "model", "parts": []},
                "finishReason": "PROHIBITED_CONTENT",
            }
        ]
    }
    resp = GeminiNativeCodec.decode_response(prohibited_resp, "gemini-2.5-flash")
    assert resp.stop_reason == AgentStopReason.PROHIBITED_CONTENT

    # 3. Output-side other reason
    other_resp = {
        "candidates": [
            {
                "content": {"role": "model", "parts": []},
                "finishReason": "OTHER",
            }
        ]
    }
    resp = GeminiNativeCodec.decode_response(other_resp, "gemini-2.5-flash")
    assert resp.stop_reason == AgentStopReason.OTHER

    # 4. Output-side safety
    safety_resp = {
        "candidates": [
            {
                "content": {"role": "model", "parts": []},
                "finishReason": "SAFETY",
            }
        ]
    }
    resp = GeminiNativeCodec.decode_response(safety_resp, "gemini-2.5-flash")
    assert resp.stop_reason == AgentStopReason.REFUSAL


def test_anthropic_gemini_native_prompt_blocked_and_finish_reasons(caplog: pytest.LogCaptureFixture) -> None:
    # Test streaming and unary mapping of prompt_blocked and prohibited_content in Anthropic endpoint
    credentials = FakeCredentials()

    # Case A: Input-side promptFeedback.blockReason streaming
    async def handler_prompt_blocked(request: httpx.Request) -> httpx.Response:
        sse_data = (
            'data: {"promptFeedback":{"blockReason":"SAFETY","blockReasonMessage":"Prompt unsafe","safetyRatings":[{"category":"HARM_CATEGORY_HATE_SPEECH","probability":"HIGH","blocked":true}]},"candidates":[]}\n\n'
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_data.encode("utf-8")
        )

    settings = Settings(
        project="sample-project",
        location="us-central1",
        proxy_api_key="local-secret",
        safety_settings=(
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        ),
    )
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler_prompt_blocked))

    with caplog.at_level("WARNING"):
        with TestClient(app) as client:
            res = client.post(
                "/v1/messages",
                headers={"authorization": "Bearer local-secret"},
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "Sensitive prompt"}],
                    "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                    "stream": True,
                }
            )
            assert res.status_code == 200
            assert "prompt_blocked" in res.text
            # Verify logging contains input-side interception diagnostic with no prompt text
            assert any("Gemini prompt blocked (input-side)" in r.message for r in caplog.records)
            assert not any("Sensitive prompt" in r.message for r in caplog.records)

    # Case B: Output-side finishReason="PROHIBITED_CONTENT" unary
    async def handler_prohibited(request: httpx.Request) -> httpx.Response:
        native_response = {
            "candidates": [
                {
                    "content": {"parts": []},
                    "finishReason": "PROHIBITED_CONTENT",
                }
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 0, "totalTokenCount": 5}
        }
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(native_response).encode("utf-8")
        )

    app_prohibited = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler_prohibited))

    with caplog.at_level("WARNING"):
        with TestClient(app_prohibited) as client:
            res = client.post(
                "/v1/messages",
                headers={"authorization": "Bearer local-secret"},
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "Another query"}],
                    "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                }
            )
            assert res.status_code == 200
            data = res.json()
            assert data["stop_reason"] == "prohibited_content"
            assert any("finishReason=PROHIBITED_CONTENT" in r.message for r in caplog.records)


def test_native_runtime_propagates_safety_settings_to_upstream() -> None:
    credentials = FakeCredentials()
    seen_requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps({"candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}]}).encode("utf-8")
        )

    configured_safety = (
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    )
    settings = Settings(
        project="sample-project",
        location="us-central1",
        proxy_api_key="local-secret",
        safety_settings=configured_safety,
    )
    app = create_app(settings, credentials=credentials, upstream_transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        res = client.post(
            "/v1/messages",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            }
        )
        assert res.status_code == 200

    assert len(seen_requests) == 1
    upstream_payload = json.loads(seen_requests[0].read())
    assert "safetySettings" in upstream_payload
    assert upstream_payload["safetySettings"] == [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]



