import asyncio
import json
import httpx
import pytest
from fastapi.testclient import TestClient

from vertex_proxy.app import Settings, create_app
from vertex_proxy.agent_ir import AgentEvent, AgentEventKind, AgentStopReason
from vertex_proxy.openai_responses import (
    chat_completions_to_responses,
    unary_responses_event,
    stream_responses_events,
)
from tests.test_app import AsyncBytes, FakeCredentials


def test_openai_responses_non_streaming() -> None:
    credentials = FakeCredentials()
    seen_requests: list[httpx.Request] = []

    chat_completion_response = {
        "id": "chatcmpl-999",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "google/gemini-2.5-flash",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello world from Responses API",
                    "tool_calls": [
                        {
                            "id": "call_abc",
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
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
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
        res = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "gemini-2.5-flash",
                "instructions": "You are a helpful assistant",
                "input": "Check weather in Beijing",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather info",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        )

    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "response"
    assert data["id"] == "resp_999"
    assert data["status"] == "completed"
    assert data["model"] == "gemini-2.5-flash"
    assert len(data["output"]) == 2
    assert data["output"][0]["type"] == "message"
    assert data["output"][0]["content"][0]["text"] == "Hello world from Responses API"
    assert data["output"][1]["type"] == "function_call"
    assert data["output"][1]["name"] == "get_weather"
    assert data["output"][1]["arguments"] == '{"location":"Beijing"}'
    assert data["usage"] == {
        "input_tokens": 15,
        "output_tokens": 25,
        "total_tokens": 40,
    }

    # Verify upstream request transformation
    assert len(seen_requests) == 1
    upstream = seen_requests[0]
    assert str(upstream.url) == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/sample-project/"
        "locations/us-central1/endpoints/openapi/chat/completions"
    )
    upstream_body = json.loads(upstream.content)
    assert upstream_body["model"] == "google/gemini-2.5-flash"
    assert upstream_body["messages"][0] == {
        "role": "system",
        "content": "You are a helpful assistant",
    }
    assert upstream_body["messages"][1] == {
        "role": "user",
        "content": "Check weather in Beijing",
    }


def test_openai_responses_input_text_normalization() -> None:
    credentials = FakeCredentials()
    seen_requests: list[httpx.Request] = []

    chat_completion_response = {
        "id": "chatcmpl-888",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "google/gemini-2.5-flash",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Normalized input response",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
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
        res = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer local-secret"},
            json={
                "model": "gemini-2.5-flash",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Hello world with input_text"}
                        ],
                    }
                ],
            },
        )

    assert res.status_code == 200
    assert len(seen_requests) == 1
    upstream_body = json.loads(seen_requests[0].content)
    assert upstream_body["messages"][0] == {
        "role": "user",
        "content": "Hello world with input_text",
    }


def test_openai_responses_streaming() -> None:

    credentials = FakeCredentials()
    upstream_chunks = (
        b'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"world!"}}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_11","function":{"name":"search","arguments":"{\\"query\\":"}}]}}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"AI\\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":12}}\n\n',
        b"data: [DONE]\n\n",
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=AsyncBytes(*upstream_chunks),
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
                "input": "Hi",
                "stream": True,
            },
        )

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    body = res.text

    assert "event: response.created" in body
    assert "event: response.in_progress" in body
    assert "event: response.output_item.added" in body
    assert "event: response.output_text.delta" in body
    assert '"delta":"Hello "' in body
    assert '"delta":"world!"' in body
    assert "event: response.function_call_arguments.delta" in body
    assert "event: response.output_text.done" in body
    assert "event: response.completed" in body
    assert "data: [DONE]" in body


def test_openai_responses_headers_cleanup() -> None:
    import gzip
    credentials = FakeCredentials()
    raw_data = (
        b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    compressed_data = gzip.compress(raw_data)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Type": "text/event-stream",
                "Content-Encoding": "gzip",
                "Content-Length": str(len(compressed_data)),
            },
            stream=AsyncBytes(compressed_data),
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
                "input": "Hi",
                "stream": True,
            },
        )

    assert res.status_code == 200
    assert "content-encoding" not in res.headers
    assert res.headers.get("content-length") != str(len(compressed_data))
    assert res.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert "event: response.created" in res.text


def test_openai_responses_chat_completions_to_responses_filter_mappings() -> None:
    for reason in ["content_filter", "safety", "prohibited_content", "other", "prompt_blocked"]:
        payload = {
            "id": "chatcmpl-test",
            "choices": [{"finish_reason": reason, "message": {"content": ""}}],
        }
        res = chat_completions_to_responses(payload, "gemini-2.5-flash")
        assert res["status"] == "failed"


def test_openai_responses_events_stop_reason_handling() -> None:
    async def _test():
        async def mock_events_prompt_blocked():
            yield AgentEvent(kind=AgentEventKind.RESPONSE_STARTED, data={"response_id": "resp_123"})
            yield AgentEvent(kind=AgentEventKind.COMPLETED, data={"id": "resp_123", "stop_reason": AgentStopReason.PROMPT_BLOCKED})

        res_json, _ = await unary_responses_event(mock_events_prompt_blocked(), "gemini-2.5-flash")
        assert res_json["status"] == "failed"

        async def mock_events_prohibited():
            yield AgentEvent(kind=AgentEventKind.RESPONSE_STARTED, data={"response_id": "resp_456"})
            yield AgentEvent(kind=AgentEventKind.COMPLETED, data={"id": "resp_456", "stop_reason": AgentStopReason.PROHIBITED_CONTENT})

        res_json2, _ = await unary_responses_event(mock_events_prohibited(), "gemini-2.5-flash")
        assert res_json2["status"] == "failed"

        # Streaming
        chunks = []
        async for chunk in stream_responses_events(mock_events_prompt_blocked(), "gemini-2.5-flash"):
            chunks.append(chunk.decode("utf-8"))
        stream_text = "".join(chunks)
        assert 'response.failed' in stream_text or 'status": "failed"' in stream_text or '"status":"failed"' in stream_text

    asyncio.run(_test())

