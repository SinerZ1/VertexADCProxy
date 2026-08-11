import json
import httpx
from fastapi.testclient import TestClient

from vertex_proxy.app import Settings, create_app
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
    assert "event: response.output_item.added" in body
    assert "event: response.text.delta" in body
    assert '"delta":"Hello "' in body
    assert '"delta":"world!"' in body
    assert "event: response.function_call_arguments.delta" in body
    assert "event: response.text.done" in body
    assert "event: response.completed" in body
    assert "data: [DONE]" in body
