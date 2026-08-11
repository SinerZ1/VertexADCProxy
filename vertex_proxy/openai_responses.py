from __future__ import annotations

import asyncio
import json
import secrets
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Mapping

import httpx


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _sse(event: str, value: Mapping[str, Any]) -> bytes:
    return b"event: " + event.encode() + b"\n" + b"data: " + _json_bytes(value) + b"\n\n"


def _normalize_tool(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    if "function" in tool and isinstance(tool["function"], dict):
        return tool
    name = tool.get("name")
    if isinstance(name, str) and name:
        func: dict[str, Any] = {"name": name}
        if isinstance(tool.get("description"), str):
            func["description"] = tool["description"]
        params = tool.get("parameters") or tool.get("input_schema")
        if isinstance(params, dict):
            func["parameters"] = params
        return {"type": "function", "function": func}
    return None


def _normalize_tool_choice(choice: Any) -> Any:
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict):
        if "function" in choice:
            return choice
        if choice.get("type") == "function" and isinstance(choice.get("name"), str):
            return {"type": "function", "function": {"name": choice["name"]}}
        if isinstance(choice.get("name"), str):
            return {"type": "function", "function": {"name": choice["name"]}}
    return choice


def _normalize_input_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content

    normalized: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "input_text":
            text = block.get("text", "")
            normalized.append({"type": "text", "text": text})
        elif block_type == "input_image":
            image_url = block.get("image_url")
            if isinstance(image_url, dict):
                normalized.append({"type": "image_url", "image_url": image_url})
            elif isinstance(block.get("source"), dict):
                src = block["source"]
                if src.get("type") == "base64":
                    media_type = src.get("media_type", "image/png")
                    data = src.get("data", "")
                    normalized.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    })
                elif src.get("type") == "url":
                    normalized.append({
                        "type": "image_url",
                        "image_url": {"url": src.get("url", "")},
                    })
        else:
            normalized.append(block)

    if all(b.get("type") == "text" for b in normalized if isinstance(b, dict)):
        return "".join(b.get("text", "") for b in normalized if isinstance(b, dict))
    return normalized


def responses_to_chat_completions(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    client_model = payload.get("model")
    if not isinstance(client_model, str) or not client_model.strip():
        raise ValueError("缺少有效的 model 字段")

    client_model = client_model.strip()
    model = client_model if "/" in client_model else f"google/{client_model}"

    messages: list[dict[str, Any]] = []

    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})

    input_data = payload.get("input")
    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
    elif isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                item_type = item.get("type")
                if "role" in item:
                    msg = {"role": item["role"], "content": _normalize_input_content(item.get("content"))}
                    if "tool_calls" in item:
                        msg["tool_calls"] = item["tool_calls"]
                    messages.append(msg)
                elif item_type == "message":
                    messages.append(
                        {
                            "role": item.get("role", "user"),
                            "content": _normalize_input_content(item.get("content")),
                        }
                    )
                elif item_type == "function_call":
                    call_id = (
                        item.get("call_id")
                        or item.get("id")
                        or f"call_{secrets.token_hex(8)}"
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": item.get("name", ""),
                                        "arguments": item.get("arguments", "{}"),
                                    },
                                }
                            ],
                        }
                    )
                elif item_type in {"function_call_output", "tool_result"}:
                    call_id = (
                        item.get("call_id")
                        or item.get("tool_call_id")
                        or item.get("id")
                    )
                    out = item.get("output") or item.get("content", "")
                    content_str = (
                        out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": content_str,
                        }
                    )

    if isinstance(payload.get("messages"), list):
        for msg in payload["messages"]:
            if isinstance(msg, dict) and "role" in msg:
                m = dict(msg)
                if "content" in m:
                    m["content"] = _normalize_input_content(m["content"])
                messages.append(m)

    converted: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if "temperature" in payload:
        converted["temperature"] = payload["temperature"]
    if "top_p" in payload:
        converted["top_p"] = payload["top_p"]

    max_tokens = (
        payload.get("max_output_tokens")
        or payload.get("max_completion_tokens")
        or payload.get("max_tokens")
    )
    if max_tokens is not None:
        converted["max_tokens"] = max_tokens

    if "response_format" in payload:
        converted["response_format"] = payload["response_format"]

    raw_tools = payload.get("tools")
    if isinstance(raw_tools, list):
        tools = [
            norm for tool in raw_tools if (norm := _normalize_tool(tool)) is not None
        ]
        if tools:
            converted["tools"] = tools

    if "tool_choice" in payload:
        converted["tool_choice"] = _normalize_tool_choice(payload["tool_choice"])

    if "parallel_tool_calls" in payload:
        converted["parallel_tool_calls"] = payload["parallel_tool_calls"]

    stream = payload.get("stream") is True
    converted["stream"] = stream
    if stream:
        converted["stream_options"] = {"include_usage": True}

    return client_model, converted


def chat_completions_to_responses(
    payload: Mapping[str, Any],
    client_model: str,
    response_id: str | None = None,
) -> dict[str, Any]:
    if not response_id:
        orig_id = payload.get("id", "")
        if orig_id.startswith("chatcmpl-"):
            response_id = "resp_" + orig_id[9:]
        else:
            response_id = f"resp_{secrets.token_hex(12)}"

    created_at = payload.get("created", int(datetime.now(timezone.utc).timestamp()))
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice, dict):
        choice = {}

    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    finish_reason = choice.get("finish_reason")

    status = "completed"
    if finish_reason == "length":
        status = "incomplete"
    elif finish_reason in {"content_filter", "safety"}:
        status = "failed"

    output: list[dict[str, Any]] = []

    text = message.get("content")
    if isinstance(text, str) and text:
        output.append(
            {
                "id": f"msg_{secrets.token_hex(12)}",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    }
                ],
            }
        )

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function")
            if not isinstance(func, dict):
                continue
            call_id = tc.get("id") or f"call_{secrets.token_hex(12)}"
            output.append(
                {
                    "id": call_id,
                    "type": "function_call",
                    "call_id": call_id,
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),
                }
            )

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": client_model,
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
    }


async def _iter_sse_data(
    response: httpx.Response,
    *,
    heartbeat_seconds: float = 15.0,
) -> AsyncIterator[str | None]:
    lines = response.aiter_lines().__aiter__()
    pending: asyncio.Task[str] | None = None
    data_lines: list[str] = []
    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(anext(lines))
            done, _ = await asyncio.wait({pending}, timeout=heartbeat_seconds)
            if not done:
                yield None
                continue
            try:
                line = pending.result()
            except StopAsyncIteration:
                break
            pending = None
            if not line:
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines.clear()
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            yield "\n".join(data_lines)
    finally:
        if pending is not None and not pending.done():
            pending.cancel()


async def _stream_chat_completions_as_responses(
    response: httpx.Response,
    client_model: str,
) -> AsyncIterator[bytes]:
    response_id = f"resp_{secrets.token_hex(12)}"
    created_at = int(datetime.now(timezone.utc).timestamp())

    yield _sse(
        "response.created",
        {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "status": "in_progress",
                "model": client_model,
                "output": [],
                "usage": None,
            },
        },
    )

    text_started = False
    msg_id = f"msg_{secrets.token_hex(12)}"
    accumulated_text = ""
    text_output_index = None
    next_output_index = 0

    tool_calls: dict[int, dict[str, Any]] = {}
    usage_dict: dict[str, int] | None = None

    async for raw_data in _iter_sse_data(response):
        if raw_data is None:
            yield _sse("ping", {"type": "ping"})
            continue
        if raw_data == "[DONE]":
            break
        try:
            chunk = json.loads(raw_data)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue

        if "error" in chunk:
            yield _sse("error", {"type": "error", "error": chunk["error"]})
            return

        if isinstance(chunk.get("usage"), dict):
            usage_dict = chunk["usage"]

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue

        text_delta = delta.get("content")
        if isinstance(text_delta, str) and text_delta:
            if not text_started:
                text_output_index = next_output_index
                next_output_index += 1
                text_started = True
                yield _sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "response_id": response_id,
                        "output_index": text_output_index,
                        "item": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                )
                yield _sse(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "response_id": response_id,
                        "output_index": text_output_index,
                        "content_index": 0,
                        "part": {"type": "text", "text": ""},
                    },
                )

            accumulated_text += text_delta
            yield _sse(
                "response.text.delta",
                {
                    "type": "response.text.delta",
                    "response_id": response_id,
                    "output_index": text_output_index,
                    "content_index": 0,
                    "delta": text_delta,
                },
            )

        streamed_tools = delta.get("tool_calls")
        if isinstance(streamed_tools, list):
            for tc in streamed_tools:
                if not isinstance(tc, dict):
                    continue
                idx = tc.get("index", 0)
                if not isinstance(idx, int):
                    idx = 0
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": "",
                        "name": "",
                        "arguments": "",
                        "output_index": next_output_index,
                        "started": False,
                    }
                    next_output_index += 1

                entry = tool_calls[idx]
                if isinstance(tc.get("id"), str) and tc["id"]:
                    entry["id"] = tc["id"]
                func = tc.get("function")
                if isinstance(func, dict):
                    if isinstance(func.get("name"), str) and func["name"]:
                        entry["name"] += func["name"]
                    if not entry["started"]:
                        entry["started"] = True
                        call_id = entry["id"] or f"call_{secrets.token_hex(12)}"
                        entry["id"] = call_id
                        yield _sse(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "response_id": response_id,
                                "output_index": entry["output_index"],
                                "item": {
                                    "id": call_id,
                                    "type": "function_call",
                                    "call_id": call_id,
                                    "name": entry["name"],
                                    "arguments": "",
                                },
                            },
                        )
                    if isinstance(func.get("arguments"), str) and func["arguments"]:
                        args_delta = func["arguments"]
                        entry["arguments"] += args_delta
                        yield _sse(
                            "response.function_call_arguments.delta",
                            {
                                "type": "response.function_call_arguments.delta",
                                "response_id": response_id,
                                "output_index": entry["output_index"],
                                "call_id": entry["id"],
                                "delta": args_delta,
                            },
                        )

    final_output: list[dict[str, Any]] = []

    if text_started:
        yield _sse(
            "response.text.done",
            {
                "type": "response.text.done",
                "response_id": response_id,
                "output_index": text_output_index,
                "content_index": 0,
                "text": accumulated_text,
            },
        )
        yield _sse(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "response_id": response_id,
                "output_index": text_output_index,
                "content_index": 0,
                "part": {"type": "text", "text": accumulated_text},
            },
        )
        msg_item = {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": accumulated_text}],
        }
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": text_output_index,
                "item": msg_item,
            },
        )
        final_output.append(msg_item)

    for _, entry in sorted(tool_calls.items()):
        if entry["started"]:
            yield _sse(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "response_id": response_id,
                    "output_index": entry["output_index"],
                    "call_id": entry["id"],
                    "arguments": entry["arguments"],
                },
            )
            fn_item = {
                "id": entry["id"],
                "type": "function_call",
                "call_id": entry["id"],
                "name": entry["name"],
                "arguments": entry["arguments"],
            }
            yield _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "response_id": response_id,
                    "output_index": entry["output_index"],
                    "item": fn_item,
                },
            )
            final_output.append(fn_item)

    input_tokens = usage_dict.get("prompt_tokens", 0) if usage_dict else 0
    output_tokens = usage_dict.get("completion_tokens", 0) if usage_dict else 0
    total_tokens = (
        usage_dict.get("total_tokens", input_tokens + output_tokens)
        if usage_dict
        else input_tokens + output_tokens
    )

    yield _sse(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "status": "completed",
                "model": client_model,
                "output": final_output,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            },
        },
    )
    yield b"data: [DONE]\n\n"


class OpenAIResponsesAdapter:
    """Adapts OpenAI Responses API (/v1/responses) requests to Vertex OpenAI Chat Completions API."""

    def __init__(self) -> None:
        self.client_model = ""
        self.stream = False

    def prepare(self, body: bytes, config: Any) -> tuple[str, bytes]:
        from vertex_proxy.app import _ensure_thought_signatures

        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求体必须是有效的 JSON 对象") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")

        client_model, converted = responses_to_chat_completions(payload)
        _ensure_thought_signatures(converted)

        self.client_model = client_model
        self.stream = converted.get("stream") is True

        return f"{config.openai_base_url}/chat/completions", _json_bytes(converted)

    def response_headers(
        self,
        response: httpx.Response,
        headers: Mapping[str, str],
    ) -> dict[str, str]:
        result = dict(headers)
        if response.status_code >= 400:
            return result
        for name in ("content-length", "content-encoding", "transfer-encoding"):
            result.pop(name, None)
        result["content-type"] = (
            "text/event-stream; charset=utf-8"
            if self.stream
            else "application/json; charset=utf-8"
        )
        return result

    async def transform(self, response: httpx.Response) -> AsyncIterator[bytes]:
        if response.status_code >= 400:
            async for chunk in response.aiter_raw():
                yield chunk
            return
        if self.stream:
            async for chunk in _stream_chat_completions_as_responses(
                response, self.client_model
            ):
                yield chunk
            return

        body = await response.aread()
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            yield body
            return
        if not isinstance(payload, dict):
            yield body
            return
        yield _json_bytes(chat_completions_to_responses(payload, self.client_model))
