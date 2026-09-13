from __future__ import annotations

import asyncio
import json
import secrets
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Mapping

import httpx

from vertex_proxy.agent_ir import (
    AgentEvent,
    AgentEventKind,
    AgentRequest,
    AgentResponse,
    AgentStopReason,
    AgentTool,
    AgentToolChoice,
    BackendKind,
    ToolChoiceMode,
    ToolExecution,
    ToolKind,
    parse_openai_tool,
)
from vertex_proxy.agent_state import (
    InMemoryAgentStateStore,
    PendingServerToolState,
    ProviderTurnState,
    ResponseState,
    ToolCallState,
    VertexNativeState,
    VertexOpenAIState,
)
from vertex_proxy.vertex_native import GeminiNativeCodec

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
    elif finish_reason in {"prohibited_content", "PROHIBITED_CONTENT"}:
        status = "failed"
    elif finish_reason in {"other", "OTHER"}:
        status = "failed"
    elif finish_reason in {"prompt_blocked", "PROMPT_BLOCKED"}:
        status = "failed"

    output: list[dict[str, Any]] = []

    text = message.get("content")
    if isinstance(text, str) and text:
        output.append(
            {
                "id": f"msg_{secrets.token_hex(12)}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
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
                    "status": "completed",
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
        "completed_at": created_at,
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

    resp_obj = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "completed_at": None,
        "status": "in_progress",
        "model": client_model,
        "output": [],
        "error": None,
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "usage": None,
    }

    yield _sse(
        "response.created",
        {
            "type": "response.created",
            "response": resp_obj,
        },
    )

    yield _sse(
        "response.in_progress",
        {
            "type": "response.in_progress",
            "response": resp_obj,
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
                        "output_index": text_output_index,
                        "item": {
                            "id": msg_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                )
                yield _sse(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": msg_id,
                        "output_index": text_output_index,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                            "logprobs": [],
                        },
                    },
                )

            accumulated_text += text_delta
            yield _sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": text_output_index,
                    "content_index": 0,
                    "delta": text_delta,
                    "logprobs": [],
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
                                "output_index": entry["output_index"],
                                "item": {
                                    "id": call_id,
                                    "type": "function_call",
                                    "status": "in_progress",
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
                                "item_id": entry["id"],
                                "output_index": entry["output_index"],
                                "call_id": entry["id"],
                                "delta": args_delta,
                            },
                        )

    final_output: list[dict[str, Any]] = []

    if text_started:
        yield _sse(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": msg_id,
                "output_index": text_output_index,
                "content_index": 0,
                "text": accumulated_text,
                "logprobs": [],
            },
        )
        yield _sse(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": msg_id,
                "output_index": text_output_index,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": accumulated_text,
                    "annotations": [],
                    "logprobs": [],
                },
            },
        )
        msg_item = {
            "id": msg_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": accumulated_text,
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
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
                    "item_id": entry["id"],
                    "output_index": entry["output_index"],
                    "call_id": entry["id"],
                    "arguments": entry["arguments"],
                },
            )
            fn_item = {
                "id": entry["id"],
                "type": "function_call",
                "status": "completed",
                "call_id": entry["id"],
                "name": entry["name"],
                "arguments": entry["arguments"],
            }
            yield _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
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

    completed_at = int(datetime.now(timezone.utc).timestamp())
    yield _sse(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "completed_at": completed_at,
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


def responses_to_agent_request(payload: Mapping[str, Any]) -> AgentRequest:
    client_model = payload.get("model")
    if not isinstance(client_model, str) or not client_model.strip():
        raise ValueError("缺少有效的 model 字段")

    client_model = client_model.strip()
    model = client_model if "/" in client_model else f"google/{client_model}"

    messages: list[dict[str, Any]] = []
    instructions = payload.get("instructions")
    instructions_str = instructions.strip() if isinstance(instructions, str) and instructions.strip() else None

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
                    role = item["role"]
                    if role in {"system", "developer"}:
                        if not instructions_str and isinstance(item.get("content"), str):
                            instructions_str = item["content"]
                        continue
                    msg = {"role": role, "content": _normalize_input_content(item.get("content"))}
                    if "tool_calls" in item:
                        msg["tool_calls"] = item["tool_calls"]
                    messages.append(msg)
                elif item_type == "message":
                    role = item.get("role", "user")
                    if role in {"system", "developer"}:
                        continue
                    messages.append(
                        {
                            "role": role,
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
                role = msg["role"]
                if role in {"system", "developer"}:
                    if not instructions_str and isinstance(msg.get("content"), str):
                        instructions_str = msg["content"]
                    continue
                m = dict(msg)
                if "content" in m:
                    m["content"] = _normalize_input_content(m["content"])
                messages.append(m)

    # Parse generation config
    temperature = payload.get("temperature")
    top_p = payload.get("top_p")
    max_tokens = (
        payload.get("max_output_tokens")
        or payload.get("max_completion_tokens")
        or payload.get("max_tokens")
    )
    response_format = payload.get("response_format")
    stop = payload.get("stop")
    stop_sequences = [stop] if isinstance(stop, str) else stop if isinstance(stop, list) else None

    # Convert tools to AgentTool (do not swallow UnsupportedToolError / ValueError)
    agent_tools = []
    raw_tools = payload.get("tools")
    if isinstance(raw_tools, list):
        for t in raw_tools:
            agent_tools.append(parse_openai_tool(t))

    # Parse Tool Choice
    raw_choice = payload.get("tool_choice")
    choice = AgentToolChoice(ToolChoiceMode.AUTO)
    if isinstance(raw_choice, str):
        if raw_choice == "none":
            choice = AgentToolChoice(ToolChoiceMode.NONE)
        elif raw_choice == "required":
            choice = AgentToolChoice(ToolChoiceMode.REQUIRED)
    elif isinstance(raw_choice, dict):
        fn = raw_choice.get("function", {})
        if fn.get("name"):
            choice = AgentToolChoice(ToolChoiceMode.SPECIFIC, specific_tool=fn["name"])

    parallel_tool_calls = payload.get("parallel_tool_calls", True)

    return AgentRequest(
        model=model,
        messages=messages,
        instructions=instructions_str,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
        response_format=response_format,
        tools=agent_tools,
        tool_choice=choice,
        parallel_tool_calls=parallel_tool_calls,
        previous_response_id=payload.get("previous_response_id"),
        reasoning=payload.get("reasoning") or payload.get("encrypted_content"),
        metadata={"client_model": client_model, "stream": payload.get("stream") is True}
    )


async def unary_responses_event(events_gen: AsyncIterator[AgentEvent], client_model: str) -> tuple[dict[str, Any], dict[str, str]]:
    response_id = f"resp_{secrets.token_hex(12)}"
    created_at = int(datetime.now(timezone.utc).timestamp())
    output_items: list[dict[str, Any]] = []
    text_content = ""
    usage_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    status = "completed"
    error_msg = None
    resp_headers: dict[str, str] = {}

    async for event in events_gen:
        if event.kind == AgentEventKind.RESPONSE_STARTED:
            response_id = event.data.get("response_id", response_id)
        elif event.kind == AgentEventKind.TEXT_DELTA:
            text_content += event.data.get("text", "")
        elif event.kind == AgentEventKind.TOOL_STARTED:
            if event.tool_kind == ToolKind.WEB_SEARCH:
                ws_id = event.item_id or f"ws_{secrets.token_hex(8)}"
                query = event.data.get("query", "web search")
                sources = event.data.get("sources", [])
                output_items.append({
                    "id": ws_id,
                    "type": "web_search_call",
                    "status": "completed",
                    "query": query,
                    "results": sources,
                })
            else:
                tool_calls = event.data.get("tool_calls", [])
                for tc in tool_calls:
                    call_id = tc.get("id") or tc.get("call_id") or f"call_{secrets.token_hex(8)}"
                    fn = tc.get("function", {})
                    output_items.append({
                        "id": call_id,
                        "type": "function_call",
                        "status": "completed",
                        "call_id": call_id,
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments", {}), ensure_ascii=False)
                    })
        elif event.kind == AgentEventKind.COMPLETED:
            if "id" in event.data and event.data["id"]:
                response_id = event.data["id"]
            if "headers" in event.data and isinstance(event.data["headers"], dict):
                resp_headers = {k: v for k, v in event.data["headers"].items() if k.lower() not in {"content-length", "content-encoding", "transfer-encoding"}}
            if "usage" in event.data:
                u = event.data["usage"]
                usage_dict = {
                    "input_tokens": u.get("prompt_tokens", 0),
                    "output_tokens": u.get("completion_tokens", 0),
                    "total_tokens": u.get("total_tokens", 0)
                }
            if "stop_reason" in event.data:
                sr = event.data["stop_reason"]
                sr_val = sr.value if hasattr(sr, "value") else str(sr)
                if sr_val in {"pause_turn", "max_tokens"}:
                    status = "incomplete"
                elif sr_val in {"prompt_blocked", "refusal", "prohibited_content", "other", "error"}:
                    status = "failed"
        elif event.kind == AgentEventKind.ERROR:
            status = "failed"
            error_msg = event.data.get("message")
            if event.data.get("status_code") == 400:
                raise ValueError(error_msg)

    if text_content:
        output_items.insert(0, {
            "id": f"msg_{secrets.token_hex(12)}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text_content}]
        })

    res_body = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "completed_at": created_at,
        "status": status,
        "model": client_model,
        "output": output_items,
        "usage": usage_dict
    }
    if error_msg:
        res_body["error"] = {"message": error_msg}
    return res_body, resp_headers


async def stream_responses_events(events_gen: AsyncIterator[AgentEvent], client_model: str) -> AsyncIterator[bytes]:
    response_id = f"resp_{secrets.token_hex(12)}"
    created_at = int(datetime.now(timezone.utc).timestamp())
    final_output: list[dict[str, Any]] = []

    msg_id = f"msg_{secrets.token_hex(12)}"
    text_started = False
    accumulated_text = ""
    next_output_index = 0
    text_output_index = 0
    tool_calls: dict[int, dict[str, Any]] = {}
    usage_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    started_emitted = False

    def make_in_progress_obj(rid: str) -> dict[str, Any]:
        return {
            "id": rid,
            "object": "response",
            "created_at": created_at,
            "completed_at": None,
            "status": "in_progress",
            "model": client_model,
            "output": [],
            "error": None,
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "usage": None,
        }

    completed_status = "completed"
    async for event in events_gen:
        if event.kind == AgentEventKind.RESPONSE_STARTED:
            response_id = event.data.get("response_id", response_id)
            if not started_emitted:
                resp_obj = make_in_progress_obj(response_id)
                yield _sse("response.created", {"type": "response.created", "response": resp_obj})
                yield _sse("response.in_progress", {"type": "response.in_progress", "response": resp_obj})
                started_emitted = True
            continue

        if not started_emitted:
            resp_obj = make_in_progress_obj(response_id)
            yield _sse("response.created", {"type": "response.created", "response": resp_obj})
            yield _sse("response.in_progress", {"type": "response.in_progress", "response": resp_obj})
            started_emitted = True

        if event.kind == AgentEventKind.TEXT_DELTA:
            text_delta = event.data.get("text", "")
            if text_delta:
                if not text_started:
                    text_started = True
                    text_output_index = next_output_index
                    next_output_index += 1
                    yield _sse(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": text_output_index,
                            "item": {
                                "id": msg_id,
                                "type": "message",
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [],
                            },
                        },
                    )
                accumulated_text += text_delta
                yield _sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": msg_id,
                        "output_index": text_output_index,
                        "content_index": 0,
                        "delta": text_delta,
                        "logprobs": [],
                    },
                )
        elif event.kind == AgentEventKind.TOOL_STARTED and event.tool_kind == ToolKind.WEB_SEARCH:
            ws_id = event.item_id or f"ws_{secrets.token_hex(8)}"
            ws_idx = next_output_index
            next_output_index += 1
            query = event.data.get("query", "web search")
            sources = event.data.get("sources", [])
            yield _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": ws_idx,
                    "item": {
                        "id": ws_id,
                        "type": "web_search_call",
                        "status": "in_progress",
                        "query": query,
                    },
                },
            )
            yield _sse("response.web_search_call.in_progress", {"type": "response.web_search_call.in_progress", "output_index": ws_idx, "call_id": ws_id})
            yield _sse("response.web_search_call.searching", {"type": "response.web_search_call.searching", "output_index": ws_idx, "call_id": ws_id})
            yield _sse("response.web_search_call.completed", {"type": "response.web_search_call.completed", "output_index": ws_idx, "call_id": ws_id, "results": sources})
            ws_item = {
                "id": ws_id,
                "type": "web_search_call",
                "status": "completed",
                "query": query,
                "results": sources,
            }
            yield _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": ws_idx,
                    "item": ws_item,
                },
            )
            final_output.append(ws_item)
        elif event.kind == AgentEventKind.TOOL_STARTED:
            tcs = event.data.get("tool_calls", [])
            for tc in tcs:
                idx = len(tool_calls)
                call_id = tc.get("id") or tc.get("call_id") or f"call_{secrets.token_hex(8)}"
                fn = tc.get("function", {})
                out_idx = next_output_index
                next_output_index += 1
                tool_calls[idx] = {
                    "id": call_id,
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "{}") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments", {}), ensure_ascii=False),
                    "output_index": out_idx
                }
                yield _sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": out_idx,
                        "item": {
                            "id": call_id,
                            "type": "function_call",
                            "status": "in_progress",
                            "call_id": call_id,
                            "name": fn.get("name", ""),
                            "arguments": "",
                        },
                    },
                )
                yield _sse(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": call_id,
                        "output_index": out_idx,
                        "call_id": call_id,
                        "delta": tool_calls[idx]["arguments"],
                    },
                )
        elif event.kind == AgentEventKind.COMPLETED:
            if "id" in event.data and event.data["id"]:
                response_id = event.data["id"]
            if "usage" in event.data:
                u = event.data["usage"]
                usage_dict = {
                    "input_tokens": u.get("prompt_tokens", 0),
                    "output_tokens": u.get("completion_tokens", 0),
                    "total_tokens": u.get("total_tokens", 0)
                }
            if "stop_reason" in event.data:
                sr = event.data["stop_reason"]
                sr_val = sr.value if hasattr(sr, "value") else str(sr)
                if sr_val in {"pause_turn", "max_tokens"}:
                    completed_status = "incomplete"
                elif sr_val in {"prompt_blocked", "refusal", "prohibited_content", "other", "error"}:
                    completed_status = "failed"
        elif event.kind == AgentEventKind.ERROR:
            err_msg = event.data.get("message", "Error")
            yield _sse("error", {"type": "error", "error": {"message": err_msg}})
            return

    if not started_emitted:
        resp_obj = make_in_progress_obj(response_id)
        yield _sse("response.created", {"type": "response.created", "response": resp_obj})
        yield _sse("response.in_progress", {"type": "response.in_progress", "response": resp_obj})

    if text_started:
        yield _sse(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": msg_id,
                "output_index": text_output_index,
                "content_index": 0,
                "text": accumulated_text,
                "logprobs": [],
            },
        )
        msg_item = {
            "id": msg_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": accumulated_text}],
        }
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": text_output_index,
                "item": msg_item,
            },
        )
        final_output.append(msg_item)

    for _, entry in sorted(tool_calls.items()):
        yield _sse(
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "item_id": entry["id"],
                "output_index": entry["output_index"],
                "call_id": entry["id"],
                "arguments": entry["arguments"],
            },
        )
        fn_item = {
            "id": entry["id"],
            "type": "function_call",
            "status": "completed",
            "call_id": entry["id"],
            "name": entry["name"],
            "arguments": entry["arguments"],
        }
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": entry["output_index"],
                "item": fn_item,
            },
        )
        final_output.append(fn_item)

    completed_at = int(datetime.now(timezone.utc).timestamp())
    yield _sse(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "completed_at": completed_at,
                "status": completed_status,
                "model": client_model,
                "output": final_output,
                "usage": usage_dict,
            },
        },
    )
    yield b"data: [DONE]\n\n"
