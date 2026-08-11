from __future__ import annotations

import asyncio
import json
import re
import secrets
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx


_SAFE_GEMINI_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VERTEX_SCHEMA_FIELDS = {
    "type",
    "format",
    "title",
    "description",
    "nullable",
    "default",
    "items",
    "minItems",
    "maxItems",
    "enum",
    "properties",
    "propertyOrdering",
    "required",
    "minProperties",
    "maxProperties",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "pattern",
    "example",
    "anyOf",
    "ref",
    "defs",
}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _sse(event: str, value: Mapping[str, Any]) -> bytes:
    return b"event: " + event.encode() + b"\n" + b"data: " + _json_bytes(value) + b"\n\n"


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if not isinstance(system, list):
        return ""
    return "\n".join(
        block["text"]
        for block in system
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def _vertex_function_schema(schema: Any) -> dict[str, Any]:
    """Convert JSON Schema keywords to Vertex's supported OpenAPI subset."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    converted: dict[str, Any] = {}
    for raw_key, value in schema.items():
        key = raw_key
        if raw_key == "$defs":
            key = "defs"
        elif raw_key == "$ref":
            key = "ref"
        elif raw_key == "oneOf":
            key = "anyOf"
        elif raw_key == "const":
            if "enum" not in schema:
                converted["enum"] = [value]
            continue

        if key not in _VERTEX_SCHEMA_FIELDS:
            continue
        if key in {"properties", "defs"}:
            if isinstance(value, dict):
                converted[key] = {
                    str(name): _vertex_function_schema(child)
                    for name, child in value.items()
                }
        elif key == "items":
            converted[key] = _vertex_function_schema(value)
        elif key == "anyOf":
            if isinstance(value, list):
                converted[key] = [_vertex_function_schema(item) for item in value]
        elif key == "ref" and isinstance(value, str):
            converted[key] = value.replace("#/$defs/", "#/defs/")
        else:
            converted[key] = value

    if not converted:
        return {"type": "object", "properties": {}}
    return converted


def _openai_user_block(block: Mapping[str, Any]) -> dict[str, Any] | None:
    block_type = block.get("type")
    if block_type == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block_type != "image" or not isinstance(block.get("source"), dict):
        return None

    source = block["source"]
    if source.get("type") == "base64":
        media_type = source.get("media_type", "application/octet-stream")
        data = source.get("data", "")
        if isinstance(data, str):
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            }
    if source.get("type") == "url" and isinstance(source.get("url"), str):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif block is not None:
                parts.append(json.dumps(block, ensure_ascii=False, separators=(",", ":")))
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def _content_value(blocks: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if not blocks:
        return ""
    if all(block.get("type") == "text" for block in blocks):
        return "".join(str(block.get("text", "")) for block in blocks)
    return blocks


def _user_messages(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if not isinstance(content, list):
        return [{"role": "user", "content": str(content)}]

    result: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush_pending() -> None:
        if pending:
            result.append({"role": "user", "content": _content_value(pending)})
            pending.clear()

    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            flush_pending()
            tool_call_id = block.get("tool_use_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            tool_content = _tool_result_text(block.get("content"))
            if block.get("is_error") is True:
                tool_content = f"Tool error: {tool_content}"
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_content,
                }
            )
            continue
        converted = _openai_user_block(block)
        if converted is not None:
            pending.append(converted)

    flush_pending()
    return result or [{"role": "user", "content": ""}]


def _assistant_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    if not isinstance(content, list):
        return {"role": "assistant", "content": str(content)}

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block_type == "tool_use":
            tool_id = block.get("id")
            name = block.get("name")
            if not isinstance(tool_id, str) or not isinstance(name, str):
                continue
            tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "extra_content": {
                        "google": {
                            "thought_signature": "skip_thought_signature_validator",
                        }
                    },
                    "thought_signature": "skip_thought_signature_validator",
                    "thoughtSignature": "skip_thought_signature_validator",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            block.get("input", {}),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "thought_signature": "skip_thought_signature_validator",
                        "thoughtSignature": "skip_thought_signature_validator",
                    },
                }
            )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _convert_messages(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    system = _system_text(payload.get("system"))
    if system:
        result.append({"role": "system", "content": system})

    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("messages 必须是数组")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("messages 中的每一项都必须是对象")
        role = message.get("role")
        if role == "user":
            result.extend(_user_messages(message.get("content", "")))
        elif role in {"assistant", "model"}:
            result.append(_assistant_message(message.get("content", "")))
        elif role in {"system", "developer"}:
            text = _system_text(message.get("content", ""))
            if text:
                result.append({"role": "system", "content": text})
        elif role == "tool":
            tool_call_id = message.get("tool_call_id") or message.get("tool_use_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ValueError("tool 消息缺少 tool_call_id")
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": _tool_result_text(message.get("content")),
                }
            )
        else:
            raise ValueError(
                "messages 仅支持 user、assistant、system、developer、tool 和 model 角色"
            )
    return result


def _convert_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            continue
        function: dict[str, Any] = {
            "name": tool["name"],
            "parameters": _vertex_function_schema(tool.get("input_schema")),
        }
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        converted.append({"type": "function", "function": function})
    return converted


def _convert_tool_choice(value: Any) -> tuple[Any | None, bool | None]:
    if not isinstance(value, dict):
        return None, None
    choice_type = value.get("type")
    if choice_type == "auto":
        choice: Any = "auto"
    elif choice_type == "any":
        choice = "required"
    elif choice_type == "none":
        choice = "none"
    elif choice_type == "tool" and isinstance(value.get("name"), str):
        choice = {"type": "function", "function": {"name": value["name"]}}
    else:
        choice = None
    parallel = False if value.get("disable_parallel_tool_use") is True else None
    return choice, parallel


def anthropic_to_openai(payload: Mapping[str, Any], gemini_model: str) -> dict[str, Any]:
    model = gemini_model.strip()
    if not model:
        raise ValueError(
            "Gemini 后端尚未配置；请设置 VERTEX_ANTHROPIC_GEMINI_MODEL"
        )
    if "/" not in model:
        model = f"google/{model}"

    result: dict[str, Any] = {
        "model": model,
        "messages": _convert_messages(payload),
    }
    if "max_tokens" in payload:
        result["max_tokens"] = payload["max_tokens"]
    for name in ("temperature", "top_p"):
        if name in payload:
            result[name] = payload[name]
    if "stop_sequences" in payload:
        result["stop"] = payload["stop_sequences"]

    stream = payload.get("stream") is True
    result["stream"] = stream
    if stream:
        result["stream_options"] = {"include_usage": True}

    tools = _convert_tools(payload.get("tools"))
    if tools:
        result["tools"] = tools
        tool_choice, parallel = _convert_tool_choice(payload.get("tool_choice"))
        if tool_choice is not None:
            result["tool_choice"] = tool_choice
        if parallel is not None:
            result["parallel_tool_calls"] = parallel
    return result


def _configured_gemini_model(config: Any) -> str:
    configured = str(getattr(config, "anthropic_gemini_model", "")).strip()
    if not configured:
        for model in getattr(config, "models", ()):
            candidate = str(model).strip()
            if candidate.removeprefix("google/").startswith("gemini-"):
                configured = candidate
                break
    if not configured:
        raise ValueError(
            "Gemini 后端尚未配置；请设置 VERTEX_ANTHROPIC_GEMINI_MODEL，"
            "或在 VERTEX_MODELS 中至少配置一个 Gemini 模型"
        )
    return configured


def _native_model_id(gemini_model: str) -> str:
    model = gemini_model.removeprefix("google/")
    if not _SAFE_GEMINI_MODEL.fullmatch(model):
        raise ValueError("VERTEX_ANTHROPIC_GEMINI_MODEL 不是有效的 Gemini 模型 ID")
    return model


def _native_user_parts(content: Any, tool_names: Mapping[str, str]) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            parts.append({"text": block["text"]})
        elif block_type == "image" and isinstance(block.get("source"), dict):
            source = block["source"]
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": source.get("media_type", "application/octet-stream"),
                            "data": source["data"],
                        }
                    }
                )
            elif source.get("type") == "url" and isinstance(source.get("url"), str):
                parts.append({"fileData": {"fileUri": source["url"]}})
        elif block_type == "tool_result":
            tool_id = block.get("tool_use_id")
            name = tool_names.get(tool_id, "tool") if isinstance(tool_id, str) else "tool"
            parts.append(
                {
                    "functionResponse": {
                        "name": name,
                        "response": {
                            "result": _tool_result_text(block.get("content")),
                            "is_error": block.get("is_error") is True,
                        },
                    }
                }
            )
    return parts or [{"text": ""}]


def _native_assistant_parts(
    content: Any,
    tool_names: dict[str, str],
) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append({"text": block["text"]})
        elif block.get("type") == "tool_use" and isinstance(block.get("name"), str):
            tool_id = block.get("id")
            if isinstance(tool_id, str):
                tool_names[tool_id] = block["name"]
            parts.append(
                {
                    "functionCall": {
                        "name": block["name"],
                        "args": block.get("input", {}),
                        "thought_signature": "skip_thought_signature_validator",
                        "thoughtSignature": "skip_thought_signature_validator",
                    },
                    "thought_signature": "skip_thought_signature_validator",
                    "thoughtSignature": "skip_thought_signature_validator",
                }
            )
    return parts or [{"text": ""}]


def anthropic_to_gemini_count_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    request: dict[str, Any] = {"contents": []}
    system = _system_text(payload.get("system"))
    system_parts = [system] if system else []

    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("messages 必须是数组")
    tool_names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("messages 中的每一项都必须是对象")
        role = message.get("role")
        if role in {"assistant", "model"}:
            request["contents"].append(
                {
                    "role": "model",
                    "parts": _native_assistant_parts(message.get("content", ""), tool_names),
                }
            )
        elif role == "user":
            request["contents"].append(
                {
                    "role": "user",
                    "parts": _native_user_parts(message.get("content", ""), tool_names),
                }
            )
        elif role in {"system", "developer"}:
            text = _system_text(message.get("content", ""))
            if text:
                system_parts.append(text)
        elif role == "tool":
            tool_id = message.get("tool_call_id") or message.get("tool_use_id")
            name = tool_names.get(tool_id, "tool") if isinstance(tool_id, str) else "tool"
            request["contents"].append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": name,
                                "response": {
                                    "result": _tool_result_text(message.get("content"))
                                },
                            }
                        }
                    ],
                }
            )
        else:
            raise ValueError(
                "messages 仅支持 user、assistant、system、developer、tool 和 model 角色"
            )

    if system_parts:
        request["systemInstruction"] = {
            "parts": [{"text": text} for text in system_parts]
        }

    functions = []
    tools = payload.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                continue
            declaration: dict[str, Any] = {
                "name": tool["name"],
                "parameters": _vertex_function_schema(tool.get("input_schema")),
            }
            if isinstance(tool.get("description"), str):
                declaration["description"] = tool["description"]
            functions.append(declaration)
    if functions:
        request["tools"] = [{"functionDeclarations": functions}]
    return request


def _anthropic_stop_reason(finish_reason: Any) -> str:
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "stop":
        return "end_turn"
    if finish_reason in {"content_filter", "safety"}:
        return "refusal"
    return "end_turn"


def _tool_input(arguments: Any) -> Any:
    if not isinstance(arguments, str) or not arguments:
        return {}
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {"value": arguments}


def _response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def openai_to_anthropic(payload: Mapping[str, Any], client_model: str) -> dict[str, Any]:
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice, dict):
        choice = {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}

    content: list[dict[str, Any]] = []
    text = _response_text(message.get("content"))
    if text:
        content.append({"type": "text", "text": text})

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                continue
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id") or f"toolu_{secrets.token_hex(12)}",
                    "name": function["name"],
                    "input": _tool_input(function.get("arguments")),
                }
            )

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "id": f"msg_{secrets.token_hex(12)}",
        "type": "message",
        "role": "assistant",
        "model": client_model,
        "content": content,
        "stop_reason": _anthropic_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
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


async def _stream_openai_as_anthropic(
    response: httpx.Response,
    client_model: str,
) -> AsyncIterator[bytes]:
    message_id = f"msg_{secrets.token_hex(12)}"
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": client_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    text_started = False
    next_block_index = 0
    finish_reason: Any = None
    output_tokens = 0
    tool_calls: dict[int, dict[str, str]] = {}

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

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            output_tokens = usage.get("completion_tokens", output_tokens)

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue

        text_delta = delta.get("content")
        if isinstance(text_delta, str) and text_delta:
            if not text_started:
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": next_block_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                text_started = True
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": next_block_index,
                    "delta": {"type": "text_delta", "text": text_delta},
                },
            )

        streamed_tools = delta.get("tool_calls")
        if isinstance(streamed_tools, list):
            for tool_call in streamed_tools:
                if not isinstance(tool_call, dict):
                    continue
                index = tool_call.get("index", 0)
                if not isinstance(index, int):
                    index = 0
                collected = tool_calls.setdefault(
                    index,
                    {"id": "", "name": "", "arguments": ""},
                )
                if isinstance(tool_call.get("id"), str):
                    collected["id"] = tool_call["id"]
                function = tool_call.get("function")
                if isinstance(function, dict):
                    if isinstance(function.get("name"), str):
                        collected["name"] += function["name"]
                    if isinstance(function.get("arguments"), str):
                        collected["arguments"] += function["arguments"]

    if text_started:
        yield _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": next_block_index},
        )
        next_block_index += 1

    for _, tool_call in sorted(tool_calls.items()):
        yield _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": next_block_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_call["id"] or f"toolu_{secrets.token_hex(12)}",
                    "name": tool_call["name"],
                    "input": {},
                },
            },
        )
        yield _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": next_block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": tool_call["arguments"] or "{}",
                },
            },
        )
        yield _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": next_block_index},
        )
        next_block_index += 1

    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": _anthropic_stop_reason(finish_reason),
                "stop_sequence": None,
            },
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


class GeminiAnthropicAdapter:
    """Converts Claude Code's Messages API protocol to Vertex Gemini OpenAI API."""

    def __init__(self) -> None:
        self.client_model = "claude-code-gemini"
        self.stream = False

    def prepare(self, body: bytes, config: Any) -> tuple[str, bytes]:
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求体必须是有效的 JSON 对象") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        client_model = payload.get("model")
        if not isinstance(client_model, str) or not client_model.strip():
            raise ValueError("缺少有效的 model 字段")

        self.client_model = client_model.strip()
        self.stream = payload.get("stream") is True
        converted = anthropic_to_openai(payload, _configured_gemini_model(config))
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
            async for chunk in _stream_openai_as_anthropic(response, self.client_model):
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
        yield _json_bytes(openai_to_anthropic(payload, self.client_model))


class GeminiCountTokensAdapter:
    """Converts Anthropic count_tokens requests to Gemini's native countTokens API."""

    def prepare(self, body: bytes, config: Any) -> tuple[str, bytes]:
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求体必须是有效的 JSON 对象") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        client_model = payload.get("model")
        if not isinstance(client_model, str) or not client_model.strip():
            raise ValueError("缺少有效的 model 字段")

        model = _native_model_id(_configured_gemini_model(config))
        url = (
            f"{config.native_base_url('v1')}/publishers/google/models/"
            f"{model}:countTokens"
        )
        return url, _json_bytes(anthropic_to_gemini_count_request(payload))

    def response_headers(
        self,
        response: httpx.Response,
        headers: Mapping[str, str],
    ) -> dict[str, str]:
        result = dict(headers)
        if response.status_code < 400:
            for name in ("content-length", "content-encoding", "transfer-encoding"):
                result.pop(name, None)
            result["content-type"] = "application/json; charset=utf-8"
        return result

    async def transform(self, response: httpx.Response) -> AsyncIterator[bytes]:
        if response.status_code >= 400:
            async for chunk in response.aiter_raw():
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
        total_tokens = payload.get("totalTokens")
        if not isinstance(total_tokens, int):
            total_tokens = payload.get("total_tokens", 0)
        yield _json_bytes({"input_tokens": total_tokens})
