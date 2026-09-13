from __future__ import annotations

import json
import logging
import secrets
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("vertex_proxy.agent_runtime")

from vertex_proxy.agent_ir import (
    AgentEvent,
    AgentEventKind,
    AgentRequest,
    AgentResponse,
    AgentStopReason,
    AgentTool,
    BackendKind,
    ToolChoiceMode,
    ToolExecution,
    ToolKind,
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
from vertex_proxy.vertex_native import GeminiNativeCodec, VertexNativeClient

@dataclass
class ExecutionPlan:
    backend: BackendKind
    provider_tools: list[AgentTool] = field(default_factory=list)
    proxy_tools: list[AgentTool] = field(default_factory=list)
    client_tools: list[AgentTool] = field(default_factory=list)
    api_version: str = "v1"

class CallRegistry:
    def __init__(self) -> None:
        self._calls: Dict[str, Dict[str, Any]] = {}

    def register_call(
        self,
        call_id: str,
        client_name: str,
        internal_name: str,
        tool_kind: ToolKind,
        execution: ToolExecution,
    ) -> None:
        self._calls[call_id] = {
            "client_name": client_name,
            "internal_name": internal_name,
            "tool_kind": tool_kind,
            "execution": execution,
        }

    def get_call(self, call_id: str) -> Optional[Dict[str, Any]]:
        return self._calls.get(call_id)

class AgentRuntime:
    def __init__(self, state_store: InMemoryAgentStateStore) -> None:
        self.state_store = state_store
        self.call_registry = CallRegistry()

    def plan_execution(self, request: AgentRequest, tool_mode: str = "best_effort") -> ExecutionPlan:
        plan = ExecutionPlan(backend=BackendKind.VERTEX_OPENAI)

        for tool in request.tools:
            tool_name = tool.name or "tool"
            if tool.execution == ToolExecution.PROVIDER:
                if tool_mode == "strict" and any(t.execution == ToolExecution.CLIENT for t in request.tools):
                    # In strict mode with mixed tools, map provider tool to internal pseudo-function
                    internal_name = f"__proxy_{tool_name}"
                    plan.proxy_tools.append(tool)
                    plan.backend = BackendKind.VERTEX_NATIVE
                    self.call_registry.register_call(tool_name, client_name=tool_name, internal_name=internal_name, tool_kind=tool.kind, execution=ToolExecution.PROXY)
                else:
                    plan.provider_tools.append(tool)
                    if tool.kind in {ToolKind.WEB_SEARCH, ToolKind.WEB_FETCH, ToolKind.CODE_EXECUTION}:
                        plan.backend = BackendKind.VERTEX_NATIVE
                    self.call_registry.register_call(tool_name, client_name=tool_name, internal_name=tool_name, tool_kind=tool.kind, execution=ToolExecution.PROVIDER)
            elif tool.execution == ToolExecution.PROXY:
                plan.proxy_tools.append(tool)
                self.call_registry.register_call(tool_name, client_name=tool_name, internal_name=tool_name, tool_kind=tool.kind, execution=ToolExecution.PROXY)
            else:
                plan.client_tools.append(tool)
                self.call_registry.register_call(tool_name, client_name=tool_name, internal_name=tool_name, tool_kind=tool.kind, execution=ToolExecution.CLIENT)

        if plan.backend == BackendKind.VERTEX_NATIVE:
            plan.api_version = "v1beta1"
        else:
            plan.api_version = "v1"

        return plan

    async def run(
        self,
        request: AgentRequest,
        config: Any,
        http_client: Any,
        token_provider: Any,
    ) -> AsyncIterator[AgentEvent]:
        # Generate target Response ID
        response_id = f"resp_{secrets.token_hex(12)}"
        req_id = request.metadata.get("request_id") or response_id
        yield AgentEvent(kind=AgentEventKind.RESPONSE_STARTED, data={"response_id": response_id})

        tool_mode = getattr(config, "agent_tool_mode", "best_effort")
        plan = self.plan_execution(request, tool_mode=tool_mode)
        previous_response_id = request.previous_response_id

        # 1. Recover Session Snapshot State (O(1) Snapshot Retrieve)
        history_messages: list[Dict[str, Any]] = []
        history_contents: list[Dict[str, Any]] = []
        prev_state: Optional[ResponseState] = None

        if previous_response_id:
            prev_state = await self.state_store.get_response_state(previous_response_id)
            if not prev_state:
                yield AgentEvent(
                    kind=AgentEventKind.ERROR,
                    data={"message": f"Previous response ID '{previous_response_id}' is invalid or expired.", "status_code": 400}
                )
                yield AgentEvent(kind=AgentEventKind.COMPLETED, data={"id": response_id, "stop_reason": AgentStopReason.ERROR, "usage": {}})
                return

            norm_req_model = request.model.removeprefix("google/")
            norm_state_model = prev_state.model.removeprefix("google/")
            if norm_req_model != norm_state_model:
                yield AgentEvent(
                    kind=AgentEventKind.ERROR,
                    data={"message": f"Incompatible model '{request.model}' for previous response ID '{previous_response_id}' (created with '{prev_state.model}')", "status_code": 400}
                )
                yield AgentEvent(kind=AgentEventKind.COMPLETED, data={"id": response_id, "stop_reason": AgentStopReason.ERROR, "usage": {}})
                return

            # Validate pending tool conflict in strict mode
            if tool_mode == "strict" and previous_response_id:
                has_tool_msg = any(m.get("role") == "tool" for m in request.messages)
                has_user_msg = any(m.get("role") == "user" and bool(m.get("content")) for m in request.messages)
                if has_tool_msg and has_user_msg:
                    yield AgentEvent(
                        kind=AgentEventKind.ERROR,
                        data={"message": "Cannot mix extra text in tool_result message when pending server tool is active.", "status_code": 400}
                    )
                    yield AgentEvent(kind=AgentEventKind.COMPLETED, data={"id": response_id, "stop_reason": AgentStopReason.ERROR, "usage": {}})
                    return

            if prev_state.backend == BackendKind.VERTEX_OPENAI and isinstance(prev_state.provider_snapshot, VertexOpenAIState):
                history_messages = list(prev_state.provider_snapshot.messages)
            elif prev_state.backend == BackendKind.VERTEX_NATIVE and isinstance(prev_state.provider_snapshot, VertexNativeState):
                history_contents = list(prev_state.provider_snapshot.contents)

        # 2. Execute via designated backend
        if plan.backend == BackendKind.VERTEX_OPENAI:
            merged_messages = []

            # Add system instruction for current request if present
            if request.instructions and isinstance(request.instructions, str):
                merged_messages.append({"role": "system", "content": request.instructions})

            # Append historical conversation messages (user/assistant/tool)
            history_conv_msgs = [m for m in history_messages if m.get("role") not in {"system", "developer"}]
            merged_messages.extend(history_conv_msgs)
            merged_messages.extend(request.messages)

            # Snapshot saved in ResponseState should be conversation messages only
            snapshot_messages = list(history_conv_msgs) + list(request.messages)

            try:
                token = await token_provider.token()
            except Exception as exc:
                yield AgentEvent(
                    kind=AgentEventKind.ERROR,
                    data={"message": f"Failed to acquire ADC token: {exc}", "status_code": 503}
                )
                yield AgentEvent(kind=AgentEventKind.COMPLETED, data={"id": response_id, "stop_reason": AgentStopReason.ERROR, "usage": {}})
                return

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            url = f"{config.openai_base_url}/chat/completions"

            is_stream = request.metadata.get("stream") is True
            payload = {
                "model": f"google/{request.model}" if "/" not in request.model else request.model,
                "messages": merged_messages,
                "stream": is_stream,
            }
            if request.temperature is not None:
                payload["temperature"] = request.temperature
            if request.top_p is not None:
                payload["top_p"] = request.top_p
            if request.max_tokens is not None:
                payload["max_tokens"] = request.max_tokens
            if request.stop_sequences:
                payload["stop"] = request.stop_sequences
            if request.response_format:
                payload["response_format"] = request.response_format

            if request.tools:
                payload["tools"] = [t.raw for t in plan.client_tools if t.raw]
                if request.tool_choice:
                    mode = request.tool_choice.mode
                    if mode == ToolChoiceMode.AUTO:
                        payload["tool_choice"] = "auto"
                    elif mode == ToolChoiceMode.NONE:
                        payload["tool_choice"] = "none"
                    elif mode == ToolChoiceMode.REQUIRED:
                        payload["tool_choice"] = "required"
                    elif mode == ToolChoiceMode.SPECIFIC and request.tool_choice.specific_tool:
                        payload["tool_choice"] = {"type": "function", "function": {"name": request.tool_choice.specific_tool}}
                if request.parallel_tool_calls is False:
                    payload["parallel_tool_calls"] = False

            try:
                req = http_client.build_request("POST", url, headers=headers, json=payload)
                response = await http_client.send(req, stream=is_stream)
                if response.status_code == 401:
                    await response.aclose()
                    token = await token_provider.token(force_refresh=True)
                    headers["Authorization"] = f"Bearer {token}"
                    req = http_client.build_request("POST", url, headers=headers, json=payload)
                    response = await http_client.send(req, stream=is_stream)

                if response.status_code >= 400:
                    body_text = await response.aread() if is_stream else response.text
                    yield AgentEvent(kind=AgentEventKind.ERROR, data={"message": body_text.decode("utf-8", errors="replace") if isinstance(body_text, bytes) else str(body_text), "status_code": response.status_code})
                    yield AgentEvent(kind=AgentEventKind.COMPLETED, data={"id": response_id, "stop_reason": AgentStopReason.ERROR, "usage": {}})
                    return

                resp_headers = dict(response.headers)

                if is_stream:
                    text_content = ""
                    tool_calls_dict: Dict[int, Dict[str, Any]] = {}
                    usage_info = {}

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].lstrip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except Exception:
                            continue

                        if "usage" in chunk and isinstance(chunk["usage"], dict):
                            usage_info = chunk["usage"]

                        choices = chunk.get("choices", [])
                        if choices and isinstance(choices[0], dict):
                            delta = choices[0].get("delta", {})
                            content_delta = delta.get("content")
                            if content_delta:
                                text_content += content_delta
                                yield AgentEvent(kind=AgentEventKind.TEXT_DELTA, data={"text": content_delta})

                            tcs = delta.get("tool_calls")
                            if isinstance(tcs, list):
                                for tc in tcs:
                                    idx = tc.get("index", 0)
                                    if idx not in tool_calls_dict:
                                        tool_calls_dict[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                                    entry = tool_calls_dict[idx]
                                    if tc.get("id"):
                                        entry["id"] = tc["id"]
                                    fn = tc.get("function", {})
                                    if fn.get("name"):
                                        entry["function"]["name"] += fn["name"]
                                    if fn.get("arguments"):
                                        entry["function"]["arguments"] += fn["arguments"]

                                yield AgentEvent(kind=AgentEventKind.TOOL_STARTED, tool_kind=ToolKind.FUNCTION, data={"tool_calls": list(tool_calls_dict.values())})

                    await response.aclose()

                    assistant_msg: Dict[str, Any] = {"role": "assistant"}
                    if text_content:
                        assistant_msg["content"] = text_content
                    if tool_calls_dict:
                        assistant_msg["tool_calls"] = [
                            {
                                "id": v["id"] or f"call_{secrets.token_hex(8)}",
                                "type": "function",
                                "function": v["function"]
                            }
                            for v in tool_calls_dict.values()
                        ]
                    snapshot_messages.append(assistant_msg)

                    new_state = ResponseState(
                        response_id=response_id,
                        previous_response_id=previous_response_id,
                        model=request.model,
                        backend=BackendKind.VERTEX_OPENAI,
                        provider_snapshot=VertexOpenAIState(messages=snapshot_messages),
                        previous_instructions=request.instructions,
                        reasoning=request.reasoning,
                        tool_calls={}
                    )
                    await self.state_store.save_response_state(new_state)

                    usage = {
                        "prompt_tokens": usage_info.get("prompt_tokens", 0),
                        "completion_tokens": usage_info.get("completion_tokens", 0),
                        "total_tokens": usage_info.get("total_tokens", 0),
                    }
                    yield AgentEvent(kind=AgentEventKind.COMPLETED, data={
                        "id": response_id,
                        "headers": resp_headers,
                        "stop_reason": AgentStopReason.END_TURN if not tool_calls_dict else AgentStopReason.TOOL_USE,
                        "usage": usage
                    })

                else:
                    resp_json = response.json()
                    orig_id = resp_json.get("id")
                    if orig_id:
                        if orig_id.startswith("chatcmpl-"):
                            response_id = "resp_" + orig_id[len("chatcmpl-"):]
                        else:
                            response_id = orig_id

                    choices = resp_json.get("choices", [])
                    tool_calls = None
                    if choices:
                        msg = choices[0].get("message", {})
                        content = msg.get("content")
                        if content:
                            yield AgentEvent(kind=AgentEventKind.TEXT_STARTED)
                            yield AgentEvent(kind=AgentEventKind.TEXT_DELTA, data={"text": content})
                            yield AgentEvent(kind=AgentEventKind.TEXT_COMPLETED)

                        tool_calls = msg.get("tool_calls")
                        if tool_calls:
                            yield AgentEvent(kind=AgentEventKind.TOOL_STARTED, tool_kind=ToolKind.FUNCTION, data={"tool_calls": tool_calls})

                        snapshot_messages.append(msg)

                    new_state = ResponseState(
                        response_id=response_id,
                        previous_response_id=previous_response_id,
                        model=request.model,
                        backend=BackendKind.VERTEX_OPENAI,
                        provider_snapshot=VertexOpenAIState(messages=snapshot_messages),
                        previous_instructions=request.instructions,
                        reasoning=request.reasoning,
                        tool_calls={}
                    )
                    await self.state_store.save_response_state(new_state)

                    usage_meta = resp_json.get("usage", {})
                    usage = {
                        "prompt_tokens": usage_meta.get("prompt_tokens", 0),
                        "completion_tokens": usage_meta.get("completion_tokens", 0),
                        "total_tokens": usage_meta.get("total_tokens", 0),
                    }

                    yield AgentEvent(kind=AgentEventKind.COMPLETED, data={
                        "id": response_id,
                        "headers": resp_headers,
                        "stop_reason": AgentStopReason.END_TURN if not tool_calls else AgentStopReason.TOOL_USE,
                        "usage": usage
                    })
            except Exception as e:
                yield AgentEvent(kind=AgentEventKind.ERROR, data={"message": str(e)})
                yield AgentEvent(kind=AgentEventKind.COMPLETED, data={"id": response_id, "stop_reason": AgentStopReason.ERROR, "usage": {}})

        elif plan.backend == BackendKind.VERTEX_NATIVE:
            native_client = VertexNativeClient(http_client, config, token_provider)

            saved_tool_calls: Dict[str, ToolCallState] = {}
            if prev_state and prev_state.tool_calls:
                saved_tool_calls.update(prev_state.tool_calls)

            # Build comprehensive tool call ID to tool name mapping from history and declared tools
            known_tool_calls: Dict[str, str] = {}
            for k, tc_state in saved_tool_calls.items():
                known_tool_calls[k] = tc_state.name
                if tc_state.openai_call_id:
                    known_tool_calls[tc_state.openai_call_id] = tc_state.name
                if tc_state.anthropic_tool_use_id:
                    known_tool_calls[tc_state.anthropic_tool_use_id] = tc_state.name

            for msg in request.messages:
                if msg.get("role") in {"assistant", "model"}:
                    for tc in msg.get("tool_calls", []):
                        cid = tc.get("id") or tc.get("call_id")
                        fname = tc.get("function", {}).get("name") if isinstance(tc.get("function"), dict) else None
                        if cid and fname:
                            known_tool_calls[cid] = fname

            declared_names = {t.name for t in (plan.client_tools + plan.provider_tools) if t.name}

            current_contents = []
            i = 0
            raw_msgs = request.messages
            while i < len(raw_msgs):
                msg = raw_msgs[i]
                role = msg.get("role")
                if role in {"system", "developer"}:
                    i += 1
                    continue
                elif role == "user":
                    content = msg.get("content")
                    parts = []
                    if isinstance(content, str):
                        parts.append({"text": content})
                    elif isinstance(content, list):
                        for p in content:
                            if isinstance(p, dict):
                                if p.get("type") == "text":
                                    parts.append({"text": p.get("text", "")})
                                elif p.get("type") == "image":
                                    src = p.get("source", {})
                                    if src.get("type") == "base64":
                                        parts.append({"inlineData": {"mimeType": src.get("media_type", "image/png"), "data": src.get("data", "")}})
                    current_contents.append({"role": "user", "parts": parts})
                    i += 1
                elif role in {"assistant", "model"}:
                    content = msg.get("content")
                    parts = []
                    if isinstance(content, str) and content:
                        parts.append({"text": content})
                    tool_calls = msg.get("tool_calls")
                    if tool_calls:
                        for tc in tool_calls:
                            fn = tc.get("function", {})
                            args = fn.get("arguments", "{}")
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except Exception:
                                    args = {}
                            part_dict: Dict[str, Any] = {
                                "functionCall": {
                                    "name": fn.get("name"),
                                    "args": args
                                }
                            }
                            sig = tc.get("thought_signature") or tc.get("thoughtSignature")
                            if sig:
                                part_dict["functionCall"]["thoughtSignature"] = sig
                            parts.append(part_dict)
                    current_contents.append({"role": "model", "parts": parts})
                    i += 1
                elif role == "tool":
                    fn_resps = []
                    while i < len(raw_msgs) and raw_msgs[i].get("role") == "tool":
                        tmsg = raw_msgs[i]
                        tc_id = tmsg.get("tool_call_id") or tmsg.get("id")
                        content_str = tmsg.get("content", "")
                        if not isinstance(content_str, str):
                            content_str = json.dumps(content_str, ensure_ascii=False)

                        fn_name = known_tool_calls.get(tc_id)
                        if not fn_name and tc_id in declared_names:
                            fn_name = tc_id
                        if not fn_name:
                            yield AgentEvent(
                                kind=AgentEventKind.ERROR,
                                data={"message": f"Invalid or missing tool_call_id '{tc_id}'", "status_code": 400}
                            )
                            yield AgentEvent(kind=AgentEventKind.COMPLETED, data={"id": response_id, "stop_reason": AgentStopReason.ERROR, "usage": {}})
                            return

                        fn_resps.append({
                            "functionResponse": {
                                "name": fn_name,
                                "response": {"result": content_str}
                            }
                        })
                        i += 1
                    current_contents.append({"role": "user", "parts": fn_resps})
                else:
                    i += 1

            merged_contents = list(history_contents) + current_contents

            system_instruction = None
            if request.instructions and isinstance(request.instructions, str):
                system_instruction = request.instructions
            else:
                for m in request.messages:
                    if m.get("role") in {"system", "developer"}:
                        system_instruction = m.get("content")
                        break

            safety_settings = getattr(config, "safety_settings", None)
            native_payload = GeminiNativeCodec.encode_request(
                contents=merged_contents,
                tools=plan.provider_tools + plan.client_tools,
                model=request.model,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                system_instruction=system_instruction,
                safety_settings=safety_settings,
                request_id=req_id,
            )

            # Delete any resolved pending tool states
            for msg in request.messages:
                if msg.get("role") == "tool":
                    tc_id = msg.get("tool_call_id") or msg.get("id")
                    if tc_id:
                        await self.state_store.delete_pending_tool_state(tc_id)

            max_iterations = getattr(config, "agent_max_iterations", 5)
            is_stream = request.metadata.get("stream") is True

            try:
                if is_stream:
                    chunk_iter = await native_client.stream_generate_content(
                        model=request.model.removeprefix("google/"),
                        payload=native_payload,
                        api_version=plan.api_version
                    )
                    accumulated_parts = []
                    usage_info = {}
                    stop_reason = AgentStopReason.END_TURN
                    log_id = f"[{req_id}] " if req_id else ""

                    async for chunk in chunk_iter:
                        if "usageMetadata" in chunk:
                            um = chunk["usageMetadata"]
                            usage_info = {
                                "prompt_tokens": um.get("promptTokenCount", 0),
                                "completion_tokens": um.get("candidatesTokenCount", 0),
                                "total_tokens": um.get("totalTokenCount", 0),
                            }

                        prompt_feedback = chunk.get("promptFeedback")
                        if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
                            block_reason = prompt_feedback.get("blockReason")
                            block_msg = prompt_feedback.get("blockReasonMessage")
                            safety_ratings = prompt_feedback.get("safetyRatings", [])
                            stop_reason = AgentStopReason.PROMPT_BLOCKED
                            LOGGER.warning(
                                "%sGemini prompt blocked (input-side): blockReason=%s, message=%s, safetyRatings=%s",
                                log_id,
                                block_reason,
                                block_msg,
                                safety_ratings,
                            )

                        candidates = chunk.get("candidates", [])
                        if not candidates:
                            continue
                        candidate = candidates[0]
                        finish_reason = candidate.get("finishReason")
                        candidate_ratings = candidate.get("safetyRatings", [])
                        if finish_reason == "MAX_TOKENS":
                            stop_reason = AgentStopReason.MAX_TOKENS
                        elif finish_reason in {"SAFETY", "RECITATION"}:
                            stop_reason = AgentStopReason.REFUSAL
                            LOGGER.warning(
                                "%sGemini generation blocked (output-side): finishReason=%s, safetyRatings=%s",
                                log_id,
                                finish_reason,
                                candidate_ratings,
                            )
                        elif finish_reason == "PROHIBITED_CONTENT":
                            stop_reason = AgentStopReason.PROHIBITED_CONTENT
                            LOGGER.warning(
                                "%sGemini generation blocked (output-side): finishReason=PROHIBITED_CONTENT, safetyRatings=%s",
                                log_id,
                                candidate_ratings,
                            )
                        elif finish_reason == "OTHER":
                            stop_reason = AgentStopReason.OTHER
                            LOGGER.warning(
                                "%sGemini generation blocked (output-side): finishReason=OTHER, safetyRatings=%s",
                                log_id,
                                candidate_ratings,
                            )
                        elif finish_reason and finish_reason != "STOP":
                            stop_reason = AgentStopReason.REFUSAL
                            LOGGER.warning(
                                "%sGemini generation stopped (output-side): finishReason=%s, safetyRatings=%s",
                                log_id,
                                finish_reason,
                                candidate_ratings,
                            )

                        content = candidate.get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            accumulated_parts.extend(parts)

                        for part in parts:
                            if "text" in part and part["text"]:
                                yield AgentEvent(kind=AgentEventKind.TEXT_DELTA, data={"text": part["text"]})
                            elif "functionCall" in part:
                                fc = part["functionCall"]
                                stop_reason = AgentStopReason.TOOL_USE
                                tc_id = f"call_{secrets.token_hex(12)}"
                                tc_name = fc.get("name", "")
                                tc_args = fc.get("args", {})
                                sig = part.get("thought_signature") or fc.get("thought_signature") or part.get("thoughtSignature") or fc.get("thoughtSignature")

                                tc_obj = {
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {
                                        "name": tc_name,
                                        "arguments": json.dumps(tc_args, ensure_ascii=False) if isinstance(tc_args, dict) else str(tc_args)
                                    }
                                }
                                if sig:
                                    tc_obj["thought_signature"] = sig
                                    tc_obj["thoughtSignature"] = sig

                                saved_tool_calls[tc_id] = ToolCallState(
                                    canonical_call_id=tc_id,
                                    provider_turn_id=f"turn_{secrets.token_hex(8)}",
                                    part_index=len(saved_tool_calls),
                                    name=tc_name,
                                    openai_call_id=tc_id,
                                    anthropic_tool_use_id=tc_id
                                )
                                self.call_registry.register_call(tc_id, client_name=tc_name, internal_name=tc_name, tool_kind=ToolKind.FUNCTION, execution=ToolExecution.CLIENT)

                                if tool_mode == "strict":
                                    await self.state_store.save_pending_tool_state(
                                        PendingServerToolState(
                                            tool_use_id=tc_id,
                                            kind=ToolKind.FUNCTION,
                                            input=tc_args if isinstance(tc_args, dict) else {},
                                            provider_state=VertexNativeState(contents=merged_contents)
                                        )
                                    )

                                yield AgentEvent(kind=AgentEventKind.TOOL_STARTED, tool_kind=ToolKind.FUNCTION, data={"tool_calls": [tc_obj]})

                        grounding_metadata = candidate.get("groundingMetadata")
                        if grounding_metadata:
                            chunks_meta = grounding_metadata.get("groundingChunks", [])
                            sources = []
                            for c in chunks_meta:
                                web = c.get("web")
                                if isinstance(web, dict) and web.get("uri"):
                                    sources.append({"title": web.get("title") or web["uri"], "url": web["uri"]})
                            if sources:
                                yield AgentEvent(kind=AgentEventKind.TOOL_STARTED, tool_kind=ToolKind.WEB_SEARCH, data={"query": "web search", "sources": sources})
                                yield AgentEvent(kind=AgentEventKind.CITATION, tool_kind=ToolKind.WEB_SEARCH, data={"sources": sources})

                    if accumulated_parts:
                        merged_contents.append({"role": "model", "parts": accumulated_parts})

                    # If server iteration limit is reached in strict mode
                    if max_iterations <= 1 and stop_reason == AgentStopReason.TOOL_USE and tool_mode == "strict":
                        stop_reason = AgentStopReason.PAUSE_TURN

                    new_state = ResponseState(
                        response_id=response_id,
                        previous_response_id=previous_response_id,
                        model=request.model,
                        backend=BackendKind.VERTEX_NATIVE,
                        provider_snapshot=VertexNativeState(contents=merged_contents),
                        previous_instructions=request.instructions,
                        reasoning=request.reasoning,
                        tool_calls=saved_tool_calls
                    )
                    await self.state_store.save_response_state(new_state)

                    yield AgentEvent(kind=AgentEventKind.COMPLETED, data={
                        "id": response_id,
                        "stop_reason": stop_reason,
                        "usage": usage_info
                    })

                else:
                    native_resp = await native_client.generate_content(
                        model=request.model.removeprefix("google/"),
                        payload=native_payload,
                        api_version=plan.api_version
                    )

                    agent_resp = GeminiNativeCodec.decode_response(
                        native_resp,
                        request.model,
                        request_id=req_id,
                    )

                    # Also check for grounding metadata and yield web search events
                    candidates = native_resp.get("candidates", [])
                    if candidates:
                        gm = candidates[0].get("groundingMetadata")
                        if gm:
                            chunks_meta = gm.get("groundingChunks", [])
                            sources = []
                            for c in chunks_meta:
                                web = c.get("web")
                                if isinstance(web, dict) and web.get("uri"):
                                    sources.append({"title": web.get("title") or web["uri"], "url": web["uri"]})
                            if sources:
                                yield AgentEvent(kind=AgentEventKind.TOOL_STARTED, tool_kind=ToolKind.WEB_SEARCH, data={"query": "web search", "sources": sources})
                                yield AgentEvent(kind=AgentEventKind.CITATION, tool_kind=ToolKind.WEB_SEARCH, data={"sources": sources})

                    for item in agent_resp.output:
                        if item["type"] == "message":
                            yield AgentEvent(kind=AgentEventKind.TEXT_STARTED)
                            yield AgentEvent(kind=AgentEventKind.TEXT_DELTA, data={"text": item["content"]})
                            yield AgentEvent(kind=AgentEventKind.TEXT_COMPLETED)
                        elif item["type"] == "tool_calls":
                            yield AgentEvent(kind=AgentEventKind.TOOL_STARTED, tool_kind=ToolKind.FUNCTION, data={"tool_calls": item["tool_calls"]})
                            turn_id = f"turn_{secrets.token_hex(8)}"
                            for idx, tc in enumerate(item["tool_calls"]):
                                call_id = tc["id"]
                                tc_name = tc["function"]["name"]
                                st = ToolCallState(
                                    canonical_call_id=call_id,
                                    provider_turn_id=turn_id,
                                    part_index=idx,
                                    name=tc_name,
                                    openai_call_id=call_id,
                                    anthropic_tool_use_id=call_id
                                )
                                saved_tool_calls[call_id] = st
                                self.call_registry.register_call(call_id, client_name=tc_name, internal_name=tc_name, tool_kind=ToolKind.FUNCTION, execution=ToolExecution.CLIENT)

                                if tool_mode == "strict":
                                    args_dict = {}
                                    try:
                                        args_dict = json.loads(tc["function"]["arguments"])
                                    except Exception:
                                        pass
                                    await self.state_store.save_pending_tool_state(
                                        PendingServerToolState(
                                            tool_use_id=call_id,
                                            kind=ToolKind.FUNCTION,
                                            input=args_dict,
                                            provider_state=VertexNativeState(contents=merged_contents)
                                        )
                                    )

                    if candidates:
                        content_block = candidates[0].get("content", {})
                        merged_contents.append(content_block)

                    actual_stop_reason = agent_resp.stop_reason
                    if max_iterations <= 1 and actual_stop_reason == AgentStopReason.TOOL_USE and tool_mode == "strict":
                        actual_stop_reason = AgentStopReason.PAUSE_TURN

                    new_state = ResponseState(
                        response_id=response_id,
                        previous_response_id=previous_response_id,
                        model=request.model,
                        backend=BackendKind.VERTEX_NATIVE,
                        provider_snapshot=VertexNativeState(contents=merged_contents),
                        previous_instructions=request.instructions,
                        reasoning=request.reasoning,
                        tool_calls=saved_tool_calls
                    )
                    await self.state_store.save_response_state(new_state)

                    yield AgentEvent(kind=AgentEventKind.COMPLETED, data={
                        "id": response_id,
                        "stop_reason": actual_stop_reason,
                        "usage": agent_resp.usage
                    })

            except Exception as e:
                yield AgentEvent(kind=AgentEventKind.ERROR, data={"message": str(e)})
                yield AgentEvent(kind=AgentEventKind.COMPLETED, data={"id": response_id, "stop_reason": AgentStopReason.ERROR, "usage": {}})
