from __future__ import annotations

import json
import secrets
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from vertex_proxy.agent_ir import (
    AgentEvent,
    AgentEventKind,
    AgentResponse,
    AgentStopReason,
    AgentTool,
    ToolKind,
)
from vertex_proxy.anthropic_gemini import _vertex_function_schema

@dataclass
class ProviderCapabilities:
    native_tool_combination: bool = True
    server_side_tool_invocations: bool = True
    google_search: bool = True
    url_context: bool = True
    code_execution: bool = True

def _to_native_parameters(parameters: Any) -> Dict[str, Any]:
    return _vertex_function_schema(parameters)

class GeminiNativeCodec:
    @staticmethod
    def encode_request(
        contents: list[Dict[str, Any]],
        tools: list[AgentTool],
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        native_tools = []
        client_functions = []
        google_search_enabled = False
        url_context_enabled = False
        code_execution_enabled = False

        for tool in tools:
            if tool.kind == ToolKind.WEB_SEARCH:
                google_search_enabled = True
            elif tool.kind == ToolKind.WEB_FETCH:
                url_context_enabled = True
            elif tool.kind == ToolKind.CODE_EXECUTION:
                code_execution_enabled = True
            elif tool.kind == ToolKind.FUNCTION:
                client_functions.append({
                    "name": tool.name,
                    "description": tool.config.get("description", "") if tool.config else "",
                    "parameters": _to_native_parameters(tool.input_schema)
                })

        # Pack Google native tools
        if client_functions:
            native_tools.append({"functionDeclarations": client_functions})
        if google_search_enabled:
            native_tools.append({"googleSearch": {}})
        if url_context_enabled:
            native_tools.append({"urlContext": {}})
        if code_execution_enabled:
            native_tools.append({"codeExecution": {}})

        native_payload: Dict[str, Any] = {
            "contents": contents,
        }

        if system_instruction:
            native_payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        generation_config: Dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if top_p is not None:
            generation_config["topP"] = top_p
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        if generation_config:
            native_payload["generationConfig"] = generation_config

        if native_tools:
            native_payload["tools"] = native_tools

        return native_payload

    @staticmethod
    def decode_response(
        native_resp: Dict[str, Any],
        model: str,
    ) -> AgentResponse:
        response_id = f"resp_{secrets.token_hex(16)}"
        candidates = native_resp.get("candidates", [])
        output_items = []
        stop_reason = AgentStopReason.END_TURN

        for candidate in candidates:
            finish_reason = candidate.get("finishReason")
            if finish_reason == "MAX_TOKENS":
                stop_reason = AgentStopReason.MAX_TOKENS
            elif finish_reason == "SAFETY" or finish_reason == "RECITATION":
                stop_reason = AgentStopReason.REFUSAL

            content = candidate.get("content", {})
            parts = content.get("parts", [])

            text_segments = []
            tool_calls = []

            for part in parts:
                if "text" in part:
                    text_segments.append(part["text"])
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    # Extract original signature
                    thought_sig = part.get("thought_signature") or fc.get("thought_signature") or part.get("thoughtSignature") or fc.get("thoughtSignature")
                    tc_id = f"call_{secrets.token_hex(12)}"

                    tc_obj = {
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": fc.get("name"),
                            "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False)
                        }
                    }
                    if thought_sig:
                        tc_obj["thought_signature"] = thought_sig
                        tc_obj["thoughtSignature"] = thought_sig
                    tool_calls.append(tc_obj)
                    stop_reason = AgentStopReason.TOOL_USE

            final_text = "".join(text_segments)

            # Extract citations from Grounding Metadata
            grounding_metadata = candidate.get("groundingMetadata")
            if grounding_metadata:
                citations = []
                seen_urls = set()
                chunks = grounding_metadata.get("groundingChunks", [])
                for chunk in chunks:
                    web = chunk.get("web")
                    if isinstance(web, dict):
                        uri = web.get("uri")
                        title = web.get("title") or uri
                        if uri and uri not in seen_urls:
                            citations.append(f"- [{title}]({uri})")
                            seen_urls.add(uri)
                if citations:
                    final_text += "\n\nSources:\n" + "\n".join(citations)

            if final_text:
                output_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": final_text
                })

            if tool_calls:
                output_items.append({
                    "type": "tool_calls",
                    "tool_calls": tool_calls
                })

        usage_meta = native_resp.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_meta.get("promptTokenCount", 0),
            "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
            "total_tokens": usage_meta.get("totalTokenCount", 0),
        }

        return AgentResponse(
            id=response_id,
            output=output_items,
            stop_reason=stop_reason,
            usage=usage
        )

    @staticmethod
    def decode_stream_chunk(
        chunk: Dict[str, Any],
    ) -> list[AgentEvent]:
        events = []
        candidates = chunk.get("candidates", [])
        if not candidates:
            return events

        for candidate in candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            for part in parts:
                if "text" in part:
                    events.append(AgentEvent(
                        kind=AgentEventKind.TEXT_DELTA,
                        data={"text": part["text"]}
                    ))
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    # If model has produced function Call inside stream
                    events.append(AgentEvent(
                        kind=AgentEventKind.TOOL_STARTED,
                        tool_kind=ToolKind.FUNCTION,
                        item_id=fc.get("name"),
                        data={"name": fc.get("name")}
                    ))
                    events.append(AgentEvent(
                        kind=AgentEventKind.TOOL_ARGUMENT_DELTA,
                        tool_kind=ToolKind.FUNCTION,
                        item_id=fc.get("name"),
                        data={"arguments": json.dumps(fc.get("args", {}), ensure_ascii=False)}
                    ))
                    events.append(AgentEvent(
                        kind=AgentEventKind.TOOL_COMPLETED,
                        tool_kind=ToolKind.FUNCTION,
                        item_id=fc.get("name")
                    ))

            # Citations / Grounding metadata stream translation
            grounding_metadata = candidate.get("groundingMetadata")
            if grounding_metadata:
                seen_urls = set()
                chunks = grounding_metadata.get("groundingChunks", [])
                sources = []
                queries = grounding_metadata.get("webSearchQueries", [])
                search_query = queries[0] if queries and isinstance(queries, list) else "web search"
                for chunk_item in chunks:
                    web = chunk_item.get("web")
                    if isinstance(web, dict):
                        uri = web.get("uri")
                        title = web.get("title") or uri
                        if uri and uri not in seen_urls:
                            sources.append({"title": title, "url": uri})
                            seen_urls.add(uri)
                if sources:
                    events.append(AgentEvent(
                        kind=AgentEventKind.TOOL_STARTED,
                        tool_kind=ToolKind.WEB_SEARCH,
                        item_id=f"ws_{secrets.token_hex(8)}",
                        data={"query": search_query, "sources": sources}
                    ))
                    events.append(AgentEvent(
                        kind=AgentEventKind.CITATION,
                        tool_kind=ToolKind.WEB_SEARCH,
                        data={"sources": sources}
                    ))

        return events

class VertexNativeClient:
    def __init__(self, http_client: Any, config: Any, token_provider: Any) -> None:
        self.client = http_client
        self.config = config
        self.token_provider = token_provider

    async def generate_content(
        self,
        model: str,
        payload: Dict[str, Any],
        api_version: str = "v1beta1",
    ) -> Dict[str, Any]:
        token = await self.token_provider.token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        url = (
            f"{self.config.native_base_url(api_version)}/publishers/google/models/"
            f"{model}:generateContent"
        )
        response = await self.client.post(url, headers=headers, json=payload)

        # Automatic retry exactly once on upstream 401
        if response.status_code == 401:
            token = await self.token_provider.token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            response = await self.client.post(url, headers=headers, json=payload)

        response.raise_for_status()
        return response.json()

    async def stream_generate_content(
        self,
        model: str,
        payload: Dict[str, Any],
        api_version: str = "v1beta1",
    ) -> AsyncIterator[Dict[str, Any]]:
        token = await self.token_provider.token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        url = (
            f"{self.config.native_base_url(api_version)}/publishers/google/models/"
            f"{model}:streamGenerateContent?alt=sse"
        )

        req = self.client.build_request("POST", url, headers=headers, json=payload)
        response = await self.client.send(req, stream=True)

        if response.status_code == 401:
            await response.aclose()
            token = await self.token_provider.token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            req = self.client.build_request("POST", url, headers=headers, json=payload)
            response = await self.client.send(req, stream=True)

        response.raise_for_status()

        async def chunk_generator():
            try:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_str = line[5:].lstrip()
                        if data_str:
                            yield json.loads(data_str)
            finally:
                await response.aclose()

        return chunk_generator()
