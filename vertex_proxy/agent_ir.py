from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

class BackendKind(Enum):
    VERTEX_OPENAI = "vertex_openai"
    VERTEX_NATIVE = "vertex_native"

class ToolKind(Enum):
    FUNCTION = "function"
    WEB_SEARCH = "web_search"
    WEB_FETCH = "web_fetch"
    CODE_EXECUTION = "code_execution"
    FILE_SEARCH = "file_search"
    COMPUTER = "computer"
    MCP = "mcp"
    TOOL_SEARCH = "tool_search"
    ADVISOR = "advisor"
    SHELL = "shell"
    APPLY_PATCH = "apply_patch"
    UNKNOWN = "unknown"

class ToolExecution(Enum):
    CLIENT = "client"      # Executed on client-side
    PROVIDER = "provider"  # Executed on Vertex AI provider-side
    PROXY = "proxy"        # Intercepted and executed by the proxy runtime

@dataclass
class AgentTool:
    kind: ToolKind
    execution: ToolExecution
    name: Optional[str]
    input_schema: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None

class ToolChoiceMode(Enum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
    SPECIFIC = "specific"
    ALLOWED = "allowed"

@dataclass
class AgentToolChoice:
    mode: ToolChoiceMode
    specific_tool: Optional[str] = None
    allowed_tools: Optional[list[str]] = None

@dataclass
class AgentRequest:
    model: str
    messages: list[Dict[str, Any]]
    tools: list[AgentTool] = field(default_factory=list)
    tool_choice: AgentToolChoice = field(default_factory=lambda: AgentToolChoice(ToolChoiceMode.AUTO))
    parallel_tool_calls: bool = True
    instructions: Optional[Any] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    response_format: Optional[Dict[str, Any]] = None
    reasoning: Optional[Any] = None  # Preserved Responses reasoning / encrypted opaque state
    previous_response_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AgentStopReason(Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    PAUSE_TURN = "pause_turn"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    ERROR = "error"

@dataclass
class AgentResponse:
    id: str
    output: list[Any]
    stop_reason: AgentStopReason
    usage: Any

class AgentEventKind(Enum):
    RESPONSE_STARTED = "response_started"
    TEXT_STARTED = "text_started"
    TEXT_DELTA = "text_delta"
    TEXT_COMPLETED = "text_completed"
    TOOL_STARTED = "tool_started"
    TOOL_PROGRESS = "tool_progress"
    TOOL_ARGUMENT_DELTA = "tool_argument_delta"
    TOOL_RESULT = "tool_result"
    TOOL_COMPLETED = "tool_completed"
    CITATION = "citation"
    USAGE = "usage"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentEvent:
    kind: AgentEventKind
    tool_kind: Optional[ToolKind] = None
    item_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

class UnsupportedToolError(ValueError):
    """Raised when an unsupported hosted or remote tool is declared by the client."""
    pass

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

def parse_openai_tool(tool: Dict[str, Any]) -> AgentTool:
    if not isinstance(tool, dict):
        raise UnsupportedToolError("Tool declaration must be a dictionary")

    tool_type = tool.get("type")

    # 1. Standard Client Custom Tools (Function Calling)
    if tool_type == "function":
        fn = tool.get("function")
        if not isinstance(fn, dict) or not fn.get("name"):
            raise UnsupportedToolError("Invalid function tool specification")
        return AgentTool(
            kind=ToolKind.FUNCTION,
            execution=ToolExecution.CLIENT,
            name=fn["name"],
            input_schema=fn.get("parameters"),
            raw=tool
        )

    # 2. Server-side / Provider Hosted Tools
    if tool_type in {"web_search", "web_search_preview", "web_search_preview_2025_03_11"}:
        return AgentTool(
            kind=ToolKind.WEB_SEARCH,
            execution=ToolExecution.PROVIDER,
            name="web_search",
            config=tool.get("web_search_options"),
            raw=tool
        )

    if tool_type == "web_fetch":
        return AgentTool(
            kind=ToolKind.WEB_FETCH,
            execution=ToolExecution.PROVIDER,
            name="web_fetch",
            raw=tool
        )

    if tool_type == "code_interpreter":
        return AgentTool(
            kind=ToolKind.CODE_EXECUTION,
            execution=ToolExecution.PROVIDER,
            name="code_interpreter",
            raw=tool
        )

    # Unhandled / Unsupported types
    if tool_type in {"file_search", "mcp", "shell", "local_shell", "apply_patch"}:
        raise UnsupportedToolError(f"Unsupported hosted tool type: {tool_type}")

    raise UnsupportedToolError(f"Unknown tool type: {tool_type}")

def parse_anthropic_tool(tool: Dict[str, Any]) -> AgentTool:
    if not isinstance(tool, dict):
        raise UnsupportedToolError("Tool declaration must be a dictionary")

    tool_type = tool.get("type")
    name = tool.get("name", "")

    # Server-side tools are typically prefixed or identified by type/name
    if isinstance(tool_type, str):
        if tool_type.startswith("web_search_") or tool_type == "web_search" or name == "web_search":
            return AgentTool(
                kind=ToolKind.WEB_SEARCH,
                execution=ToolExecution.PROVIDER,
                name="web_search",
                raw=tool
            )
        if tool_type.startswith("web_fetch_") or tool_type == "web_fetch" or name == "web_fetch":
            return AgentTool(
                kind=ToolKind.WEB_FETCH,
                execution=ToolExecution.PROVIDER,
                name="web_fetch",
                raw=tool
            )
        if tool_type.startswith("code_execution_") or tool_type == "code_execution" or name == "code_execution":
            return AgentTool(
                kind=ToolKind.CODE_EXECUTION,
                execution=ToolExecution.PROVIDER,
                name="code_execution",
                raw=tool
            )

    if name in {"web_search", "web_fetch", "code_execution"}:
        kind_map = {
            "web_search": ToolKind.WEB_SEARCH,
            "web_fetch": ToolKind.WEB_FETCH,
            "code_execution": ToolKind.CODE_EXECUTION,
        }
        return AgentTool(
            kind=kind_map[name],
            execution=ToolExecution.PROVIDER,
            name=name,
            raw=tool
        )

    # Standard client-side tool mapping
    if name and "input_schema" in tool:
        fn_dict: Dict[str, Any] = {
            "name": name,
            "parameters": _vertex_function_schema(tool.get("input_schema")),
        }
        if isinstance(tool.get("description"), str):
            fn_dict["description"] = tool["description"]
        raw_openai = {"type": "function", "function": fn_dict}
        return AgentTool(
            kind=ToolKind.FUNCTION,
            execution=ToolExecution.CLIENT,
            name=name,
            input_schema=tool.get("input_schema"),
            config={"description": tool.get("description", "")} if isinstance(tool.get("description"), str) else None,
            raw=raw_openai
        )

    raise UnsupportedToolError(f"Unsupported or invalid Anthropic tool: {tool}")
