from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from vertex_proxy.agent_ir import BackendKind, ToolKind

@dataclass
class VertexOpenAIState:
    messages: list[Dict[str, Any]]

@dataclass
class VertexNativeState:
    contents: list[Dict[str, Any]]

@dataclass
class ProviderTurnState:
    turn_id: str
    model_content: Dict[str, Any]  # The full {"role": "model", "parts": [...]} content block
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)

@dataclass
class ToolCallState:
    canonical_call_id: str
    provider_turn_id: str
    part_index: int
    name: str = ""
    openai_call_id: Optional[str] = None
    anthropic_tool_use_id: Optional[str] = None

@dataclass
class PendingServerToolState:
    tool_use_id: str
    kind: ToolKind
    input: Dict[str, Any]
    provider_state: Any  # Snapshot state (VertexOpenAIState or VertexNativeState)
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)

@dataclass
class ResponseState:
    response_id: str
    previous_response_id: Optional[str]
    model: str
    backend: BackendKind
    provider_snapshot: Union[VertexOpenAIState, VertexNativeState]
    previous_instructions: Optional[Any] = None
    reasoning: Optional[Any] = None
    tool_calls: Dict[str, ToolCallState] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)

class InMemoryAgentStateStore:
    def __init__(
        self,
        max_response_states: int = 1000,
        max_pending_tools: int = 2000,
        max_provider_turns: int = 2000,
        ttl_seconds: float = 3600.0,
    ) -> None:
        self._response_states: Dict[str, ResponseState] = {}
        self._pending_tools: Dict[str, PendingServerToolState] = {}
        self._provider_turns: Dict[str, ProviderTurnState] = {}

        self._max_response_states = max_response_states
        self._max_pending_tools = max_pending_tools
        self._max_provider_turns = max_provider_turns
        self._ttl_seconds = ttl_seconds

        self._lock = asyncio.Lock()

    async def _evict_expired(self) -> None:
        now = time.time()
        expired_responses = [
            rid for rid, s in self._response_states.items()
            if now - s.last_accessed_at > self._ttl_seconds
        ]
        for rid in expired_responses:
            del self._response_states[rid]

        expired_tools = [
            tid for tid, t in self._pending_tools.items()
            if now - t.last_accessed_at > self._ttl_seconds
        ]
        for tid in expired_tools:
            del self._pending_tools[tid]

        expired_turns = [
            turn_id for turn_id, turn in self._provider_turns.items()
            if now - turn.last_accessed_at > self._ttl_seconds
        ]
        for turn_id in expired_turns:
            del self._provider_turns[turn_id]

    async def _enforce_limits(self) -> None:
        # LRU truncation for ResponseState based on last_accessed_at
        if len(self._response_states) > self._max_response_states:
            sorted_states = sorted(self._response_states.items(), key=lambda x: x[1].last_accessed_at)
            to_remove = len(sorted_states) - self._max_response_states
            for i in range(to_remove):
                del self._response_states[sorted_states[i][0]]

        # LRU truncation for PendingServerToolState
        if len(self._pending_tools) > self._max_pending_tools:
            sorted_tools = sorted(self._pending_tools.items(), key=lambda x: x[1].last_accessed_at)
            to_remove = len(sorted_tools) - self._max_pending_tools
            for i in range(to_remove):
                del self._pending_tools[sorted_tools[i][0]]

        # LRU truncation for ProviderTurnState
        if len(self._provider_turns) > self._max_provider_turns:
            sorted_turns = sorted(self._provider_turns.items(), key=lambda x: x[1].last_accessed_at)
            to_remove = len(sorted_turns) - self._max_provider_turns
            for i in range(to_remove):
                del self._provider_turns[sorted_turns[i][0]]

    async def save_response_state(self, state: ResponseState) -> None:
        async with self._lock:
            await self._evict_expired()
            state.last_accessed_at = time.time()
            self._response_states[state.response_id] = state
            await self._enforce_limits()

    async def get_response_state(self, response_id: str) -> Optional[ResponseState]:
        async with self._lock:
            await self._evict_expired()
            state = self._response_states.get(response_id)
            if state is None and response_id.startswith("msg_"):
                state = self._response_states.get("resp_" + response_id[4:])
            elif state is None and response_id.startswith("resp_"):
                state = self._response_states.get("msg_" + response_id[5:])
            if state:
                state.last_accessed_at = time.time()
            return state

    async def save_pending_tool_state(self, state: PendingServerToolState) -> None:
        async with self._lock:
            await self._evict_expired()
            state.last_accessed_at = time.time()
            self._pending_tools[state.tool_use_id] = state
            await self._enforce_limits()

    async def get_pending_tool_state(self, tool_use_id: str) -> Optional[PendingServerToolState]:
        async with self._lock:
            await self._evict_expired()
            tool_state = self._pending_tools.get(tool_use_id)
            if tool_state:
                tool_state.last_accessed_at = time.time()
            return tool_state

    async def delete_pending_tool_state(self, tool_use_id: str) -> None:
        async with self._lock:
            if tool_use_id in self._pending_tools:
                del self._pending_tools[tool_use_id]

    async def save_provider_turn(self, turn: ProviderTurnState) -> None:
        async with self._lock:
            await self._evict_expired()
            turn.last_accessed_at = time.time()
            self._provider_turns[turn.turn_id] = turn
            await self._enforce_limits()

    async def get_provider_turn(self, turn_id: str) -> Optional[ProviderTurnState]:
        async with self._lock:
            await self._evict_expired()
            turn = self._provider_turns.get(turn_id)
            if turn:
                turn.last_accessed_at = time.time()
            return turn
