# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development & Test Commands

- **Run all tests:** `pytest` or `.\.venv\Scripts\pytest`
- **Run a single test file:** `pytest tests/test_app.py`
- **Run a specific test:** `pytest tests/test_app.py -k test_function_name`
- **Run local proxy server:** `uvicorn vertex_proxy.app:app --host 127.0.0.1 --port 8000`
- **Install package with dev dependencies:** `pip install -e .[test,gui]`
- **Run GUI desktop client:** `python gui.py`
- **Build GUI executable:** `python build_gui.py`
- **Build Docker image:** `docker build -t vertex-adc-proxy .`

## Environment Setup Requirements

The server requires the following environment variables to start:
- `GOOGLE_CLOUD_PROJECT`: GCP Project ID.
- `VERTEX_LOCATION`: Vertex AI region (e.g. `asia-east1`, `us-central1`, or `global`).

Optional runtime configuration:
- `VERTEX_PROXY_API_KEY`: Client authorization key (`Authorization: Bearer <key>` or `X-API-Key`).
- `VERTEX_AGENT_TOOL_MODE`: Tool dispatch mode (`best_effort` for direct Native tools combination, `strict` for strict mixed-tool defer/resume state machine).
- `VERTEX_AGENT_MAX_ITERATIONS`: Server-side autonomous iteration limit before triggering `pause_turn` (default `5`).
- `VERTEX_ANTHROPIC_BACKEND`: `/v1/messages` target backend (`gemini` for protocol translation to Gemini, `claude` for direct Vertex Claude passthrough).
- `VERTEX_ANTHROPIC_GEMINI_MODEL`: Model name used when `VERTEX_ANTHROPIC_BACKEND=gemini`.
- `VERTEX_MODELS`: Comma-separated list returned by `GET /v1/models`.
- `VERTEX_PROXY_LOG_MODE`: Logging granularity (`full`, `messages`, `errors`, `none`).

## Architecture Overview

`vertex-proxy` is a FastAPI reverse proxy that exposes OpenAI and Anthropic compatible endpoints, converting and routing requests to Google Cloud Vertex AI using Application Default Credentials (ADC).

### Key Modules

- **`vertex_proxy/app.py`**: Entry point and FastAPI application.
  - **ADC & Token Management**: Acquires Google ADC access tokens and caches them; automatically forces token refresh and retries once on upstream HTTP 401 errors.
  - **Upstream Host Translation**: Maps `VERTEX_LOCATION=global` to `aiplatform.googleapis.com` and regional locations to `{location}-aiplatform.googleapis.com`.
  - **Endpoints & Routing**:
    - `/v1/*`: OpenAI Chat Completions compatibility endpoint.
    - `/v1/responses`: OpenAI Responses API endpoint routed via unified `AgentRuntime`.
    - `/v1/messages` & `/v1/messages/count_tokens`: Anthropic Messages API compatibility endpoint routed via `AgentRuntime`.
    - `/vertex/{api_version}/*`: Direct passthrough to Vertex REST API (`v1`, `v1beta1`).
- **`vertex_proxy/agent_ir.py`**: Intermediate Representation (IR) data models (`AgentRequest`, `AgentResponse`, `AgentEvent`, `AgentTool`, `UnsupportedToolError`) and schema converters.
- **`vertex_proxy/agent_runtime.py`**: Unified Agent Runtime with dual-backend execution planning (Vertex OpenAI and Vertex Native), `CallRegistry`, multi-tool merging, and autonomous server iteration control.
- **`vertex_proxy/agent_state.py`**: Thread-safe `InMemoryAgentStateStore` with sliding TTL and LRU capacity limits for `ResponseState`, `PendingServerToolState`, and `ProviderTurnState`.
- **`vertex_proxy/vertex_native.py`**: Native REST client and Codec supporting `googleSearch`, `codeExecution`, `urlContext`, `functionDeclarations`, and real Native SSE streaming.
- **`vertex_proxy/anthropic_gemini.py`**: Inbound parsing and outbound Unary/SSE formatting for Anthropic Messages API.
- **`vertex_proxy/openai_responses.py`**: Inbound parsing and outbound Unary/SSE formatting for OpenAI Responses API.
- **`vertex_proxy/gui.py`**: PyQt6 desktop UI for managing the proxy process, inspecting ADC credentials, and configuring proxies/keys on Windows.

### Testing Architecture

- Unit and integration tests in `tests/` use `httpx.MockTransport` and `FakeCredentials` to mock Google ADC authentication and upstream HTTP calls. Tests run locally without live GCP network calls or valid credentials.
