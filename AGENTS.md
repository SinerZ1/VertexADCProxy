# Agent Guide: Vertex AI ADC Proxy

High-signal repository facts and commands to avoid common mistakes.

## Quick Commands
- **Run tests:** `.\.venv\Scripts\pytest` or `pytest`
- **Run local server:** `uvicorn vertex_proxy.app:app --host 127.0.0.1 --port 8000`
- **Docker build:** `docker build -t vertex-adc-proxy .`

## Environment Prerequisites
The application fails to start if either of these is missing or invalid:
- `GOOGLE_CLOUD_PROJECT` (e.g. `your-project-id`)
- `VERTEX_LOCATION` (e.g. `asia-east1`, `us-central1`, or `global`)

### Optional Configurations
- `VERTEX_PROXY_API_KEY`: If set, incoming requests must use `Authorization: Bearer <key>` or `X-API-Key` headers. Otherwise, a 401 is returned.
- `VERTEX_MODELS`: Comma-separated list returned by `/v1/models`.
- `VERTEX_PROXY_LOG_MODE`: Logging mode configuration. Can be set to `full` (default, logs full request/response), `messages` (logs request messages only), `errors` (logs responses containing `content_filter`), or `none` (logs no request/response).

## Architectural Mechanics
- **Upstream Host Translation:**
  - If `VERTEX_LOCATION` is `global` -> `aiplatform.googleapis.com`
  - Otherwise -> `{location}-aiplatform.googleapis.com`
- **Path Rewriting:**
  - `/v1/{path}` -> `/v1/projects/{project}/locations/{location}/endpoints/openapi/{path}` (OpenAI endpoint)
  - `/vertex/{api_version}/{path}` -> `/{api_version}/projects/{project}/locations/{location}/{path}` (Native REST endpoint, supports `v1` and `v1beta1`)
- **Model Name Normalization:**
  - For incoming `/v1/*` requests, if the JSON body has `"model": "google/<model_name>"`, the proxy automatically strips the `google/` prefix before forwarding to Vertex.
- **Token Management & Upstream 401s:**
  - Tokens are retrieved using Application Default Credentials (ADC) and cached.
  - If the upstream Vertex AI endpoint returns a `401 Unauthorized`, the proxy automatically forces an immediate token refresh and retries the request exactly once.
- **Request and Response Logging:**
  - Log modes are configured via `VERTEX_PROXY_LOG_MODE` env variable:
    - `full`: Logs both request and response in full detail.
    - `messages`: Logs only the request body's messages key array, with fallback to formatting the entire request body. Responses are not logged.
    - `errors`: Logs only the completed responses containing `content_filter` error.
    - `none`: Completely disables request and response logs.
  - JSON formatting: Any valid JSON logs are automatically parsed and formatted using json.dumps with 2-space indentation.
  - Signature filtering: Headers containing credentials or signatures (including `authorization`, `x-api-key`, `x-goog-api-key`, or names with `signature` or `auth`) are automatically stripped from logs for safety.
  - Automatic Decompression: Upstream response streams compressed with `gzip` or `deflate` are decompressed automatically before logging to prevent garbled outputs.
  - Dynamic Encoding Detection: Decodes response bodies utilizing the `charset` specified in headers, falling back to `utf-8`.

## Testing Context
- **No live GCP access needed:** Tests in `tests/test_app.py` mock ADC and the HTTP transport layer using `FakeCredentials` and `httpx.MockTransport`. No real Google API credentials or network connections are utilized during testing.
