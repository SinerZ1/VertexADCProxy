from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping

import google.auth
import httpx
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest
from starlette.background import BackgroundTask

LOGGER = logging.getLogger("vertex_proxy")
_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_SAFE_RESOURCE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_REQUEST_HEADERS_TO_DROP = _HOP_BY_HOP_HEADERS | {
    "authorization",
    "content-length",
    "host",
    "x-api-key",
    "x-goog-api-key",
    "x-goog-user-project",
}


def _positive_float(value: str, name: str, *, allow_zero: bool = False) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是数字") from exc
    if parsed < 0 or (parsed == 0 and not allow_zero):
        qualifier = "非负" if allow_zero else "正数"
        raise RuntimeError(f"{name} 必须是{qualifier}")
    return parsed


@dataclass(frozen=True)
class Settings:
    project: str
    location: str
    proxy_api_key: str | None = None
    models: tuple[str, ...] = ()
    connect_timeout: float = 10.0
    read_timeout: float | None = 300.0
    token_refresh_skew: float = 300.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        source = os.environ if env is None else env
        project = source.get("GOOGLE_CLOUD_PROJECT", "").strip()
        location = source.get("VERTEX_LOCATION", "").strip().lower()
        if not project:
            raise RuntimeError("缺少环境变量 GOOGLE_CLOUD_PROJECT")
        if not location:
            raise RuntimeError("缺少环境变量 VERTEX_LOCATION")
        if not _SAFE_RESOURCE.fullmatch(project):
            raise RuntimeError("GOOGLE_CLOUD_PROJECT 格式不合法")
        if not _SAFE_RESOURCE.fullmatch(location):
            raise RuntimeError("VERTEX_LOCATION 格式不合法")

        read_timeout = _positive_float(
            source.get("VERTEX_READ_TIMEOUT", "300"),
            "VERTEX_READ_TIMEOUT",
            allow_zero=True,
        )
        raw_models = source.get("VERTEX_MODELS", "")
        models = tuple(dict.fromkeys(item.strip() for item in raw_models.split(",") if item.strip()))
        return cls(
            project=project,
            location=location,
            proxy_api_key=source.get("VERTEX_PROXY_API_KEY") or None,
            models=models,
            connect_timeout=_positive_float(
                source.get("VERTEX_CONNECT_TIMEOUT", "10"),
                "VERTEX_CONNECT_TIMEOUT",
            ),
            read_timeout=None if read_timeout == 0 else read_timeout,
            token_refresh_skew=_positive_float(
                source.get("VERTEX_TOKEN_REFRESH_SKEW", "300"),
                "VERTEX_TOKEN_REFRESH_SKEW",
                allow_zero=True,
            ),
        )

    @property
    def upstream_host(self) -> str:
        if self.location == "global":
            return "aiplatform.googleapis.com"
        return f"{self.location}-aiplatform.googleapis.com"

    @property
    def openai_base_url(self) -> str:
        return (
            f"https://{self.upstream_host}/v1/projects/{self.project}"
            f"/locations/{self.location}/endpoints/openapi"
        )

    def native_base_url(self, api_version: str) -> str:
        return (
            f"https://{self.upstream_host}/{api_version}/projects/{self.project}"
            f"/locations/{self.location}"
        )


class AdcTokenProvider:
    """Serializes ADC refreshes and refreshes shortly before expiry."""

    def __init__(
        self,
        credentials: Credentials,
        auth_request: GoogleAuthRequest,
        refresh_skew: float,
    ) -> None:
        self._credentials = credentials
        self._auth_request = auth_request
        self._refresh_skew = timedelta(seconds=refresh_skew)
        self._lock = asyncio.Lock()

    def _needs_refresh(self) -> bool:
        if not self._credentials.token:
            return True
        expiry = self._credentials.expiry
        if expiry is None:
            return False
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return expiry <= datetime.now(timezone.utc) + self._refresh_skew

    async def token(self, *, force_refresh: bool = False) -> str:
        async with self._lock:
            if force_refresh or self._needs_refresh():
                await asyncio.to_thread(self._credentials.refresh, self._auth_request)
            token = self._credentials.token
            if not token:
                raise RuntimeError("ADC 未返回 access token")
            return token


def _request_headers(request: Request, token: str) -> dict[str, str]:
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in _REQUEST_HEADERS_TO_DROP
    }
    headers["authorization"] = f"Bearer {token}"
    return headers


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        name: value
        for name, value in response.headers.items()
        if name.lower() not in _HOP_BY_HOP_HEADERS
    }


def _format_json(text: str) -> str:
    if not text:
        return text
    try:
        import json
        parsed = json.loads(text)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        return text


def _decompress_response(body: bytes, headers: Mapping[str, str]) -> bytes:
    content_encoding = headers.get("content-encoding", "").lower()
    if not content_encoding or not body:
        return body
    try:
        if "gzip" in content_encoding:
            import gzip
            return gzip.decompress(body)
        elif "deflate" in content_encoding:
            import zlib
            try:
                return zlib.decompress(body)
            except zlib.error:
                return zlib.decompress(body, -zlib.MAX_WBITS)
        elif "br" in content_encoding:
            try:
                import brotli
                return brotli.decompress(body)
            except ImportError:
                pass
    except Exception as exc:
        LOGGER.warning("Decompression failed: %s", exc)
    return body


def _error(status_code: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type}},
    )


def _authorized(request: Request, expected_key: str | None) -> bool:
    if expected_key is None:
        return True
    supplied = request.headers.get("x-api-key", "")
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    return bool(supplied) and secrets.compare_digest(supplied, expected_key)


def _adc_credentials() -> Credentials:
    credentials, _ = google.auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
    return credentials


def create_app(
    settings: Settings | None = None,
    *,
    credentials: Credentials | None = None,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        config = settings or Settings.from_env()
        adc = credentials or _adc_credentials()

        # requests and httpx both honor HTTP_PROXY/HTTPS_PROXY with trust_env enabled.
        auth_session = requests.Session()
        auth_session.trust_env = True
        auth_request = GoogleAuthRequest(session=auth_session)
        token_provider = AdcTokenProvider(adc, auth_request, config.token_refresh_skew)
        timeout = httpx.Timeout(
            connect=config.connect_timeout,
            read=config.read_timeout,
            write=config.read_timeout,
            pool=config.connect_timeout,
        )
        client = httpx.AsyncClient(
            timeout=timeout,
            trust_env=True,
            follow_redirects=False,
            transport=upstream_transport,
        )
        application.state.settings = config
        application.state.token_provider = token_provider
        application.state.upstream_client = client
        LOGGER.info(
            "Vertex proxy ready: project=%s location=%s upstream=%s",
            config.project,
            config.location,
            config.upstream_host,
        )
        try:
            yield
        finally:
            await client.aclose()
            auth_session.close()

    application = FastAPI(
        title="Vertex AI ADC Proxy",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @application.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/v1/models")
    async def list_models(request: Request):
        config: Settings = request.app.state.settings
        if not _authorized(request, config.proxy_api_key):
            return _error(401, "Invalid proxy API key", "authentication_error")
        now = int(datetime.now(timezone.utc).timestamp())
        return {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "created": now, "owned_by": "google"}
                for model in config.models
            ],
        }

    async def proxy(request: Request, upstream_url: str):
        req_id = secrets.token_hex(4)
        config: Settings = request.app.state.settings
        log_mode = os.environ.get("VERTEX_PROXY_LOG_MODE", "full")

        body = await request.body()

        if not _authorized(request, config.proxy_api_key):
            LOGGER.warning(
                "[%s] Unauthorized request: %s %s",
                req_id,
                request.method,
                request.url,
            )
            if log_mode == "full":
                LOGGER.info(
                    "[%s] Completed response: Status 401 | Body: Invalid proxy API key",
                    req_id,
                )
            return _error(401, "Invalid proxy API key", "authentication_error")

        if body and "application/json" in request.headers.get("content-type", "").lower():
            try:
                import json
                data = json.loads(body)
                if isinstance(data, dict) and "model" in data and isinstance(data["model"], str):
                    model_name = data["model"]
                    if "/" not in model_name:
                        data["model"] = f"google/{model_name}"
                        body = json.dumps(data).encode("utf-8")
            except Exception as exc:
                LOGGER.warning("Failed to preprocess request body model prefix: %s", exc)

        req_headers = {}
        for name, value in request.headers.items():
            name_lower = name.lower()
            if name_lower in {"authorization", "x-api-key", "x-goog-api-key"} or "signature" in name_lower or "auth" in name_lower:
                continue
            req_headers[name] = value

        req_body_str = ""
        if body:
            try:
                req_body_str = body.decode("utf-8", errors="replace")
            except Exception:
                req_body_str = "<binary or undecodable body>"

        if log_mode == "full":
            LOGGER.info(
                "[%s] Received request: %s %s | Headers: %s | Body: %s",
                req_id,
                request.method,
                request.url,
                req_headers,
                _format_json(req_body_str),
            )
        elif log_mode == "messages":
            messages = None
            if body:
                try:
                    import json
                    data = json.loads(body)
                    if isinstance(data, dict) and "messages" in data:
                        messages = data["messages"]
                except Exception:
                    pass
            if messages is not None:
                try:
                    import json
                    msg_str = json.dumps(messages, indent=2, ensure_ascii=False)
                except Exception:
                    msg_str = str(messages)
                LOGGER.info("[%s] Request messages: %s", req_id, msg_str)
            else:
                LOGGER.info("[%s] Request Body: %s", req_id, _format_json(req_body_str))

        client: httpx.AsyncClient = request.app.state.upstream_client
        token_provider: AdcTokenProvider = request.app.state.token_provider

        try:
            token = await token_provider.token()
            upstream_request = client.build_request(
                request.method,
                upstream_url,
                params=request.query_params.multi_items(),
                headers=_request_headers(request, token),
                content=body,
            )
            response = await client.send(upstream_request, stream=True)
            if response.status_code == 401:
                await response.aclose()
                token = await token_provider.token(force_refresh=True)
                upstream_request = client.build_request(
                    request.method,
                    upstream_url,
                    params=request.query_params.multi_items(),
                    headers=_request_headers(request, token),
                    content=body,
                )
                response = await client.send(upstream_request, stream=True)
        except Exception as exc:
            LOGGER.warning("Vertex upstream request failed: %s", exc.__class__.__name__)
            if log_mode == "full":
                LOGGER.info(
                    "[%s] Completed response: Status 502 | Body: Vertex upstream unavailable",
                    req_id,
                )
            return _error(502, "Vertex upstream unavailable", "upstream_error")

        async def logged_stream_generator():
            accumulated_chunks = []
            try:
                async for chunk in response.aiter_raw():
                    accumulated_chunks.append(chunk)
                    yield chunk
            finally:
                if log_mode == "full" or log_mode == "errors":
                    decompressed_body = _decompress_response(b"".join(accumulated_chunks), response.headers)
                    content_type = response.headers.get("content-type", "").lower()
                    charset = "utf-8"
                    if "charset=" in content_type:
                        try:
                            charset = content_type.split("charset=")[-1].strip().split(";")[0]
                        except Exception:
                            charset = "utf-8"
                    
                    try:
                        resp_body_str = decompressed_body.decode(charset, errors="replace")
                    except Exception:
                        resp_body_str = "<binary or undecodable response>"
                    
                    is_cf_error = "content_filter" in resp_body_str
                    if log_mode == "full" or (log_mode == "errors" and is_cf_error):
                        resp_headers = dict(_response_headers(response))
                        LOGGER.info(
                            "[%s] Completed response: Status %s | Headers: %s | Body: %s",
                            req_id,
                            response.status_code,
                            resp_headers,
                            _format_json(resp_body_str),
                        )

        return StreamingResponse(
            logged_stream_generator(),
            status_code=response.status_code,
            headers=_response_headers(response),
            background=BackgroundTask(response.aclose),
        )

    @application.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def openai_proxy(request: Request, path: str):
        config: Settings = request.app.state.settings
        return await proxy(request, f"{config.openai_base_url}/{path}")

    @application.api_route(
        "/vertex/{api_version}/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def native_proxy(request: Request, api_version: str, path: str):
        if api_version not in {"v1", "v1beta1"}:
            return _error(404, "Only v1 and v1beta1 are supported", "invalid_request_error")
        config: Settings = request.app.state.settings
        return await proxy(request, f"{config.native_base_url(api_version)}/{path}")

    return application


app = create_app()
