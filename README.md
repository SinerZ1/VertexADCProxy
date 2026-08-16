# Vertex AI ADC 反向代理

这是一个 Python/FastAPI 反向代理，用于将本地 OpenAI 兼容请求和 Anthropic Messages 请求转发到 Vertex AI。代理本身通过 Application Default Credentials (ADC) 获取并自动刷新 Access Token，客户端无需直接接触 Google 凭据。

支持功能：

- OpenAI 兼容入口 `/v1/*`，包括 `/v1/chat/completions` 和 OpenAI 新 Responses API `/v1/responses` 的普通及 SSE 流式响应
- Anthropic Messages 入口 `/v1/messages` 和 `/v1/messages/count_tokens`，统一接入 Agent Runtime 协议转换层
- 统一 Agent Runtime 调度：自动规划 Vertex OpenAI 与 Vertex Native (REST) 双后端执行
- 状态化会话与快照恢复：支持 `previous_response_id` 进行 O(1) 状态恢复，透明兼容 `resp_` 与 `msg_` 标识，并支持滑动 TTL 与 LRU 淘汰
- 原生工具与混合工具支持：支持 `googleSearch`、`codeExecution`、`urlContext` 与 `functionDeclarations` 原生组合及真正的 SSE 流式传输
- Strict 混合工具调度与 `pause_turn`：支持 `VERTEX_AGENT_TOOL_MODE=strict` 状态挂起/恢复、输入冲突防护与自主迭代上限控制
- Vertex 原生 REST 入口 /vertex/v1/* 和 /vertex/v1beta1/*
- 自动读取 GOOGLE_CLOUD_PROJECT 与 VERTEX_LOCATION
- ADC Token 提前刷新；上游返回 401 时强制刷新并重试一次
- ADC 刷新和 Vertex HTTP 请求均支持读取 HTTP_PROXY、HTTPS_PROXY、NO_PROXY
- 可选的本地 API Key 认证，防止代理接口在本地或网络中未授权访问
- 完善的客户端请求和上游响应日志记录（支持全量、不输出、仅 messages 和仅错误 4 种模式，自动美化 JSON 输出）

## 安装与启动

支持 Python 3.10 或更高版本的 Windows 环境。

1. 克隆或进入项目目录：
   ```cmd
   cd vertex-proxy
   ```

2. 创建并激活虚拟环境：
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. 安装项目依赖：
   ```cmd
   pip install -e .
   ```

4. 配置环境变量（Windows 命令行 CMD 格式）：
   ```cmd
   set GOOGLE_CLOUD_PROJECT=your-project-id
   set VERTEX_LOCATION=asia-east1
   set HTTP_PROXY=http://127.0.0.1:7890
   set HTTPS_PROXY=http://127.0.0.1:7890

   rem 建议：配置此 API Key 保护代理，客户端将其作为 OpenAI API Key 使用
   set VERTEX_PROXY_API_KEY=change-me

   rem 可选：仅用于 GET /v1/models 的本地模型列表响应
   set VERTEX_MODELS=google/gemini-2.5-flash,google/gemini-2.5-pro

   rem Claude Code 使用 Gemini：Anthropic Messages <-> Vertex Gemini 协议转换
   set VERTEX_ANTHROPIC_BACKEND=gemini
   set VERTEX_ANTHROPIC_GEMINI_MODEL=google/gemini-2.5-pro

   rem 可选：日志记录输出模式 (full:全量, messages:仅消息体, none:不输出)
   set VERTEX_PROXY_LOG_MODE=full
   ```

   如果使用 PowerShell，请使用以下格式配置环境变量：
   ```powershell
   $env:GOOGLE_CLOUD_PROJECT="your-project-id"
   $env:VERTEX_LOCATION="asia-east1"
   $env:HTTP_PROXY="http://127.0.0.1:7890"
   $env:HTTPS_PROXY="http://127.0.0.1:7890"
   $env:VERTEX_PROXY_API_KEY="change-me"
   $env:VERTEX_MODELS="google/gemini-2.5-flash,google/gemini-2.5-pro"
   $env:VERTEX_ANTHROPIC_BACKEND="gemini"
   $env:VERTEX_ANTHROPIC_GEMINI_MODEL="google/gemini-2.5-pro"
   $env:VERTEX_PROXY_LOG_MODE="full"
   ```

5. 启动服务：
   ```cmd
   uvicorn vertex_proxy.app:app --host 127.0.0.1 --port 8000
   ```

在本机使用用户 Application Default Credentials (ADC)：
```cmd
gcloud auth application-default login
```

在 GCE、GKE、Cloud Run 等云端环境中，可以直接使用绑定至工作负载的服务账号。运行身份至少需要拥有调用目标 Vertex 模型的 IAM 权限。

如果代理需要对外提供服务，请监听内网地址或部署在安全网关之后，并务必配置 VERTEX_PROXY_API_KEY。切勿将未设置客户端认证的实例直接暴露在公网上。

## OpenAI 兼容调用

Vertex 的 Gemini 模型名使用 `google/` 前缀；请求未携带 provider 前缀时，代理会自动补上 `google/`。

在 Windows 命令行中使用 curl 发送测试请求（在 CMD 中需要对双引号进行转义）：
```cmd
curl http://127.0.0.1:8000/v1/chat/completions ^
  -H "Authorization: Bearer change-me" ^
  -H "Content-Type: application/json" ^
  -d "{\"model\": \"google/gemini-2.5-flash\", \"messages\": [{\"role\": \"user\", \"content\": \"你好\"}], \"stream\": true}"
```

Python OpenAI SDK 调用示例：
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="change-me",  # 对应 VERTEX_PROXY_API_KEY
)

stream = client.chat.completions.create(
    model="google/gemini-2.5-flash",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)
for chunk in stream:
    print(chunk)
```

GET /v1/models 接口不会向上游 Google 发送查询，而是直接返回 VERTEX_MODELS 环境变量中配置的本地模型列表，从而避免产生额外的 Vertex API 调用请求。

### OpenAI Responses API (/v1/responses)

代理支持 OpenAI 新的 Responses API 端点 `/v1/responses`（如 `client.responses.create(...)`）：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="change-me",
)

# 非流式调用
response = client.responses.create(
    model="google/gemini-2.5-flash",
    instructions="You are a helpful assistant",
    input="Hello!",
)
print(response.output)

# 流式调用 (SSE)
stream = client.responses.create(
    model="google/gemini-2.5-flash",
    instructions="You are a helpful assistant",
    input="Hello!",
    stream=True,
)
for chunk in stream:
    print(chunk)
```

`/v1/responses` 会自动将 `instructions`、`input`（字符串或多轮消息/tool_result/function_call 数组）、`tools`（OpenAI function 格式或简化 name/parameters 格式）、`response_format` 等参数转换为 Vertex OpenAI `chat/completions` 请求，并将上游响应与 SSE 流式事件双向转换为标准的 `response` 对象及 `response.created` / `response.text.delta` / `response.function_call_arguments.delta` / `response.completed` 事件。

## Claude Code / Anthropic Messages 调用

不需要启用 Vertex Claude。默认推荐的调用链是：

```text
Claude Code
  -> Anthropic Messages API (/v1/messages)
  -> 本代理进行消息、工具调用和 SSE 事件双向转换
  -> Vertex OpenAI 兼容端点 (/endpoints/openapi/chat/completions)
  -> Gemini
```

转换层支持 system 指令、文本和图片内容、Claude Code 工具 schema、`tool_use` / `tool_result` 多轮调用、普通响应、流式响应、停止原因与 token 用量。流式期间代理会生成 Anthropic SSE 事件，并在 Gemini 暂时无输出时发送 `ping`。`/v1/messages/count_tokens` 会转到 Gemini 原生 `countTokens` API。

先配置实际调用的 Gemini 模型：

```powershell
$env:GOOGLE_CLOUD_PROJECT="your-project-id"
$env:VERTEX_LOCATION="global"
$env:VERTEX_PROXY_API_KEY="change-me"
$env:VERTEX_ANTHROPIC_BACKEND="gemini"
$env:VERTEX_ANTHROPIC_GEMINI_MODEL="google/gemini-2.5-pro"

$env:ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
$env:ANTHROPIC_AUTH_TOKEN="change-me"
$env:ANTHROPIC_CUSTOM_MODEL_OPTION="claude-code-gemini"
$env:ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="Gemini via Vertex Proxy"
$env:ANTHROPIC_MODEL="claude-code-gemini"

claude
```

`ANTHROPIC_MODEL` 只决定 Claude Code 启动时选择并发送给 Gateway 的客户端模型名；Gemini 模式下代理不会把它当作 Vertex 路由目标，而是始终调用 `VERTEX_ANTHROPIC_GEMINI_MODEL`。所以这三个 `ANTHROPIC_CUSTOM_*` / `ANTHROPIC_MODEL` 设置都是客户端显示和选择层面的可选配置，不设置也不影响代理最终调用的 Gemini。若没有单独设置 `VERTEX_ANTHROPIC_GEMINI_MODEL`，代理会使用 `VERTEX_MODELS` 中的第一个 Gemini 模型。

这里必须使用 `ANTHROPIC_BASE_URL`，不要同时设置 `CLAUDE_CODE_USE_VERTEX=1`；后者会绕过本项目的 Anthropic 协议转换层。

如果以后启用了 Vertex Claude，也可以切换到原始协议直通模式：

```powershell
$env:VERTEX_ANTHROPIC_BACKEND="claude"
$env:ANTHROPIC_MODEL="claude-sonnet-4-6"
```

Claude 直通模式会根据 `stream` 调用 Vertex `rawPredict` 或 `streamRawPredict`，并支持以下模型名规则：

- 当前 Google Cloud Claude 模型 ID（例如 `claude-sonnet-4-6`）会直接使用。
- Anthropic 带发布日期的模型名会转换为对应 Google Cloud ID，例如 `claude-sonnet-4-5-20250929` 转为 `claude-sonnet-4-5`。
- 特殊名称或企业内部别名可通过 `VERTEX_ANTHROPIC_MODEL_MAP` 配置，多个映射使用逗号分隔，例如 `team-sonnet=claude-sonnet-4-6,team-haiku=claude-haiku-4-5`。

Claude 直通模式下，如果 Vertex 不支持 Claude Code 发送的实验性字段，可暂时设置：

```powershell
$env:CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS="1"
```

Claude Code 请求可能包含源码和工具结果。生产环境建议将 `VERTEX_PROXY_LOG_MODE` 设置为 `errors` 或 `none`，避免记录完整请求和响应。

协议参考：[Claude Code LLM Gateway protocol](https://code.claude.com/docs/en/llm-gateway-protocol)、[Vertex OpenAI Chat Completions](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-chat-completions-streaming)、[Gemini function calling](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling)、[Gemini countTokens](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/projects.locations.publishers.models/countTokens)。

## Vertex 原生 REST 调用

本地请求路径会自动补全 GCP 项目和区域。

测试原生 REST 接口（以 Windows CMD 格式为例）：
```cmd
curl http://127.0.0.1:8000/vertex/v1/publishers/google/models/gemini-2.5-flash:generateContent ^
  -H "Authorization: Bearer change-me" ^
  -H "Content-Type: application/json" ^
  -d "{\"contents\": [{\"role\": \"user\", \"parts\": [{\"text\": \"你好\"}]}]}"
```

对应的上游目标地址为：
```text
https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/gemini-2.5-flash:generateContent
```

当 `VERTEX_LOCATION` 设置为 `global` 时，上游主机地址会调整为 `aiplatform.googleapis.com`；设置为 `us` 或 `eu` 时会使用对应的 Google Cloud multi-region host。

## 可选配置

| 环境变量 | 默认值 | 说明 |
| --- | ---: | --- |
| `VERTEX_PROXY_API_KEY` | 未设置 | 设置后要求请求携带 `Authorization: Bearer ...` 或 `X-API-Key` 请求头 |
| `VERTEX_MODELS` | 空 | `/v1/models` 接口返回的逗号分隔的模型列表 |
| `VERTEX_AGENT_TOOL_MODE` | `best_effort` | 代理工具调度模式：`best_effort`（直接组合 Native 工具）或 `strict`（严格混合工具状态机与挂起） |
| `VERTEX_AGENT_MAX_ITERATIONS` | `5` | Strict 模式下服务侧自主迭代轮次上限，超限触发 `pause_turn` |
| `VERTEX_ANTHROPIC_BACKEND` | `claude` | `/v1/messages` 的后端：`gemini` 启用协议转换，`claude` 使用 Vertex Claude 直通 |
| `VERTEX_ANTHROPIC_GEMINI_MODEL` | 空 | Gemini 转换模式实际调用的模型；未设置时选取 `VERTEX_MODELS` 中第一个 Gemini 模型 |
| `VERTEX_ANTHROPIC_MODEL_MAP` | 空 | Anthropic 客户端模型名到 Google Cloud Claude 模型 ID 的逗号分隔映射 |
| `VERTEX_PROXY_LOG_MODE` | `full` | 日志记录输出模式（full: 全量输出, messages: 仅输出 request 消息体, errors: 仅输出非 200 或 content_filter 的 Response 错误, none: 不输出） |
| `VERTEX_CONNECT_TIMEOUT` | `10` | 连接上游服务的超时时间，单位为秒 |
| `VERTEX_READ_TIMEOUT` | `300` | 读取上游服务的超时时间，单位为秒；设为 `0` 表示无限制 |
| `VERTEX_TOKEN_REFRESH_SKEW` | `300` | Token 到期前多少秒执行主动刷新操作 |

健康检查接口：
```cmd
curl http://127.0.0.1:8000/healthz
```
健康检查接口不会触发 ADC 刷新，也不会向上游 Vertex 发送任何请求。

## Docker 容器化运行

在 Windows Docker 环境下运行：
```cmd
docker build -t vertex-adc-proxy .
docker run --rm -p 8000:8000 ^
  -e GOOGLE_CLOUD_PROJECT ^
  -e VERTEX_LOCATION ^
  -e HTTP_PROXY ^
  -e HTTPS_PROXY ^
  -e VERTEX_PROXY_API_KEY ^
  -e VERTEX_MODELS ^
  -e VERTEX_ANTHROPIC_BACKEND ^
  -e VERTEX_ANTHROPIC_GEMINI_MODEL ^
  -e VERTEX_ANTHROPIC_MODEL_MAP ^
  -e VERTEX_PROXY_LOG_MODE ^
  -v "%USERPROFILE%/.config/gcloud:/root/.config/gcloud:ro" ^
  vertex-adc-proxy
```
生产环境部署时请优先使用工作负载身份（Workload Identity）或绑定服务账号，切勿将个人 ADC 凭证文件直接打包到 Docker 镜像中。

## 本地测试

```cmd
pip install -e .[test]
.\.venv\Scripts\pytest -q
```

## Windows 桌面 GUI 客户端

本项目额外提供了一个基于 PyQt6 封装的 Windows 桌面客户端，支持现代化深色模式用户界面。

### 核心功能

- 一键服务启停：在独立后台线程中安全启动/停止 FastAPI 代理服务，完全不影响界面交互和流畅度。
- 本地凭据检测 (ADC)：自动扫描电脑中的 Google ADC 授权文件，实时显示当前 ADC 的有效状态。
- 网络连接测试 (GCP Connection Test)：支持在界面中快速选择模型向 Google Cloud 发送 PING 握手包，快速检测本地凭据和网络代理连通性。
- Claude Code 后端设置：可在 GUI 中选择 Gemini 协议转换或 Vertex Claude 直通，并独立指定 Claude Code 实际使用的 Gemini 模型。
- 网络代理配置：内置独立的 HTTP_PROXY / HTTPS_PROXY 设置，支持独立开启、隔离或借用全局系统代理。
- 自定义 API Key：支持一键生成并安全显示/隐藏以 sk- 开头的本地保护密钥。
- 动态日志记录配置：针对Request & Response内容支持在“全量输出”、“不输出”、“仅输出 Request messages”以及“仅输出 Response 错误”之间切换。
- 系统托盘运行：关闭窗口时自动最小化至系统右下角托盘在后台静默运行，支持托盘气泡通知与完整的右键上下文菜单。
- 配置自动持久化：自动在本地保存所有偏好设置，下次启动时自动加载并一键复原。

### 运行与开发

1. 安装 GUI 扩展依赖：
   ```cmd
   pip install -e .[gui]
   ```

2. 直接启动桌面客户端：
   ```cmd
   python gui.py
   ```

### 独立单文件打包 (.exe)

运行项目根目录下的自动化打包脚本，即可一键生成无命令行黑窗口的单文件独立运行程序：
```cmd
python build_gui.py
```
打包成功后，可在项目根目录下的 `dist` 文件夹中找到 `VertexADCProxy.exe`。双击即可直接在没有 Python 环境的纯净 Windows 操作系统中运行。
