# Vertex AI ADC 反向代理

这是一个 Python/FastAPI 反向代理，用于将本地 OpenAI 兼容的请求转发到 Vertex AI。代理本身通过 Application Default Credentials (ADC) 获取并自动刷新 Access Token，客户端无需直接接触 Google 凭据。

支持功能：

- OpenAI 兼容入口 /v1/*，包括 /v1/chat/completions 的普通及 SSE 流式响应
- Vertex 原生 REST 入口 /vertex/v1/* 和 /vertex/v1beta1/*
- 自动读取 GOOGLE_CLOUD_PROJECT 与 VERTEX_LOCATION
- ADC Token 提前刷新；上游返回 401 时强制刷新并重试一次
- ADC 刷新和 Vertex HTTP 请求均支持读取 HTTP_PROXY、HTTPS_PROXY、NO_PROXY
- 可选的本地 API Key 认证，防止代理接口在本地或网络中未授权访问
- 完善的客户端请求和上游响应日志记录（支持全量、不输出、仅输出 messages 部分 3 种粒度选择，自动美化 JSON 输出）

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

Vertex 的 Gemini 模型名默认使用 google/ 前缀，代理不会改写请求体。

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

当 VERTEX_LOCATION 环境变量设置为 global 时，上游主机地址将自动调整为 aiplatform.googleapis.com。

## 可选配置

| 环境变量 | 默认值 | 说明 |
| --- | ---: | --- |
| `VERTEX_PROXY_API_KEY` | 未设置 | 设置后要求请求携带 `Authorization: Bearer ...` 或 `X-API-Key` 请求头 |
| `VERTEX_MODELS` | 空 | `/v1/models` 接口返回的逗号分隔的模型列表 |
| `VERTEX_PROXY_LOG_MODE` | `full` | 日志记录输出模式（full: 全量输出, messages: 仅输出 request 消息体, none: 不输出） |
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
- 网络代理配置：内置独立的 HTTP_PROXY / HTTPS_PROXY 设置，支持独立开启、隔离或借用全局系统代理。
- 自定义 API Key：支持一键生成并安全显示/隐藏以 sk- 开头的本地保护密钥。
- 动态日志记录配置：针对Request & Response内容支持在“全量输出”、“不输出”及“仅输出 Request messages”之间切换。
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
