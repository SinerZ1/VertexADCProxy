# Vertex AI ADC 反向代理

这是一个 Python/FastAPI 反向代理，把本地 OpenAI 兼容请求转发到 Vertex AI。代理本身通过 Application Default Credentials（ADC）获取并自动刷新 access token；客户端不需要接触 Google 凭据。

支持：

- OpenAI 兼容入口 `/v1/*`，包括 `/v1/chat/completions` 的普通及 SSE 流式响应
- Vertex 原生 REST 入口 `/vertex/v1/*` 和 `/vertex/v1beta1/*`
- 自动读取 `GOOGLE_CLOUD_PROJECT`、`VERTEX_LOCATION`
- ADC token 提前刷新；上游返回 401 时强制刷新并重试一次
- ADC 刷新和 Vertex HTTP 请求均读取 `HTTP_PROXY`、`HTTPS_PROXY`、`NO_PROXY`
- 可选的本地 API key，避免代理裸奔

## 安装与启动

需要 Python 3.10 或更高版本。

```bash
cd vertex-proxy
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 你已有的配置
export GOOGLE_CLOUD_PROJECT="your-project-id"
export VERTEX_LOCATION="asia-east1"  # global 也支持
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"

# 强烈建议：保护这个代理；客户端把它当作 OpenAI API key 使用
export VERTEX_PROXY_API_KEY="change-me"

# 可选，仅用于 GET /v1/models 的本地模型列表
export VERTEX_MODELS="google/gemini-2.5-flash,google/gemini-2.5-pro"

uvicorn vertex_proxy.app:app --host 127.0.0.1 --port 8000
```

在本机使用用户 ADC：

```bash
gcloud auth application-default login
```

在 GCE、GKE、Cloud Run 等环境中，可直接使用绑定到工作负载的服务账号。运行身份至少需要调用目标 Vertex 模型的 IAM 权限。

如果代理要对外提供服务，请监听内网地址或放在受认证的网关之后，并设置 `VERTEX_PROXY_API_KEY`。不要把没有客户端认证的实例直接暴露到公网。

## OpenAI 兼容调用

Vertex 的 Gemini 模型名使用 `google/` 前缀；代理不会改写请求体。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer ${VERTEX_PROXY_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-2.5-flash",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

Python OpenAI SDK：

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

`GET /v1/models` 不向 Google 查询，而是返回 `VERTEX_MODELS` 中配置的模型，因而不会产生 Vertex 请求。

## Vertex 原生 REST 调用

本地路径会自动补上项目和区域：

```bash
curl http://127.0.0.1:8000/vertex/v1/publishers/google/models/gemini-2.5-flash:generateContent \
  -H "Authorization: Bearer ${VERTEX_PROXY_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"role": "user", "parts": [{"text": "你好"}]}]
  }'
```

对应上游为：

```text
https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT/locations/LOCATION/publishers/google/models/gemini-2.5-flash:generateContent
```

当 `VERTEX_LOCATION=global` 时，上游主机自动改为 `aiplatform.googleapis.com`。

## 可选配置

| 环境变量 | 默认值 | 说明 |
| --- | ---: | --- |
| `VERTEX_PROXY_API_KEY` | 未设置 | 设置后要求 `Authorization: Bearer ...` 或 `X-API-Key` |
| `VERTEX_MODELS` | 空 | `/v1/models` 返回的逗号分隔模型列表 |
| `VERTEX_CONNECT_TIMEOUT` | `10` | 上游连接超时，单位秒 |
| `VERTEX_READ_TIMEOUT` | `300` | 上游读取超时；设为 `0` 表示不限制 |
| `VERTEX_TOKEN_REFRESH_SKEW` | `300` | token 到期前多少秒刷新 |

健康检查：

```bash
curl http://127.0.0.1:8000/healthz
```

健康检查不会触发 ADC 刷新，也不会访问 Vertex。

## Docker

```bash
docker build -t vertex-adc-proxy .
docker run --rm -p 8000:8000 \
  -e GOOGLE_CLOUD_PROJECT \
  -e VERTEX_LOCATION \
  -e HTTP_PROXY \
  -e HTTPS_PROXY \
  -e VERTEX_PROXY_API_KEY \
  -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
  vertex-adc-proxy
```

生产环境优先使用工作负载身份或附加服务账号，不要把个人 ADC 文件打进镜像。

## 测试

```bash
pip install -e '.[test]'
pytest -q
```

## Windows 桌面 GUI 客户端

本项目额外提供了一个基于 **PyQt6** 封装的 Windows 桌面客户端，具有现代化的深色模式界面。

### 🌟 核心功能
- **一键服务启停**：在独立后台线程中安全启动/停止 FastAPI 代理服务，完全不影响界面交互。
- **本地凭据检测 (ADC)**：自动定位电脑中的 Google ADC 授权文件，并显示当前 ADC 状态。
- **网络连接测试 (GCP Connection Test)**：支持选择模型（如 `gemini-3.5-flash`、`gemini-3.1-flash` 等）并向 Google Cloud 发送 PING 握手包，快速检测凭据和网络代理连通性。
- **网络代理配置**：内置独立的 `HTTP_PROXY` / `HTTPS_PROXY` 设置，可选开启，隔离或借用全局网络代理。
- **自定义 API Key**：支持一键生成及显示/隐藏 `sk-` 开头的本地保护密钥。
- **系统托盘运行**：关闭窗口自动最小化至托盘在后台默默运行，支持托盘气泡消息通知及右键完整菜单。
- **配置自动持久化**：自动保存您的所有偏好设置，下次启动一键复原。

### 🚀 运行与开发

1. **安装 GUI 扩展依赖**：
   ```bash
   pip install -e ".[gui]"
   ```

2. **直接启动桌面客户端**：
   ```bash
   python gui.py
   ```

### 📦 独立单文件打包 (.exe)

运行项目根目录下的自动化打包脚本，即可一键生成无命令行黑窗口的单文件独立运行程序：
```bash
python build_gui.py
```
打包成功后，可在项目根目录下的 `dist/` 文件夹中找到 **`VertexADCProxy.exe`**。双击即可直接在没有 Python 环境的纯净 Windows 操作系统中运行！
