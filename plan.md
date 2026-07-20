# Vertex AI ADC Proxy - Windows 桌面客户端封装计划书

本计划书旨在为 `vertex-proxy` 项目扩展一个基于 **PyQt6** 的现代化 Windows 桌面客户端（带有图形界面 GUI），支持配置持久化、ADC 状态自动检测、一键连接测试、系统托盘隐藏以及单文件打包成独立 `.exe`。

---

## 🎨 1. GUI 界面设计与排版

我们将使用 **PyQt6** 构建一个紧凑、专业、意在契合 Windows 11 Fluent 风格的深色主题窗口：
- **主题风格**：背景深碳灰 (`#1e1e1e`)、输入框与按键中灰 (`#2d2d2d`)、圆角边框、高对比白色文字、淡蓝色 (`#0078d4`) 动作高亮。
- **界面分区**：
  1. **服务配置区 (Service Configuration)**：
     - **反代端口**：`QLineEdit`（限制仅输入数字，范围 1~65535），默认值 `10101`。
     - **API Key**：`QLineEdit`（支持密码模式，一键切换明文/密文），配备「生成 API Key」按钮（自动生成 `sk-` 开头的 16 位随机 Hex 密钥，格式如 `sk-6e19c25f...`）。
  2. **GCP 与 ADC 凭证区 (GCP & ADC Settings)**：
     - **GCP 项目 ID (GOOGLE_CLOUD_PROJECT)**：`QLineEdit`，可手动填入，留空时自动读取环境变量。
     - **Vertex 区域 (VERTEX_LOCATION)**：可编辑下拉框 `QComboBox`，默认可选 `us-central1`, `asia-east1`, `global` 等，留空自动读取环境变量。
     - **服务账号 JSON**：支持点击「浏览」按钮，弹出 Windows 文件选择框指定 `GOOGLE_APPLICATION_CREDENTIALS` 本地 JSON 路径。
     - **ADC 文件状态栏**：
       - 启动时与运行中自动搜索 ADC 凭证路径：优先读取 `GOOGLE_APPLICATION_CREDENTIALS` 环境变量或用户配置的文件；若未找到，再检查 Windows 标准路径：`C:\Users\%username%\AppData\Roaming\gcloud\application_default_credentials.json`。
       - **状态显示**：绿色文字显示「已找到 ADC 凭证：[路径]」或红色文字警告「未找到 ADC 凭证，请登录/指定密钥」。
     - **连接测试模块**：
       - **测试模型下拉框**：默认支持 `gemini-3.5-flash`（推荐）、`gemini-3.1-flash-lite`、`gemini-3.1-flash`、`gemini-3.1-pro-preview`。支持用户手动输入自定义模型。
       - **「测试连接」按钮**：在 ADC 文件存在时启用。点击后在子线程发起一次上游 API 握手，验证凭据、项目 ID、区域、代理以及所选模型的连通性，并弹出成功/失败弹窗（含具体的错误原因）。
  3. **网络代理区 (Network Proxy Settings)**：
     - **使用代理** 复选框：选中时启用代理输入。
     - **HTTP_PROXY** / **HTTPS_PROXY**：输入框，可填入如 `http://127.0.0.1:7890`。留空则自动检测并读取系统环境变量。
  4. **服务状态与控制 (Service Controller & Log Console)**：
     - **控制按钮**：「启动代理」与「停止代理」大按钮。
     - **日志控制台**：程序员专属等宽字体 (`Consolas`) 的只读日志框，配备「清除日志」按钮，自动滚动，流式呈现 API 调用状况。
     - **一键复制**：「复制本地 API Base URL` 按钮（例如：复制 `http://127.0.0.1:10101/v1` 到剪贴板）。

---

## ⚙️ 2. 技术架构与核心算法实现

### 2.1 ADC 文件智能定位逻辑
优先读取环境变量或用户指定的密钥文件；若未找到，再扫描 AppData 路径：
`C:\Users\<username>\AppData\Roaming\gcloud\application_default_credentials.json`。

### 2.2 上游连通性异步测试逻辑
在独立线程中执行，加载 ADC 凭证获取 access token，并向 Vertex AI 的特定模型发送一笔轻量级的生文（generateContent）请求。若返回 HTTP 200，说明项目、区域、凭据及网络链路畅通。

### 2.3 异步 FastAPI/Uvicorn 服务器线程
在子线程（`QThread`）中运行 Uvicorn 服务器。当主界面退出或用户点击“停止”时，修改 `server.should_exit = True` 触发其优雅停机。

### 2.4 日志拦截与 ANSI 过滤
继承 `logging.Handler` 实现日志数据中转，通过 Qt 信号传递日志文本，并引入正则过滤，剔除 Uvicorn 默认带有的终端彩色字符（如 `\x1b[32m`）。

### 2.5 系统托盘集成
关闭窗口时，默认将应用隐藏至系统托盘，并提供右键功能菜单（显示窗口、启动/停止代理、彻底退出）。

---

## 📦 3. 打包单文件 exe 方案 (PyInstaller)
编写 `build_gui.py` 自动化打包脚本。使用 `--onefile` 和 `--noconsole` 参数，利用 `--collect-all` 聚合 `uvicorn`, `fastapi`, `starlette`, `httpx` 等核心隐藏依赖，保证打包的 exe 文件在纯净 Windows 电脑上可以一键双击运行。
