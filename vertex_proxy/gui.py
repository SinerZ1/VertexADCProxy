from __future__ import annotations

import sys
import os
import json
import re
import secrets
import logging
import asyncio
from pathlib import Path
from typing import Mapping

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QLineEdit, QComboBox, QCheckBox, QPushButton, 
    QGroupBox, QTextEdit, QFileDialog, QMessageBox, QSystemTrayIcon, QMenu, QStyle
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QAction, QIntValidator

# Logger setup
LOGGER = logging.getLogger("vertex_proxy_gui")

# Regex to strip ANSI escape codes from logging outputs
_ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub('', text)


def locate_adc_file(configured_path: str | None = None) -> tuple[bool, str]:
    """Locate Google Application Default Credentials (ADC) file."""
    # 1. Check user configured credentials path
    if configured_path:
        p = Path(configured_path)
        if p.exists() and p.is_file():
            return True, str(p)
            
    # 2. Check GOOGLE_APPLICATION_CREDENTIALS environment variable
    env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path:
        p = Path(env_path)
        if p.exists() and p.is_file():
            return True, str(p)
            
    # 3. Check Windows AppData standard path
    appdata = os.environ.get("APPDATA") # C:\Users\<username>\AppData\Roaming
    if appdata:
        p = Path(appdata) / "gcloud" / "application_default_credentials.json"
        if p.exists() and p.is_file():
            return True, str(p)
            
    # 4. Try home directory AppData path as fallback
    p_home = Path.home() / "AppData" / "Roaming" / "gcloud" / "application_default_credentials.json"
    if p_home.exists() and p_home.is_file():
        return True, str(p_home)
        
    return False, ""


def get_config_path() -> Path:
    """Gets the path to the configuration file, prioritizing local folder."""
    local_path = Path("proxy_config.json")
    try:
        # Check if writable
        if not local_path.exists():
            local_path.touch()
        return local_path
    except OSError:
        return Path.home() / ".vertex_proxy_config.json"


def load_config() -> dict:
    """Loads configuration from JSON file."""
    path = get_config_path()
    if path.exists() and path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            LOGGER.warning("Failed to load config: %s", e)
    return {}


def save_config(config_dict: dict) -> None:
    """Saves configuration to JSON file."""
    path = get_config_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOGGER.error("Failed to save config: %s", e)


class LogSignaler(QObject):
    log_written = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, signaler: LogSignaler):
        super().__init__()
        self.signaler = signaler

    def emit(self, record):
        try:
            msg = self.format(record)
            clean_msg = strip_ansi(msg)
            self.signaler.log_written.emit(clean_msg + "\n")
        except Exception:
            self.handleError(record)


# Beautiful fluent dark-mode theme stylesheet
DARK_STYLE = """
QMainWindow {
    background-color: #1a1a1a;
}
QWidget {
    font-family: "Segoe UI", "Segoe UI CLI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
    color: #e3e3e3;
}
QLabel {
    color: #e3e3e3;
}
QLabel#title_label {
    font-size: 18px;
    font-weight: bold;
    color: #ffffff;
}
QLabel#subtitle_label {
    font-size: 12px;
    color: #888888;
}
QLineEdit, QComboBox {
    background-color: #262626;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 5px 8px;
    color: #ffffff;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #0078d4;
    background-color: #2d2d2d;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 25px;
    border-left: none;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #cccccc;
    width: 0;
    height: 0;
}
QComboBox::down-arrow:on {
    border-top: none;
    border-bottom: 5px solid #0078d4;
}
QComboBox QAbstractItemView {
    background-color: #262626;
    border: 1px solid #3d3d3d;
    selection-background-color: #0078d4;
    selection-color: #ffffff;
    color: #ffffff;
    outline: none;
}
QCheckBox {
    spacing: 5px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    background-color: #262626;
    border: 1px solid #3d3d3d;
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background-color: #0078d4;
    border: 1px solid #0078d4;
}
QPushButton {
    background-color: #2e2e2e;
    border: 1px solid #444444;
    border-radius: 4px;
    padding: 5px 12px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #383838;
    border: 1px solid #555555;
}
QPushButton:pressed {
    background-color: #252525;
}
QPushButton#btn_start {
    background-color: #0078d4;
    color: #ffffff;
    border: 1px solid #0078d4;
}
QPushButton#btn_start:hover {
    background-color: #1084d9;
}
QPushButton#btn_start:pressed {
    background-color: #006cc1;
}
QPushButton#btn_stop {
    background-color: #d83b01;
    color: #ffffff;
    border: 1px solid #d83b01;
}
QPushButton#btn_stop:hover {
    background-color: #e84a1a;
}
QPushButton#btn_stop:pressed {
    background-color: #c43300;
}
QGroupBox {
    border: 1px solid #333333;
    border-radius: 6px;
    margin-top: 15px;
    padding-top: 15px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 5px;
    color: #0078d4;
    font-weight: bold;
}
QTextEdit#log_view {
    background-color: #0d0d0d;
    border: 1px solid #262626;
    border-radius: 4px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    color: #00ff00;
    padding: 5px;
}
QScrollBar:vertical {
    border: none;
    background: #1a1a1a;
    width: 8px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #444444;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #555555;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    border: none;
    background: none;
}
"""


class ConnectionTestThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, project: str, location: str, model: str, creds_path: str, use_proxy: bool, http_proxy: str, https_proxy: str):
        super().__init__()
        self.project = project
        self.location = location
        self.model = model
        self.creds_path = creds_path
        self.use_proxy = use_proxy
        self.http_proxy = http_proxy
        self.https_proxy = https_proxy

    def run(self):
        # Temp override env variables
        original_env = {}
        vars_to_set = {}
        if self.creds_path:
            vars_to_set["GOOGLE_APPLICATION_CREDENTIALS"] = self.creds_path

        if self.use_proxy:
            if self.http_proxy:
                vars_to_set["HTTP_PROXY"] = self.http_proxy
                vars_to_set["http_proxy"] = self.http_proxy
            if self.https_proxy:
                vars_to_set["HTTPS_PROXY"] = self.https_proxy
                vars_to_set["https_proxy"] = self.https_proxy
        else:
            vars_to_set["HTTP_PROXY"] = ""
            vars_to_set["HTTPS_PROXY"] = ""
            vars_to_set["http_proxy"] = ""
            vars_to_set["https_proxy"] = ""

        for k, v in vars_to_set.items():
            original_env[k] = os.environ.get(k)
            if v:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

        try:
            import google.auth
            from google.auth.transport.requests import Request as GoogleAuthRequest
            import requests

            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            try:
                credentials, project_from_creds = google.auth.default(scopes=scopes)
            except Exception as e:
                self.finished.emit(False, f"获取 Google ADC 默认凭证失败: {e}\n\n请验证您的 ADC 文件。")
                return

            resolved_project = self.project or project_from_creds or os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not resolved_project:
                self.finished.emit(False, "缺少 GCP 项目 ID。请在界面输入 GOOGLE_CLOUD_PROJECT。")
                return

            resolved_location = self.location or os.environ.get("VERTEX_LOCATION") or "us-central1"

            # Refresh token
            try:
                auth_request = GoogleAuthRequest()
                credentials.refresh(auth_request)
                token = credentials.token
                if not token:
                    self.finished.emit(False, "刷新 token 失败: ADC 未返回 Access Token。")
                    return
            except Exception as e:
                self.finished.emit(False, f"刷新 Access Token 失败: {e}\n\n原因可能是网络不通，或 ADC 凭证已过期/无效。")
                return

            # Construct Native Vertex endpoint
            host = "aiplatform.googleapis.com" if resolved_location == "global" else f"{resolved_location}-aiplatform.googleapis.com"
            url = f"https://{host}/v1/projects/{resolved_project}/locations/{resolved_location}/publishers/google/models/{self.model}:generateContent"

            # Setup session
            session = requests.Session()
            if self.use_proxy:
                session.proxies = {
                    "http": self.http_proxy,
                    "https": self.https_proxy,
                }
            else:
                session.trust_env = False

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            body = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "ping"}]
                    }
                ]
            }

            try:
                response = session.post(url, json=body, headers=headers, timeout=12)
            except Exception as e:
                self.finished.emit(False, f"向 Vertex AI 发送请求网络异常:\n{e}\n\n请检查代理和网络设置。")
                return

            if response.status_code == 200:
                self.finished.emit(True, f"测试成功！\n\n- 项目 ID: {resolved_project}\n- 区域: {resolved_location}\n- 选用模型: {self.model}\n- 响应成功: HTTP 200 OK")
            else:
                self.finished.emit(False, f"上游 API 响应错误 (HTTP {response.status_code})\n\n内容: {response.text}\n\n请检查项目、区域及模型是否开通或匹配。")

        finally:
            # Restore original env variables
            for k, v in original_env.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)


class UvicornServerThread(QThread):
    status_changed = pyqtSignal(str, str)

    def __init__(self, port: int, api_key: str, project: str, location: str, models: str, connect_timeout: float, read_timeout: float, use_proxy: bool, http_proxy: str, https_proxy: str, creds_path: str):
        super().__init__()
        self.port = port
        self.api_key = api_key
        self.project = project
        self.location = location
        self.models = models
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.use_proxy = use_proxy
        self.http_proxy = http_proxy
        self.https_proxy = https_proxy
        self.creds_path = creds_path
        self.server = None

    def run(self):
        original_env = {}
        env_vars = {}
        if self.project:
            env_vars["GOOGLE_CLOUD_PROJECT"] = self.project
        if self.location:
            env_vars["VERTEX_LOCATION"] = self.location
        if self.api_key:
            env_vars["VERTEX_PROXY_API_KEY"] = self.api_key
        else:
            env_vars["VERTEX_PROXY_API_KEY"] = ""
            
        env_vars["VERTEX_MODELS"] = self.models
        env_vars["VERTEX_CONNECT_TIMEOUT"] = str(self.connect_timeout)
        env_vars["VERTEX_READ_TIMEOUT"] = str(self.read_timeout)
        
        if self.creds_path:
            env_vars["GOOGLE_APPLICATION_CREDENTIALS"] = self.creds_path

        if self.use_proxy:
            if self.http_proxy:
                env_vars["HTTP_PROXY"] = self.http_proxy
                env_vars["http_proxy"] = self.http_proxy
            if self.https_proxy:
                env_vars["HTTPS_PROXY"] = self.https_proxy
                env_vars["https_proxy"] = self.https_proxy
        else:
            env_vars["HTTP_PROXY"] = ""
            env_vars["HTTPS_PROXY"] = ""
            env_vars["http_proxy"] = ""
            env_vars["https_proxy"] = ""

        # Set environment variables
        for k, v in env_vars.items():
            original_env[k] = os.environ.get(k)
            if v:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

        try:
            import uvicorn
            import asyncio
            from vertex_proxy.app import Settings, create_app
            
            try:
                settings = Settings.from_env()
                app = create_app(settings)
            except Exception as e:
                self.status_changed.emit("error", f"初始化配置失败: {e}")
                return

            config = uvicorn.Config(
                app=app,
                host="127.0.0.1",
                port=self.port,
                log_config=None,
                loop="asyncio"
            )
            self.server = uvicorn.Server(config)
            
            self.status_changed.emit("running", "")
            
            # Start asyncio loop inside the QThread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.server.serve())
            except Exception as e:
                self.status_changed.emit("error", f"服务器运行异常: {e}")
                return
                
        except Exception as e:
            self.status_changed.emit("error", f"启动服务器失败 (可能是端口被占用): {e}")
            return
        finally:
            self.status_changed.emit("stopped", "")
            # Restore environment variables
            for k, v in original_env.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)

    def stop(self):
        if self.server:
            self.server.should_exit = True


class VertexProxyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.server_thread = None
        self.test_thread = None
        self.log_signaler = LogSignaler()
        self.log_signaler.log_written.connect(self.append_log)
        
        # Connect log signaler to logging
        setup_logging(self.log_signaler)
        
        self.init_ui()
        self.load_settings()
        self.update_adc_status()
        self.setup_tray()

    def init_ui(self):
        self.setWindowTitle("Vertex AI ADC Proxy 桌面版")
        self.resize(800, 650)
        self.setStyleSheet(DARK_STYLE)

        # Set Window icon using standard system icons to avoid missing external file
        self.setWindowIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))

        # Main Widget and Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Left Column: Configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        main_layout.addWidget(left_panel, stretch=4)

        # Title
        title_widget = QWidget()
        title_vbox = QVBoxLayout(title_widget)
        title_vbox.setContentsMargins(0, 0, 0, 5)
        title_vbox.setSpacing(3)
        self.lbl_title = QLabel("Vertex AI ADC Proxy")
        self.lbl_title.setObjectName("title_label")
        self.lbl_subtitle = QLabel("本地 OpenAI 兼容反向代理服务 (FastAPI + PyQt6)")
        self.lbl_subtitle.setObjectName("subtitle_label")
        title_vbox.addWidget(self.lbl_title)
        title_vbox.addWidget(self.lbl_subtitle)
        left_layout.addWidget(title_widget)

        # Group 1: Service Configuration
        grp_service = QGroupBox("服务配置 (Service Configuration)")
        grid_service = QGridLayout(grp_service)
        grid_service.setSpacing(10)
        
        grid_service.addWidget(QLabel("反代端口:"), 0, 0)
        self.txt_port = QLineEdit("10101")
        self.txt_port.setValidator(QIntValidator(1, 65535))
        self.txt_port.setToolTip("本地监听端口，默认 10101")
        grid_service.addWidget(self.txt_port, 0, 1)

        grid_service.addWidget(QLabel("API Key:"), 1, 0)
        key_hbox = QHBoxLayout()
        self.txt_api_key = QLineEdit()
        self.txt_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_api_key.setPlaceholderText("留空则免密码访问（不推荐）")
        self.txt_api_key.setToolTip("可选本地 API Key，防止代理被盗用。客户端作为 Bearer API Key 填入")
        key_hbox.addWidget(self.txt_api_key)
        
        self.btn_toggle_key = QPushButton("显示")
        self.btn_toggle_key.clicked.connect(self.toggle_api_key_visibility)
        key_hbox.addWidget(self.btn_toggle_key)
        
        self.btn_gen_key = QPushButton("生成")
        self.btn_gen_key.clicked.connect(self.generate_api_key)
        key_hbox.addWidget(self.btn_gen_key)
        grid_service.addLayout(key_hbox, 1, 1)
        
        left_layout.addWidget(grp_service)

        # Group 2: GCP & ADC Settings
        grp_gcp = QGroupBox("Google Cloud 凭据与 ADC")
        grid_gcp = QGridLayout(grp_gcp)
        grid_gcp.setSpacing(10)

        grid_gcp.addWidget(QLabel("项目 ID (GCP Project):"), 0, 0)
        self.txt_project = QLineEdit()
        self.txt_project.setPlaceholderText("留空自动读取环境变量 GOOGLE_CLOUD_PROJECT")
        self.txt_project.setToolTip("Google Cloud 项目 ID (GOOGLE_CLOUD_PROJECT)")
        grid_gcp.addWidget(self.txt_project, 0, 1)

        grid_gcp.addWidget(QLabel("Vertex 区域 (Location):"), 1, 0)
        self.cmb_location = QComboBox()
        self.cmb_location.setEditable(True)
        self.cmb_location.addItems(["us-central1", "asia-east1", "us-east4", "europe-west1", "global"])
        self.cmb_location.setPlaceholderText("留空自动读取 VERTEX_LOCATION")
        self.cmb_location.setToolTip("Vertex AI 服务区域，如 us-central1 或 global")
        grid_gcp.addWidget(self.cmb_location, 1, 1)

        grid_gcp.addWidget(QLabel("服务账号 JSON 文件:"), 2, 0)
        creds_hbox = QHBoxLayout()
        self.txt_creds_path = QLineEdit()
        self.txt_creds_path.setPlaceholderText("可选。指定则设置 GOOGLE_APPLICATION_CREDENTIALS")
        self.txt_creds_path.setToolTip("选择服务账号的 .json 密钥文件以自定义 ADC 凭据")
        creds_hbox.addWidget(self.txt_creds_path)
        btn_browse_creds = QPushButton("浏览")
        btn_browse_creds.clicked.connect(self.browse_creds_file)
        creds_hbox.addWidget(btn_browse_creds)
        grid_gcp.addLayout(creds_hbox, 2, 1)

        # ADC status labels
        grid_gcp.addWidget(QLabel("本地 ADC 凭据状态:"), 3, 0)
        self.lbl_adc_status = QLabel("检测中...")
        self.lbl_adc_status.setObjectName("adc_status_label")
        self.lbl_adc_status.setWordWrap(True)
        grid_gcp.addWidget(self.lbl_adc_status, 3, 1)

        # Connection testing controls
        grid_gcp.addWidget(QLabel("网络连接测试模型:"), 4, 0)
        test_hbox = QHBoxLayout()
        self.cmb_test_model = QComboBox()
        self.cmb_test_model.setEditable(True)
        self.cmb_test_model.addItems([
            "gemini-3.1-flash-lite",
            "gemini-3.1-flash",
            "gemini-3.1-pro-preview",
            "gemini-3.5-flash",
        ])
        self.cmb_test_model.setToolTip("用于测试 GCP 连通性的模型，支持自定义输入")
        test_hbox.addWidget(self.cmb_test_model)
        
        self.btn_test_conn = QPushButton("测试连接")
        self.btn_test_conn.clicked.connect(self.test_connectivity)
        self.btn_test_conn.setToolTip("向 Google Cloud 目标模型发送轻量级请求，测试 ADC 凭据与网络代理是否正常")
        test_hbox.addWidget(self.btn_test_conn)
        grid_gcp.addLayout(test_hbox, 4, 1)

        left_layout.addWidget(grp_gcp)

        # Group 3: Network Proxy Settings
        grp_proxy = QGroupBox("网络代理设置 (Network Proxy)")
        grid_proxy = QGridLayout(grp_proxy)
        grid_proxy.setSpacing(10)

        self.chk_use_proxy = QCheckBox("启用自定义网络代理")
        self.chk_use_proxy.setToolTip("选中此项启用自定义代理；否则程序将不通过代理或读取系统全局代理")
        self.chk_use_proxy.toggled.connect(self.toggle_proxy_fields)
        grid_proxy.addWidget(self.chk_use_proxy, 0, 0, 1, 2)

        grid_proxy.addWidget(QLabel("HTTP_PROXY:"), 1, 0)
        self.txt_http_proxy = QLineEdit()
        self.txt_http_proxy.setPlaceholderText("例如 http://127.0.0.1:7890 (留空读取系统环境变量)")
        grid_proxy.addWidget(self.txt_http_proxy, 1, 1)

        grid_proxy.addWidget(QLabel("HTTPS_PROXY:"), 2, 0)
        self.txt_https_proxy = QLineEdit()
        self.txt_https_proxy.setPlaceholderText("例如 http://127.0.0.1:7890 (留空读取系统环境变量)")
        grid_proxy.addWidget(self.txt_https_proxy, 2, 1)

        left_layout.addWidget(grp_proxy)
        
        # Spacer
        left_layout.addStretch()

        # Right Column: Service Controls & Log view
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        main_layout.addWidget(right_panel, stretch=5)

        # Status and Controls Area
        ctrl_panel = QWidget()
        ctrl_hbox = QHBoxLayout(ctrl_panel)
        ctrl_hbox.setContentsMargins(0, 0, 0, 0)
        ctrl_hbox.setSpacing(10)

        self.btn_start = QPushButton(" 启动代理 ")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self.start_proxy)
        ctrl_hbox.addWidget(self.btn_start)

        self.btn_stop = QPushButton(" 停止代理 ")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_proxy)
        ctrl_hbox.addWidget(self.btn_stop)

        self.btn_copy_url = QPushButton("复制 API URL")
        self.btn_copy_url.clicked.connect(self.copy_api_url)
        ctrl_hbox.addWidget(self.btn_copy_url)

        right_layout.addWidget(ctrl_panel)

        # Server Status Bar label
        status_bar = QHBoxLayout()
        status_bar.addWidget(QLabel("当前状态: "))
        self.lbl_server_status = QLabel("已停止")
        self.lbl_server_status.setStyleSheet("color: #888888; font-weight: bold;")
        status_bar.addWidget(self.lbl_server_status)
        status_bar.addStretch()
        right_layout.addLayout(status_bar)

        # Group 4: Log console
        grp_logs = QGroupBox("实时日志输出 (Live Logs)")
        logs_vbox = QVBoxLayout(grp_logs)
        logs_vbox.setContentsMargins(8, 12, 8, 8)
        
        self.log_view = QTextEdit()
        self.log_view.setObjectName("log_view")
        self.log_view.setReadOnly(True)
        logs_vbox.addWidget(self.log_view)

        # Log bottom action buttons
        logs_actions = QHBoxLayout()
        self.chk_auto_scroll = QCheckBox("自动滚动")
        self.chk_auto_scroll.setChecked(True)
        logs_actions.addWidget(self.chk_auto_scroll)
        logs_actions.addStretch()

        btn_clear_log = QPushButton("清除日志")
        btn_clear_log.clicked.connect(self.clear_logs)
        logs_actions.addWidget(btn_clear_log)

        btn_export_log = QPushButton("导出日志")
        btn_export_log.clicked.connect(self.export_logs)
        logs_actions.addWidget(btn_export_log)
        
        logs_vbox.addLayout(logs_actions)
        right_layout.addWidget(grp_logs)

    def toggle_api_key_visibility(self):
        if self.txt_api_key.echoMode() == QLineEdit.EchoMode.Password:
            self.txt_api_key.setEchoMode(QLineEdit.EchoMode.Normal)
            self.btn_toggle_key.setText("隐藏")
        else:
            self.txt_api_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.btn_toggle_key.setText("显示")

    def generate_api_key(self):
        # Generates sk-6e19c25f... style api key (16 random hex characters after sk-)
        key = f"sk-{secrets.token_hex(16)}"
        self.txt_api_key.setText(key)
        self.txt_api_key.setEchoMode(QLineEdit.EchoMode.Normal)
        self.btn_toggle_key.setText("隐藏")
        self.append_log(f"系统消息 - 已生成新 API Key: {key}\n")

    def browse_creds_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择服务账号 JSON 密钥文件", "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.txt_creds_path.setText(file_path)
            self.update_adc_status()

    def toggle_proxy_fields(self, checked: bool):
        self.txt_http_proxy.setEnabled(checked)
        self.txt_https_proxy.setEnabled(checked)

    def update_adc_status(self):
        configured_path = self.txt_creds_path.text().strip() or None
        exists, path = locate_adc_file(configured_path)
        if exists:
            self.lbl_adc_status.setText(f"已找到凭据: {Path(path).name}")
            self.lbl_adc_status.setStyleSheet("color: #107c10; font-weight: bold;")
            self.btn_test_conn.setEnabled(True)
        else:
            self.lbl_adc_status.setText("未检测到本地 ADC 凭据，建议导入服务账号 JSON。")
            self.lbl_adc_status.setStyleSheet("color: #d83b01; font-weight: bold;")
            self.btn_test_conn.setEnabled(False)

    def append_log(self, text: str):
        self.log_view.moveCursor(self.log_view.textCursor().MoveOperation.End)
        self.log_view.insertPlainText(text)
        if self.chk_auto_scroll.isChecked():
            self.log_view.ensureCursorVisible()

    def clear_logs(self):
        self.log_view.clear()

    def export_logs(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出日志", "vertex_proxy.log", "Log Files (*.log);;Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.log_view.toPlainText())
                QMessageBox.information(self, "导出成功", "日志已成功保存！")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"保存日志时遇到错误: {e}")

    def load_settings(self):
        config = load_config()
        if not config:
            # Set initial default fields
            self.chk_use_proxy.setChecked(False)
            self.toggle_proxy_fields(False)
            return

        self.txt_port.setText(str(config.get("port", "10101")))
        self.txt_api_key.setText(config.get("api_key", ""))
        self.txt_project.setText(config.get("project", ""))
        
        location = config.get("location", "")
        if location:
            index = self.cmb_location.findText(location)
            if index >= 0:
                self.cmb_location.setCurrentIndex(index)
            else:
                self.cmb_location.setEditText(location)
                
        self.txt_creds_path.setText(config.get("creds_path", ""))
        self.chk_use_proxy.setChecked(config.get("use_proxy", False))
        self.toggle_proxy_fields(self.chk_use_proxy.isChecked())
        self.txt_http_proxy.setText(config.get("http_proxy", ""))
        self.txt_https_proxy.setText(config.get("https_proxy", ""))

    def save_current_settings(self):
        config = {
            "port": int(self.txt_port.text().strip() or "10101"),
            "api_key": self.txt_api_key.text().strip(),
            "project": self.txt_project.text().strip(),
            "location": self.cmb_location.currentText().strip(),
            "creds_path": self.txt_creds_path.text().strip(),
            "use_proxy": self.chk_use_proxy.isChecked(),
            "http_proxy": self.txt_http_proxy.text().strip(),
            "https_proxy": self.txt_https_proxy.text().strip(),
        }
        save_config(config)

    def test_connectivity(self):
        self.save_current_settings()
        
        project = self.txt_project.text().strip()
        location = self.cmb_location.currentText().strip()
        model = self.cmb_test_model.currentText().strip()
        
        # Locate ADC
        configured_path = self.txt_creds_path.text().strip() or None
        exists, creds_path = locate_adc_file(configured_path)
        
        if not exists:
            QMessageBox.warning(self, "测试失败", "未找到本地 ADC 凭据，请提供有效的服务账号密钥文件或配置。")
            return

        self.btn_test_conn.setEnabled(False)
        self.btn_test_conn.setText("测试中...")
        self.append_log(f"系统消息 - 启动网络连通性测试 (目标模型: {model})...\n")

        self.test_thread = ConnectionTestThread(
            project=project,
            location=location,
            model=model,
            creds_path=creds_path,
            use_proxy=self.chk_use_proxy.isChecked(),
            http_proxy=self.txt_http_proxy.text().strip(),
            https_proxy=self.txt_https_proxy.text().strip()
        )
        self.test_thread.finished.connect(self.on_test_finished)
        self.test_thread.start()

    def on_test_finished(self, success: bool, message: str):
        self.btn_test_conn.setEnabled(True)
        self.btn_test_conn.setText("测试连接")
        
        if success:
            QMessageBox.information(self, "连接测试成功", message)
            self.append_log("系统消息 - 连通性测试：成功。\n")
        else:
            QMessageBox.critical(self, "连接测试失败", message)
            self.append_log(f"系统消息 - 连通性测试：失败。详情:\n{message}\n")
        
        self.update_adc_status()

    def start_proxy(self):
        self.save_current_settings()
        
        # Verify ADC
        configured_path = self.txt_creds_path.text().strip() or None
        exists, creds_path = locate_adc_file(configured_path)
        if not exists:
            # We can still try starting but print a warning
            self.append_log("警告 - 未检测到本地 ADC 密钥文件，服务器可能因为无凭证报错！\n")
            
        try:
            port = int(self.txt_port.text().strip() or "10101")
        except ValueError:
            QMessageBox.critical(self, "启动失败", "反向代理端口不合法！")
            return

        api_key = self.txt_api_key.text().strip() or None
        project = self.txt_project.text().strip()
        location = self.cmb_location.currentText().strip()
        
        # Standard models configuration returned for /v1/models (from user configuration or default)
        models = "gemini-3.5-flash,gemini-3.1-flash-lite,gemini-3.1-flash,gemini-3.1-pro-preview"
        
        # Default timeouts
        connect_timeout = 10.0
        read_timeout = 300.0

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.toggle_inputs(False)

        self.append_log(f"系统消息 - 正在启动本地代理服务器 (端口: {port})...\n")

        self.server_thread = UvicornServerThread(
            port=port,
            api_key=api_key,
            project=project,
            location=location,
            models=models,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            use_proxy=self.chk_use_proxy.isChecked(),
            http_proxy=self.txt_http_proxy.text().strip(),
            https_proxy=self.txt_https_proxy.text().strip(),
            creds_path=creds_path
        )
        self.server_thread.status_changed.connect(self.on_server_status_changed)
        self.server_thread.start()

    def stop_proxy(self):
        if self.server_thread and self.server_thread.isRunning():
            self.append_log("系统消息 - 正在停止代理服务器...\n")
            self.server_thread.stop()
            # Wait for thread to finish
            self.server_thread.wait(2000)

    def on_server_status_changed(self, status: str, detail: str):
        if status == "running":
            self.lbl_server_status.setText("运行中")
            self.lbl_server_status.setStyleSheet("color: #107c10; font-weight: bold;")
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_stop.setObjectName("btn_stop")
            self.btn_stop.setStyleSheet("") # Apply standard QSS stop style
            self.toggle_inputs(False)
            self.show_tray_message("代理服务器已启动", f"监听地址: http://127.0.0.1:{self.txt_port.text().strip()}")
        elif status == "stopped":
            self.lbl_server_status.setText("已停止")
            self.lbl_server_status.setStyleSheet("color: #888888; font-weight: bold;")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.toggle_inputs(True)
            self.show_tray_message("代理服务器已停止", "本地反向代理服务已经关闭")
        elif status == "error":
            self.lbl_server_status.setText("发生错误")
            self.lbl_server_status.setStyleSheet("color: #d83b01; font-weight: bold;")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.toggle_inputs(True)
            self.append_log(f"错误 - {detail}\n")
            QMessageBox.critical(self, "服务器错误", f"运行服务时遭遇异常:\n{detail}")
            self.show_tray_message("代理服务器运行出错", detail)

    def toggle_inputs(self, enabled: bool):
        self.txt_port.setEnabled(enabled)
        self.txt_api_key.setEnabled(enabled)
        self.btn_gen_key.setEnabled(enabled)
        self.txt_project.setEnabled(enabled)
        self.cmb_location.setEnabled(enabled)
        self.txt_creds_path.setEnabled(enabled)
        self.chk_use_proxy.setEnabled(enabled)
        if enabled:
            self.toggle_proxy_fields(self.chk_use_proxy.isChecked())
        else:
            self.toggle_proxy_fields(False)

    def copy_api_url(self):
        port = self.txt_port.text().strip() or "10101"
        url = f"http://127.0.0.1:{port}/v1"
        clipboard = QApplication.clipboard()
        clipboard.setText(url)
        self.append_log(f"系统消息 - 已复制 API Base URL: {url}\n")

    # Tray Integration
    def setup_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        
        tray_menu = QMenu()
        show_action = QAction("显示主界面", self)
        show_action.triggered.connect(self.show_normal_window)
        
        start_action = QAction("启动代理", self)
        start_action.triggered.connect(self.start_proxy)
        
        stop_action = QAction("停止代理", self)
        stop_action.triggered.connect(self.stop_proxy)
        
        exit_action = QAction("彻底退出", self)
        exit_action.triggered.connect(self.terminate_app)

        tray_menu.addAction(show_action)
        tray_menu.addSeparator()
        tray_menu.addAction(start_action)
        tray_menu.addAction(stop_action)
        tray_menu.addSeparator()
        tray_menu.addAction(exit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.on_tray_activated)
        self.tray_icon.show()

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger: # Left click
            self.show_normal_window()

    def show_normal_window(self):
        self.show()
        self.activateWindow()
        self.raise_()

    def show_tray_message(self, title: str, body: str):
        if self.tray_icon and self.tray_icon.isVisible():
            self.tray_icon.showMessage(title, body, QSystemTrayIcon.MessageIcon.Information, 3000)

    def closeEvent(self, event):
        # Instead of closing, hide in tray
        if self.tray_icon and self.tray_icon.isVisible():
            event.ignore()
            self.hide()
            self.show_tray_message("已最小化至系统托盘", "代理服务仍在后台运行，双击托盘图标可重新打开。")
        else:
            self.terminate_app()

    def terminate_app(self):
        self.save_current_settings()
        if self.server_thread and self.server_thread.isRunning():
            self.server_thread.stop()
            self.server_thread.wait(2000)
        self.tray_icon.hide()
        QApplication.quit()
        sys.exit(0)


def setup_logging(signaler: LogSignaler):
    """Sets up Python logging interceptor to forward to PyQt signal."""
    handler = QtLogHandler(signaler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%H:%M:%S')
    handler.setFormatter(formatter)
    
    # Configure target loggers
    loggers = [
        logging.getLogger("uvicorn"),
        logging.getLogger("uvicorn.error"),
        logging.getLogger("uvicorn.access"),
        logging.getLogger("vertex_proxy"),
        logging.getLogger("vertex_proxy_gui"),
    ]
    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def main():
    # Allow PyQt window scaling for high DPI displays
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False) # Keep running when window is closed (hidden in tray)
    
    window = VertexProxyApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
