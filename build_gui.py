import sys
import os

def main():
    print("=========================================")
    print("  Vertex AI ADC Proxy 桌面版打包程序  ")
    print("=========================================")
    
    # 1. Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("错误: 未检测到 pyinstaller。")
        print("请先在虚拟环境中运行: pip install -e .[gui]")
        sys.exit(1)
        
    print("正在使用 PyInstaller 打包单文件独立 exe...")
    print("这可能需要几分钟，请耐心等待...\n")
    
    # 2. Build commands
    import PyInstaller.__main__
    
    args = [
        "gui.py",                      # 脚本入口
        "--name=VertexADCProxy",        # 生成的可执行文件名
        "--onefile",                   # 打包为单文件
        "--noconsole",                 # 隐藏命令行黑窗口
        "--clean",                     # 清理缓存
        "--collect-all", "uvicorn",    # 完整收集 uvicorn 依赖与元数据
        "--collect-all", "fastapi",    # 完整收集 fastapi 依赖与元数据
        "--collect-all", "starlette",  # 完整收集 starlette 依赖与元数据
        "--collect-all", "httpx",      # 完整收集 httpx 依赖与元数据
    ]
    
    try:
        PyInstaller.__main__.run(args)
        print("\n=========================================")
        print("[SUCCESS] GUI Desktop App compiled successfully!")
        print("Executable path: dist/VertexADCProxy.exe")
        print("=========================================")
    except Exception as e:
        print(f"\nError during compilation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
