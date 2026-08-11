import sys

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

    # 2. Ensure icon.ico exists from Vertex Proxy.png
    from pathlib import Path
    png_path = Path("Vertex Proxy.png")
    ico_path = Path("icon.ico")

    if png_path.exists() and (not ico_path.exists() or png_path.stat().st_mtime > ico_path.stat().st_mtime):
        print("正在从 Vertex Proxy.png 生成/更新高清 icon.ico 图标文件...")
        try:
            from PIL import Image
            img = Image.open(png_path)
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
            img.save(ico_path, format="ICO", sizes=sizes)
            print("已使用 Pillow 成功生成多分辨率高清 icon.ico")
        except Exception:
            try:
                from PyQt6.QtWidgets import QApplication
                from PyQt6.QtGui import QPixmap
                from PyQt6.QtCore import Qt
                _app = QApplication.instance() or QApplication([])
                pixmap = QPixmap(str(png_path))
                if not pixmap.isNull():
                    pixmap.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation).save(str(ico_path), "ICO")
                    print("成功生成 icon.ico (Qt Smooth)")
            except Exception as e:
                print(f"警告: 转换 icon.ico 失败: {e}")

    print("正在使用 PyInstaller 打包单文件独立 exe...")
    print("这可能需要几分钟，请耐心等待...\n")
    
    # 3. Build commands
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

    if ico_path.exists():
        args.append(f"--icon={ico_path}")
        args.append(f"--add-data={ico_path};.")
    if png_path.exists():
        args.append(f"--add-data={png_path};.")
    
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
