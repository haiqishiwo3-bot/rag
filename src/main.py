#!/usr/bin/env python3
# ==========================================
# RAG Project - Main Entry Point
# 统一入口文件
# ==========================================

import os
import sys
import uvicorn
from dotenv import load_dotenv
# 导入应用
from controller.RagController import app
# 加载环境变量
load_dotenv()

# 添加 src 目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def main():
    """
    启动 RAG 服务
    """
    # 从环境变量读取配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")
    

    
    # 启动信息
    print("=" * 50)
    print("  RAG Project - Starting Server")
    print("=" * 50)
    print(f"  Host:   {host}")
    print(f"  Port:   {port}")
    print(f"  Workers: {workers}")
    print(f"  Reload: {reload}")
    print(f"  Log:    {log_level}")
    print("=" * 50)
    print(f"\n  API Docs: http://{host}:{port}/docs")
    print(f"  Redoc:    http://{host}:{port}/redoc")
    print("\n  Press CTRL+C to quit\n")
    
    # 启动服务
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
