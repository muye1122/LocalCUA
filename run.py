#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LocalCUA 统一入口。

用法：
  python run.py                 启动 Web UI（默认 http://127.0.0.1:7860）
  python run.py --host 0.0.0.0 --port 8080
  python run.py --cli           终端模式，无需浏览器，直接在命令行驱动 Agent
  python run.py --make-sandbox  生成 Windows Sandbox 隔离配置 agent_sandbox.wsb
"""

import os
import sys
import argparse
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv_local():
    """零依赖读取项目根目录的 .env（仅在本进程注入 CU_ 等环境变量）。"""
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(root, ".env")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


_load_dotenv_local()

from agent.config import load_config
from agent.safety import ensure_dpi_aware


def cmd_web(host, port):
    from ui.web import main
    main(host=host, port=port)


def cmd_cli():
    """终端模式：在命令行直接下达任务、处理确认与回复，无需浏览器。"""
    import json
    from agent.core import AgentSession

    cfg = load_config()
    stop = threading.Event()

    def confirm(action):
        print("⚠️ 高风险操作待确认：", json.dumps(action, ensure_ascii=False))
        return input("允许执行？[y/N] ").strip().lower() in ("y", "yes")

    def log(msg):
        print(msg)

    sess = AgentSession(cfg, log=log, confirm=confirm, stop_event=stop)

    task = input("请输入任务> ").strip()
    if not task:
        print("（空任务，退出）")
        return

    reply = None
    try:
        while True:
            res = sess.run(task, reply=reply)
            reply = None
            if res == "NEED_REPLY":
                reply = input("模型在等你回复> ").strip()
                if not reply:
                    print("（空回复，继续）")
                continue
            print(f"—— 结束：{res}")
            break
    except KeyboardInterrupt:
        stop.set()
        print("\n⛔ 已请求停止，Agent 将在下一步前停下。")


def cmd_make_sandbox():
    from tools.sandbox import write_wsb
    path = write_wsb()
    print("已生成 Windows Sandbox 配置：", path)
    print("双击该 .wsb 即可在隔离桌面中启动 LocalCUA（沙箱内需自行装好 Python / 依赖 / Ollama）。")


def main():
    ensure_dpi_aware()  # 高 DPI 缩放修正：在任何截屏/点击前声明
    parser = argparse.ArgumentParser(description="LocalCUA - 本地电脑操作智能体")
    parser.add_argument("--cli", action="store_true", help="终端模式（无需浏览器）")
    parser.add_argument("--make-sandbox", action="store_true", help="生成 Windows Sandbox 配置")
    parser.add_argument("--host", default=None, help="Web 绑定地址（默认读 config.yaml）")
    parser.add_argument("--port", type=int, default=None, help="Web 端口（默认读 config.yaml）")
    args = parser.parse_args()

    if args.make_sandbox:
        cmd_make_sandbox()
        return
    if args.cli:
        cmd_cli()
        return
    cmd_web(args.host, args.port)


if __name__ == "__main__":
    main()
