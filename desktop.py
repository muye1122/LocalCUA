#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LocalCUA 桌面版启动器：双击 bat → 弹出一个无地址栏、无标签栏的独立窗口。

为什么不写 tkinter：当前 .venv 和系统 Python 都没有 tkinter（参考项目用的是
另一个 venv）。用浏览器的 --app 模式把现有 Web UI 拉成独立窗口，复用全部现成
功能（任务输入 / 实时日志 / 思考栏 / 停止 / 高风险确认 / 回复框），一行都不用重写。

--user-data-dir 是必须的：不加的话 Edge 会把 --app 窗口交给已有的浏览器进程，
Popen 立刻返回，我们就无法在用户关闭窗口时把 Web 服务一起停掉。

关掉独立窗口 = 停服务 = 日志窗口自动关闭；端口若已有服务则直接复用，
所以上次异常退出留下的 uvicorn 不会造成端口冲突。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
LOG = os.path.join(ROOT, "data", "desktop.log")


def log(msg: str):
    """pythonw / 最小化窗口看不到控制台，所有线索落到文件里。"""
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass


def config_port():
    try:
        import yaml
        with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        w = cfg.get("web") or {}
        return str(w.get("host", "127.0.0.1")), int(w.get("port", 7860))
    except Exception:
        return "127.0.0.1", 7860


def port_alive(port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def find_browser():
    """优先 Edge，其次 Chrome；找不到就退回普通浏览器标签页。"""
    px = os.environ.get("ProgramFiles(x86)", "")
    pf = os.environ.get("ProgramFiles", "")
    la = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        os.path.join(px, "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(pf, "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(la, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(pf, "Google", "Chrome", "Application", "chrome.exe"),
        shutil.which("msedge"),
        shutil.which("chrome"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def check():
    host, port = config_port()
    b = find_browser()
    print(f"服务地址   : http://{host}:{port}")
    print(f"端口已占用 : {port_alive(port)}（已占用则直接复用，不重复起服务）")
    print(f"浏览器     : {b or '未找到 Edge/Chrome，将退回默认浏览器'}")
    print(f"日志       : {LOG}")
    return 0 if b else 1


def main():
    host, port = config_port()
    url = f"http://{host}:{port}"
    log(f"启动：{url}")

    svc = None
    if port_alive(port):
        log("端口已有服务，直接复用（可能是上次异常退出留下的）")
    else:
        svc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "ui.web:app",
             "--host", host, "--port", str(port), "--log-level", "warning"],
            cwd=ROOT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        log(f"已拉起 Web 服务 pid={svc.pid}")
        for _ in range(80):
            if port_alive(port, timeout=0.5):
                break
            if svc.poll() is not None:
                log("❌ 服务进程提前退出，请手动跑 run_web.bat 看报错")
                return
            time.sleep(0.25)
        else:
            log("❌ 服务 20 秒内没起来，请手动跑 run_web.bat 排查")
            return

    browser = find_browser()
    if not browser:
        log("⚠ 没找到 Edge/Chrome，退回默认浏览器（普通标签页）")
        import webbrowser
        webbrowser.open(url)
        if svc:
            svc.wait()   # 没有独立窗口可等，服务一直跑，关日志窗即停
        return

    # 独立 profile ⇒ 独立进程 ⇒ Popen.wait() 能真正等到「用户关掉这个窗口」
    profile = os.path.join(os.environ.get("LOCALAPPDATA", ROOT),
                          "LocalCUA", "DesktopProfile")
    os.makedirs(profile, exist_ok=True)
    args = [
        browser,
        f"--app={url}",
        f"--user-data-dir={profile}",
        "--window-size=1080,820",
        "--window-position=120,80",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    log("打开独立窗口：" + os.path.basename(browser))
    win = subprocess.Popen(args)
    win.wait()
    log("独立窗口已关闭，停止 Web 服务")
    if svc and svc.poll() is None:
        svc.terminate()
        try:
            svc.wait(timeout=5)
        except Exception:
            svc.kill()


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(check())
    try:
        main()
    except Exception:
        log("崩溃：\n" + traceback.format_exc())
        sys.exit(1)
