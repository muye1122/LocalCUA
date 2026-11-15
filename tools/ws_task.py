#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通过本地 Web UI 的 WebSocket 提交任务，并实时打印日志。

用法：
    python tools/ws_task.py --task "打开记事本输入你好" [--timeout 300]
这个脚本只发任务、收日志，不碰鼠标键盘 —— 用来验证「界面输入 -> agent 全自动执行」。
"""
import argparse
import asyncio
import json
import sys

import websockets


async def run(task: str, timeout: float, url: str):
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "task", "text": task}, ensure_ascii=False))
        loop = asyncio.get_event_loop()
        end = loop.time() + timeout
        while True:
            left = end - loop.time()
            if left <= 0:
                print("\n[超时] 结束观察")
                break
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(left, 5))
            except asyncio.TimeoutError:
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                print(raw)
                continue
            t = msg.get("type")
            if t == "log":
                print(msg.get("msg", ""), flush=True)
            elif t == "finished":
                print(f"[结束] {msg.get('result')}", flush=True)
                break
            elif t == "error":
                print(f"[错误] {msg.get('msg', '')}", flush=True)
                break
            elif t == "need_reply":
                print("[模型提问] 自动回复：继续执行", flush=True)
                await ws.send(json.dumps({"type": "reply", "text": "继续执行，自行判断"},
                                         ensure_ascii=False))
            elif t == "confirm_request":
                act = msg.get("action") or {}
                print(f"[确认请求] {json.dumps(act, ensure_ascii=False)[:120]} -> 自动拒绝",
                      flush=True)
                await ws.send(json.dumps({"type": "confirm", "allow": False}))
            else:
                print(f"[{t}] {json.dumps(msg, ensure_ascii=False)[:200]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--url", default="ws://127.0.0.1:7860/ws")
    a = ap.parse_args()
    asyncio.run(run(a.task, a.timeout, a.url))


if __name__ == "__main__":
    main()
