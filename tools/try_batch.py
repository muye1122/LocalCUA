#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""探针：模型在新提示词下会不会主动给 actions 数组（动作序列）。

不碰鼠标键盘，只跟 Ollama 说话。用法：
    python tools/try_batch.py
"""
import json
import sys
import urllib.request

sys.path.insert(0, ".")

from agent.core import AgentSession   # noqa: E402


def chat(system, user, model="qwen3vl:8b"):
    body = json.dumps({
        "model": model, "stream": False,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["message"]["content"]


CASES = [
    ("打开浏览器搜索今天的天气",
     "任务：打开浏览器搜索今天的天气\n当前子任务：整体任务\n"
     "【系统窗口状态】前台窗口：桌面；其他打开的顶层窗口：\n"
     "可交互元素清单：[1] Edge (200,300) 48x48\n[2] 开始 (5,1040) 40x40\n"
     "请根据当前截图，给出下一步动作（只输出 JSON）。"),
    ("在资源管理器里跳转到 D:\\测试 并新建文件夹 叫 报告",
     "任务：在资源管理器里跳转到 D:\\测试 并新建文件夹 叫 报告\n"
     "当前子任务：整体任务\n【系统窗口状态】前台窗口：测试 - 文件资源管理器；\n"
     "可交互元素清单：[1] 新建 (60,40) 60x28\n"
     "请根据当前截图，给出下一步动作（只输出 JSON）。"),
    ("在记事本里输入 hello world 然后按回车",
     "任务：在记事本里输入 hello world 然后按回车\n当前子任务：整体任务\n"
     "【系统窗口状态】前台窗口：无标题 - 记事本；\n"
     "请根据当前截图，给出下一步动作（只输出 JSON）。"),
]


def main():
    sess = AgentSession({"safety": {}, "grounding": {"backend": "none"},
                         "planner": {"enabled": False}, "validator": {"enabled": False},
                         "agent": {"max_steps": 1}})
    system = sess._system_prompt(1280, 800)
    for name, user in CASES:
        print("=" * 70)
        print("任务：", name)
        raw = chat(system, user)
        print("模型返回：", raw.strip()[:400])
        try:
            d = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        except Exception:
            print("→ 解析不出 JSON")
            continue
        n = len(d.get("actions") or [])
        print(f"→ 动作数 {n}，plan={d.get('plan')!r}")


if __name__ == "__main__":
    main()
