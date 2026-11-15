#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""纯文本对照：不给图，只给元素清单，看 7B 靠文字+坐标能否选对编号。
若纯文本也错 -> 瓶颈是空间推理/常识；若纯文本对 -> 瓶颈是'看图找编号'。
"""
import json
import os
import sys

import requests

OLLAMA = "http://127.0.0.1:11434"

with open("data/icons/elements.json", encoding="utf-8") as f:
    elements = json.load(f)["elements"]
SW, SH = 2560, 1600

# 按"阅读顺序"重排编号：y 为主、x 为辅（上到下、左到右），并把像素换成百分比
rows = sorted(elements, key=lambda e: (e["center"][1] // 80, e["center"][0]))
lines = []
for i, e in enumerate(rows, 1):
    cx, cy = e["center"]
    x_pct = round(cx / SW * 100)
    y_pct = round(cy / SH * 100)
    w = max(1, e["bbox"][2] - e["bbox"][0])
    h = max(1, e["bbox"][3] - e["bbox"][1])
    kind = "输入框" if w > 100 and h < 80 and w / max(h, 1) > 3 else "元素"
    lines.append(f"编号{i}: {kind}, 位置 x={x_pct}%(横向) y={y_pct}%(纵向), 尺寸{w}x{h}")

lst = "\n".join(lines)
system = (
    f"屏幕是 2560x1600。下面是屏幕上全部 {len(rows)} 个可点击元素的清单，"
    "x% 是离左边的距离，y% 是离顶部的距离（0%=最上/最左，100%=最下/最右）。\n"
    + lst +
    "\n\n根据用户的描述选出对应编号。只回答：编号=<数字>"
)

targets = [
    "任务栏上的搜索框（任务栏在最底部，搜索框有放大镜图标和'搜索'文字）",
    "任务栏最左边的 Windows 开始按钮（Windows 徽标，任务栏最左）",
    "左侧边栏顶部的'新建任务'按钮",
    "页面底部的聊天输入框（很宽的框，中间偏左）",
    "右上角的最小化按钮（窗口右上角三个按钮中最左那个）",
]

answers = []
for t in targets:
    payload = {"model": "qwen3vl:8b", "prompt": f"{system}\n\n要点击：{t}",
               "stream": False, "options": {"temperature": 0, "num_predict": 40}}
    r = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=300)
    a = r.json().get("response", "").strip().splitlines()[0]
    print(f"[{t[:20]}...] -> {a}")
    answers.append({"target": t, "answer": a})

with open("data/icons/text_result.json", "w", encoding="utf-8") as f:
    json.dump(answers, f, ensure_ascii=False, indent=2)
