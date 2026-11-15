#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""集成冒烟：真实截图 -> YoloCaptionGrounding -> resolve_desc -> 换算屏幕坐标。
不真点鼠标，只输出定位结果与耗时，人工看 verify.png 复核。
"""
import json
import sys
import time

import pyautogui
from PIL import Image, ImageDraw

sys.path.insert(0, ".")
from agent.config import load_config  # noqa: E402
from agent.models import ModelClient  # noqa: E402
from agent.grounding import build_grounding  # noqa: E402

OUT = "data/smoke"
TARGETS = [
    "任务栏上的搜索框",
    "任务栏最左边的 Windows 开始按钮",
    "左侧边栏顶部的新建任务按钮",
    "右上角的最小化按钮",
    "页面底部中间的聊天输入框",
]


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    model = ModelClient(cfg)
    g = build_grounding(cfg, model)

    full = pyautogui.screenshot()
    # 与主循环一致：压到 1280 宽
    scale = 1280 / full.size[0]
    img_s = full.resize((1280, int(full.size[1] * scale)), Image.LANCZOS)

    t0 = time.time()
    els, _ = g.annotate(img_s.copy())
    print(f"[检测] {len(els)} 个元素, {time.time()-t0:.1f}s")

    results = []
    vis = full.convert("RGB")
    d = ImageDraw.Draw(vis)
    for desc in TARGETS:
        t0 = time.time()
        r = g.resolve_desc(desc, img_s)
        dt = time.time() - t0
        if r:
            x, y, eid, label, score = r
            # 缩放坐标系 -> 屏幕坐标
            sx, sy = x / scale, y / scale
            print(f"[{desc}]\n   -> #{eid} '{label}' 分{score} 截图({x:.0f},{y:.0f}) 屏幕({sx:.0f},{sy:.0f})  {dt:.1f}s")
            d.line([(sx - 20, sy), (sx + 20, sy)], fill=(255, 0, 0), width=3)
            d.line([(sx, sy - 20), (sx, sy + 20)], fill=(255, 0, 0), width=3)
            results.append({"target": desc, "id": eid, "label": label,
                            "score": score, "screen": [round(sx), round(sy)], "sec": round(dt, 1)})
        else:
            print(f"[{desc}]\n   -> 未匹配  {dt:.1f}s")
            results.append({"target": desc, "miss": True, "sec": round(dt, 1)})

    vis = vis.resize((1280, int(vis.size[1] * 1280 / vis.size[0])), Image.LANCZOS)
    vis.save(f"{OUT}/verify.png")
    with open(f"{OUT}/result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[复核图] {OUT}/verify.png")


if __name__ == "__main__":
    main()
