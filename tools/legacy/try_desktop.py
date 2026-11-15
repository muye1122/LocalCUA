#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证「桌面图标优先」这条路：读清单 → 匹配 → 打印坐标（默认不点击）。

  python tools/legacy/try_desktop.py                 # 只读验证，打印所有图标 + pyautogui 坐标
  python tools/legacy/try_desktop.py Dev-C++         # 只验证匹配这一个名字
  python tools/legacy/try_desktop.py Dev-C++ --click # 真的 Win+D + 双击（会打开程序）
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.core import AgentSession  # noqa: E402

a = AgentSession.__new__(AgentSession)          # 不跑 __init__：这里只需要读图标的两个方法
a.log = lambda m: print(m, flush=True)

args = [x for x in sys.argv[1:] if not x.startswith("--")]
click = "--click" in sys.argv
target = args[0] if args else ""

t0 = time.time()
items, sw = a._desktop_icons()
print(f"读到 {len(items)} 个桌面图标，物理屏宽 {sw}，耗时 {time.time() - t0:.2f}s\n")

for it in items:
    x, y = a._icon_to_py_coords(it, sw)
    print(f"  {it['name']:<40s} 物理({it['x']},{it['y']}) -> py({x:.0f},{y:.0f})")

if target:
    hit, score = a._match_desktop_icon(target, items)
    if hit:
        x, y = a._icon_to_py_coords(hit, sw)
        print(f"\n匹配「{target}」=> 「{hit['name']}」 分{score} 点击坐标 ({x:.0f},{y:.0f})")
        if click:
            import pyautogui
            a._show_desktop(settle=0.6)
            pyautogui.doubleClick(x, y)
            title = a._wait_app_window(target, max_wait=6.0)
            print("打开结果：", title or "没检测到窗口")
    else:
        print(f"\n匹配「{target}」=> 没找到，应改走开始菜单搜索")
