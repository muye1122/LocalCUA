#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""坐标标定实验：验证 pyautogui 的绝对/相对移动与屏幕像素是否 1:1。

思路（来自使用者）：
  1. 先把光标推到右上角极限 -> 得到一个确定的锚点
  2. 再往左移动 n 个单位 -> 读回真实位置
  3. 看 n 与"屏幕宽度的几分之几"的对应关系，就知道 1 单位等于多少像素

顺带检查 DPI 感知、截图分辨率、多屏偏移，输出一份标定报告 json。
运行： ./.venv/Scripts/python.exe tools/calibrate.py [--move-pixels 640]
"""
import ctypes
import json
import sys
import time

import pyautogui

pyautogui.FAILSAFE = False       # 标定过程会主动贴边，必须关掉急停
pyautogui.PAUSE = 0.05

user32 = ctypes.windll.user32


def dpi_info():
    """返回 (系统DPI, 进程是否已感知)"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
        aware = True
    except Exception:
        aware = False
    hdc = user32.GetDC(0)
    gdi32 = ctypes.windll.gdi32
    LOGPIXELSX = 88
    dpi = gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
    user32.ReleaseDC(0, hdc)
    return dpi, aware


def pos():
    p = pyautogui.position()
    return (p[0], p[1])


def main():
    move_pixels = 640
    if "--move-pixels" in sys.argv:
        move_pixels = int(sys.argv[sys.argv.index("--move-pixels") + 1])

    dpi, aware = dpi_info()
    w, h = pyautogui.size()
    origin = pos()

    # 截图分辨率（模型真正看到的像素）
    from PIL import ImageGrab
    shot = ImageGrab.grab(all_screens=True)
    shot_size = shot.size

    report = {
        "pyautogui_size": [w, h],
        "screenshot_size": list(shot_size),
        "dpi": dpi,
        "dpi_aware": aware,
        "dpi_scale": round(dpi / 96.0, 3),
        "origin": list(origin),
        "anchor_tests": [],
        "relative_tests": [],
    }

    try:
        # --- 1. 极值锚点：推到右上角 ---
        for name, tx, ty in [("right-top", w - 1, 0), ("left-bottom", 0, h - 1)]:
            pyautogui.moveTo(tx, ty)
            time.sleep(0.15)
            got = pos()
            report["anchor_tests"].append({
                "target": [tx, ty], "got": list(got),
                "err": [got[0] - tx, got[1] - ty],
                "on_edge": got[0] in (0, w - 1) or got[1] in (0, h - 1),
            })

        # --- 2. 相对移动：从右上角出发往左 n 像素 ---
        pyautogui.moveTo(w - 1, 0)
        time.sleep(0.15)
        start = pos()
        for n in (move_pixels, move_pixels // 2, move_pixels // 4):
            pyautogui.moveTo(w - 1, 0)
            time.sleep(0.1)
            pyautogui.moveRel(-n, 0, duration=0)
            time.sleep(0.15)
            got = pos()
            report["relative_tests"].append({
                "requested_px_left": n,
                "start": list(start),
                "got": list(got),
                "actual_px_left": start[0] - got[0],
                "px_err": (start[0] - got[0]) - n,
                # 换算成"占屏幕宽度的百分比"，模型只需给这个量
                "pct_of_screen": round(n / w * 100, 2),
            })

        # --- 3. 百分比映射验证：百分比 -> 像素 ---
        pct_tests = []
        for pct in (10, 25, 50, 75):
            tx = int(round((w - 1) * pct / 100.0))
            pyautogui.moveTo(tx, h // 2)
            time.sleep(0.12)
            got = pos()
            pct_tests.append({
                "pct": pct, "target_x": tx, "got_x": got[0],
                "err": got[0] - tx,
                "back_pct": round(got[0] / (w - 1) * 100, 2),
            })
        report["pct_tests"] = pct_tests
    finally:
        pyautogui.moveTo(*origin)

    # 屏幕分配：模型给百分比时我们用它换算
    report["note"] = "pct->px 使用 round((w-1)*pct/100)，反向 pct = px/(w-1)*100"

    out = "data/calibration.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
