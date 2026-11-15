#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照组实验：同一张图，三种问法，看模型到底差在哪。
  A. 不带网格，问像素坐标
  B. 带网格，问百分比（= grid_probe 的问法）
  C. 带网格，先问"任务栏在哪"，再问坐标（分步引导）
"""
import os
import sys

import pyautogui
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from grid_probe import draw_grid, ask_model, to_b64  # noqa: E402

OUT = "data/grid_probe"


def main():
    os.makedirs(OUT, exist_ok=True)
    full = pyautogui.screenshot()
    sw, sh = full.size
    small = full.resize((1280, int(sh * 1280 / sw)), Image.LANCZOS)

    grid_img = draw_grid(small)
    grid_img.save(f"{OUT}/ab_gridded.png")
    small.save(f"{OUT}/ab_plain.png")

    q_pixel = (
        "这是一张 1280x800 的屏幕截图。请定位任务栏最左边的 Windows 开始按钮的中心。\n"
        "只回答像素坐标，格式：x=数字, y=数字。不要解释。"
    )
    q_pct = (
        "这张截图上有粉色百分比网格，上下左右边缘标着 0~100 刻度。\n"
        "请定位任务栏最左边的 Windows 开始按钮的中心。\n"
        "只回答百分比坐标，格式：x=数字%, y=数字%。不要解释。"
    )
    q_step1 = "这张截图里任务栏在哪里？用一句话说明它在图片的哪个位置（上/下/左/右）。"
    q_step2 = (
        "任务栏在最底部。现在看任务栏最左边的 Windows 开始按钮。\n"
        "图片上有粉色百分比网格和 0~100 刻度。"
        "只回答按钮中心的百分比坐标，格式：x=数字%, y=数字%。不要解释。"
    )

    tests = [
        ("A_无网格_像素", small, q_pixel),
        ("B_网格_百分比", grid_img, q_pct),
        ("C1_分步_先说位置", grid_img, q_step1),
        ("C2_分步_再要坐标", grid_img, q_step2),
    ]
    for name, img, q in tests:
        ans = ask_model(img, q)
        print(f"[{name}] {ans.strip()}")
        print("-" * 40)


if __name__ == "__main__":
    main()
