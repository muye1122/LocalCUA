#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""两步定位实验：
  第 1 步：全图(1280宽)让模型粗定位 -> 得到粗坐标
  第 2 步：以粗坐标为中心裁一块局部图，放大 N 倍，让模型在局部图里再定位
  把局部坐标换算回全图像素，画红叉输出复核图。

用法：
  ./.venv/Scripts/python.exe tools/two_step.py --target "任务栏最左边的 Windows 开始按钮"
  可选 --crop 320 --zoom 3 --rounds 2（可递归多轮，直到 crop 内误差 < 若干 px）
"""
import argparse
import json
import os
import re
import sys

import pyautogui
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from grid_probe import ask_model  # noqa: E402

OUT = "data/two_step"


def parse_xy(text: str):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) < 2:
        return None
    return float(nums[0]), float(nums[1])


def clamp_crop(cx, cy, half_w, half_h, w, h):
    x0 = max(0, min(int(cx - half_w), w - 1))
    y0 = max(0, min(int(cy - half_h), h - 1))
    x1 = max(x0 + 1, min(int(cx + half_w), w))
    y1 = max(y0 + 1, min(int(cy + half_h), h))
    return x0, y0, x1, y1


def draw_cross(im, x, y, color=(255, 0, 0), size=14, width=3):
    d = ImageDraw.Draw(im)
    d.line([(x - size, y), (x + size, y)], fill=color, width=width)
    d.line([(x, y - size), (x, y + size)], fill=color, width=width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="任务栏最左边的 Windows 开始按钮")
    ap.add_argument("--crop", type=int, default=320, help="局部图边长（全图像素）")
    ap.add_argument("--zoom", type=int, default=3, help="局部图放大倍数")
    ap.add_argument("--rounds", type=int, default=2, help="局部化轮数")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    full = pyautogui.screenshot()
    sw, sh = full.size
    small = full.resize((1280, int(sh * 1280 / sw)), Image.LANCZOS)
    small.save(f"{args.out}/step0_full.png")
    to_full = sw / small.size[0]  # 全图/小图 比例

    # ---- 第 1 步：粗定位 ----
    q1 = (
        f"这是一张 {small.size[0]}x{small.size[1]} 的屏幕截图。请定位：{args.target}\n"
        "只回答它的中心像素坐标，格式：x=数字, y=数字。不要解释。"
    )
    a1 = ask_model(small, q1)
    p = parse_xy(a1) or (0, 0)
    print(f"[第1步-粗定位] 模型答 {a1.strip()}  -> 小图坐标 {p}")
    cx, cy = p[0] * to_full, p[1] * to_full  # 换到全图像素
    print(f"[第1步] 换算到全图 ({cx:.0f}, {cy:.0f}) / {sw}x{sh}")

    trail = []
    img = full.convert("RGB")
    x0, y0, x1, y1 = 0, 0, sw, sh  # 当前"关注窗口"在全图中的位置

    # ---- 第 2..N 步：局部放大定位 ----
    for r in range(args.rounds):
        cur = img.crop((x0, y0, x1, y1))
        cw, chh = cur.size
        half = args.crop // 2
        # 在当前窗口内再取局部
        lx0, ly0, lx1, ly1 = clamp_crop(cx - x0, cy - y0, half, half, cw, chh)
        patch = cur.crop((lx0, ly0, lx1, ly1))
        pw, ph = patch.size
        big = patch.resize((pw * args.zoom, ph * args.zoom), Image.LANCZOS)
        big.save(f"{args.out}/step{r+1}_patch.png")

        q = (
            f"这是一张 {big.size[0]}x{big.size[1]} 的放大截图（原图 {pw}x{ph} 的 {args.zoom} 倍）。\n"
            f"请定位：{args.target}\n"
            "只回答它在放大图中的中心像素坐标，格式：x=数字, y=数字。不要解释。"
        )
        a = ask_model(big, q)
        p2 = parse_xy(a) or (pw * args.zoom / 2, ph * args.zoom / 2)
        print(f"[第{r+2}步-局部] 模型答 {a.strip()}  -> 放大图坐标 ({p2[0]:.0f},{p2[1]:.0f})")
        # 放大图 -> patch -> 当前窗口 -> 全图
        gx = x0 + lx0 + p2[0] / args.zoom
        gy = y0 + ly0 + p2[1] / args.zoom
        print(f"[第{r+2}步] 换算回全图 ({gx:.0f}, {gy:.0f})")
        trail.append({"round": r + 2, "raw": a.strip(), "global": [round(gx), round(gy)]})
        cx, cy = gx, gy
        # 更新关注窗口（下一轮在更小范围内找）
        x0, y0, x1, y1 = clamp_crop(cx, cy, args.crop, args.crop, sw, sh)

    # ---- 输出复核图 ----
    vis = full.convert("RGB").copy()
    draw_cross(vis, int(cx), int(cy))
    # 也画一个轨迹缩略图
    for t in trail:
        draw_cross(vis, t["global"][0], t["global"][1], color=(0, 128, 255), size=10, width=2)
    vis.save(f"{args.out}/verify.png")

    result = {"target": args.target, "coarse": [round(p[0] * to_full), round(p[1] * to_full)],
              "final": [round(cx), round(cy)], "trail": trail,
              "screen": [sw, sh], "verify": f"{args.out}/verify.png"}
    with open(f"{args.out}/result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[复核图] {args.out}/verify.png  红叉=最终位置 蓝叉=每步轨迹")


if __name__ == "__main__":
    main()
