#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""网格标尺 + 百分比定位 A/B 实测。

目的：验证"让模型读百分比"是否比"让模型猜绝对像素"更准。
做法：
  1. 截全屏，压到 1280 宽（模型上下文限制）
  2. 画 10% 网格 + 边框百分比刻度（0..100）
  3. 问模型：某个元素在 x=?% y=?%
  4. 把百分比换算成真实像素，再截图裁一小块贴回来，人眼复核

用法：
  ./.venv/Scripts/python.exe tools/grid_probe.py --target "任务栏最左边的开始按钮"
"""
import argparse
import base64
import io
import json
import os
import re
import sys

import pyautogui
from PIL import Image, ImageDraw

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("CU_MODEL", "qwen3vl:8b")
GRID_COLOR = (255, 0, 128)
TEXT_BG = (255, 255, 255)


def draw_grid(img: Image.Image, step_pct: int = 10) -> Image.Image:
    """在图上叠加百分比网格与刻度标签。"""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    for p in range(step_pct, 100, step_pct):
        x = int(w * p / 100)
        y = int(h * p / 100)
        d.line([(x, 0), (x, h)], fill=GRID_COLOR, width=1)
        d.line([(0, y), (w, y)], fill=GRID_COLOR, width=1)
    # 顶部/左侧刻度文字
    for p in range(0, 101, step_pct):
        x = min(int(w * p / 100), w - 22)
        y = min(int(h * p / 100), h - 12)
        d.rectangle([x, 0, x + 22, 12], fill=TEXT_BG)
        d.text((x + 2, 1), str(p), fill=(0, 0, 0))
        d.rectangle([0, y, 22, y + 12], fill=TEXT_BG)
        d.text((2, y + 1), str(p), fill=(0, 0, 0))
    return im


def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask_model(img: Image.Image, question: str) -> str:
    import requests
    payload = {
        "model": MODEL,
        "prompt": question,
        "images": [to_b64(img)],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 160, "num_ctx": 4096},
    }
    r = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("response", "")


def parse_pct(text: str):
    """从模型回答里抓 x/y 百分比。容忍 "x=12.5%, y=98%" / "12.5, 98" 等写法。"""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) < 2:
        return None
    return float(nums[0]), float(nums[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="任务栏最左边的 Windows 开始按钮")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--out", default="data/grid_probe")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    full = pyautogui.screenshot()
    sw, sh = full.size
    scale = args.width / sw
    small = full.resize((args.width, int(sh * scale)), Image.LANCZOS)
    gridded = draw_grid(small)
    gridded.save(f"{args.out}/gridded.png")

    q = (
        "这张截图上有粉色的百分比网格，上边缘和左边缘标了 0 到 100 的刻度。\n"
        f"请定位：{args.target}\n"
        "只回答它的中心点百分比坐标，格式严格为：x=数字%, y=数字%\n"
        "例如 x=50%, y=98%。不要解释。"
    )
    answer = ask_model(gridded, q)
    print("[模型回答]", answer.strip())

    pct = parse_pct(answer)
    if not pct:
        print("没解析出百分比，退出")
        return 1
    px, py = pct
    px = max(0, min(100, px))
    py = max(0, min(100, py))
    real_x = int(round((sw - 1) * px / 100))
    real_y = int(round((sh - 1) * py / 100))
    print(f"[换算] {px}%,{py}%  ->  屏幕像素 ({real_x}, {real_y})  屏幕 {sw}x{sh}")

    # 裁一块 400x260 的局部图，标红十字，方便人眼复核准不准
    half_w, half_h = 200, 130
    box = (max(0, real_x - half_w), max(0, real_y - half_h),
           min(sw, real_x + half_w), min(sh, real_y + half_h))
    crop = full.crop(box).convert("RGB")
    d = ImageDraw.Draw(crop)
    cx, cy = real_x - box[0], real_y - box[1]
    d.line([(cx - 20, cy), (cx + 20, cy)], fill=(255, 0, 0), width=3)
    d.line([(cx, cy - 20), (cx, cy + 20)], fill=(255, 0, 0), width=3)
    crop.save(f"{args.out}/verify.png")

    result = {"target": args.target, "answer": answer.strip(), "pct": [px, py],
              "pixel": [real_x, real_y], "screen": [sw, sh],
              "verify": f"{args.out}/verify.png"}
    with open(f"{args.out}/result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[复核图] {args.out}/verify.png  （红叉是模型指的位置）")
    print(f"[网格图] {args.out}/gridded.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
