#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方案：逐元素起名 + 文本匹配。

  1. YOLO 检测全部元素
  2. 每个元素裁小图 -> 问 qwen3vl:8b"这是什么？8个字以内"
  3. 目标描述 vs 元素描述 文本匹配（关键词重合）
  4. 命中元素的 bbox 中心就是点击坐标
"""
import json
import os
import re
import sys
import time

import numpy as np
import onnxruntime as ort
import pyautogui
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from grid_probe import ask_model  # noqa: E402
from icon_detect import detect  # noqa: E402

OUT = "data/caption"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "icon_detect", "model.onnx")
CACHE = os.path.join(OUT, "cache.json")


def caption_element(img, bbox, pad_ratio=0.25):
    x1, y1, x2, y2 = bbox
    pw = int((x2 - x1) * pad_ratio) + 8
    ph = int((y2 - y1) * pad_ratio) + 8
    crop = img.crop((max(0, x1 - pw), max(0, y1 - ph),
                     min(img.size[0], x2 + pw), min(img.size[1], y2 + ph)))
    scale = min(3.0, max(1.0, 320 / max(crop.size)))
    if scale > 1:
        crop = crop.resize((int(crop.size[0] * scale), int(crop.size[1] * scale)), Image.LANCZOS)
    ans = ask_model(crop, "这是电脑屏幕上的一个界面元素。它是什么/上面有什么文字？回答不超过12个字。").strip()
    return ans.splitlines()[0][:24] if ans else ""


STOPWORDS = set("的请帮我点点打开那个这个点击一个在把和是了上面左边的按钮图标".split())


def keywords(text):
    words = re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) >= 1}


def bigrams(text):
    words = re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", text.lower())
    grams = set()
    for w in words:
        if len(w) == 1:
            grams.add(w)
        for i in range(len(w) - 1):
            grams.add(w[i:i + 2])
    return grams


def position_bonus(target, cx, cy, sw, sh):
    """程序解析目标里的位置词，给符合的元素加分（模型只管认东西，位置归程序算）。"""
    bonus = 0.0
    xr, yr = cx / sw, cy / sh
    if "任务栏" in target or "底部" in target:
        bonus += 0.5 if yr > 0.9 else -0.3
    if "顶部" in target:
        bonus += 0.4 if yr < 0.12 else -0.2
    if "右上角" in target:
        bonus += 0.5 if (xr > 0.85 and yr < 0.1) else -0.3
    if "左下" in target:
        bonus += 0.5 if (xr < 0.4 and yr > 0.85) else -0.2
    if "左边" in target or "左侧" in target or "左上" in target:
        bonus += 0.3 if xr < 0.4 else -0.2
    if "右边" in target or "右侧" in target:
        bonus += 0.3 if xr > 0.6 else -0.2
    if "最左" in target:
        bonus += 0.3 if xr < 0.05 else -0.2
    if "输入框" in target:
        bonus += 0.2 if yr > 0.75 else 0.0
    return bonus


def match_score(target_kw, caption, center, sw, sh, target_text=""):
    cap_gr = bigrams(caption)
    if not target_kw or not cap_gr:
        return 0.0
    ov = len(target_kw & cap_gr)
    score = ov / min(len(target_kw), len(cap_gr))
    if target_text:
        score += position_bonus(target_text, center[0], center[1], sw, sh)
    return score


def main():
    os.makedirs(OUT, exist_ok=True)
    full = pyautogui.screenshot()
    sw, sh = full.size
    sess = ort.InferenceSession(os.path.abspath(MODEL_PATH), providers=["CPUExecutionProvider"])
    dets = detect(full, sess, conf=0.3)
    print(f"[检测] {len(dets)} 个元素")

    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            cache = json.load(f)

    # 只对一部分候选起名：全部 67 个太慢；先跑全部看总耗时（每次约1.5s）
    t0 = time.time()
    caps = []
    for i, d in enumerate(dets):
        bbox = [round(v) for v in d[:4]]
        key = "|".join(map(str, bbox)) + f"|{sw}x{sh}"
        if key in cache:
            cap = cache[key]
        else:
            cap = caption_element(full, bbox)
            cache[key] = cap
        caps.append(cap)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    print(f"[起名] 耗时 {time.time()-t0:.0f}s（含缓存命中）")

    targets = [
        ("任务栏上的搜索框", True),
        ("任务栏最左边的 Windows 开始按钮", True),
        ("左侧边栏顶部的'新建任务'按钮", True),
        ("页面底部的聊天输入框", True),
        ("右上角的最小化按钮", True),
    ]

    results = []
    for desc, _ in targets:
        tkw = bigrams(desc)
        scored = []
        for i, ((x1, y1, x2, y2, s), cap) in enumerate(zip(dets, caps)):
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            sc = match_score(tkw, cap, (cx, cy), sw, sh, target_text=desc)
            scored.append((sc, i, cap))
        scored.sort(reverse=True)
        best = scored[0]
        x1, y1, x2, y2, _ = dets[best[1]]
        center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
        print(f"\n[{desc}]")
        for sc, i, cap in scored[:4]:
            print(f"   匹配{sc:.2f} #{i+1} '{cap}'")
        print(f"   -> 选 #{best[1]+1} '{best[2]}' 坐标 {center}")
        results.append({"target": desc, "picked": best[1] + 1, "caption": best[2],
                        "score": best[0], "center": center})

    # 标注复核图
    vis = full.convert("RGB").copy()
    d = ImageDraw.Draw(vis)
    for r in results:
        cx, cy = r["center"]
        d.line([(cx - 20, cy), (cx + 20, cy)], fill=(255, 0, 0), width=3)
        d.line([(cx, cy - 20), (cx, cy + 20)], fill=(255, 0, 0), width=3)
    vis = vis.resize((1280, int(sh * 1280 / sw)), Image.LANCZOS)
    vis.save(f"{OUT}/verify.png")
    with open(f"{OUT}/result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[复核图] {OUT}/verify.png 红叉=最终选择")


if __name__ == "__main__":
    main()
