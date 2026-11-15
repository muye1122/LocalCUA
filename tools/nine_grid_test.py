#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 7B 的定性能力：九宫格位置回答 + 局部候选选择。

第 1 步：全图问"目标在九宫格哪格"（左上/中上/.../右下），程序把该格内的 YOLO 元素筛出来
第 2 步：把目标格裁出来放大，候选元素画编号，模型在少量候选里选
"""
import json
import os
import sys

import pyautogui
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from grid_probe import ask_model, to_b64  # noqa: E402
from icon_detect import detect  # noqa: E402
import onnxruntime as ort  # noqa: E402

OUT = "data/nine"
CELLS = ["左上", "中上", "右上", "左中", "中心", "右中", "左下", "中下", "右下"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "icon_detect", "model.onnx")


def cell_bbox(name, sw, sh):
    c = CELLS.index(name)
    col, row = c % 3, c // 3
    # 边缘格放宽：任务栏贴底，给出格的边界留 5% 余量
    x0, x1 = col * sw / 3, (col + 1) * sw / 3
    y0, y1 = row * sh / 3, (row + 1) * sh / 3
    pad_x, pad_y = sw * 0.16, sh * 0.12
    return (max(0, x0 - pad_x), max(0, y0 - pad_y),
            min(sw, x1 + pad_x), min(sh, y1 + pad_y))


def main():
    os.makedirs(OUT, exist_ok=True)
    full = pyautogui.screenshot()
    sw, sh = full.size
    sess = ort.InferenceSession(os.path.abspath(MODEL_PATH), providers=["CPUExecutionProvider"])
    dets = detect(full, sess, conf=0.3)

    targets = [
        ("任务栏上的搜索框", "搜索框"),
        ("任务栏最左边的 Windows 开始按钮", "开始按钮"),
        ("左侧边栏顶部的'新建任务'按钮", "新建任务"),
        ("页面底部的聊天输入框", "输入框"),
        ("右上角的最小化按钮", "最小化按钮"),
    ]

    small = full.resize((1280, int(sh * 1280 / sw)), Image.LANCZOS)
    results = []
    for desc, short in targets:
        # --- 第 1 步：分轴定性（x 轴、y 轴各问一次，比九宫格单选容易）---
        qx = (f"这是一张电脑屏幕截图。要找的目标是：{desc}\n"
              "目标的中心在屏幕横向的哪个位置？只回答一个词：左 / 中 / 右")
        ax = ask_model(small, qx).strip().splitlines()[0]
        qy = (f"这是一张电脑屏幕截图。要找的目标是：{desc}\n"
              "目标的中心在屏幕纵向的哪个位置？只回答一个词：上 / 中 / 下")
        ay = ask_model(small, qy).strip().splitlines()[0]
        col = 0 if "左" in ax else (2 if "右" in ax else 1)
        row = 0 if "上" in ay else (2 if "右" in ay or "下" in ay else 1)
        cell = CELLS[row * 3 + col]
        print(f"[{short}] 横向:{ax} 纵向:{ay} -> {cell}")
        if not cell:
            results.append({"target": short, "cell": None})
            continue

        # --- 程序按格过滤候选 ---
        bx0, by0, bx1, by1 = cell_bbox(cell, sw, sh)
        cands = [d for d in dets
                 if bx0 <= (d[0] + d[2]) / 2 < bx1 and by0 <= (d[1] + d[3]) / 2 < by1]
        print(f"[{short}] 格子内候选: {len(cands)} 个")

        # --- 第 2 步：裁格放大，候选画编号 ---
        crop = full.crop((int(bx0), int(by0), int(bx1), int(by1))).convert("RGB")
        scale = min(2.0, 1000 / max(crop.size))
        crop = crop.resize((int(crop.size[0] * scale), int(crop.size[1] * scale)), Image.LANCZOS)
        if crop.size[0] > 1568:
            crop = crop.resize((1568, int(crop.size[1] * 1568 / crop.size[0])), Image.LANCZOS)
        d = ImageDraw.Draw(crop)
        try:
            font = ImageFont.truetype("consola.ttf", 26)
        except Exception:
            font = ImageFont.load_default()
        cand_info = []
        for i, (x1, y1, x2, y2, s) in enumerate(cands):
            rx1, ry1 = (x1 - bx0) * scale, (y1 - by0) * scale
            rx2, ry2 = (x2 - bx0) * scale, (y2 - by0) * scale
            d.rectangle([rx1, ry1, rx2, ry2], outline=(230, 57, 70), width=3)
            ty0, ty1 = max(0, ry1 - 28), max(20, ry1)
            d.rectangle([rx1, ty0, rx1 + 40, ty1], fill=(230, 57, 70))
            d.text((rx1 + 5, max(0, ry1 - 26)), str(i + 1), fill=(255, 255, 255), font=font)
            cand_info.append({"n": i + 1, "bbox": [round(v, 1) for v in (x1, y1, x2, y2)]})
        crop.save(f"{OUT}/{short}_candidates.png")

        n_list = "、".join(str(c["n"]) for c in cand_info)
        q2 = (f"这张图里标了 {len(cand_info)} 个编号框（编号：{n_list}）。\n"
              f"要找的目标是：{desc}\n"
              "目标是哪个编号？只回答：编号=<数字>")
        a2 = ask_model(crop, q2).strip().splitlines()[0]
        import re
        m = re.search(r"\d+", a2)
        pick = int(m.group()) if m else None
        print(f"[{short}] 候选选择答: {a2}  -> 编号{pick}")
        final = None
        if pick and 1 <= pick <= len(cand_info):
            b = cand_info[pick - 1]["bbox"]
            final = [round((b[0] + b[2]) / 2), round((b[1] + b[3]) / 2)]
            print(f"[{short}] 最终坐标: {final}")
        results.append({"target": short, "cell": cell, "n_cands": len(cand_info),
                        "picked": pick, "final": final})

    with open(f"{OUT}/result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n==== 汇总 ====")
    for r in results:
        print(f"  {r['target']}: 格={r.get('cell')} 候选={r.get('n_cands')} "
              f"选={r.get('picked')} 坐标={r.get('final')}")


if __name__ == "__main__":
    main()
