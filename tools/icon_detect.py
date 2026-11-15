#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OmniParser icon_detect（ONNX 版）：把屏幕截图里的可点击元素全检测出来。

YOLO 专项定位模型，bbox 精度 ±3px，彻底解决 7B 模型"猜坐标不准"的问题：
模型只负责在编号清单里"选编号"，坐标由 YOLO 提供。

用法：
  ./.venv/Scripts/python.exe tools/icon_detect.py [--conf 0.3] [--out data/icons]
输出：data/icons/annotated.png（编号标注图）+ data/icons/elements.json
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import pyautogui
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "icon_detect", "model.onnx")
IMG_SIZE = 640
COLORS = [(230, 57, 70), (29, 130, 216), (42, 157, 143), (231, 111, 81), (154, 106, 195)]


def letterbox(img: Image.Image):
    """等比缩放补灰边到 640x640，返回 (np数组, 缩放参数)。"""
    w, h = img.size
    r = min(IMG_SIZE / w, IMG_SIZE / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (114, 114, 114))
    dx, dy = (IMG_SIZE - nw) // 2, (IMG_SIZE - nh) // 2
    canvas.paste(resized, (dx, dy))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0  # HWC
    arr = arr.transpose(2, 0, 1)[None]  # 1,3,640,640
    return arr, r, dx, dy


def nms(boxes, scores, iou_thr=0.45):
    idx = scores.argsort()[::-1]
    keep = []
    while idx.size:
        i = idx[0]
        keep.append(i)
        if idx.size == 1:
            break
        rest = idx[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.clip(a1 + a2 - inter, 1e-9, None)
        idx = rest[iou <= iou_thr]
    return keep


def detect(img: Image.Image, session: ort.InferenceSession, conf: float = 0.3):
    """返回原图坐标系的 [x1,y1,x2,y2,score] 列表。"""
    arr, r, dx, dy = letterbox(img)
    out = session.run(None, {"images": arr})[0]  # (1,5,8400)
    pred = out[0].T  # (8400,5) cx,cy,w,h,score
    mask = pred[:, 4] >= conf
    pred = pred[mask]
    if len(pred) == 0:
        return []
    # cxcywh -> xy1xy2
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep = nms(boxes, pred[:, 4])
    boxes, scores = boxes[keep], pred[keep, 4]
    # 去掉 letterbox 灰边 -> 原图坐标
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / r
    iw, ih = img.size
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, iw)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, ih)
    order = scores.argsort()[::-1]
    return [[round(float(v), 1) for v in boxes[i]] + [round(float(scores[i]), 3)] for i in order]


def annotate(img: Image.Image, dets, out_path):
    vis = img.convert("RGB").copy()
    try:
        font = ImageFont.truetype("consola.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(vis)
    for i, (x1, y1, x2, y2, s) in enumerate(dets):
        c = COLORS[i % len(COLORS)]
        d.rectangle([x1, y1, x2, y2], outline=c, width=3)
        d.rectangle([x1, max(0, y1 - 24), x1 + 52, y1], fill=c)
        d.text((x1 + 4, max(0, y1 - 22)), str(i + 1), fill=(255, 255, 255), font=font)
    vis.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", default="data/icons")
    ap.add_argument("--img", default=None, help="用已有图片代替截屏")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sess = ort.InferenceSession(os.path.abspath(MODEL_PATH), providers=["CPUExecutionProvider"])

    import time
    if args.img:
        img = Image.open(args.img)
    else:
        t0 = time.time()
        img = pyautogui.screenshot()
        shot_ms = (time.time() - t0) * 1000

    t0 = time.time()
    dets = detect(img, sess, args.conf)
    det_ms = (time.time() - t0) * 1000

    annotate(img, dets, f"{args.out}/annotated.png")
    elements = [{"id": i + 1, "bbox": [int(v) for v in b[:4]],
                 "center": [int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)],
                 "score": b[4]} for i, b in enumerate(dets)]
    with open(f"{args.out}/elements.json", "w", encoding="utf-8") as f:
        json.dump({"size": img.size, "elements": elements}, f, ensure_ascii=False, indent=1)

    print(f"[截屏] {img.size[0]}x{img.size[1]}" + (f" {shot_ms:.0f}ms" if not args.img else ""))
    print(f"[检测] {len(dets)} 个元素  耗时 {det_ms:.0f}ms")
    for e in elements[:12]:
        print(f"  #{e['id']:>3} bbox={e['bbox']} center={e['center']} score={e['score']}")
    if len(elements) > 12:
        print(f"  ... 共 {len(elements)} 个，完整清单见 {args.out}/elements.json")
    print(f"[标注图] {args.out}/annotated.png")


if __name__ == "__main__":
    main()
