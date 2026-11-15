#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端验证：YOLO 检测元素 + 7B 模型选编号，看能否选对。

对比两条路线：
  路线1（旧）：模型自己猜坐标  -> 实测误差 ±30~90px
  路线2（新）：YOLO 出编号清单 -> 模型做选择题 -> 坐标由 bbox 中心提供
"""
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from grid_probe import ask_model  # noqa: E402

OUT = "data/icons"


def build_list_text(elements, max_n=40):
    lines = []
    for e in elements[:max_n]:
        x1, y1, x2, y2 = e["bbox"]
        cx, cy = e["center"]
        lines.append(f"  编号{e['id']}: 位置({cx},{cy}) 大小{max(1,x2-x1)}x{max(1,y2-y1)}")
    return "\n".join(lines)


def main():
    img = Image.open(f"{OUT}/annotated.png")
    img = img.resize((1280, int(img.size[1] * 1280 / img.size[0])), Image.LANCZOS)
    with open(f"{OUT}/elements.json", encoding="utf-8") as f:
        elements = json.load(f)["elements"]
    lst = build_list_text(elements)

    targets = [
        "任务栏上的搜索框",
        "任务栏最左边的 Windows 开始按钮",
        "左侧边栏的'新建任务'按钮",
        "页面底部的输入框",
        "右上角的最小化按钮",
    ]

    system = (
        "你是屏幕操作助手。屏幕截图上每个可点击元素都有彩色编号框。\n"
        "可点击元素清单（编号: 中心坐标 大小）：\n" + lst + "\n\n"
        "任务：根据用户的描述，选出对应元素的编号。\n"
        "只回答一行：编号=<数字>。不要解释。"
    )

    results = []
    for t in targets:
        ans = ask_model(img, f"{system}\n\n用户要点击：{t}")
        first_line = ans.strip().splitlines()[0] if ans.strip() else ""
        print(f"[{t}] -> {first_line}")
        results.append({"target": t, "answer": ans.strip()})

    with open(f"{OUT}/select_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
