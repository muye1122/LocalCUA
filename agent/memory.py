#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轨迹记忆：把每一步的动作记录成 JSONL，截图做环形缓冲（只留最近 N 张）。

目录结构：data/trajectories/<时间戳>_<任务>/
    trajectory.jsonl   逐步记录（文本，很小，保留用于复盘）
    step_001_before.png ... （滚动保留最近 max_images 张，超出自动删最旧）
    meta.json

任务结束时调用 cleanup_images()：本目录内所有截图全删。
"""

import json
import os
from collections import deque
from datetime import datetime

from PIL import Image


class TrajectoryMemory:
    def __init__(self, base_dir: str = "data/trajectories", max_images: int = 50):
        self.base_dir = base_dir
        self.max_images = int(max_images)
        self.dir = None
        self.path = None
        self.steps = []
        self._images = deque()      # 环形缓冲：只保留最近 max_images 张
        self.saved_count = 0

    def start(self, task: str):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() else "_" for c in task)[:40]
        self.dir = os.path.join(self.base_dir, f"{ts}_{safe}")
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "trajectory.jsonl")
        self.steps = []
        self._images.clear()
        self.saved_count = 0
        with open(os.path.join(self.dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"task": task, "started": ts}, f, ensure_ascii=False, indent=2)

    def save_image(self, img: Image.Image, name: str) -> str:
        """保存截图；超过 max_images 张时删掉最旧的一张（滚动窗口）。"""
        p = os.path.join(self.dir, name)
        img.save(p)
        self.saved_count += 1
        self._images.append(p)
        while len(self._images) > self.max_images:
            old = self._images.popleft()
            try:
                os.remove(old)
            except OSError:
                pass
        return p

    def cleanup_images(self) -> int:
        """删除本次运行目录下的所有截图，返回删除数量。
        保留 trajectory.jsonl / meta.json（纯文本、体积很小，便于复盘）。

        CU_KEEP_SHOTS=1 时保留截图：排查"模型到底看到了什么"只有这条路。
        """
        if os.environ.get("CU_KEEP_SHOTS"):
            self._images.clear()
            return 0
        self._images.clear()
        if not self.dir or not os.path.isdir(self.dir):
            return 0
        removed = 0
        for fn in os.listdir(self.dir):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    os.remove(os.path.join(self.dir, fn))
                    removed += 1
                except OSError:
                    pass
        return removed

    def add(self, step_idx, action, note="", before=None, after=None,
            elapsed=None, think="", result=""):
        """记一步。

        除动作本身，还落三个字段供事后分析（tools/analyze_runs.py 会用）：
          elapsed —— 这一步真正花了多少秒（定位性能退化的第一信号）
          think   —— 模型自报的 plan（看它是不是在复述任务名）
          result  —— ok / nochange / blocked / done …（失败模式聚类用）
        """
        rec = {
            "step": step_idx,
            "action": action,
            "note": note,
            # 截图在结束时会被删除，这里不写路径以免误导
            "before": None,
            "after": None,
        }
        if elapsed is not None:
            rec["elapsed"] = round(float(elapsed), 2)
        if think:
            rec["think"] = think
        if result:
            rec["result"] = result
        self.steps.append(rec)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def list_runs(base_dir: str = "data/trajectories"):
        if not os.path.isdir(base_dir):
            return []
        return sorted(
            [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))],
            reverse=True,
        )

    @staticmethod
    def load(run_dir: str):
        path = os.path.join(run_dir, "trajectory.jsonl")
        if not os.path.isfile(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
