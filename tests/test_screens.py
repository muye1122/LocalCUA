#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""新功能测试：
  1) 截图滚动保留（超过上限删最旧）
  2) 任务结束清理所有截图
  3) 动作后「无新内容」判定：复用截图、不重复截屏
  4) 兜底像素差检测能识破模型误判
"""

import os
import tempfile
import threading

from PIL import Image

from agent.core import AgentSession
from agent.memory import TrajectoryMemory
from conftest import FakeModel


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, action):
        self.calls.append(action)
        a = action.get("action")
        if a == "done":
            return False
        if a == "ask":
            return "ASK"
        return True


def _cfg():
    return {
        "grounding": {"backend": "none"},
        "planner": {"enabled": True},
        "validator": {"enabled": False},
        "agent": {"max_steps": 6, "step_pause": 0, "max_screenshots": 3,
                  "type_guard": False, "dialog_guard": False,
                  "wait_idle": False, "home_desktop": False, "frog_eye": False},
        "safety": {"confirm_risky": False},
    }


def test_memory_rolling_limit():
    d = tempfile.mkdtemp()
    mem = TrajectoryMemory(base_dir=d, max_images=3)
    mem.start("滚动测试")
    img = Image.new("RGB", (8, 8), (1, 2, 3))
    for i in range(5):
        mem.save_image(img, f"f{i}.png")
    pngs = [f for f in os.listdir(mem.dir) if f.endswith(".png")]
    assert len(pngs) == 3
    assert "f0.png" not in pngs and "f1.png" not in pngs   # 最旧两张被删
    assert "f4.png" in pngs


def test_cleanup_on_finish():
    d = tempfile.mkdtemp()
    shots = []

    def shot():
        shots.append(1)
        return Image.new("RGB", (64, 48), (255, 255, 255))

    model = FakeModel(replies=[
        "1. 一步",
        '{"action":"click","x":10,"y":10,"expect_change":false}',
        '{"action":"done","result":"完成"}',
    ])
    sess = AgentSession(_cfg(), model=model, executor=FakeExecutor(),
                        screenshot_fn=shot, stop_event=threading.Event())
    sess.memory = TrajectoryMemory(base_dir=d, max_images=50)
    assert sess.run("测试任务") == "DONE"

    # 截图全部被清理，只留文字轨迹
    left = os.listdir(sess.memory.dir)
    assert not [f for f in left if f.endswith(".png")]
    assert "trajectory.jsonl" in left


def test_no_new_content_reuses_screenshot():
    """expect_change=false 且画面未变 -> 下一步复用旧截图，不再新增截屏。"""
    d = tempfile.mkdtemp()
    shots = []

    def shot():
        shots.append(1)
        return Image.new("RGB", (64, 48), (255, 255, 255))  # 画面恒定

    model = FakeModel(replies=[
        "1. 一步",
        '{"action":"wait","ms":1,"expect_change":false}',
        '{"action":"done","result":"完成"}',
    ])
    sess = AgentSession(_cfg(), model=model, executor=FakeExecutor(),
                        screenshot_fn=shot, stop_event=threading.Event())
    sess.memory = TrajectoryMemory(base_dir=d, max_images=50)
    sess.run("测试任务")

    # 步1：开头 1 次 + 动作后兜底抽检 1 次；步2 done 不截屏 -> 共 2 次
    assert len(shots) == 2


def test_changed_detects_difference():
    a = Image.new("RGB", (640, 480), (255, 255, 255))
    b = Image.new("RGB", (640, 480), (0, 0, 0))
    assert AgentSession._changed(a, a) is False
    assert AgentSession._changed(a, b) is True


def test_expect_change_false_but_screen_changed():
    """模型说不会变、实际变了 -> 兜底判定应认为「变了」，下一步用新图。"""
    a = Image.new("RGB", (640, 480), (250, 250, 250))
    b = Image.new("RGB", (640, 480), (250, 250, 250))
    # 只改一小块区域，模拟弹窗
    from PIL import ImageDraw
    d = ImageDraw.Draw(b)
    d.rectangle([100, 100, 500, 400], fill=(30, 30, 30))
    assert AgentSession._changed(a, b) is True


def test_reused_screen_skips_ocr():
    """画面没变时 OCR 必须跳过 —— 它每步 2.0 秒，是 YOLO 的 8 倍。

    真机实测（2560×1600）：YOLO 235ms、OCR 2000ms。原先只有 YOLO 受
    _reused_screen 保护，OCR 无条件是每步都跑，连「点一下已经选中的按钮」
    这种画面根本没动的步骤也照跑一遍，白占整步时间的约三成。
    """
    d = tempfile.mkdtemp()
    ocr_calls = []

    class G:
        def __init__(self):
            self._last_elements = []

        def annotate(self, img):
            return ([{"id": 1, "label": "按钮", "bbox": [10, 10, 40, 30],
                      "center": [25, 20], "score": 0.9}], img.copy())

        def ocr_enrich(self, img, elements, scale=1.0):
            ocr_calls.append(1)
            return []

    model = FakeModel(replies=[
        "1. 一步",
        '{"action":"wait","ms":1,"expect_change":false}',
        '{"action":"done","result":"完成"}',
    ])
    sess = AgentSession(_cfg(), model=model, executor=FakeExecutor(),
                        screenshot_fn=lambda: Image.new("RGB", (64, 48), (255, 255, 255)),
                        stop_event=threading.Event())
    sess.grounding = G()
    sess.memory = TrajectoryMemory(base_dir=d, max_images=50)
    sess.run("测试任务")
    assert len(ocr_calls) == 1, "画面没变的第二步不该再跑 OCR"
