"""轨迹记忆测试：开始、存图、记录、列举、回放。"""

import os
import tempfile

from PIL import Image

from agent.memory import TrajectoryMemory


def test_record_and_load():
    d = tempfile.mkdtemp()
    mem = TrajectoryMemory(base_dir=d)
    mem.start("测试任务")
    img = Image.new("RGB", (10, 10), (0, 0, 0))
    bp = mem.save_image(img, "a.png")
    assert os.path.exists(bp)
    mem.add(1, {"action": "click", "x": 5, "y": 5}, "ok", bp, None)

    runs = TrajectoryMemory.list_runs(d)
    assert len(runs) == 1
    loaded = TrajectoryMemory.load(os.path.join(d, runs[0]))
    assert len(loaded) == 1
    assert loaded[0]["action"]["action"] == "click"
    assert loaded[0]["note"] == "ok"
