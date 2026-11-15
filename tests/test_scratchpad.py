"""草稿纸测试：保证「任务之间不混」「只回填最近几步」「不堆积文件」。"""
import os

from agent.scratchpad import Scratchpad


def test_start_writes_task_and_isolates_runs(tmp_path):
    sp = Scratchpad(root=str(tmp_path))
    sp.start("打开记事本")
    sp.note(1, "点开始按钮", "已执行")
    p1 = sp.path
    assert os.path.exists(p1)
    assert "打开记事本" in open(p1, encoding="utf-8").read()

    sp.start("给文件传输助手发消息")   # 第二个任务
    p2 = sp.path
    assert p1 != p2, "每个任务必须是新文件"
    body2 = open(p2, encoding="utf-8").read()
    assert "给文件传输助手发消息" in body2
    assert "打开记事本" not in body2, "上一个任务的内容不能混进来"


def test_recent_returns_only_last_n(tmp_path):
    sp = Scratchpad(root=str(tmp_path))
    sp.start("t")
    for i in range(1, 11):
        sp.note(i, f"动作{i}", "已执行")
    got = sp.recent(3)
    assert "动作8" in got and "动作9" in got and "动作10" in got
    assert "动作7" not in got
    assert got.count("\n") == 3


def test_prune_keeps_recent_max_files(tmp_path):
    sp = Scratchpad(root=str(tmp_path))
    sp.MAX_FILES = 3
    for i in range(6):
        sp.start(f"任务{i}")
    left = sorted(f for f in os.listdir(tmp_path) if f.endswith(".md"))
    assert len(left) == 3


def test_note_survives_unwritable_path(tmp_path):
    """草稿纸写不进去也不能把主流程拖垮。"""
    sp = Scratchpad(root=os.path.join(str(tmp_path), "不存在的盘"))
    sp.path = "Z:/definitely/not/here/x.md"
    sp.note(1, "点按钮", "已执行")   # 不抛异常
    assert len(sp.lines) == 1
