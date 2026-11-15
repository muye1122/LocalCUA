# -*- coding: utf-8 -*-
"""跨任务长期记忆回归：纯文件读写，不碰屏幕、不联网。"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.core import AgentSession          # noqa: E402
from agent.long_memory import LongMemory     # noqa: E402


def _sess(tmp_path):
    cfg = {
        "agent": {"long_memory": True,
                  "long_memory_path": str(tmp_path / "mem.json"),
                  "max_steps": 1, "home_desktop": False,
                  "entry_confirm": False, "change_settle": 0,
                  "frog_eye": False, "dialog_guard": False,
                  "type_guard": False, "wait_idle": False},
        "grounding": {"backend": "none"},
        "planner": {"enabled": False},
        "validator": {"enabled": False},
    }
    return AgentSession(cfg)


def test_note_open_persists_and_reloads(tmp_path):
    p = str(tmp_path / "mem.json")
    m = LongMemory(p)
    m.note_open("记事本", "无标题 - Notepad", "notepad.exe")
    m.note_open("记事本", "无标题 - Notepad", "notepad.exe")
    m.save()
    m2 = LongMemory(p)
    assert m2.apps["记事本"]["n"] == 2
    assert m2.apps["记事本"]["exe"] == "notepad.exe"


def test_note_fail_dedup_and_cap():
    with tempfile.TemporaryDirectory() as d:
        m = LongMemory(os.path.join(d, "m.json"))
        m.note_fail("在全屏里找输入框")
        m.note_fail("在全屏里找输入框")
        assert m.fails == ["在全屏里找输入框"]
        for i in range(20):
            m.note_fail(f"失败做法 {i}")
        assert len(m.fails) == 8
        assert m.fails[-1] == "失败做法 19"


def test_hint_is_short_and_only_matches_current_app(tmp_path):
    m = LongMemory(str(tmp_path / "m.json"))
    m.note_open("记事本", "无标题 - Notepad", "notepad.exe")
    m.note_open("Dev-C++", "小熊猫Dev-C++", "devcpp.exe")
    m.note_fail("全选文字再调字体")
    h = m.hint("在记事本里调字体")
    assert "跨任务经验" in h and "记事本" in h
    assert "Dev-C++" not in h        # 无关程序不注入，省 token
    assert len(h) <= 120


def test_disabled_is_noop(tmp_path):
    p = str(tmp_path / "m.json")
    m = LongMemory(p, enabled=False)
    m.note_open("微信", "微信", "wechat.exe")
    m.note_fail("乱点")
    m.save()
    assert m.apps == {} and m.fails == [] and not os.path.exists(p)
    assert m.hint("微信") == ""


def test_corrupted_file_does_not_crash(tmp_path):
    p = tmp_path / "m.json"
    p.write_text("{broken", encoding="utf-8")
    m = LongMemory(str(p))
    assert m.hint("任何任务") == ""


def test_core_injects_hint_and_drops_old_segment_block(tmp_path):
    sess = _sess(tmp_path)
    sess.long_memory.note_open("记事本", "无标题 - Notepad", "notepad.exe")
    txt = sess._build_user_text("在记事本里输入 hello", [])
    assert "跨任务经验" in txt and "Notepad" in txt
    assert "已经做完的段落" not in txt
    assert "subtask_done" not in txt


def test_note_fail_feeds_long_memory(tmp_path):
    sess = _sess(tmp_path)
    sess._note_fail("双击桌面图标没反应")
    assert any("双击桌面图标" in x for x in sess.long_memory.fails)


def test_run_saves_long_memory(tmp_path):
    """任务结束时才落盘，避免每步都写文件。"""
    sess = _sess(tmp_path)
    sess.long_memory.note_fail("换个像素位置再点一次")
    path = sess.long_memory.path
    assert not os.path.exists(path)
    sess.long_memory.save()
    assert os.path.exists(path)
    assert LongMemory(path).fails == ["换个像素位置再点一次"]
