"""规划器测试：拆解、当前子任务、标记完成、重规划、关闭。"""

from agent.planner import Planner
from conftest import FakeModel


def test_decompose_enabled():
    m = FakeModel(replies=["1. 打开应用\n2. 输入内容\n3. 保存"])
    p = Planner({"planner": {"enabled": True}}, m)
    subs = p.decompose("做个事")
    assert subs == ["打开应用", "输入内容", "保存"]
    assert p.current() == "打开应用"
    p.mark_done()
    assert p.current() == "输入内容"
    p.mark_done()
    p.mark_done()
    assert p.is_complete()


def test_decompose_disabled():
    p = Planner({"planner": {"enabled": False}}, FakeModel())
    assert p.decompose("单步任务") == ["单步任务"]


def test_parse_bullets():
    p = Planner({"planner": {"enabled": True}}, FakeModel(replies=["- a\n- b\n- c"]))
    assert p.decompose("x") == ["a", "b", "c"]


def test_replan_preserves_done_and_appends():
    m = FakeModel(replies=["1. 打开\n2. 输入", "1. 重试\n2. 收尾"])
    p = Planner({"planner": {"enabled": True, "replan_on_fail": True}}, m)
    p.decompose("原任务")
    before = list(p.subtasks)
    p.idx = 1  # 假设已完成第 1 步
    p.replan("原任务", "点错了")
    assert p.subtasks[:1] == before[:1]
    assert p.subtasks[1:] == ["重试", "收尾"]


def test_replan_off_advances():
    m = FakeModel(replies=["1. 打开"])
    p = Planner({"planner": {"enabled": True, "replan_on_fail": False}}, m)
    p.decompose("原任务")
    p.idx = 1
    p.replan("原任务", "fail")
    assert p.idx == 2
