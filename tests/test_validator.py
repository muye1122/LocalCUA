"""校验器测试：关闭时直接通过；开启时按 VLM 返回 YES/NO 判定。"""

from PIL import Image

from agent.validator import Validator
from conftest import FakeModel

_IMG = Image.new("RGB", (4, 4), (0, 0, 0))


def test_disabled_returns_ok():
    v = Validator({"validator": {"enabled": False}}, FakeModel())
    ok, note = v.verify("click", _IMG, _IMG)
    assert ok is True and note == ""


def test_enabled_yes():
    v = Validator({"validator": {"enabled": True}}, FakeModel(replies=["YES 成功"]))
    ok, note = v.verify("click", _IMG, _IMG)
    assert ok is True and "成功" in note


def test_enabled_no():
    v = Validator({"validator": {"enabled": True}}, FakeModel(replies=["NO 没点中"]))
    ok, note = v.verify("click", _IMG, _IMG)
    assert ok is False and "没点中" in note


def test_enabled_call_failure_is_tolerated():
    def boom(messages, temperature=None):
        raise RuntimeError("model down")

    v = Validator({"validator": {"enabled": True}}, boom)
    ok, note = v.verify("click", _IMG, _IMG)
    assert ok is True and "校验调用失败" in note
