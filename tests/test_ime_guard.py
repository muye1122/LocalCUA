# -*- coding: utf-8 -*-
"""中文输入法守卫回归：纯假对象，不碰真机 IME / 屏幕 / 键鼠。

真机 bug：用户给一个「，」，发出去是「，，」。
根因是中文态 IME 拦截 KEYEVENTF_UNICODE 注入的全角标点后自己又补一个，
不是 clean_message_text 能管的（那发生在敲入之前）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import agent.actions as acts                     # noqa: E402
from agent.actions import _ImeGuard              # noqa: E402  (真实类，非 fixture 替换版)
from agent.core import clean_message_text        # noqa: E402


class FakeUser:
    def GetForegroundWindow(self):
        return 999


class FakeImm:
    def __init__(self, open_status=True, himc=1234):
        self.open = open_status
        self.himc = himc
        self.calls = []

    def ImmGetContext(self, hwnd):
        self.calls.append(("get", hwnd))
        return self.himc

    def ImmGetOpenStatus(self, h):
        self.calls.append(("status", h))
        return 1 if self.open else 0

    def ImmSetOpenStatus(self, h, flag):
        self.calls.append(("set", h, bool(flag)))
        return 1

    def ImmReleaseContext(self, hwnd, h):
        self.calls.append(("release", hwnd, h))
        return 1


def test_guard_closes_ime_then_restores():
    imm = FakeImm(open_status=True)
    with _ImeGuard(imm32=imm, user32=FakeUser()):
        assert ("set", 1234, False) in imm.calls
    assert ("set", 1234, True) in imm.calls
    assert ("release", 999, 1234) in imm.calls


def test_guard_restores_even_when_body_raises():
    imm = FakeImm(open_status=True)
    try:
        with _ImeGuard(imm32=imm, user32=FakeUser()):
            raise RuntimeError("敲字中途炸了")
    except RuntimeError:
        pass
    assert ("set", 1234, True) in imm.calls, "异常也必须把输入法还回去"
    assert ("release", 999, 1234) in imm.calls


def test_guard_leaves_ime_alone_when_already_off():
    imm = FakeImm(open_status=False)
    with _ImeGuard(imm32=imm, user32=FakeUser()):
        pass
    assert not any(c[0] == "set" and c[2] is False for c in imm.calls)
    assert ("release", 999, 1234) in imm.calls


def test_guard_noops_without_ime_context():
    imm = FakeImm(himc=0)
    with _ImeGuard(imm32=imm, user32=FakeUser()):
        pass
    assert imm.calls == [("get", 999)]


def test_type_unicode_uses_guard_by_default(monkeypatch):
    class Spy:
        entered = 0
        exited = 0

        def __enter__(self):
            Spy.entered += 1
            return self

        def __exit__(self, *a):
            Spy.exited += 1
            return False

    monkeypatch.setattr(acts, "_ImeGuard", Spy)
    acts._type_unicode("，", interval=0)
    assert Spy.entered == 1 and Spy.exited == 1


def test_type_unicode_can_opt_out(monkeypatch):
    class Spy:
        entered = 0

        def __enter__(self):
            Spy.entered += 1
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(acts, "_ImeGuard", Spy)
    acts._type_unicode("，", interval=0, ime_guard=False)
    assert Spy.entered == 0


def test_clean_message_text_still_collapses_a_double_comma():
    """发送前的字符串清洗仍然负责用户本来就打重的标点，两者分工不冲突。"""
    assert clean_message_text("你好，，") == "你好，"
    assert clean_message_text("，") == "，"


def test_single_punctuation_must_be_detectable(monkeypatch):
    """真机「一个逗号变两个」的真因：单个标点太小，旧差分阈值检不出 → 重敲。

    2560x1600 上一个约 10x10 的逗号只占 0.0024% 像素；
    默认参数 (thr=0.003,max_w=320) 和 type 分支的 thr=0.0001 都看不见，
    只有原分辨率 + thr=0.00001 能检出。_send_message 现在用的就是这组参数。
    """
    from PIL import Image, ImageDraw
    from agent.core import AgentSession

    a = Image.new("RGB", (2560, 1600), "white")
    b = a.copy()
    ImageDraw.Draw(b).ellipse([1280, 800, 1290, 810], fill="black")
    assert AgentSession._changed(a, b) is False
    assert AgentSession._changed(a, b, thr=0.0001, max_w=None) is False
    assert AgentSession._changed(a, b, thr=0.00001, max_w=None) is True
