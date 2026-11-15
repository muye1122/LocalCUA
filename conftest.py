import os
import sys

import pytest

# 让 pytest 能直接 import agent 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def no_real_gui(monkeypatch):
    """跑测试绝不碰真机屏幕 / 键鼠 / 截图。

    有些用例会走到 _show_desktop / _open_app 这类真实模板，默认会按 Win+D、
    点鼠标、截屏。人在上课时这些动作会打断他，所以统一换成空操作；
    需要断言「按了哪些键」的用例会自己再 monkeypatch 覆盖。
    """
    try:
        import pyautogui
        from PIL import Image

        noop = lambda *a, **k: None
        for name in ("click", "doubleClick", "rightClick", "moveTo", "moveRel",
                     "dragTo", "dragRel", "mouseDown", "mouseUp", "scroll",
                     "hotkey", "press", "keyDown", "keyUp", "typewrite", "write"):
            if hasattr(pyautogui, name):
                monkeypatch.setattr(pyautogui, name, noop, raising=False)
        monkeypatch.setattr(
            pyautogui, "screenshot",
            lambda *a, **k: Image.new("RGB", (64, 48), "white"), raising=False)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def isolate_cursor(monkeypatch):
    """测试不该依赖真机光标位置。

    真机跑完常把光标停在 (0,0)，pyautogui 的急停保护会让任何鼠标调用直接抛
    FailSafeException；而沙箱里 SetCursorPos 可能被拒绝，根本挪不开（实测返回 0）。
    统一钉死成屏幕中间的安全坐标，并在测试进程里关掉急停检查。
    """
    try:
        import pyautogui
        monkeypatch.setattr(pyautogui, "FAILSAFE", False)
        # 三道保险：有些调用链读的是 FAILSAFE_POINTS 或当次 position()，
        # 只改 FAILSAFE 挡不住。屏幕角落锁死时把这两样也钉成安全值。
        monkeypatch.setattr(pyautogui, "FAILSAFE_POINTS", [])
        _pt = pyautogui.position()
        _safe = type(_pt)(640, 400)
        monkeypatch.setattr(pyautogui, "position", lambda: _safe)
    except Exception:
        pass
    try:
        # SendInput 在没有交互桌面 / 会话受限时会被拒绝，测试不该因此挂掉。
        # _send_input 本来就是为可测抽出来的；需要断言按键的用例会自己再覆盖。
        import agent.actions as acts
        monkeypatch.setattr(acts, "_send_input", lambda kb: None)
    except Exception:
        pass
    try:
        import agent.core as core
        monkeypatch.setattr(core.AgentSession, "_cursor_pos",
                            staticmethod(lambda: (640, 400)))
    except Exception:
        pass


class FakeModel:
    """测试用假模型：按序返回预设回复，不做任何网络调用。"""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = 0

    def chat(self, messages, temperature=None, max_tokens=None):
        self.calls += 1
        if self.replies:
            return self.replies.pop(0)
        return "done"

    def chat_coder(self, messages, temperature=0.1, max_tokens=None):
        """写代码走同一条假通道（真 ModelClient 会优先用 coder 模型）。"""
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)


@pytest.fixture(autouse=True)
def no_real_ime(monkeypatch):
    """测试绝不改动用户的中文输入法状态。

    _type_unicode 现在会临时关 IME 再敲字；单元测试里换成空 guard，
    避免跑测试时改动使用者的输入法状态。测 _ImeGuard 本身的用例会自己注入假对象。
    """
    try:
        import agent.actions as acts

        class _NoIme:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(acts, "_ImeGuard", _NoIme, raising=False)
    except Exception:
        pass
