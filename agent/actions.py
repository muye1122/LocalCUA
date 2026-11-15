#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""动作执行：把模型给出的结构化动作真正落到鼠标键盘上。

关键点：优先用 grounding 解析出的 element_id 得到元素中心坐标，
比模型直接给的绝对坐标更准（这是 v2 点得准的核心之一）。
中文输入走 Windows 剪贴板粘贴（ctypes 实现，无需额外依赖）。
"""

import time
import ctypes
from contextlib import nullcontext

try:
    import pyautogui
except Exception:  # pragma: no cover - 仅在没有 GUI / 显示器时触发
    pyautogui = None


class _ImeGuard:
    """临时关掉中文输入法（IME）再敲字，敲完无条件恢复。

    真机 bug：用户给一个「，」，发出去变成「，，」。
    锅不在 clean_message_text —— 那是敲入之前的字符串清洗，够不着这里。
    真正的原因是微软拼音在中文态下拦截 KEYEVENTF_UNICODE 注入的全角标点，
    自己又补一个。关掉 IME 后 UNICODE 事件直达窗口，一个字符就是一个字符。

    只在拿到 IME 上下文、且当前确实处于打开状态时才改动；__exit__ 里恢复，
    try/finally 语义由 with 保证，绝不把用户的输入法留在英文态。
    imm32 / user32 可注入，测试用假对象，不碰真机。
    """

    def __init__(self, imm32=None, user32=None):
        self._imm = imm32
        self._u = user32
        self._hwnd = None
        self._himc = None
        self._was_open = False

    def __enter__(self):
        try:
            if self._imm is None or self._u is None:
                # 64 位下句柄是指针；ctypes 默认 restype=c_int 会截断，必须显式声明
                imm = ctypes.windll.imm32
                u = ctypes.windll.user32
                u.GetForegroundWindow.restype = ctypes.c_void_p
                imm.ImmGetContext.restype = ctypes.c_void_p
                imm.ImmGetContext.argtypes = [ctypes.c_void_p]
                imm.ImmGetOpenStatus.argtypes = [ctypes.c_void_p]
                imm.ImmSetOpenStatus.argtypes = [ctypes.c_void_p, ctypes.c_bool]
                imm.ImmReleaseContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            else:
                imm, u = self._imm, self._u
            hwnd = u.GetForegroundWindow()
            if not hwnd:
                return self
            himc = imm.ImmGetContext(hwnd)
            if not himc:
                return self
            was = bool(imm.ImmGetOpenStatus(himc))
            if was:
                imm.ImmSetOpenStatus(himc, False)
            self._hwnd, self._himc, self._was_open = hwnd, himc, was
        except Exception:
            pass   # 拿不到 IME（非 Windows / 无窗口）就按原样敲，绝不因此中断任务
        return self

    def __exit__(self, *exc):
        try:
            if self._himc:
                imm = self._imm or ctypes.windll.imm32
                if self._was_open:
                    imm.ImmSetOpenStatus(self._himc, True)
                imm.ImmReleaseContext(self._hwnd, self._himc)
        except Exception:
            pass
        return False


def _park_if_corner():
    """鼠标停在屏幕边缘/角落时，pyautogui 的急停保护会让每一次点击都失败。
    这里用 Win32 直接挪到屏幕中心（绕过 pyautogui 的检查），保证动作能继续。"""
    if pyautogui is None:
        return
    try:
        x, y = pyautogui.position()
        w, h = pyautogui.size()
    except Exception:
        return
    if x <= 1 or y <= 1 or x >= w - 2 or y >= h - 2:
        ctypes.windll.user32.SetCursorPos(w // 2, h // 2)


# ---- SendInput 结构体：模块级定义一次（原来每次敲字都重建一遍类，纯浪费）----
class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _INPUTUNION(ctypes.Union):
    # mi 必须在：union 大小由最大成员决定
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_ulong), ("u", _INPUTUNION)]


INPUT_KEYBOARD = 1
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002
VK_RETURN = 0x0D


def _send_input(kb: _KEYBDINPUT):
    """发一条键盘事件。单独抽出来是为了可测：测试里替换它就能断言"发了哪些键"。"""
    ii = _INPUT(type=INPUT_KEYBOARD)
    ii.ki = kb
    if ctypes.windll.user32.SendInput(1, ctypes.byref(ii), ctypes.sizeof(_INPUT)) != 1:
        raise RuntimeError(f"按键未被系统接受（wVk={kb.wVk}, wScan={kb.wScan}）")


def _type_unicode(text: str, interval: float = 0.04, ime_guard: bool = True):
    """用 SendInput 的 KEYEVENTF_UNICODE 逐字符模拟真实按键敲入文字。
    不是剪贴板粘贴——每个字符都是一次独立的键盘事件，效果等同手打（支持中文）。

    换行单独走 VK_RETURN：U+000A 在很多控件（含记事本）里会被当成"无字符"吞掉，
    整段多行文字会挤成一行 —— 写诗、写代码全毁在这一条上。

    ime_guard：中文输入法会把 UNICODE 注入的全角标点二次加工成双份
    （真机：给一个「，」发出去是「，，」）。敲字期间临时关 IME，结束立刻恢复。
    传 False 可退回旧行为。
    """
    with (_ImeGuard() if ime_guard else nullcontext()):
        for ch in str(text):
            if ch == "\n":
                _send_input(_KEYBDINPUT(VK_RETURN, 0, 0, 0, None))
                time.sleep(min(0.02, interval * 2))
                _send_input(_KEYBDINPUT(VK_RETURN, 0, KEYEVENTF_KEYUP, 0, None))
                time.sleep(interval * 2)
                continue
            if ch == "\r":
                continue                            # \r\n 里的 \r 已经由上面的回车代表
            _send_input(_KEYBDINPUT(0, ord(ch), KEYEVENTF_UNICODE, 0, None))
            time.sleep(min(0.01, interval * 0.25))
            _send_input(_KEYBDINPUT(0, ord(ch), KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, None))
            # 给目标窗口消息处理时间：UWP（微信/计算器）要 0.04，原生控件（Dev-C++）0.01 就够
            time.sleep(interval)


class ActionExecutor:
    def __init__(self, safety, grounding, log=print):
        self.safety = safety
        self.grounding = grounding
        self.log = log
        # 可选的「测试护栏」：core 在 no_send 模式下挂一个判定函数。
        # 放在执行器最底层，模板 / 序列 / 模型直接发的动作都绕不过去。
        self.send_guard = None

    def _resolve_xy(self, action):
        """优先 element_id（grounding 中心点），其次模型给的 x/y。"""
        eid = action.get("element_id")
        if eid is not None:
            pt = self.grounding.resolve(eid)
            if pt:
                return float(pt[0]), float(pt[1])
        x, y = action.get("x"), action.get("y")
        if x is None or y is None:
            return None
        return float(x), float(y)

    def execute(self, action):
        if self.send_guard is not None and self.send_guard(action):
            self.log("[NO-SEND] ⛔ 已拦截发送类动作，未对外发送（测试护栏开启，视为执行成功）")
            return True
        if pyautogui is None:  # pragma: no cover
            self.log("[跳过] pyautogui 不可用（非 Windows / 无 GUI 环境）")
            return True
        _park_if_corner()  # 鼠标停在边缘会让后续每一次点击都触发急停
        a = action.get("action")
        if a == "done":
            self.log(f"[完成] {action.get('result', '')}")
            return False
        if a == "ask":
            self.log(f"[询问模型] {action.get('question', '')}")
            return "ASK"
        if a == "wait":
            ms = int(action.get("ms", 1000))
            self.log(f"[等待] {ms} ms")
            time.sleep(ms / 1000.0)
            return True

        if a in ("click", "double_click", "right_click"):
            xy = self._resolve_xy(action)
            if xy is None:
                self.log("[跳过] 该动作缺少坐标或元素")
                return True
            x, y = self.safety.clamp(*xy)
            self.log(f"[{a}] ({x}, {y})")
            try:
                if a == "click":
                    pyautogui.click(x, y)
                elif a == "double_click":
                    pyautogui.doubleClick(x, y)
                else:
                    pyautogui.rightClick(x, y)
            except Exception as e:  # 含 FailSafeException（鼠标被甩到角落）
                self.log(f"[动作失败] {type(e).__name__}: {e}")
                return True
            return True

        if a == "drag":
            f = self._resolve_point(action.get("from")) or self._resolve_xy(action)
            t = self._resolve_point(action.get("to"))
            if f is None or t is None:
                self.log("[跳过] drag 缺少起点/终点")
                return True
            x1, y1 = self.safety.clamp(*f)
            x2, y2 = self.safety.clamp(*t)
            self.log(f"[拖拽] ({x1},{y1}) -> ({x2},{y2})")
            pyautogui.moveTo(x1, y1)
            pyautogui.dragTo(x2, y2, duration=0.4)
            return True

        if a == "type":
            text = str(action.get("text", ""))
            self.log(f"[输入·逐字敲] {text!r}")
            _type_unicode(text)
            return True

        if a == "key":
            keys = [k.strip().lower() for k in str(action.get("key", "")).split("+") if k.strip()]
            self.log(f"[按键] {'+'.join(keys)}")
            if not keys:
                return True
            if len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)
            return True

        if a == "scroll":
            dx = int(action.get("dx", 0) or 0)
            dy = int(action.get("dy", 0) or 0)
            # 模型的约定是「dy 正 = 向下」，而 pyautogui.scroll 正好相反（正 = 向上）。
            # 以前直接把 dy 透传：模型想往下翻，画面却往上跑，它就认为"滚动没用"，
            # 于是宁可在原地乱点也不用这个动作（真机 37 步里 scroll 出现 0 次）。
            notches = max(-20, min(20, dy))
            # 滚动只对「鼠标下面的窗口」生效，所以必须先移到可滚动区域里。
            # 以前固定滚屏幕中心：光标被停靠在左上角时，那一下滚的是别人的窗口。
            at = action.get("_at")
            if not at:
                at = (self.safety.W // 2, self.safety.H // 2)
            self.log(f"[滚动] dy={dy} dx={dx} @({int(at[0])},{int(at[1])})")
            if notches:
                pyautogui.moveTo(int(at[0]), int(at[1]))
                pyautogui.scroll(-notches)
            if dx:
                # 横向滚动没有独立 API，只能按住 shift 再滚轮
                pyautogui.keyDown("shift")
                try:
                    pyautogui.scroll(max(-20, min(20, dx)))
                finally:
                    pyautogui.keyUp("shift")
            return True

        self.log(f"[跳过] 未知动作: {a}")
        return True

    @staticmethod
    def _resolve_point(spec):
        if not isinstance(spec, (list, tuple)) or len(spec) < 2:
            return None
        try:
            return float(spec[0]), float(spec[1])
        except Exception:
            return None
