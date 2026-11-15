#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""安全层：坐标越界检查、急停开关、高风险操作识别。

为便于在无显示器的 CI 环境导入与测试，pyautogui 改为按需懒导入；
若不可用（如 Linux CI），自动回退到默认分辨率 (1920,1080) 并关闭 FAILSAFE。
"""

import json
import sys

try:
    import pyautogui
    _HAS_PG = True
except Exception:  # pragma: no cover - 仅在没有 GUI / 显示器时触发
    pyautogui = None
    _HAS_PG = False


def ensure_dpi_aware():
    """Windows 高 DPI 缩放修正。

    在 125% / 150% 缩放的笔记本上，若不声明 DPI 感知，pyautogui 拿到的屏幕尺寸
    与 PIL 截图的像素尺寸会不一致，导致「截图坐标」和「点击坐标」错位——这是
    Windows CUA 最常见的第一个 issue。这里在读取屏幕尺寸前声明
    PROCESS_PER_MONITOR_DPI_AWARE（或退化为进程级 DPI 感知）。非 Windows 直接跳过。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        try:
            # 2 = PROCESS_PER_MONITOR_DPI_AWARE（Win8.1+，多显示器最准确）
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            # 老系统退化：SetProcessDPIAware（仅进程级，Vista+）
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class Safety:
    def __init__(self, cfg: dict):
        ensure_dpi_aware()  # 必须在读取屏幕尺寸前声明，否则坐标会错位
        s = (cfg.get("safety", {}) or {})
        self.failsafe = bool(s.get("failsafe", True))
        self.clamp_coords = bool(s.get("clamp_coords", True))
        self.confirm_risky = bool(s.get("confirm_risky", True))
        self.risk_keywords = [str(k).lower() for k in (s.get("risk_keywords", []) or [])]

        if _HAS_PG:
            try:
                if self.failsafe:
                    pyautogui.FAILSAFE = True   # 鼠标甩到左上角(0,0)即急停
                pyautogui.PAUSE = 0.05
                self.W, self.H = pyautogui.size()
            except Exception:  # pragma: no cover
                self.W, self.H = 1920, 1080
                self.failsafe = False
        else:  # pragma: no cover
            self.W, self.H = 1920, 1080
            self.failsafe = False

    def clamp(self, x, y):
        """坐标越界保护：超出屏幕范围会被夹回边界。"""
        if not self.clamp_coords:
            try:
                return int(round(float(x))), int(round(float(y)))
            except Exception:
                return 0, 0
        try:
            x = max(0, min(self.W - 1, int(round(float(x)))))
            y = max(0, min(self.H - 1, int(round(float(y)))))
        except Exception:
            x, y = 0, 0
        return x, y

    def is_risky(self, action: dict) -> bool:
        """命中风险关键词（如删除/发送/付款）返回 True，需人工确认。"""
        if not self.confirm_risky:
            return False
        blob = json.dumps(action, ensure_ascii=False).lower()
        return any(k in blob for k in self.risk_keywords)
