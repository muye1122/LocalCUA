#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跨任务长期记忆：教程里「长期记忆库 / 跨会话记忆」在 GUI agent 上的最小落地。

现在的记忆都是任务内的：history 最近 6 条、草稿纸只留最近 10 个任务、
_fail_log 任务一结束就清空。于是「Dev-C++ 冷启动要点开始菜单兜底」
「在记事本里全选文字再调字体没用」这类经验，下次开机全忘，
8B 会把同样的弯路再走一遍。

只存两类**程序能直接利用的事实**，不存截图、不存对话、不联网：
  ① 某个程序这次是怎么开起来的（窗口标题 / exe / 成功次数）
  ② 试过没用的做法（最近 N 条，去重）

注入提示词时最多约 130 字。上下文只有 8192，真机撞到 8033 那次
prefill 从 3.6s 飙到 20s+，所以这里宁可少给，绝不把长期记忆做成新的负担。
"""
from __future__ import annotations

import json
import os

MAX_APPS = 12
MAX_FAILS = 8
FAIL_CHARS = 56


class LongMemory:
    def __init__(self, path: str = "data/long_memory.json", enabled: bool = True):
        self.path = path
        self.enabled = bool(enabled)
        self.apps: dict = {}
        self.fails: list = []
        self._dirty = False
        if self.enabled:
            self.load()

    def load(self):
        try:
            if not os.path.exists(self.path):
                return
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f) or {}
            self.apps = dict(d.get("apps") or {})
            self.fails = [str(x) for x in (d.get("fails") or [])][-MAX_FAILS:]
        except Exception:
            self.apps, self.fails = {}, []   # 文件坏了不能让机器人挂掉

    def save(self):
        if not self.enabled or not self._dirty:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"apps": self.apps, "fails": self.fails},
                          f, ensure_ascii=False, indent=2)
            self._dirty = False
        except Exception:
            pass

    def note_open(self, app: str, title: str = "", exe: str = ""):
        """进场确认成功时记一笔：这个程序的窗口到底叫什么。

        下次 _find_app_window 认不出标题时，这些是排查线索；注入给模型时
        也能让它别再重复发 open_app。
        """
        if not self.enabled:
            return
        a = (app or "").strip()
        if not a:
            return
        old = self.apps.get(a) or {}
        self.apps[a] = {
            "title": (title or "").strip()[:60],
            "exe": (exe or "").strip()[:40],
            "n": int(old.get("n") or 0) + 1,
        }
        if len(self.apps) > MAX_APPS:
            # 留最近用过的：按成功次数降序，不常用的先淘汰
            keep = sorted(self.apps.items(), key=lambda kv: kv[1].get("n", 0),
                          reverse=True)[:MAX_APPS]
            self.apps = dict(keep)
        self._dirty = True

    def note_fail(self, text: str):
        """"这招试过没用"跨任务保留。同一条不重复记。"""
        if not self.enabled:
            return
        t = " ".join(str(text or "").split())[:FAIL_CHARS]
        if not t:
            return
        self.fails = [x for x in self.fails if x != t]
        self.fails.append(t)
        del self.fails[:-MAX_FAILS]
        self._dirty = True

    def hint(self, task: str = "") -> str:
        """给提示词的一小段事实，总长严格受限。"""
        if not self.enabled or (not self.apps and not self.fails):
            return ""
        parts = []
        t = str(task or "")
        for app, info in self.apps.items():
            if app and app in t and (info.get("title") or info.get("exe")):
                parts.append(f"「{app}」窗口是 {info.get('title') or info.get('exe')}")
                break   # 只给一条，别把上下文撑满
        if self.fails:
            # 只给最近 1 条：_build_user_text 里已有本任务的失败清单，
            # 这里再堆两条就是重复占用上下文。
            parts.append("试过没用：" + self.fails[-1])
        if not parts:
            return ""
        note = "【跨任务经验（只作参考，仍以当前截图为准）】" + "；".join(parts) + "。\n"
        return note[:120]
