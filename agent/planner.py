#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""规划器：把大任务拆成有序子任务清单，长任务不跑偏；失败时支持重规划。"""

import re

from agent.models import ModelClient


def _subtask_key(text: str) -> str:
    """子步骤指纹：去掉序号/标点/常见虚词，取动作+对象的核心字，用于判重。"""
    t = re.sub(r"^[\d\.\)\、\s]+", "", (text or "").strip())
    t = re.sub(r"[\s，。、,.!?；：:；的了对把在将然后接着再]", "", t)
    return t[:24]


def _is_dup(a: str, b: str) -> bool:
    """判断两个子步骤是不是同一个动作的细分（7B 常把「点搜索框」拆成点框/点图标/点文字）。"""
    ka, kb = _subtask_key(a), _subtask_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    # 前 4 个字相同 = 同一个对象的连续动作
    if len(ka) >= 4 and len(kb) >= 4 and ka[:4] == kb[:4]:
        return True
    ga = {ka[i:i + 2] for i in range(len(ka) - 1)}
    gb = {kb[i:i + 2] for i in range(len(kb) - 1)}
    if not ga or not gb:
        return False
    return len(ga & gb) / min(len(ga), len(gb)) >= 0.7


class Planner:
    def __init__(self, cfg: dict, model: ModelClient):
        p = (cfg.get("planner", {}) or {})
        self.enabled = bool(p.get("enabled", False))
        self.replan_on_fail = bool(p.get("replan_on_fail", True))
        self.max_subtasks = int(p.get("max_subtasks", 6))
        self.model = model
        self.subtasks = []
        self.idx = 0

    def decompose(self, task: str):
        """把任务拆成子步骤清单。关闭规划器时整任务即单步。"""
        if not self.enabled:
            self.subtasks = [task]
            self.idx = 0
            return self.subtasks

        prompt = (
            "你是一个任务规划器。请把下面的电脑操作任务拆成有序的子步骤，"
            f"不超过 {self.max_subtasks} 步，每步都必须推进任务进展（不要把一个点击拆成好几步）。"
            "每步一行，用 '1. ' 开头，只写动作本身不要解释。\n"
            f"任务：{task}"
        )
        try:
            text = self.model.chat([{"role": "user", "content": prompt}], temperature=0.1)
            items = self._parse(text)
            self.subtasks = self._dedupe(items) if items else [task]
        except Exception:
            self.subtasks = [task]
        self.idx = 0
        return self.subtasks

    def _dedupe(self, items):
        """去掉语义重复的子步骤（7B 常把「点搜索框」拆成点框/点图标/点文字），并限步数。"""
        out = []
        for it in items:
            if not _subtask_key(it):
                continue
            if any(_is_dup(it, prev) for prev in out):
                continue
            out.append(it)
        if not out:
            return items
        return out[: self.max_subtasks]

    @staticmethod
    def _parse(text: str):
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)[.)]\s+(.+)", line)
            if m:
                items.append(m.group(2).strip())
                continue
            m = re.match(r"^[-*]\s+(.+)", line)
            if m:
                items.append(m.group(1).strip())
        return items

    def current(self):
        if self.idx < len(self.subtasks):
            return self.subtasks[self.idx]
        return None

    def mark_done(self):
        self.idx += 1

    def is_complete(self):
        return self.idx >= len(self.subtasks)

    def replan(self, task: str, failure_note: str):
        """某子任务失败后，基于失败原因重规划剩余步骤。"""
        if not self.replan_on_fail:
            self.idx += 1
            return
        prompt = (
            f"原任务：{task}\n已完成的子步骤到第 {self.idx} 步。\n"
            f"上一步失败原因：{failure_note}\n"
            "请重新给出剩余的有序子步骤（每步一行，'1. ' 开头），只写动作。"
        )
        try:
            text = self.model.chat([{"role": "user", "content": prompt}], temperature=0.1)
            items = self._parse(text)
            if items:
                self.subtasks = self.subtasks[: self.idx] + items
        except Exception:
            self.idx += 1
