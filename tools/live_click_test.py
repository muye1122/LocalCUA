#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""真机闭环冒烟：让 agent 用 element_desc 真实点击一次，再截屏确认。"""
import sys
import threading

sys.path.insert(0, ".")
from agent.config import load_config  # noqa: E402
from agent.core import AgentSession  # noqa: E402

cfg = load_config()
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


session = AgentSession(cfg, log=log)
session.max_steps = 3
result = session.run("单击屏幕底部任务栏上的搜索框（有放大镜图标和'搜索'两个字），点中后立即 done")
print("\n==== 结果:", result)
