#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模板路由冒烟测试：给一批任务句，打印每句命中哪条确定性模板。

不操作鼠标键盘，纯解析层。用途：
  1. 快速回答「我的 agent 现在能一句话搞定哪些任务」
  2. 改动路由正则后，一眼看出有没有把原来命中的句式改丢
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.core import AgentSession  # noqa: E402

CASES = [
    # (任务句, 期望模板)
    ("打开记事本", "open_app"),
    ("请帮我打开计算器", "open_app"),
    ("打开浏览器搜索今天的天气", "search_web"),
    ("查一下明天北京的天气", "search_web"),
    ("帮我搜一下 python 教程", "search_web"),
    ("打开 www.qq.com", "open_url"),
    ("访问 https://www.baidu.com", "open_url"),
    ("关闭记事本", "close_app"),
    ("关掉计算器", "close_app"),
    ("打开 D:\\测试", "navigate"),
    ("打开 C 盘", "navigate"),
    ("在桌面新建一个文件夹叫报告", "nav_newfile"),
    ("打开D:" + chr(92) + "测试，在里面新建一个文件", "nav_newfile"),
    ("在微信里给文件传输助手发送一条消息：你好", "send_message"),
    ("给文件传输助手发：早", "send_message"),
    ("对示例联系人A说你好呀", "send_message"),
    ("在微信里给示例联系人B发送一条消息：你好", "BLOCKED(白名单)"),
    ("使用计算机算出123*123，再发给示例联系人A", "calc_send"),
    ("打开计算器，算128*64", "calc_show"),
    ("用C语言写一个程序，求1到100的和", "write_code"),
    ("打开dev-c++，写一个求15以内阶乘的函数", "write_code"),
    ("打开桌面上的 test.txt", "navigate"),   # 桌面上没这个文件时由导航接管；文件存在则 open_file
    ("打开记事本，输入三行字：春天来了、花开满树、燕子归来，每行一个", "open_and_type"),
    ("打开记事本，输入一段话", "generic"),   # 没给出具体内容，不该瞎打
    ("打开记事本，写一首诗", "generic"),
    ("打开微信，输入 你好", "generic"),
    ("写成 一个求两数最大公约数的函数", "generic"),
    ("把桌面的文件拖到回收站", "generic"),
]


def route(sess, task):
    if sess._parse_search_intent(task):
        return "search_web"
    if sess._parse_url_intent(task):
        return "open_url"
    if sess._parse_file_intent(task) and sess._resolve_existing_file(
            sess._parse_file_intent(task)):
        return "open_file"
    if sess._parse_open_and_type_intent(task):
        return "open_and_type"
    if sess._parse_open_intent(task):
        return "open_app"
    if sess._parse_close_intent(task):
        return "close_app"
    calc = sess._parse_calc_send_intent(task)
    if calc:
        ok, _why = AgentSession._send_target_allowed(calc["app"], calc["target"])
        return "calc_send" if ok else "BLOCKED(白名单)"
    if sess._parse_calc_show_intent(task):
        return "calc_show"
    if sess._parse_code_task(task):
        return "write_code"
    nav = sess._parse_nav_task(task)
    if nav:
        m = AgentSession._NAV_NEWFILE_RE.search(nav[1] or "")
        return "navigate" if not m else "nav_newfile"
    intent = sess._route_task(task)
    if intent and intent.get("blocked"):
        return "BLOCKED(白名单)"
    if intent:
        return "send_message"
    return "generic"


def main():
    import threading
    sess = AgentSession(
        {"safety": {}, "grounding": {"backend": "none"}, "planner": {"enabled": False},
         "validator": {"enabled": False}, "agent": {"max_steps": 1}},
        log=lambda m: None, stop_event=threading.Event())
    print("=" * 62)
    print("模板路由一览（命中 = 程序接管，1 步完成）")
    print("=" * 62)
    bad = 0
    for task, want in CASES:
        got = route(sess, task)
        mark = "✅" if got == want else "❌"
        if got != want:
            bad += 1
        print(f"{mark} {got:<18} {task}")
    print("-" * 62)
    print(f"{'全部命中' if bad == 0 else f'{bad} 句不符合预期'}，共 {len(CASES)} 句")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
