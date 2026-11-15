"""真机演示：给 LocalCUA 一个任务，看它自己循环「截屏 -> 思考 -> 动作」。

用法:
    python tools/legacy/live_demo.py "打开记事本并输入你好"
    python tools/legacy/live_demo.py "打开记事本" --steps 12

模型 ask 时自动回一句提示（免人工输入），方便无人值守跑演示。
"""

import argparse
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.config import load_config
from agent.core import AgentSession
from agent.safety import ensure_dpi_aware

AUTO_REPLY = "别问我，你自己看着屏幕决定。可以先用开始菜单搜索目标程序。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--steps", type=int, default=10)
    a = ap.parse_args()

    ensure_dpi_aware()
    cfg = load_config()
    cfg.setdefault("agent", {})["max_steps"] = a.steps
    cfg["agent"]["step_pause"] = float(os.environ.get("CU_PAUSE", "1.2"))

    stop = threading.Event()
    sess = AgentSession(cfg, log=lambda m: print(m, flush=True),
                        confirm=lambda act: True, stop_event=stop)

    reply = None
    for _ in range(3):
        res = sess.run(a.task, reply=reply)
        if res == "NEED_REPLY":
            reply = AUTO_REPLY
            continue
        print(f"=== 结束：{res}", flush=True)
        return


if __name__ == "__main__":
    main()
