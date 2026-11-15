#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""无人值守跑一条任务（给回归测试用）。

用法:
    python tools/try_auto.py "打开D:\\测试，在里面新建一个文件"

与 run.py --cli 的区别：不交互、不询问风险确认、把每步耗时直接打出来。
用于"跑一批任务看会不会翻车"，日志好抓。
"""
import os
import sys
import time
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    task = sys.argv[1]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 无人值守：别在风险确认上卡住（这批任务里没有真危险动作）
    cfg.setdefault("safety", {})["confirm_risky"] = False

    from agent.core import AgentSession

    s = AgentSession(cfg)
    t0 = time.time()
    s.log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
    status = s.run(task)
    print(f"RESULT {status} 总耗时 {time.time() - t0:.1f}s", flush=True)
    return 0 if str(status).upper().endswith("DONE") else 1


if __name__ == "__main__":
    sys.exit(main())
