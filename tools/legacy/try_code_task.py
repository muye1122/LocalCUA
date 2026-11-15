#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""真机验证：写代码模板一条龙（模型写码 → 编译 → 开 Dev-C++ → 敲 → 另存 → 校验）。"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml

from agent.core import AgentSession

TASK = sys.argv[1] if len(sys.argv) > 1 else \
    "打开dev-c++，写一个求15以内阶乘的函数。命名为“阶乘”保存到桌面。"


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    sess = AgentSession(cfg)
    sess.log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    job = sess._parse_code_task(TASK)
    print("解析:", job, flush=True)
    if not job:
        print("❌ 意图没识别出来")
        return 1
    t0 = time.time()
    ok = sess._write_code_task(job)
    print(f"RESULT ok={ok} saved_by={getattr(sess, 'ok_saved_by', None)} "
          f"耗时={time.time() - t0:.1f}s", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
