"""通用任务运行器：直接跑一条自然语言任务，日志打到终端。

用法：
    python tools/run_task.py --task "打开记事本，输入一段话" [--max-steps 12] [--no-plan]
    python tools/run_task.py --task "给示例联系人A发消息说你好" --no-send   # 验证链路，绝不真发
"""
import argparse
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.config import load_config            # noqa: E402
from agent.core import AgentSession                 # noqa: E402
from agent.safety import ensure_dpi_aware          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--no-plan", action="store_true", help="不拆分子任务（单步推进）")
    ap.add_argument("--no-send", action="store_true",
                    help="测试护栏：拦截 send_message 与对话窗口里的回车，绝不对外发消息")
    ap.add_argument("--config", default=None,
                    help="默认按优先级找：根目录 config.yaml 优先于 agent/config.yaml")
    args = ap.parse_args()

    ensure_dpi_aware()
    cfg = load_config(args.config) or {}
    if args.max_steps:
        cfg.setdefault("agent", {})["max_steps"] = args.max_steps
    if args.no_plan:
        cfg.setdefault("planner", {})["enabled"] = False
    if args.no_send:
        cfg.setdefault("agent", {})["no_send"] = True

    stop = threading.Event()
    sess = AgentSession(cfg, log=lambda m: print(m, flush=True), confirm=None, stop_event=stop)

    try:
        res = sess.run(args.task)
        print(f"—— 结束：{res}", flush=True)
    except KeyboardInterrupt:
        stop.set()
        print("\n⛔ 已中断", flush=True)


if __name__ == "__main__":
    main()
