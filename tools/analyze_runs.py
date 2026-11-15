#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轨迹分析：把 data/trajectories 里的历史跑批成一张体检表。

用法：
    python tools/analyze_runs.py                # 最近 30 次
    python tools/analyze_runs.py --n 100        # 最近 100 次
    python tools/analyze_runs.py --task 微信    # 只看任务名含"微信"的
    python tools/analyze_runs.py --slow 15      # 只列出平均每步 >15 秒的跑

它回答四个问题：
  1. 每步到底花多少秒（哪个环节在拖）
  2. 失败模式里谁出现最多（nochange / 重复动作 / 无坐标）
  3. 思考（plan）是不是在复述任务名（复述=等于没思考）
  4. 步数分布 —— 1 步完成的任务占多少
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_run(d):
    meta = {}
    mp = os.path.join(d, "meta.json")
    if os.path.isfile(mp):
        try:
            meta = json.load(open(mp, encoding="utf-8"))
        except Exception:
            meta = {}
    steps = []
    tp = os.path.join(d, "trajectory.jsonl")
    if os.path.isfile(tp):
        for line in open(tp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except Exception:
                pass
    return meta, steps


def act_name(rec):
    a = rec.get("action") or {}
    if isinstance(a.get("actions"), list):
        return "batch"
    return str(a.get("action") or "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--task", default="")
    ap.add_argument("--slow", type=float, default=0)
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "trajectories"))
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print("没有轨迹目录：", args.dir)
        return 1

    runs = sorted((os.path.join(args.dir, d) for d in os.listdir(args.dir)
                   if os.path.isdir(os.path.join(args.dir, d))), reverse=True)
    rows = []
    for d in runs:
        meta, steps = load_run(d)
        task = meta.get("task") or os.path.basename(d).split("_", 2)[-1]
        if args.task and args.task not in task:
            continue
        elapsed = [s["elapsed"] for s in steps if isinstance(s.get("elapsed"), (int, float))]
        avg = sum(elapsed) / len(elapsed) if elapsed else 0.0
        if args.slow and avg < args.slow:
            continue
        rows.append({
            "dir": d,
            "task": task,
            "steps": len(steps),
            "elapsed": elapsed,
            "avg": avg,
            "total": sum(elapsed),
            "recs": steps,
        })
        if len(rows) >= args.n:
            break

    if not rows:
        print("没有匹配的轨迹（可能还没跑过，或全是模板任务——模板任务不写步骤）")
        return 0

    print("=" * 78)
    print(f"轨迹体检表（{len(rows)} 次，最近在前）")
    print("=" * 78)
    print(f"{'任务':<34}{'步数':>5}{'总时长':>8}{'每步':>7}  动作分布")
    print("-" * 78)
    for r in rows:
        acts = Counter(act_name(s) for s in r["recs"])
        brief = " ".join(f"{k}×{v}" for k, v in acts.most_common(4))
        name = r["task"][:32]
        print(f"{name:<34}{r['steps']:>5}{r['total']:>7.0f}s{r['avg']:>6.1f}s  {brief}")

    all_elapsed = [e for r in rows for e in r["elapsed"]]
    if all_elapsed:
        all_elapsed.sort()
        def pct(p):
            i = min(len(all_elapsed) - 1, int(len(all_elapsed) * p))
            return all_elapsed[i]
        print("-" * 78)
        print(f"单步耗时：中位 {pct(0.5):.1f}s   p90 {pct(0.9):.1f}s   "
              f"最快 {all_elapsed[0]:.1f}s   最慢 {all_elapsed[-1]:.1f}s")

    # 失败模式
    res = Counter()
    for r in rows:
        for s in r["recs"]:
            v = s.get("result")
            if v:
                res[v] += 1
    if res:
        print("结果分布：" + "  ".join(f"{k} {v}" for k, v in res.most_common()))

    # 思考重复率：同一个 plan 反复出现 = 模型在复述任务名
    dup_total, dup_runs = 0, 0
    for r in rows:
        thinks = [s.get("think") for s in r["recs"] if s.get("think")]
        if len(thinks) >= 2:
            dup_runs += 1
            if len(set(thinks)) == 1:
                dup_total += 1
    if dup_runs:
        print(f"思考重复：{dup_total}/{dup_runs} 次跑的 plan 从头到尾完全没变"
              f"（{'严重' if dup_total * 2 > dup_runs else '尚可'}）")

    # 步数分布
    ORDER = {"0": 0, "1": 1, "2-3": 2, "4-6": 3, "7-10": 4, "11+": 5}
    buck = defaultdict(int)
    for r in rows:
        n = r["steps"]
        buck["0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else "4-6" if n <= 6
             else "7-10" if n <= 10 else "11+"] += 1
    print("步数分布：" + "  ".join(f"{k}步 {v}次" for k, v in
                                   sorted(buck.items(), key=lambda kv: ORDER[kv[0]])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
