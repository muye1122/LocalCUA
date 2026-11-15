#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比测试：本地 coder 模型写 C 代码的一次成功率（真编译、真运行）。

  python tools/try_coder.py                # 跑内置的 4 道题
  python tools/try_coder.py --model qwen2.5vl:7b   # 换成视觉模型对照
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

GCC_DIR = r"D:\dev-C++\Dev-Cpp\MinGW64\bin"

CASES = [
    ("阶乘（long long 陷阱）",
     "用C语言写一个程序，输出15的阶乘。只输出最终结果，不要多余文字。",
     "1307674368000"),
    ("冒泡排序",
     "用C语言写一个程序，读入10个整数，用冒泡排序从小到大排序后输出，数字之间用空格分隔。",
     None),
    ("百元买百鸡（之前7稿全错）",
     "用C语言写一个程序求解百元买百鸡：公鸡每只5文钱，母鸡每只3文钱，小鸡3只1文钱，"
     "用100文钱买100只鸡，问公鸡、母鸡、小鸡各买多少只？输出所有解。",
     None),
    ("遍历二叉树（之前栈溢出）",
     "用C语言写一个程序，用数组 {4,2,5,1,3,6,7} 建一棵二叉搜索树，然后中序遍历输出。",
     "1 2 3 4 5 6 7"),
]


def gen(model, prompt, timeout=300):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.2, "num_ctx": 8192}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return r.get("response", ""), time.time() - t0, r.get("eval_count", 0)


def clean(code):
    if "```" not in code:
        return code
    for part in code.split("```"):
        p = part.strip()
        if p.startswith("c\n") or p.startswith("C\n"):
            p = p[2:]
        if "#include" in p or "int main" in p:
            return p.strip()
    return code


def try_run(model, title, prompt, expect):
    code, dt, ntok = gen(model, prompt)
    code = clean(code)
    d = Path(tempfile.mkdtemp())
    src = d / "t.c"
    src.write_text(code, encoding="utf-8")
    env = dict(os.environ)
    env["PATH"] = GCC_DIR + os.pathsep + env.get("PATH", "")
    cp = subprocess.run([os.path.join(GCC_DIR, "gcc.exe"), "-std=c99", "-O2",
                         str(src), "-o", str(d / "t.exe")],
                        capture_output=True, text=True, env=env)
    if cp.returncode != 0:
        print(f"  ❌ {title}: 编译失败 {cp.stderr[:200]}")
        return False
    try:
        rp = subprocess.run([str(d / "t.exe")], capture_output=True, text=True,
                            timeout=10, input="5 3 9 1 7 2 8 4 6 0\n")
    except subprocess.TimeoutExpired:
        print(f"  ❌ {title}: 运行超时（死循环）")
        return False
    out = (rp.stdout or "").strip()
    status = "编译+运行 OK"
    if rp.returncode != 0:
        status = f"崩溃 rc={rp.returncode}"
    ok = None if expect is None else (expect in out.replace("\n", " "))
    flag = "" if expect is None else ("✅ 答案对" if ok else f"❌ 答案错（期望 {expect}）")
    print(f"  {title}: {dt:5.1f}s / {ntok} tok / {ntok/dt if dt else 0:4.1f} tok/s  {status} {flag}")
    print(f"     输出：{out[:120]!r}")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    a = ap.parse_args()
    only = [c for c in CASES]
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        only = [c for c in CASES if sys.argv[1] in c[0]]
    print(f"模型：{a.model}（首题含冷加载，会慢一些）\n")
    win = 0
    for title, prompt, expect in only:
        try:
            win += try_run(a.model, title, prompt, expect)
        except Exception as e:
            print(f"  ❌ {title}: {e}")
    checked = [c for c in only if c[2] is not None]
    if checked:
        print(f"\n可判卷题：{win}/{len(checked)} 通过")


if __name__ == "__main__":
    main()
