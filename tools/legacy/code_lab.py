#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
code_lab —— 验证「模型写 C 代码 → 编译 → 读报错 → 自己改」这条闭环。

为什么需要它（演示前的实测工具，不是最终产品）：
  1. 评委现场出题，代码必然不在任何模板里 —— 只能靠模型现写；
  2. 但模型第一次写出来的代码经常有编译错（少分号、类型不匹配）；
  3. 项目没有 OCR，读不到 Dev-C++ 窗口里的报错 ——
     所以编译验证走 Dev-C++ 自带的 MinGW gcc 命令行，stderr 直接进程序。
  这样"写-编-改"能闭环，而且全程离线。

用法：
  python tools/legacy/code_lab.py "写一个函数判断字符串是否回文，并在 main 里测试"
  python tools/legacy/code_lab.py --rounds 3 "用冒泡排序对数组排序"
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from agent.models import ModelClient  # noqa: E402

OUT_DIR = ROOT / "data" / "code_lab"

# Dev-C++ 自带的 MinGW64；找不到再退回 PATH 里的 gcc
GCC_CANDIDATES = [
    r"D:\dev-C++\Dev-Cpp\MinGW64\bin\gcc.exe",
    r"C:\Program Files (x86)\Dev-Cpp\MinGW64\bin\gcc.exe",
    r"C:\Program Files\Dev-Cpp\MinGW64\bin\gcc.exe",
    "gcc",
]

SYSTEM = (
    "你是资深 C 语言程序员。用户会给你一道题目，你要写出完整、可直接编译运行的 C 代码。\n"
    "硬性要求：\n"
    "1) 只用标准 C（C11），只 include <stdio.h> <stdlib.h> <string.h> 这类标准头；\n"
    "2) 必须包含 main 函数，并在 main 里用具体例子验证结果（printf 输出）；\n"
    "3) 不要用中文变量名，不要用 C++ 特性（如 cout、引用、类）；\n"
    "4) 只输出一个 ```c 代码块，代码块外不要写任何解释。\n"
)


def find_gcc():
    for c in GCC_CANDIDATES:
        if c == "gcc" or os.path.exists(c):
            return c
    return None


def extract_code(raw: str) -> str:
    """从模型输出里抠出 C 代码块；没有围栏就把整段当代码。"""
    t = (raw or "").strip()
    m = re.search(r"```(?:c|cpp|C)?\s*\n(.*?)```", t, re.S)
    if m:
        return m.group(1).strip()
    # 没有闭合围栏（输出被截断）：取 ``` 之后的全部
    i = t.find("```")
    if i != -1:
        body = t[i + 3:]
        body = re.sub(r"^(?:c|cpp)\s*\n", "", body, flags=re.I)
        return body.strip()
    return t


def compile_c(gcc: str, src: Path, exe: Path, timeout: float = 30.0):
    """返回 (ok, message)。message 是编译器的原始报错，直接喂回模型。

    ⚠️ 必须在 PATH 里带上 gcc 所在目录：这是 MSYS2 版的 MinGW64，
    cc1.exe / collect2.exe 是**子进程**，它们要加载同目录的 libgcc_s_seh-1.dll、
    libwinpthread-1.dll。bin 不在 PATH 时这些子进程会静默退出 ——
    gcc 主进程只返回 1、stderr 一个字都没有，看起来像"编译失败但没有原因"。
    """
    cmd = [gcc, str(src), "-std=c11", "-Wall", "-o", str(exe)]
    env = os.environ.copy()
    gcc_dir = str(Path(gcc).parent)
    env["PATH"] = gcc_dir + os.pathsep + env.get("PATH", "")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace", env=env)
    except subprocess.TimeoutExpired:
        return False, "编译超时（30s）"
    except Exception as exc:
        return False, f"编译器调用失败：{exc}"
    if p.returncode == 0:
        warn = (p.stderr or "").strip()
        return True, warn
    msg = ((p.stderr or "") + (p.stdout or "")).strip()
    if not msg:
        msg = ("编译器退出码 %d 但没有任何报错输出 —— 多半是 gcc 子进程加载不到 DLL，"
               "确认 PATH 里含 %s" % (p.returncode, gcc_dir))
    return False, msg


def run_exe(exe: Path, timeout: float = 10.0, stdin_data: str = None):
    """运行编译产物。stdin_data=None 时立刻给 EOF ——
    否则 scanf 会一直等键盘输入，看起来像"死循环"（实际是没人喂数据）。"""
    try:
        p = subprocess.run([str(exe)], capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace",
                           input=(stdin_data if stdin_data is not None else ""))
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, ("运行超时（可能有死循环，或程序在等键盘输入 —— 用 --stdin 喂数据）")
    except Exception as exc:
        return -1, f"运行失败：{exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="题目描述")
    ap.add_argument("--rounds", type=int, default=3, help="最多修几轮（默认 3）")
    ap.add_argument("--run", action="store_true", default=True, help="编译成功后运行（默认开）")
    ap.add_argument("--stdin", default=None, help="程序的标准输入（题目要求读键盘时必给）")
    args = ap.parse_args()

    gcc = find_gcc()
    if not gcc:
        print("找不到 gcc。请确认 Dev-C++ 装在 D:\\dev-C++\\Dev-Cpp（自带 MinGW64）。")
        return 2
    print(f"[环境] gcc = {gcc}")

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    mc = ModelClient(cfg)
    print(f"[模型] 主={mc.model}  代码={mc.coder_model or '（未配置，用主模型）'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"题目：{args.task}"}]
    history = []

    total_t0 = time.time()
    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        raw = mc.chat_coder(messages, temperature=0.1, max_tokens=900)
        code = extract_code(raw)
        dt = time.time() - t0
        src = OUT_DIR / f"try_{rnd}.c"
        exe = OUT_DIR / f"try_{rnd}.exe"
        src.write_text(code, encoding="utf-8")
        print(f"\n=== 第 {rnd} 轮：模型生成 {dt:.1f}s，{len(code)} 字符，{code.count(chr(10)) + 1} 行 ===")

        ok, msg = compile_c(gcc, src, exe)
        if ok:
            print(f"[编译] ✅ 通过（{dt:.1f}s 生成）")
            if msg:
                print(f"[警告] {msg[:400]}")
            print(f"[产物] {src}")
            if args.run:
                rc, out = run_exe(exe, stdin_data=args.stdin)
                print(f"[运行] 退出码 {rc}")
                print("------- 程序输出 -------")
                print(out[:1500] or "（无输出）")
                print("------------------------")
            total = time.time() - total_t0
            print(f"\n[合计] {total:.1f}s，{rnd} 轮通过")
            return 0

        print("[编译] ❌ 失败，报错如下（将回填给模型自修）")
        print(msg[:1200])
        history.append({"round": rnd, "error": msg[:800], "chars": len(code)})
        messages.append({"role": "assistant", "content": code})
        messages.append({"role": "user", "content":
                         "以上代码编译失败，编译器报错：\n" + msg[:1500] +
                         "\n请修正后重新输出**完整**代码（只输出一个 ```c 代码块，不要解释）。"})

    total = time.time() - total_t0
    print(f"\n[失败] {args.rounds} 轮都没编过，合计 {total:.1f}s")
    (OUT_DIR / "last_failure.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1


if __name__ == "__main__":
    sys.exit(main())
