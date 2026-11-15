#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
try_devcpp —— 最小验证：能不能把 C 代码逐字敲进 Dev-C++ 的编辑区。

只做四件事：打开 Dev-C++ → Ctrl+N → 点编辑区 → 逐字敲一段代码。
不保存、不编译（编译已经用 gcc 验证过了），所以不会弹任何对话框。

跑之前：把鼠标放一边别动，脚本会接管 5~10 秒。

用法：
  python tools/try_devcpp.py

成功后你会看到 Dev-C++ 里出现一段求斐波那契前 10 项的 C 代码，
截图存在 data/devcpp_test/ 下。
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

CODE = """#include <stdio.h>

int main() {
    int i, a = 0, b = 1, t;
    for (i = 0; i < 10; i++) {
        printf("%d ", a);
        t = a + b;
        a = b;
        b = t;
    }
    printf("\\n");
    return 0;
}
"""


DEVCPP_EXE = Path(r"D:\dev-C++\Dev-Cpp\devcpp.exe")
DEVCPP_EXE_ALT = Path(r"C:\Program Files (x86)\Dev-Cpp\devcpp.exe")


def launch_devcpp(sess):
    """先走"点开始菜单搜程序"（演示时更像人在操作），失败再直接启动 exe。"""
    if sess._open_app("Dev-C++"):
        return True
    for exe in (DEVCPP_EXE, DEVCPP_EXE_ALT):
        if exe.exists():
            print(f"      开始菜单没搜到，直接启动 {exe}")
            import subprocess
            subprocess.Popen([str(exe)])
            for _ in range(20):
                time.sleep(0.5)
                if sess._focus_program("Dev-C++"):
                    return True
    return False


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    from agent.core import AgentSession

    sess = AgentSession(cfg)
    sess.log = print
    out_dir = ROOT / "data" / "devcpp_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] 打开 Dev-C++…")
    if not launch_devcpp(sess):
        print("❌ 没能打开 Dev-C++。请确认 D:\\dev-C++\\Dev-Cpp\\devcpp.exe 存在，"
              "或先手动开一次让它出现在任务栏。")
        return 1

    print("[2/5] Ctrl+N 新建源文件…")
    import pyautogui
    time.sleep(0.8)
    pyautogui.hotkey("ctrl", "n")
    time.sleep(0.8)

    print("[3/5] 定位编辑区…")
    c = sess._focused_control_center()
    if c:
        print(f"      焦点控件 @({c[0]:.0f},{c[1]:.0f}) {c[2]:.0f}x{c[3]:.0f}px，点它")
        pyautogui.click(int(c[0]), int(c[1]))
        time.sleep(0.4)
    else:
        print("      ⚠️ 拿不到焦点控件，直接开始敲（可能敲到别处，注意看）")

    print("[4/5] 逐行敲代码（换行用回车，避免多行注入出错）…")
    from agent.actions import _type_unicode
    t0 = time.time()
    for idx, line in enumerate(CODE.split("\n"), 1):
        if line:
            _type_unicode(line)
        pyautogui.press("enter")
        print(f"      第 {idx:2d}/{len(CODE.split(chr(10)))} 行  已用 {time.time() - t0:.1f}s")
    print(f"      敲完共 {time.time() - t0:.1f}s")

    print("[5/5] 截图存档…")
    time.sleep(0.6)
    img = sess.screenshot_fn()
    p = out_dir / f"devcpp_{time.strftime('%H%M%S')}.png"
    img.save(p)
    print(f"\n✅ 截图：{p}")
    print("   请看一眼 Dev-C++：如果代码整齐出现、花括号引号都对，这条路就通了。")
    print("   如果出现乱码/丢字符/换行错位，把截图发我。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
