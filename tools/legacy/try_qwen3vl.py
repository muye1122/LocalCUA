"""qwen3vl:8b 实测：速度、看的准不准、给不给 actions 数组。

不碰鼠标键盘，只问模型。用法：
    .venv/Scripts/python.exe tools/try_qwen3vl.py [模型名]
"""
import base64
import io
import json
import sys
import time
import urllib.request

sys.path.insert(0, ".")
from agent.core import AgentSession  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3vl:8b"


def shot_b64():
    from PIL import Image
    import pyautogui
    img = pyautogui.screenshot()
    img.thumbnail((1280, 1280))
    b = io.BytesIO()
    img.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode(), img.size


def chat(system, user, image=None, num_ctx=8192):
    msg = {"role": "user", "content": user}
    if image:
        msg["images"] = [image]
    body = json.dumps({
        "model": MODEL, "stream": False,
        "messages": [{"role": "system", "content": system}, msg],
        "options": {"temperature": 0.1, "num_ctx": num_ctx},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d["message"]["content"], time.time() - t0, d


sess = AgentSession({"safety": {}, "grounding": {"backend": "none"},
                     "planner": {"enabled": False}, "validator": {"enabled": False},
                     "agent": {"max_steps": 1}})
system = sess._system_prompt(1280, 720)

cases = [
    ("打开浏览器搜索今天的天气", None),
    ("在文件资源管理器里新建一个文件夹叫报告", None),
    ("打开记事本输入 hello world", None),
]

print("=" * 74)
print(f"模型：{MODEL}")
for name, _ in cases:
    user = (f"任务：{name}\n当前子任务：整体任务\n"
            "【系统窗口状态】前台窗口：桌面\n可交互元素清单（空）\n请给出下一步。")
    raw, dt, _ = chat(system, user)
    acts = None
    try:
        s = raw[raw.index("{"): raw.rindex("}") + 1]
        d = json.loads(s)
        acts = d.get("actions")
    except Exception:
        d = {}
    print("-" * 74)
    print(f"任务：{name}   ({dt:.1f}s)")
    print("原文：", raw.strip()[:240].replace("\n", " "))
    print(f"→ plan={d.get('plan')!r}  动作数={len(acts or [])}")

# 视觉：给一张真截图，问它在哪、下一步点哪
img, size = shot_b64()
print("=" * 74)
print(f"视觉测试（截图 {size[0]}x{size[1]}）")
user = ("任务：打开浏览器搜索今天的天气\n当前子任务：整体任务\n"
        f"这是当前屏幕截图（{size[0]}x{size[1]}）。请指出你下一步要点哪里。")
raw, dt, d = chat(system, user, image=img)
print(f"耗时 {dt:.1f}s   eval_count={d.get('eval_count')}  "
      f"eval_duration={d.get('eval_duration', 0) / 1e9:.1f}s  "
      f"prompt_eval={d.get('prompt_eval_count')}")
print("原文：", raw.strip()[:400].replace("\n", " "))
