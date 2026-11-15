"""定位耗时探针：不碰鼠标键盘，只量 [定位] 这一步到底花多久。

真机问题：一步 66~124 秒，其中 40~90 秒都花在 resolve_desc 逐个元素起名上。
用法：
    .venv/Scripts/python.exe tools/try_locate.py "任务栏上的 Edge 图标"
"""
import sys
import time

sys.path.insert(0, ".")
from agent.config import load_config
from agent.core import AgentSession
from agent.models import ModelClient
from agent.yolo_grounding import YoloCaptionGrounding

TARGETS = sys.argv[1:] or ["任务栏上的 Edge 图标", "任务栏最右侧的微信图标",
                           "保存按钮", "左侧聊天列表里的第一个会话"]

cfg = load_config()
model = ModelClient(cfg)
g = YoloCaptionGrounding(cfg["grounding"]["yolo_caption"], model=model)

import pyautogui  # noqa: E402
img = pyautogui.screenshot()
img_s, scale = AgentSession._fit_image(AgentSession(cfg, log=lambda m: None), img)

t0 = time.time()
els, _ = g.annotate(img_s.copy())
t_det = time.time() - t0
t1 = time.time()
new = g.ocr_enrich(img, els, scale=scale)
els = els + new
t_ocr = time.time() - t1
t2 = time.time()
g.ensure_captions_batch(img_s, els, max_n=40)
named = sum(1 for e in els[:40] if e.get("label"))
t_cap = time.time() - t2
print(f"检测 {t_det:.1f}s  OCR {t_ocr:.1f}s  批量认名 {t_cap:.1f}s  "
      f"（{len(els)} 个元素，{named} 个有名字）")

for tgt in TARGETS:
    t3 = time.time()
    r = g.resolve_desc(tgt, img_s)
    dt = time.time() - t3
    if r:
        print(f"[定位] {tgt!r} -> #{r[2]} {r[3]!r} 分{r[4]}  {dt:.1f}s")
    else:
        print(f"[定位] {tgt!r} -> 未匹配（宁可不点）  {dt:.1f}s")
