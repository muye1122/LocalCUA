"""真机验证「单击→双击」改写：找一个看得见的元素，走完整动作管线点它，看结果。

用法：
    python tools/try_dblclick.py 测试
    python tools/try_dblclick.py 测试 --click      # 真的点下去（默认只测算，不动鼠标）

默认是干跑（打印坐标和改写结果，不发鼠标事件）—— 想真点加 --click。
"""

import sys
import time

import yaml

sys.path.insert(0, r"D:\LocalCUA")

from agent.core import AgentSession   # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = sys.argv[1]
    do_click = "--click" in sys.argv

    cfg = yaml.safe_load(open(r"D:\LocalCUA\config.yaml", encoding="utf-8"))
    s = AgentSession(cfg)
    s.log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)

    img = s.screenshot_fn()
    img_s, scale = s._fit_image(img)
    els, _ = s.grounding.annotate(img_s)
    new = s.grounding.ocr_enrich(img, els, scale=scale)
    els = els + new
    s.grounding.ensure_captions_batch(img_s, els, max_n=40)
    s._last_elements = els

    print(f"前台：{s._window_info()[0]}　类名：{s._foreground_class()}")
    print(f"文件列表矩形：{s._explorer_list_rect()}")
    hit = [e for e in els if target in (e.get("label") or "")]
    if not hit:
        print(f"✗ 元素清单里没有「{target}」，先看看别的：")
        for e in els[:25]:
            print(f"   [{e['id']}] {e.get('label')} {e['bbox']}")
        return 1

    el = hit[0]
    x1, y1, x2, y2 = el["bbox"]
    act = {"action": "click", "element_id": el["id"], "expect_change": True}
    print(f"目标：[{el['id']}] {el['label']} 中心({(x1+x2)//2},{(y1+y2)//2})")

    act = s._to_screen_coords(act, scale)
    act.pop("element_id", None)
    act = s._prefer_double_click(act)
    print(f"管线结果：{act}")

    if not do_click:
        print("（干跑，没有真的点。加 --click 才动鼠标）")
        return 0

    s.executor.execute(act)
    time.sleep(1.5)
    print(f"点完之后前台：{s._window_info()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
