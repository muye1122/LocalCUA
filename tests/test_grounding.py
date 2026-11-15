"""Grounding 层测试：Noop / VLM 自检测 / OmniParser 适配 / 路由。"""

import json
import os
import tempfile
from unittest.mock import patch

from PIL import Image

from agent.grounding import (
    NoopGrounding,
    VLMGrounding,
    OmniParserGrounding,
    UITarsGrounding,
    build_grounding,
)
from conftest import FakeModel


def _img():
    return Image.new("RGB", (100, 80), (255, 255, 255))


def test_noop_grounding():
    g = NoopGrounding({})
    els, ann = g.annotate(_img())
    assert els == [] and ann is not None
    assert g.resolve(1) is None


def test_build_grounding_dispatch():
    assert isinstance(build_grounding({"grounding": {"backend": "none"}}), NoopGrounding)
    assert isinstance(build_grounding({"grounding": {"backend": "vlm"}}, model=object()), VLMGrounding)
    assert isinstance(build_grounding({"grounding": {"backend": "omniparser"}}), OmniParserGrounding)
    assert isinstance(build_grounding({"grounding": {"backend": "ui_tars"}}), UITarsGrounding)
    # 默认 backend 应为 vlm
    assert isinstance(build_grounding({}), VLMGrounding)


def test_vlm_grounding_parse_and_resolve():
    model = FakeModel(replies=['[{"id":1,"label":"OK","bbox":[10,10,50,40]}]'])
    g = VLMGrounding({"grounding": {"vlm": {"max_elements": 5}}}, model)
    els, ann = g.annotate(_img())
    assert len(els) == 1
    assert els[0]["id"] == 1 and els[0]["label"] == "OK"
    assert els[0]["bbox"] == [10, 10, 50, 40]
    cx, cy = g.resolve(1)
    assert (cx, cy) == (30, 25)
    # 标注图应为新对象
    assert ann is not None


def test_vlm_grounding_robust_to_prose_and_fences():
    reply = '这里有一些元素：\n```json\n[{"id":1,"label":"btn","bbox":[0,0,10,10]}]\n```\n结束'
    g = VLMGrounding({}, FakeModel(replies=[reply]))
    els, _ = g.annotate(_img())
    assert len(els) == 1
    assert els[0]["bbox"] == [0, 0, 10, 10]


def test_vlm_grounding_clamps_out_of_bounds():
    g = VLMGrounding({}, FakeModel(replies=['[{"id":1,"label":"x","bbox":[-5,-5,200,200]}]']))
    els, _ = g.annotate(_img())
    assert els[0]["bbox"] == [0, 0, 100, 80]


def test_vlm_grounding_filters_tiny_boxes():
    g = VLMGrounding({}, FakeModel(replies=['[{"id":1,"label":"x","bbox":[0,0,1,1]}]']))
    els, _ = g.annotate(_img())
    assert els == []


def test_vlm_grounding_empty_on_bad_json():
    g = VLMGrounding({}, FakeModel(replies=["我看不出有什么可点"]))
    els, _ = g.annotate(_img())
    assert els == []


def test_omniparser_grounding(monkeypatch):
    payload = {"elements": [{"id": 2, "label": "save", "bbox": [5, 5, 20, 20]}]}

    class FakeResp:
        def read(self):
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch("urllib.request.urlopen", lambda req, timeout=0: FakeResp()):
        g = OmniParserGrounding({"grounding": {"omniparser": {"server_url": "http://x"}}})
        els, _ = g.annotate(_img())
    assert len(els) == 1 and els[0]["id"] == 2
    cx, cy = g.resolve(2)
    assert (cx, cy) == (12.5, 12.5)


def test_omniparser_falls_back_on_error(monkeypatch):
    def boom(req, timeout=0):
        raise OSError("conn refused")

    with patch("urllib.request.urlopen", boom):
        g = OmniParserGrounding({"grounding": {"omniparser": {"server_url": "http://x"}}})
        els, _ = g.annotate(_img())
    assert els == []


def test_uitars_grounding_is_passthrough():
    g = UITarsGrounding({})
    els, _ = g.annotate(_img())
    assert els == []
    assert g.resolve(1) is None


# ---------- yolo_caption 的起名开销 ----------
# 真机实测：起名每步 6~12 秒，占整步时间三分之一；而其中 4~5 秒是图像编码的
# 固定开销，跟「问几个元素」无关。所以省时间的唯一办法是「少发请求」。

def _yc(cfg_extra=None):
    from agent.yolo_grounding import YoloCaptionGrounding
    cfg = {"grounding": {"yolo_caption": cfg_extra or {}}}
    g = YoloCaptionGrounding(cfg, model=None)
    return g


def test_el_key_ignores_position():
    """同一个图标挪到别处必须命中同一条缓存。

    旧键把 bbox 拼进去了，窗口一移、窗口一缩，满屏「×」「⋯」全成新元素，
    每次都要重新问模型一遍。
    """
    from agent.yolo_grounding import YoloCaptionGrounding as Y
    img = Image.new("RGB", (200, 200), "white")
    for i in range(30, 180, 10):
        img.putpixel((i, i), (i, i, i))
    a = img.crop((0, 0, 32, 32))
    b = img.crop((0, 0, 32, 32))          # 内容相同
    assert Y._el_key(a, [10, 10, 42, 42]) == Y._el_key(b, [500, 300, 532, 332])


def test_el_key_distinguishes_content():
    from agent.yolo_grounding import YoloCaptionGrounding as Y
    a = Image.new("RGB", (32, 32), "white")
    b = Image.new("RGB", (32, 32), "black")
    assert Y._el_key(a) != Y._el_key(b)


def _grounding_with_tmp_cache(extra):
    """用临时缓存的 grounding：测试绝不能写生产缓存文件（data/caption_cache.json）。"""
    import tempfile
    from agent.yolo_grounding import _CaptionCache
    g = _yc(extra)
    g.cache = _CaptionCache(os.path.join(tempfile.mkdtemp(), "cache.json"))
    return g


def test_skip_caption_when_few_unlabeled():
    """只差一两个图标没名字时，不该为它付一次完整模型调用。"""
    from PIL import ImageDraw
    g = _grounding_with_tmp_cache({"skip_below": 3})
    calls = []

    class M:
        def chat(self, *a, **k):
            calls.append(1)
            return ""

    g.model = M()
    # 每个元素区域涂成不同颜色：纯色图会让所有裁剪图哈希相同，缓存一命中就测不出意图
    img = Image.new("RGB", (400, 400), "white")
    d = ImageDraw.Draw(img)

    def el(i, x, y):
        d.rectangle([x, y, x + 24, y + 24], fill=(i * 30 % 256, (255 - i * 30) % 256, 90))
        return {"id": i, "label": "", "bbox": [x, y, x + 24, y + 24]}

    g.ensure_captions_batch(img, [el(1, 20, 20), el(2, 200, 200)], max_n=40)
    assert calls == [], "2 个缺名元素不值得发一次 6~12 秒的请求"

    many = [el(i, i * 40, i * 40) for i in range(1, 8)]
    g.cache = type(g.cache)(os.path.join(tempfile.mkdtemp(), "cache.json"))  # 换新缓存
    g.ensure_captions_batch(img, many, max_n=40)
    assert len(calls) == 1, "7 个缺名元素值得问一次"


# ---------- 只对变化区域跑 OCR 时，坐标要补裁剪偏移 ----------
# 「青蛙眼」为了省时间会把整屏裁成一小块再 OCR。不补偏移的话，读出来的文字
# 坐标会整体偏向左上角，名字全回填到错元素上 —— 比不省时间糟糕得多。

class _FakeOCRResult:
    def __init__(self, boxes, txts):
        self.boxes = boxes
        self.txts = txts


def test_ocr_enrich_applies_crop_origin():
    from agent.yolo_grounding import YoloCaptionGrounding

    g = _yc({})
    box = [[10, 20], [60, 20], [60, 40], [10, 40]]
    with patch.object(YoloCaptionGrounding, "_get_ocr",
                      classmethod(lambda cls: (lambda arr: _FakeOCRResult([box], ["保存"])))):
        new = g.ocr_enrich(Image.new("RGB", (200, 100), "white"), [],
                           scale=0.5, origin=(100, 50))
    assert len(new) == 1
    b = new[0]["bbox"]
    assert b[0] == (10 + 100) * 0.5 and b[1] == (20 + 50) * 0.5, \
        f"裁剪偏移没补上，坐标会是 {b} 而不是 (55,35) 起"
    assert b[2] == (60 + 100) * 0.5 and b[3] == (40 + 50) * 0.5
    assert new[0]["label"] == "保存"


def test_ocr_enrich_no_origin_keeps_old_behaviour():
    from agent.yolo_grounding import YoloCaptionGrounding

    g = _yc({})
    box = [[10, 20], [60, 20], [60, 40], [10, 40]]
    with patch.object(YoloCaptionGrounding, "_get_ocr",
                      classmethod(lambda cls: (lambda arr: _FakeOCRResult([box], ["保存"])))):
        new = g.ocr_enrich(Image.new("RGB", (200, 100), "white"), [], scale=1.0)
    assert new[0]["bbox"] == [10, 20, 60, 40]


def test_ocr_enrich_merges_previous_texts_for_double_click_check():
    """裁剪时只看了一小块，屏幕别处的文件名不该凭空消失。

    last_texts 是「点击目标附近有没有带扩展名的文件名」的判断依据 ——
    它一丢，主文件夹页里的文件就会被单击当成文件夹处理。
    """
    from agent.yolo_grounding import YoloCaptionGrounding

    g = _yc({})
    with patch.object(YoloCaptionGrounding, "_get_ocr",
                      classmethod(lambda cls: (lambda arr: _FakeOCRResult([], [])))):
        g.ocr_enrich(Image.new("RGB", (200, 100), "white"), [], scale=1.0,
                     origin=(0, 0), extra_texts=[(1, 2, 3, 4, "报表.xlsx")])
    assert [t[4] for t in g.last_texts] == ["报表.xlsx"]
