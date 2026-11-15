#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grounding（元素定位）层 —— LocalCUA 相比“纯坐标”方案最大的升级点。

核心思路：先把屏幕解析成「带编号的可交互元素列表」，让模型输出元素编号，
再由 grounding 把编号解析成精确坐标。点中率显著高于模型直接猜坐标。

四种 backend：
  - vlm         : 默认。复用主视觉模型自己框出可交互元素，零额外依赖、开箱即用
  - none        : 不解析，回到“像素+绝对坐标”行为（最快）
  - omniparser  : 调用独立的 OmniParser 服务，返回图标/文字/可点区域的 bbox+标签（高精度）
  - ui_tars     : 使用 UI-TARS 系模型，输出 0~factor 归一化坐标（factor 默认 1000），再做检测

每个 Grounding 实现两个方法：
  annotate(img) -> (elements, annotated_img)
      elements: [{"id": int, "label": str, "bbox": [x1,y1,x2,y2]}, ...]（坐标基于原图像素）
      annotated_img: 画了框+编号的图（用于调试/界面展示）
  resolve(element_id) -> (x, y)   把元素编号解析成中心点坐标
"""

import io
import json
import base64
import urllib.request
from abc import ABC, abstractmethod

from PIL import Image, ImageDraw


class BaseGrounding(ABC):
    def __init__(self, cfg=None):
        self._last_elements = []

    @abstractmethod
    def annotate(self, img: Image.Image):
        ...

    @abstractmethod
    def resolve(self, element_id):
        ...

    # 以下为通用实现，子类可直接复用
    def _resolve(self, element_id):
        for el in self._last_elements:
            if str(el["id"]) == str(element_id):
                x1, y1, x2, y2 = el["bbox"]
                return ((x1 + x2) / 2, (y1 + y2) / 2)
        return None

    @staticmethod
    def _draw(img, elements):
        draw = ImageDraw.Draw(img)
        for el in elements:
            x1, y1, x2, y2 = el["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, max(0, y1 - 12)), f"[{el['id']}] {el.get('label', '')}", fill="red")
        return img

    @property
    def last_elements(self):
        return self._last_elements


class NoopGrounding(BaseGrounding):
    """backend=none：不做元素解析，保持原版行为。模型直接给坐标。"""

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def annotate(self, img: Image.Image):
        self._last_elements = []
        return [], img

    def resolve(self, element_id):
        return None


class VLMGrounding(BaseGrounding):
    """
    backend=vlm（默认）：复用主视觉模型，让它“看截图 -> 列出可交互元素 + 包围盒”。
    无需任何额外服务或权重，只要有能看图说话的本地 VL 模型即可。
    解析到 bbox 后，动作模型只需输出 element_id，grounding 取其中心坐标执行，
    比模型直接给绝对坐标更稳；若清单里没有合适元素，动作模型仍可回退到 x/y 坐标。
    """

    DETECT_PROMPT = (
        "You are a strict UI element detector. Examine the screenshot and list the "
        "interactive/clickable UI elements (buttons, links, text inputs, checkboxes, "
        "icons, tabs, menu items). Return ONLY a JSON array, each item formatted as "
        '{"id": int starting at 1, "label": short description, '
        '"bbox": [x1, y1, x2, y2]} in absolute pixel coordinates of the screenshot. '
        "Omit uncertain items. Do not output any text outside the JSON array."
    )

    def __init__(self, cfg: dict, model=None):
        super().__init__()
        self.model = model
        g = (cfg.get("grounding", {}) or {}).get("vlm", {}) or {}
        self.max_elements = int(g.get("max_elements", 40))

    @staticmethod
    def _parse(raw: str, size):
        W, H = size
        t = (raw or "").strip()
        s, e = t.find("["), t.rfind("]")
        if s == -1 or e == -1:
            s, e = t.find("{"), t.rfind("}")
        if s != -1 and e != -1:
            t = t[s : e + 1]
        try:
            data = json.loads(t)
        except Exception:
            # 输出被 max_tokens 截断时，抢救最后一个完整的对象
            data = None
            start = t.find("[")
            if start != -1:
                last = t.rfind("}")
                while last != -1:
                    try:
                        data = json.loads(t[start : last + 1] + "]")
                        break
                    except Exception:
                        last = t.rfind("}", 0, last)
            if data is None:
                return []
        if isinstance(data, dict):
            data = data.get("elements", []) or []
        els = []
        for i, item in enumerate(data, start=1):
            try:
                bbox = item.get("bbox") or item.get("box")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = (float(v) for v in bbox)
                x1, y1 = max(0, min(W, x1)), max(0, min(H, y1))
                x2, y2 = max(0, min(W, x2)), max(0, min(H, y2))
                if x2 - x1 < 3 or y2 - y1 < 3:
                    continue
                els.append({"id": i, "label": str(item.get("label", "")),
                            "bbox": [x1, y1, x2, y2]})
            except Exception:
                continue
            if len(els) >= 40:
                break
        return els

    def _detect(self, img: Image.Image):
        if self.model is None:
            return []
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        content = [
            {"type": "text", "text": self.DETECT_PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
        ]
        try:
            # 元素清单比动作 JSON 长得多，必须给足输出长度，否则会被截断成非法 JSON
            raw = self.model.chat([{"role": "user", "content": content}],
                                  temperature=0.0, max_tokens=1200)
        except Exception as e:
            print(f"[grounding] VLM 检测失败，降级为无元素: {e}")
            return []
        return self._parse(raw, img.size)[: self.max_elements]

    def annotate(self, img: Image.Image):
        els = self._detect(img)
        self._last_elements = els
        return els, self._draw(img.copy(), els)

    def resolve(self, element_id):
        return self._resolve(element_id)


class OmniParserGrounding(BaseGrounding):
    """
    调用一个独立运行的 OmniParser 服务（见 README 的启动方式），提供更高精度的检测。
    服务约定：POST {server_url}/parse  { "image": "<base64 png>" }
            返回 { "elements": [ {"id":..,"label":..,"bbox":[x1,y1,x2,y2]} ] }
    """

    def __init__(self, cfg: dict):
        super().__init__()
        g = (cfg.get("grounding", {}) or {}).get("omniparser", {}) or {}
        self.server_url = g.get("server_url", "http://localhost:8000").rstrip("/")

    def annotate(self, img: Image.Image):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        payload = json.dumps({"image": b64}).encode("utf-8")
        req = urllib.request.Request(
            self.server_url + "/parse",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            elements = data.get("elements", [])
        except Exception as e:
            print(f"[grounding] OmniParser 调用失败，降级为无元素: {e}")
            elements = []
        self._last_elements = elements
        return elements, self._draw(img, elements)

    def resolve(self, element_id):
        return self._resolve(element_id)


class UITarsGrounding(BaseGrounding):
    """
    UI-TARS 系模型原生 grounding。UI-TARS（字节）在 action 层面直接输出
    0~factor（默认 1000）归一化坐标。此 backend 复用主模型做一次「元素检测」，
    但要求模型按归一化空间返回 bbox；随后按 factor 与截图尺寸换算成像素坐标。
    需要后端是一个 UI-TARS 系 / 支持 0-1000 归一化输出的视觉模型。
    """

    DETECT_PROMPT = (
        "You are a strict UI element detector for a UI-TARS style model. Examine the "
        "screenshot and list interactive elements. Return ONLY a JSON array of objects "
        '{"id": int starting at 1, "label": short text, "bbox": [x1,y1,x2,y2]} '
        "where coordinates are in the 0-1000 NORMALIZED space (the model's native "
        "output range), NOT pixels. Omit uncertain items. No text outside JSON."
    )

    def __init__(self, cfg: dict, model=None):
        super().__init__()
        self.model = model
        g = (cfg.get("grounding", {}) or {}).get("ui_tars", {}) or {}
        self.factor = int(g.get("factor", 1000))   # 归一化上限，UI-TARS 为 1000
        # 原来默认 "qwen25vl" 指向已删除的 qwen2.5vl:7b；这个字段目前没被用到，
        # 留个空默认值，别让死配置继续暗示某个不存在的模型。
        self.model_type = g.get("model_type", "")

    @staticmethod
    def _parse(raw: str, size, factor):
        W, H = size
        t = (raw or "").strip()
        s, e = t.find("["), t.rfind("]")
        if s == -1 or e == -1:
            s, e = t.find("{"), t.rfind("}")
        if s != -1 and e != -1:
            t = t[s : e + 1]
        try:
            data = json.loads(t)
        except Exception:
            return []
        if isinstance(data, dict):
            data = data.get("elements", []) or []
        scale_x, scale_y = W / factor, H / factor
        els = []
        for i, item in enumerate(data, start=1):
            try:
                bbox = item.get("bbox") or item.get("box")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = (float(v) * scale_x for v in bbox[:2])
                x2b, y2b = (float(v) * scale_y for v in bbox[2:4])
                x1, y1 = max(0, min(W, x1)), max(0, min(H, y1))
                x2, y2 = max(0, min(W, x2b)), max(0, min(H, y2b))
                if x2 - x1 < 3 or y2 - y1 < 3:
                    continue
                els.append({"id": i, "label": str(item.get("label", "")),
                            "bbox": [x1, y1, x2, y2]})
            except Exception:
                continue
            if len(els) >= 40:
                break
        return els

    def _detect(self, img: Image.Image):
        if self.model is None:
            return []
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        content = [
            {"type": "text", "text": self.DETECT_PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
        ]
        try:
            raw = self.model.chat([{"role": "user", "content": content}], temperature=0.0)
        except Exception as e:
            print(f"[grounding] UI-TARS 检测失败，降级为无元素: {e}")
            return []
        return self._parse(raw, img.size, self.factor)

    def annotate(self, img: Image.Image):
        els = self._detect(img)
        self._last_elements = els
        return els, self._draw(img.copy(), els)

    def resolve(self, element_id):
        return self._resolve(element_id)


def build_grounding(cfg: dict, model=None) -> BaseGrounding:
    backend = (cfg.get("grounding", {}) or {}).get("backend", "vlm")
    if backend == "omniparser":
        return OmniParserGrounding(cfg)
    if backend == "ui_tars":
        return UITarsGrounding(cfg, model)
    if backend == "vlm":
        return VLMGrounding(cfg, model)
    if backend == "yolo_caption":
        from agent.yolo_grounding import YoloCaptionGrounding
        return YoloCaptionGrounding(cfg, model)
    return NoopGrounding()
