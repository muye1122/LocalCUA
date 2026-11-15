#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YoloCaptionGrounding —— 实测后的最优组合：

  YOLO(OmniParser icon_detect, ONNX) 负责「哪里有可点元素」  -> bbox 精度 ±3px, 200ms
  程序(bigram匹配+位置启发)     负责「描述 -> 哪个元素」     -> 定量计算零幻觉
  本地 VLM(7B)                  只负责「给元素起名」与「说出要点什么」-> 定性描述, 它擅长

实测依据（tools/caption_test.py 等实验）：
  - 让 7B 猜坐标：误差 ±30~90px，两步局部放大也无改善
  - 让 7B 选编号（看图对号入座 / 纯文本清单）：均不稳定
  - 逐元素裁小图让 7B 起名：可靠（小图定性是它强项）
  - 程序用 bigram+位置启发做目标匹配：搜索框/开始按钮/新建任务全部精确命中

坐标换算已标定（tools/calibrate.py）：pyautogui 与截图像素 1:1，无 DPI 陷阱。
"""

import hashlib
import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agent.grounding import BaseGrounding

_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models", "icon_detect", "model.onnx")
_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "caption_cache.json")

_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort
        _SESSION = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])
    return _SESSION


# ---------- YOLO 检测（与 tools/icon_detect.py 同逻辑） ----------

def _letterbox(img, size=640):
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (dx, dy))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None], r, dx, dy


def _nms(boxes, scores, iou_thr=0.45):
    idx = scores.argsort()[::-1]
    keep = []
    while idx.size:
        i = idx[0]
        keep.append(i)
        if idx.size == 1:
            break
        rest = idx[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.clip(a1 + a2 - inter, 1e-9, None)
        idx = rest[iou <= iou_thr]
    return keep


def detect_elements(img: Image.Image, conf: float = 0.3):
    """返回原图坐标 [x1,y1,x2,y2,score]，按阅读顺序（上->下、左->右）排序编号。"""
    arr, r, dx, dy = _letterbox(img)
    out = _session().run(None, {"images": arr})[0][0].T  # (8400,5)
    pred = out[out[:, 4] >= conf]
    if len(pred) == 0:
        return []
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep = _nms(boxes, pred[:, 4])
    boxes, scores = boxes[keep], pred[keep, 4]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / r
    iw, ih = img.size
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, iw)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, ih)
    dets = [[float(v) for v in b] + [float(s)] for b, s in zip(boxes, scores)]
    dets.sort(key=lambda d: (d[1] // 60, d[0]))  # 阅读顺序
    return dets


# ---------- 文本匹配 ----------

def _bigrams(text):
    words = re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", (text or "").lower())
    grams = set()
    for w in words:
        if len(w) == 1:
            grams.add(w)
        for i in range(len(w) - 1):
            grams.add(w[i:i + 2])
    return grams


def position_bonus(target, cx, cy, sw, sh):
    """程序解析目标描述里的位置词，给符合的元素加分。定量归程序，零幻觉。"""
    bonus = 0.0
    xr, yr = cx / sw, cy / sh
    if "任务栏" in target:
        bonus += 0.5 if yr > 0.9 else -0.5
    if "底部" in target:
        bonus += 0.4 if yr > 0.8 else -0.3
    if "顶部" in target:
        bonus += 0.4 if yr < 0.12 else -0.3
    if "右上角" in target:
        bonus += 0.5 if (xr > 0.8 and yr < 0.12) else -0.4
    if "左上角" in target:
        bonus += 0.5 if (xr < 0.3 and yr < 0.12) else -0.4
    if "右下" in target:
        bonus += 0.4 if (xr > 0.6 and yr > 0.8) else -0.3
    if "左下" in target:
        bonus += 0.4 if (xr < 0.45 and yr > 0.8) else -0.3
    if "最左" in target:
        bonus += 0.4 if xr < 0.06 else -0.3
    if "最右" in target:
        bonus += 0.4 if xr > 0.94 else -0.3
    if "左边" in target or "左侧" in target or "左上" in target:
        bonus += 0.25 if xr < 0.4 else -0.2
    if "右边" in target or "右侧" in target or "右上" in target:
        bonus += 0.25 if xr > 0.6 else -0.2
    if "输入框" in target or "搜索框" in target:
        bonus += 0.15 if yr > 0.7 else 0.0
    return bonus


def match_score(target, caption, cx, cy, sw, sh):
    tg, cg = _bigrams(target), _bigrams(caption)
    if not tg or not cg:
        return position_bonus(target, cx, cy, sw, sh)
    ov = len(tg & cg)
    return ov / min(len(tg), len(cg)) + position_bonus(target, cx, cy, sw, sh)


# ---------- 元素起名（带缓存） ----------

class _CaptionCache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        try:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    self.data = json.load(f)
        except Exception:
            self.data = {}

    def key(self, bbox, size):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        return f"{x1},{y1},{x2},{y2}|{size[0]}x{size[1]}"

    def get(self, key):
        return self.data.get(key)

    def put(self, key, val):
        self.data[key] = val
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False)
        except Exception:
            pass


class YoloCaptionGrounding(BaseGrounding):
    """backend=yolo_caption：YOLO 检测 + VLM 起名 + 程序匹配。"""

    CAPTION_PROMPT = (
        "这是电脑屏幕上某个界面元素的放大截图。请用不超过14个字描述："
        "1)上面的文字（原样抄）；2)若没有文字，说明这是什么图标/按钮，"
        "尽量说出它属于哪个软件（如 Windows 徽标、微信、浏览器等），以及颜色形状。"
        "直接描述，不要开头语。"
    )

    def __init__(self, cfg: dict, model=None):
        super().__init__()
        self.model = model
        g = (cfg.get("grounding", {}) or {}).get("yolo_caption", {}) or {}
        self.conf = float(g.get("conf", 0.3))
        self.max_elements = int(g.get("max_elements", 80))
        # 缺名元素少于这个数就不批量起名了：一次调用固定要付 4~5 秒的图像
        # 编码开销，跟问几个无关。为 1 个图标花 8 秒不值 —— 模型可以用坐标点它。
        self.skip_below = int(g.get("skip_below", 3))
        self.cache = _CaptionCache(_CACHE_PATH)
        self._captions = []          # 与 _last_elements 对齐
        self.last_batch = (0, 0)     # 最近一次批量起名的 (认出几个, 问了几个)；
                                     # 必须在这里初始化：缓存全命中时会提前返回，不赋值
        self._screen_size = None

    # ---- 起名 ----
    def describe(self, img: Image.Image, bbox):
        x1, y1, x2, y2 = bbox
        pw = int((x2 - x1) * 0.3) + 6
        ph = int((y2 - y1) * 0.3) + 6
        crop = img.crop((max(0, int(x1 - pw)), max(0, int(y1 - ph)),
                         min(img.size[0], int(x2 + pw)), min(img.size[1], int(y2 + ph))))
        scale = min(4.0, max(1.0, 320 / max(crop.size)))
        if scale > 1.05:
            crop = crop.resize((int(crop.size[0] * scale), int(crop.size[1] * scale)), Image.LANCZOS)
        # 缓存键必须含内容哈希：同一位置屏幕内容变了要重新起名
        h = hashlib.md5(crop.tobytes()).hexdigest()[:12]
        key = f"{h}|{int(x1)},{int(y1)}"
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        cap = ""
        if self.model is not None:
            try:
                raw = self.model.chat([{"role": "user", "content": [
                    {"type": "text", "text": self.CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": _img_data_url(crop)}},
                ]}], temperature=0.0, max_tokens=60)
                cap = (raw or "").strip().splitlines()[0][:40]
            except Exception as e:
                print(f"[grounding] 起名失败: {e}")
        self.cache.put(key, cap)
        return cap

    # ---- 核心接口 ----
    _OCR = None

    @classmethod
    def _get_ocr(cls):
        """本地 OCR（PaddleOCR 的 onnx 版，模型随包安装、离线可用）。"""
        if cls._OCR is None:
            try:
                from rapidocr import RapidOCR
                cls._OCR = RapidOCR()
            except Exception as e:
                print(f"[grounding] OCR 不可用（{e}），回退纯 7B 起名")
                cls._OCR = False
        return cls._OCR or None

    def ocr_enrich(self, img: Image.Image, elements: list, scale: float = 1.0,
                   max_new: int = 40, origin=(0, 0), extra_texts=()) -> list:
        """本地 OCR 读屏幕文字，做两件事：
        ① 把名字回填到 YOLO 元素上（按位置匹配）；
        ② 把"没有任何元素框的独立文字"变成可点元素（返回新增的元素列表）。

        ②是治"找不到文件夹"的根本：icon_detect 只认图标/按钮，纯文字区域不检测
        —— 资源管理器"详细信息"视图里文件名列一个框都没有，模型手里的清单
        永远没有文件名，"测试"文件夹在它的世界里不存在，只能盯着别处乱点。
        而文件名本身就是可点目标（点一下选中、双击打开）。

        为什么有了 7B 批量起名还要 OCR：7B 认屏幕小字不稳（认不出就编编号），
        OCR 是查表式的 —— 读什么是什么，全屏 2 秒、零模型调用。

        img 用原始分辨率截图（缩放图上 12px 的字 OCR 读不清）；
        elements 的 bbox 是缩放图坐标，文字框坐标乘 scale 换算过去再匹配。

        origin：img 是整屏裁下来的一块时，给出裁剪左上角在原图里的物理坐标。
            「青蛙眼」只对变化区域跑 OCR 就靠它 —— 不补这个偏移，读出来的文字
            坐标会整体偏向左上角，名字全回填到错元素上。
        extra_texts：裁剪时上一帧里、落在裁剪框之外的那些文字框。只给
            _prefer_double_click 做"附近有没有文件名"的存在性判断，混进来即可。
        """
        ocr = self._get_ocr()
        if ocr is None:
            return []
        try:
            r = ocr(np.array(img))
            txts = getattr(r, "txts", None)
            boxes = getattr(r, "boxes", None)
            # txts/boxes 可能是 ndarray，直接 `or []` 会对多元素数组报
            # "truth value is ambiguous"，必须先判 None
            txts = [] if txts is None else list(txts)
            boxes = [] if boxes is None else list(boxes)
        except Exception as e:
            print(f"[grounding] OCR 失败: {e}")
            return []
        texts = []
        ox, oy = (int(origin[0]), int(origin[1])) if origin else (0, 0)
        for b, t in zip(boxes, txts):
            xs = [p[0] for p in b]
            ys = [p[1] for p in b]
            # OCR 跑在原图上，文字框坐标要【乘】scale 换算到缩放图坐标系
            # （2560 -> 1536 是 ×0.6；写成除法会把坐标放大 1.67 倍，名字全体错位）
            # 裁剪图则先加回裁剪偏移，才是整屏坐标。
            texts.append((((min(xs) + ox) * scale), ((min(ys) + oy) * scale),
                          ((max(xs) + ox) * scale), ((max(ys) + oy) * scale), str(t)))
        # 留给 _prefer_double_click 用：点击目标附近有没有"带扩展名的文件名"。
        # 真机教训：主文件夹页的「最近使用的文件」里列的是文件，单击升格成双击
        # 会把别人的 Excel 直接打开——文件夹没有扩展名，文件有，一查便知。
        self.last_texts = list(extra_texts) + texts

        # ① 名字回填
        for el in elements:
            if el.get("label"):
                continue
            x1, y1, x2, y2 = [int(v) for v in el["bbox"]]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            best, bd = None, 1e18
            for bx1, by1, bx2, by2, t in texts:
                bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
                # 文字中心横向落在元素范围（放宽 10px），纵向允许到元素下方 40px ——
                # 资源管理器里文件夹图标的文字标签就挂在图标下面
                if bx1 - 10 <= cx <= bx2 + 10 and y1 - 12 <= bcy <= y2 + 40:
                    d = abs(bcx - cx) + abs(bcy - cy)
                    if d < bd:
                        bd, best = d, t
            t = (best or "").strip()
            # 图标会被 OCR 读成 □、Q 之类的符号，那不是真名字，丢掉留给 7B
            if len(t) >= 2 and re.search(r"[\u4e00-\u9fffA-Za-z0-9]{2}", t):
                el["label"] = t[:24]

        # ② 没有元素框覆盖的独立文字 → 新的可点元素
        covered = [(int(e["bbox"][0]) - 10, int(e["bbox"][1]) - 12,
                    int(e["bbox"][2]) + 10, int(e["bbox"][3]) + 40) for e in elements]
        date_re = re.compile(r"^\d{4}[/\-.]")
        next_id = max((e["id"] for e in elements), default=0) + 1
        new = []
        for bx1, by1, bx2, by2, t in texts:
            t = t.strip()
            if len(t) < 2 or date_re.match(t) or t.replace(".", "").replace("/", "").isdigit():
                continue   # 日期、纯数字、单字符不是"目标"
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if any(a - 6 <= cx <= c + 6 and b - 6 <= cy <= d + 6 for a, b, c, d in covered):
                continue   # 已经是某个元素的标签/落在已检测元素上
            if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]{2}", t):
                continue
            new.append({"id": next_id, "label": t[:24], "bbox": [bx1, by1, bx2, by2]})
            next_id += 1
            if len(new) >= max_new:
                break
        return new

    BATCH_PROMPT = (
        "这张图是从电脑屏幕上裁下来的多个界面元素，每个的左上角有红色编号。\n"
        "逐个说出它是什么：\n"
        "· 按钮 → 抄按钮上的字（如 确定、关闭、新建、上一页）\n"
        "· 文件夹或文件 → 抄它的名字（如 测试、报告.docx）\n"
        "· 图标 → 写它代表的程序或对象（如 微信、记事本、回收站、D盘）\n"
        "严格一行一个，格式：编号|名称。编号照抄红色数字。"
        # 名称限长是实打实的省时：认名是自回归生成的，20 个元素 × 每行 10 token
        # ≈ 200 token ≈ 9 秒。模型爱写成「左侧上方的新建文本文档按钮」，
        # 这一句把每行压到 6 字以内，生成量直接砍掉三分之一。
        "名称必须 2~6 个字，写最核心的那个词（如 确定、保存、微信、文件夹）。"
        "不要写解释，不要漏行，看不清就写 未知。"
    )

    def ensure_captions_batch(self, img: Image.Image, elements: list, max_n: int = 40) -> list:
        """一次调用给整批元素起名，名字回填到 el["label"]。

        为什么非做不可：只给「编号 + 坐标」的清单等于一页没有文字的坐标表，
        模型只能回去目测截图上的小字 —— 7B 认不清就编编号（真机日志里出现过
        根本不存在、也就没有坐标的 280 号），编不出来就猜坐标，于是同一个
        位置点出 20/35/42/74/77 五个不同的 x。名字就是它的眼睛。

        成本：逐元素 describe 是 N 次调用（40 个 ≈ 2 分钟），
        拼成一张网格图问一次只要 1 次调用。
        """
        if self.model is None or not elements:
            return elements
        todo = [el for el in elements[:max_n] if not el.get("label")]
        if not todo:
            return elements

        # 先查缓存：界面点一下之后大部分元素没变，命中的就不用再问模型
        prepared = []
        for el in todo:
            crop = self._el_crop(img, el["bbox"])
            if crop.size[0] < 2 or crop.size[1] < 2:
                continue
            ck = self._el_key(crop, el["bbox"])
            hit = self.cache.get(ck)
            if hit == "-":          # 之前问过、认不出来，别每步重新问一遍
                continue
            if hit:
                el["label"] = hit
                continue
            prepared.append((el, crop, ck))
        if not prepared:
            return elements
        # 只剩个把元素没名字，却要为此付一次完整的模型调用（实测 6~12 秒，
        # 其中 4~5 秒是图像编码的固定开销，跟问几个无关）。这些少数派多半是
        # 无字图标，模型照样可以用坐标点它们，不值得让整步慢七八秒。
        if self.skip_below > 0 and len(prepared) <= self.skip_below:
            self._batch_done = True      # 别让逐个补问的兜底再来一轮
            return elements
        todo = [p[0] for p in prepared]

        CELL_W, CELL_H, PAD, COLS = 200, 60, 8, 4
        rows = (len(todo) + COLS - 1) // COLS
        canvas = Image.new("RGB", (COLS * (CELL_W + PAD) + PAD, rows * (CELL_H + PAD) + PAD), "white")
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("msyh.ttc", 15)
        except Exception:
            font = ImageFont.load_default()

        placed = []
        for idx, (el, crop, ck) in enumerate(prepared):
            w, h = crop.size
            s = min(CELL_W / w, CELL_H / h)
            crop = crop.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
            cx = PAD + (idx % COLS) * (CELL_W + PAD)
            cy = PAD + (idx // COLS) * (CELL_H + PAD)
            draw.rectangle([cx - 2, cy - 2, cx + CELL_W + 2, cy + CELL_H + 2], outline="#bbb")
            canvas.paste(crop, (cx + (CELL_W - crop.size[0]) // 2, cy + (CELL_H - crop.size[1]) // 2))
            draw.rectangle([cx - 1, cy - 1, cx + 19, cy + 16], fill="white")
            draw.text((cx + 2, cy), str(el["id"]), fill="#e00000", font=font)
            placed.append((el, ck))
        if not placed:
            return elements

        try:
            raw = self.model.chat([{"role": "user", "content": [
                {"type": "text", "text": self.BATCH_PROMPT},
                {"type": "image_url", "image_url": {"url": _img_data_url(canvas)}},
            ]}], temperature=0.0,
                # 每行 12 token 足够：编号(1~2) + 竖线 + 6 个汉字(6~8)。
                # 原来给 18 是照"随便写"的预算留的，现在名称限了长度，一起收紧 ——
                # max_tokens 是上限不是目标，但它决定了模型能啰嗦到哪一步。
                max_tokens=len(placed) * 12 + 48)
        except Exception as e:
            print(f"[grounding] 批量起名失败: {e}")
            return elements

        got = {}
        for line in (raw or "").splitlines():
            line = line.strip().strip("-·*#> ")
            if "|" not in line:
                continue
            k, v = line.split("|", 1)
            k = re.sub(r"\D", "", k)
            v = v.strip().strip("`'\"“” ")
            if k.isdigit() and v and v not in ("未知", "unknown", "Unknown"):
                got[int(k)] = v[:24]
        by_id = {el["id"]: ck for el, ck in placed}
        for el, ck in placed:
            name = got.get(el["id"])
            if name:
                el["label"] = name
                self.cache.put(ck, name)
            else:
                self.cache.put(ck, "-")   # 记下"问过了但认不出"，避免每步重问
        self.last_batch = (len(got), len(placed))
        self._batch_done = True      # 这一屏已经整体问过一轮，别再逐个补问
        return elements

    def _el_crop(self, img: Image.Image, bbox, ratio: float = 0.25):
        """裁元素小图（带外扩 —— 文件夹图标的文字标签在框下方）。"""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pw, ph = int((x2 - x1) * ratio) + 4, int((y2 - y1) * ratio) + 4
        return img.crop((max(0, x1 - pw), max(0, y1 - ph),
                         min(img.size[0], x2 + pw), min(img.size[1], y2 + ph)))

    @staticmethod
    def _el_key(crop, bbox=None) -> str:
        """起名缓存的键：只看「这个图标长什么样」，不看它在屏幕哪个位置。

        原来的键把 bbox 混进去了（`b<hash>|x,y`），元素挪 1 像素就 miss。
        可内容哈希已经能识别"同一位置换了内容"，bbox 是多余的。更糟的是窗口
        一移、窗口一缩，满屏的「×」「⋯」「齿轮」全成了新元素，每次都要重新
        问一遍模型 —— 一次 6~12 秒。

        改成固定尺寸缩略图的哈希：同一个关闭按钮，在微信里、在浏览器里、
        在屏幕哪个角落，都是同一个名字，直接命中。加 v2 前缀是因为旧缓存的
        键语义不同，别混用。
        """
        try:
            small = crop.convert("RGB").resize((64, 64), Image.BILINEAR)
            return "v2:" + hashlib.md5(small.tobytes()).hexdigest()[:12]
        except Exception:
            return "v2:" + hashlib.md5(crop.tobytes()).hexdigest()[:12]

    def annotate(self, img: Image.Image):
        dets = detect_elements(img, self.conf)[: self.max_elements]
        self._screen_size = img.size
        els = []
        for i, (x1, y1, x2, y2, s) in enumerate(dets, start=1):
            els.append({"id": i, "label": "", "bbox": [x1, y1, x2, y2]})
        self._last_elements = els
        self._captions = [""] * len(els)   # 起名惰性：resolve_desc 时按需进行
        self._batch_done = False           # 批量起名跑过没有（跑过就别再逐个问）
        return els, self._draw(img.copy(), els)

    def resolve(self, element_id):
        return self._resolve(element_id)

    def _cap(self, eid) -> str:
        """按编号取名字，越界返回空串。

        真机崩溃：OCR 补进来的文字元素编号会超过 YOLO 那一轮的长度，
        `_captions[id-1]` 直接 IndexError，整个任务以「❌ list index out of range」结束
        —— 用户看到的是程序自己崩了，而不是它没做对。
        """
        try:
            i = int(eid) - 1
        except Exception:
            return ""
        return self._captions[i] if 0 <= i < len(self._captions) else ""

    def _set_cap(self, eid, text: str):
        i = int(eid) - 1
        while len(self._captions) <= i:
            self._captions.append("")
        self._captions[i] = text

    def ensure_captions(self, img: Image.Image, only_ids=None):
        """给元素起名（惰性+缓存）。only_ids 限定范围可大幅省时。"""
        # img 可能没传（只拿当前这批元素算分）：别去 annotate，直接用已有元素
        if img is not None and (self._screen_size != img.size or not self._last_elements):
            self.annotate(img)
        if not self._last_elements:
            return self._captions
        if self.model is None:      # 没有模型就只认已填好的标签
            return self._captions
        todo = only_ids or [el["id"] for el in self._last_elements]
        by_id = {el["id"]: el for el in self._last_elements}
        # 批量起名（ensure_captions_batch）把名字写在 el["label"] 上，这里必须先搬进
        # _captions 再判断——否则每元素都要重新 describe 一次，是真的调模型：
        # 12 个元素 = 12 次调用 ≈ 40~90 秒。真机上就是「一步 66s / 124s」的来源。
        left = 0
        for i in todo:
            el = by_id.get(i)
            if el is None or self._cap(i):
                continue
            if el.get("label"):
                self._set_cap(i, el["label"])
                continue
            left += 1
        if getattr(self, "_batch_done", False):
            # 批量起名已经问过一轮了（每步都跑），没名字的元素是"批量也认不出"的。
            # 再逐个 describe 就是每步白烧 4 秒 × N：真机一步 66~124 秒的主因。
            return self._captions
        if left > 8:
            # 没名字的太多就别逐个问了：位置分 + 已有的名字足够定位，
            # 为了 1 个模糊描述烧 8 次模型调用（30 秒）不划算
            return self._captions
        for i in todo:
            el = by_id.get(i)
            if el is None or self._cap(i):
                continue
            self._set_cap(i, self.describe(img, el["bbox"]))
        return self._captions

    def resolve_desc(self, target: str, img: Image.Image = None):
        """模型给的自然语言目标 -> 匹配元素 -> 中心坐标。
        提速策略：先按位置启发给所有元素打分（零成本），只对 top-K 起名；
        最高分不够好再扩大候选。"""
        if img is not None and self._screen_size != img.size:
            self.annotate(img)
        els = self._last_elements
        if not els:
            return None
        sw, sh = self._screen_size

        # 程序先算位置分（不起名，零成本）
        pre = []
        for el in els:
            cx = (el["bbox"][0] + el["bbox"][2]) / 2
            cy = (el["bbox"][1] + el["bbox"][3]) / 2
            pre.append((position_bonus(target, cx, cy, sw, sh), el))
        # 区域词强约束时只保留该区域的元素
        strong = any(kw in target for kw in ("任务栏", "右上角", "左上角", "底部", "顶部"))
        if strong:
            kept = [(b, el) for b, el in pre if b > 0]
            pre = kept or pre
        pre.sort(key=lambda t: -t[0])

        def _try(k):
            caps = self.ensure_captions(img, only_ids=[el["id"] for _, el in pre[:k]])
            best, best_score = None, -1.0
            for b, el in pre[:k]:
                cx = (el["bbox"][0] + el["bbox"][2]) / 2
                cy = (el["bbox"][1] + el["bbox"][3]) / 2
                # 名字有两套：批量起名写 el["label"]，逐元素起名写 _captions。
                # 只看 _captions 会让整屏已认名的元素变成"没名字"，定位退化成按位置瞎猜
                # （真机：'任务栏最右侧的微信图标' 被匹配到 #59 电池图标）。
                _name = el.get("label") or self._cap(el["id"])
                sc = b + self._text_part(target, _name, cx, cy, sw, sh)
                if sc > best_score:
                    best, best_score = el, sc
            return best, best_score

        best, score = _try(12)
        if best is None or score < 0.6:
            b2, s2 = _try(28)
            if s2 > score:
                best, score = b2, s2
        if best is None or score < 0.9:
            # 低分匹配宁可不点：瞎点放大镜/iCloud 之类不相关元素比点不到更糟
            # （实测标定：精准位置匹配 ≈1.15，瞎匹配 ≈0.25，阈值取中间）
            return None
        # 硬校验：目标里的「实词」必须在元素名字里出现过。
        # 真机翻车：「任务栏最右侧的微信图标」被位置分抬到 1.29，匹配到 #59「电池图标」
        # —— 位置对、东西错，点下去就是乱点，而且日志还写着"定位成功"。
        name = (best.get("label") or self._cap(best["id"]) or "").lower()
        terms = self._content_terms(target)
        if terms and not name:
            # 位置对、但这个东西没名字 = 无法确认是不是你要的。
            # 真机：'任务栏最右侧的微信图标' 靠位置分 1.15 命中一个无名元素。
            # 宁可不点（调用方会退回模型自己给的坐标），也不瞎点。
            return None
        if terms and name:
            allow = list(terms)
            for k, syns in self.SYNONYMS.items():      # 「开始按钮」≈「Windows 徽标」
                if k in target:
                    allow += [s.lower() for s in syns]
            hit = any(t in name or (len(name) >= 2 and name in t) for t in allow)
            if not hit:
                return None
        x1, y1, x2, y2 = best["bbox"]
        return ((x1 + x2) / 2, (y1 + y2) / 2, best["id"],
                best.get("label") or self._cap(best["id"]), round(score, 2))

    # 目标描述里的结构词：去掉它们剩下的才是"要什么"（微信/回收站/保存）。
    _STOP_TERMS = ("任务栏", "左上角", "右上角", "右下角", "左下角", "屏幕", "桌面",
                   "图标", "按钮", "窗口", "区域", "位置", "旁边", "最右侧", "最左侧",
                   "最右", "最左", "左侧", "右侧", "顶部", "底部", "中间", "上面",
                   "下面", "这个", "那个", "一个", "里面", "输入框", "编辑区",
                   "的", "上", "下", "里")

    @classmethod
    def _content_terms(cls, target: str):
        """目标描述里的实词（去掉位置/结构词后的关键词）。

        用来做最后一道核对：「任务栏最右侧的微信图标」→ ['微信']，
        元素名字里必须出现「微信」，否则位置分再高也不认（真机把电池图标当成微信）。
        """
        t = str(target or "").lower()
        for s in cls._STOP_TERMS:
            t = t.replace(s, " ")
        terms = [w for w in re.split(r"[\s,，。、/|()（）\[\]【】:：;；!！?？\"'“”]+", t)
                 if len(w) >= 2]
        return terms[:4]

    # 高频控件同义词：模型起名不会说"开始按钮"，会说"Windows 徽标/四色方块"。    # 目标描述里出现 key 时，把 syns 也纳入文本匹配，任一命中即算强匹配。
    SYNONYMS = {
        "开始按钮": ["windows 徽标", "四色方块", "彩色方块", "蓝色方块", "win 徽标", "win图标"],
        "搜索框": ["搜索", "放大镜", "在这里输入你要搜索的内容"],
        "搜索按钮": ["搜索", "放大镜"],
        "最小化": ["最小化", "减号", "横线", "下划线按钮"],
        "最大化": ["最大化", "方形", "方框"],
        "关闭按钮": ["关闭", "叉", "叉号", "×"],
        "记事本": ["notepad", "记事本", "文本编辑"],
        "计算器": ["calc", "计算器"],
        "浏览器": ["edge", "浏览器", "chrome"],
    }

    @classmethod
    def _target_terms(cls, target: str):
        terms = [target]
        t = target or ""
        for k, syns in cls.SYNONYMS.items():
            if k in t:
                terms.extend([t2 for t2 in syns if t2 not in terms])
        return terms

    def _text_part(self, target, caption, cx, cy, sw, sh):
        cg = _bigrams(caption or "")
        if not cg:
            return 0.0
        best = 0.0
        for term in self._target_terms(target):
            tg = _bigrams(term)
            if tg:
                best = max(best, len(tg & cg) / min(len(tg), len(cg)))
        return best

    # ---- 调试图 ----
    def annotated_with_labels(self, img: Image.Image):
        els, vis = self.annotate(img)
        draw = ImageDraw.Draw(vis)
        try:
            font = ImageFont.truetype("msyh.ttc", 14)
        except Exception:
            font = ImageFont.load_default()
        caps = self.ensure_captions(img)
        for el, cap in zip(els, caps):
            x1, y1, x2, y2 = el["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=(230, 57, 70), width=2)
            draw.text((x1 + 2, y1 + 2), f"{el['id']}:{cap[:12]}", fill=(230, 57, 70), font=font)
        return vis


def _img_data_url(img: Image.Image):
    import base64
    import io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
