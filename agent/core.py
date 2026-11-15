#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心编排：把 模型 / grounding / 规划 / 校验 / 安全 / 记忆 / 动作 串成主循环。

流程（每步）：
  1. 截屏
  2. grounding.annotate -> 元素清单（可无）
  3. 组装 prompt（任务 + 当前子任务 + 历史 + 截图 + 元素清单）发给模型
  4. 解析动作 JSON
  5. 安全闸门：高风险操作等人工确认
  6. 停顿（可干预）-> 执行动作
  7. 校验器（可选）：回截屏确认成功否，失败触发重规划
  8. 记录轨迹
  9. 模型 done / 达到最大步数 -> 结束

支持：随时 stop（stop_event）；模型 ask 时返回 NEED_REPLY，由调用方喂回复后 resume。

为便于测试，pyautogui 懒导入；screenshot / model / executor 均可注入覆盖。
"""

import ast
import json
import os
import re
import time
import threading

from pathlib import Path

from PIL import Image, ImageDraw

try:
    import pyautogui
except Exception:  # pragma: no cover - 仅在没有 GUI / 显示器时触发
    pyautogui = None

from agent.models import ModelClient, image_to_data_url
from agent.grounding import build_grounding
from agent.safety import Safety
from agent.planner import Planner
from agent.validator import Validator
from agent.actions import ActionExecutor
from agent.memory import TrajectoryMemory
from agent.scratchpad import Scratchpad
from agent.long_memory import LongMemory

# 引号类字符：任务是「对X说：<内容>」时，这对引号只是语法标记，不该进正文。
# 真机踩坑：任务写 对示例联系人A说：”你好。“ 时，弯引号被原样逐字敲进了微信，
# 且中文输入法会把注入的引号/标点二次处理，出现「，，」「。。」这种双份。
_QUOTE_CHARS = "“”‘’「」『』\"'＂＇`"
_PUNCT_RE = re.compile(r"([，。！？；：、,.!?;:])\1+")


def clean_message_text(s: str) -> str:
    """清洗要发出去的正文：去掉包裹/游离的引号，折叠重复标点，去多余空白。
    只用于「发消息」正文，不用于通用打字（记事本里写引号是合法的）。"""
    s = (s or "").strip()
    # 一层层剥掉首尾成对引号（任务里常见 ：“内容。” 这种写法）
    while len(s) >= 2 and s[0] in _QUOTE_CHARS and s[-1] in _QUOTE_CHARS:
        s = s[1:-1].strip()
    # 正文里残留的引号大多是标记，不是内容
    for q in _QUOTE_CHARS:
        s = s.replace(q, "")
    # 中文输入法会把标点敲成双份（真机：，→，， 。→。。）
    s = _PUNCT_RE.sub(r"\1", s)
    return re.sub(r"[ \t]{2,}", " ", s).strip()


# ------------------------- 算术：绝不该让 7B 去"心算" -------------------------
# 真机教训：任务「算出 123*123 再发给示例联系人A」被整句丢进通用循环后，
# 7B 先在微信里反复敲 "123*123"（把聊天输入框当计算器），21 步烧掉 303 秒，
# 既没打开计算器、也没算出结果、更没发出去。
# 算术是确定性运算，交给 AST 白名单求值；模型只负责"谁是收信人"。
_CALC_WORDS = ("算", "计算", "求值", "得出", "结果", "等于", "多少")
_EXPR_CHARS = "0123456789 \t.+-*/×÷()^%"
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                   ast.Mod, ast.Pow)


def _normalize_expr(expr: str) -> str:
    """把中文/键盘习惯的运算符统一成 Python 能算的形式。"""
    s = (expr or "").strip()
    s = s.replace("×", "*").replace("✕", "*").replace("x", "*").replace("X", "*")
    s = s.replace("÷", "/").replace("（", "(").replace("）", ")").replace("^", "**")
    s = re.sub(r"\s+", "", s)
    return s.rstrip("+-*/%.=")


def _safe_eval(expr: str):
    """只允许四则运算的表达式求值（AST 白名单，禁属性/名称/调用）。

    返回 int 或 float。表达式不合法或超范围时抛 ValueError。
    安全要点：白名单校验每一个节点，最后才 compile+eval，且 globals 里没有
    __builtins__；同时限制幂指数与结果量级，避免 9**9**9 把进程算死。
    """
    s = _normalize_expr(expr)
    if not s:
        raise ValueError("空表达式")
    if len(s) > 60:
        raise ValueError("表达式过长")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"不是合法算式：{s}") from exc
    for node in ast.walk(tree):
        # ast.walk 会把 op 字段（Add/Mult 这些运算符实例）也走进来，它们不是"写法"
        if isinstance(node, (ast.Expression, ast.operator, ast.unaryop)):
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise ValueError("只支持 + - * / % ^")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise ValueError("只支持正负号")
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("只支持数字")
            if abs(node.value) > 1e12:
                raise ValueError("数字太大")
        else:
            raise ValueError(f"表达式含不支持的写法：{type(node).__name__}")
        # 幂运算单独收紧：右操作数必须是合理的小常量
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            r = node.right
            if not isinstance(r, ast.Constant) or not isinstance(r.value, (int, float)) \
                    or abs(r.value) > 64:
                raise ValueError("指数过大")
    value = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
    if isinstance(value, complex) or not isinstance(value, (int, float)):
        raise ValueError("结果不是实数")
    if abs(value) > 1e15:
        raise ValueError("结果太大")
    return value


def _fmt_number(v) -> str:
    """120.0 -> '120'；0.30000000000000004 -> '0.3'。"""
    if isinstance(v, int):
        return str(v)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.10g}"


def _parse_expr(text: str):
    """从任务句子里抠出一个算式。必须同时满足：含运算符、以数字或 ) 结尾。
    「123*123的值」-> 123*123；「发给示例联系人A」这种没有运算符的不会被误抓。"""
    t = (text or "").replace("（", "(").replace("）", ")")
    # 起始字符允许左括号，否则「(1+2)*3」会从中间的数字开始、括号不平衡而漏掉
    for cand in re.findall(r"[0-9(][0-9 \t.+\-*/×÷()^]*", t):
        c = cand.strip()
        if len(c) < 3 or not re.search(r"[+\-*/×÷^]", c):
            continue
        if not (c[-1].isdigit() or c[-1] == ")"):
            continue
        try:
            _safe_eval(c)
        except Exception:
            continue
        return c
    return None


def _clean_desc(desc: str) -> str:
    """把「名字叫测试的文件夹」这类自然语言描述压成关键词「测试」。

    模型的 element_desc 从来不是干净的名字：它会写成"那个名叫测试的文件夹"、
    "帮我点开桌面上的 D 盘"。拿整句去比元素标签是比不中的，先削掉这些虚词。
    """
    t = str(desc or "").strip().strip("「」\"'“”《》 ")
    for w in ("我想要", "帮我", "麻烦", "点击", "双击", "打开", "进入", "找到",
              "名字叫", "名字是", "名为", "名叫", "叫做", "叫"):
        t = t.replace(w, "")
    for w in ("文件夹", "文件", "图标", "按钮", "那个", "这个"):
        t = t.replace(w, "")
    return t.strip(" 的")


def _match_label(elements, key: str):
    """按名字找元素：先精确、再互相包含。找不到返回 None。"""
    key = (key or "").strip()
    if not key:
        return None
    for el in elements:
        if (el.get("label") or "").strip() == key:
            return el
    if len(key) >= 2:
        for el in elements:
            lb = (el.get("label") or "").strip()
            if len(lb) >= 2 and (key in lb or lb in key):
                return el
    return None


def _default_screenshot():
    if pyautogui is None:  # pragma: no cover
        raise RuntimeError("pyautogui 不可用（非 Windows / 无 GUI 环境）")
    return pyautogui.screenshot()


class AgentSession:
    def __init__(self, cfg: dict, log=print, confirm=None, stop_event: threading.Event = None,
                 model=None, executor=None, screenshot_fn=None):
        self.cfg = cfg
        self.log = log
        self.confirm = confirm or (lambda action: True)  # 阻塞式确认回调
        self.stop_event = stop_event or threading.Event()

        self.model = model or ModelClient(cfg)
        self.grounding = build_grounding(cfg, self.model)
        self.safety = Safety(cfg)
        self.planner = Planner(cfg, self.model)
        self.validator = Validator(cfg, self.model)
        self.executor = executor or ActionExecutor(self.safety, self.grounding, log)
        self.screenshot_fn = screenshot_fn or _default_screenshot

        a = (cfg.get("agent", {}) or {})
        self.memory = TrajectoryMemory(max_images=int(a.get("max_screenshots", 50)))
        self.scratch = Scratchpad()   # 外部草稿纸：解决 7B 上下文装不下全程历史的问题
        self.max_steps = int(a.get("max_steps", 40))
        # 每步执行前的停顿。原意是「给人时间看到它要干什么、好在扣动扳机前喊停」，
        # 但真机上这是纯损耗：截图上一步就已经画好了，人要看的是"下一步做什么"，
        # 而这 1.5 秒 × 10 步 = 15 秒纯等待，比任何优化省下来的都多。
        # 干预手段还有三个（停止按钮 / 鼠标甩到屏幕角落急停 / 用户在用电脑就不抢），
        # 所以只留 0.5 秒给 UI 刷新和手快点的人。
        self.step_pause = float(a.get("step_pause", 0.5))
        self.history_window = int(a.get("history_window", 6))
        # 动作后是否重截：由模型在动作里自报 expect_change，这里只放兜底开关
        self.verify_screen_change = bool(a.get("verify_screen_change", True))
        # 失败后是否再单独叫一次模型做"复盘"。代价是多一次调用（3~5 秒），
        # 换来的是思考栏里出现真正针对这次失败的分析，而不是复述任务名。
        self.reflect_on_fail = bool(a.get("reflect_on_fail", True))
        # 打字闸：不许把字打进用户已打开的文件（真机改坏过 models_fetch.log）
        self.type_guard_enabled = bool(a.get("type_guard", True))
        # 对话框清障：残留的「另存为」挡在屏幕上时先按 Esc。测试里关掉（无真窗口）。
        self.dialog_guard = bool(a.get("dialog_guard", True))
        # 通用循环开始前 Win+D 回桌面，给模型一个干净起点。
        # 真机动机：8B 的注意力被眼前的东西牵着走 —— 它把 WorkBuddy 底部的输入框
        # 当成开始菜单搜索框，把「今天天气」打进对话框，连试三次白烧两分钟。
        self.home_desktop = bool(a.get("home_desktop", True))
        # 任务点名了「在浏览器里 / 用记事本」→ 先把那个程序打开。
        # 省掉通用循环第一步「我要先打开 XX」（一次完整截图 + 推理，20 秒）。
        self.preopen_target = bool(a.get("preopen_target", True))
        # 「青蛙眼」：只处理画面里变化的部分。默认开 —— 它省掉的是重复劳动
        # （同一片没动的内容反复 OCR、反复认名），不改变任何一次决策的输入。
        self.frog_eye = bool(a.get("frog_eye", True))
        # 进场确认：双击／open_app 后用帧差分 + 前台 exe 判定「你已经在应用里了」，
        # 并作为系统事实强制回灌给模型。与应用无关，不写死任何界面特征。
        self.entry_confirm = bool(a.get("entry_confirm", True))
        # 动作后期待有新画面时，不立刻截屏：给 UI 渲染一个观察窗，期间青蛙眼采样。
        # 0 = 关掉（回到「动作完立刻截」的旧行为）。真机：秒截会截到半渲染的窗口。
        # stable（默认）：连续两帧变化量相同就提前走，实测 1~1.5 秒就够；
        # fixed：旧行为，固定等满；off：完全关掉。
        self.change_settle = float(a.get("change_settle", 3.0))
        self.change_settle_mode = str(a.get("change_settle_mode", "stable") or "stable").lower()
        # 演示硬保险：只有步数上限时，「步步不同却没进展」能跑满 40 步 × 十几秒，
        # 评委只能干等。0 = 不限。
        self.max_seconds = float(os.environ.get("CU_MAX_SECONDS")
                                 or a.get("max_seconds", 180.0))
        # 测试护栏：真机验证时绝不对外发消息。上一轮跑 2 步时 agent 把「现在几点」
        # 打进 WorkBuddy 输入框并回车，就是缺这条边界。
        self.no_send = bool(a.get("no_send", False)) or os.environ.get("CU_NO_SEND") == "1"
        if self.no_send:
            self.executor.send_guard = self._no_send_guard
            self.log("[NO-SEND] 测试护栏已开启：send_message 与对话窗口里的回车都不会真正发出")
        # 变化窗口（秒）：上一帧比这还老就作废，按「整屏都变」处理。
        # 注意不能设成 1 秒 —— 一步的周期是「截图→推理 13s→执行」，两帧自然间隔
        # 25 秒上下；窗口小于它，差分永远不触发，整个功能等于没写。这里的窗口
        # 只防一种情况：任务中途被暂停很久，恢复后基准帧早就作废了。
        self.frog_eye_window = float(a.get("frog_eye_window", 120.0))
        # 给模型看时的呈现方式：
        #   popup（默认）—— 只在「变化区长得像个弹窗」时启用裁剪。用户的原意就是这样：
        #                   点完蹦出对话框那一刻最有价值（弹窗以外的网页一个字都不用重读），
        #                   平时翻页/滚动就让位给全屏，不硬凹。
        #   outline —— 每一步都裁（实测能省，但收益取决于页面变化分布）。
        #   dim —— 红框 + 压暗不变区。
        #   off —— 全关。
        self.frog_eye_model = str(a.get("frog_eye_model", "popup") or "popup").lower()
        # 微信发送白名单（config 可覆盖默认的两个会话）。放类级是因为
        # _send_target_allowed 是 classmethod：通用循环和模板层都要用同一份。
        try:
            type(self)._cfg_snapshot = cfg
        except Exception:
            pass

        self.step = 0
        self.history = []
        self.task = ""
        self.subtasks = []
        # 跨任务长期记忆：只存「程序怎么开起来的」「哪些做法试过没用」两类事实。
        # history / 草稿纸 / _fail_log 仍是任务内记忆，两者不互相替代。
        _root = Path(__file__).resolve().parent.parent
        self.long_memory = LongMemory(
            str(a.get("long_memory_path") or (_root / "data" / "long_memory.json")),
            enabled=bool(a.get("long_memory", True)))
        self._pending = None       # (截图, 是否与上一张同一画面) —— 下一轮直接拿来用
        self._reused_screen = False
        self._last_elements = None
        self._last_annotated = None
        # 「青蛙眼」用的上一帧基准：灰度半图 + 抓它的时间戳（决定还比不比得动）
        self._last_gray = None
        self._last_frame_ts = 0.0
        self._frog_els = None        # 上一帧的最终元素清单（含认名），不变区域直接复用
        self._chg_regions = []       # 本步的变化区域（物理像素），画框和写提示词都要用
        # 最近成功输入过的 [(窗口标题, 文字)]：用来认出「这是我们自己敲出来的内容」，
        # 否则打完字标题变成 `*第一行 - 记事本`，下一步会被自己的文件闸误判成别人的文件
        self._typed_recent = []
        self._type_block_reason = ""

    # ------------------------- 图像预处理 -------------------------
    MAX_IMG_W = int(__import__("os").environ.get("CU_MAX_IMG_W", "1536"))

    def _fit_image(self, img):
        """把截图缩到模型能吃的分辨率。
        本地 7B 视觉模型上下文只有 4096 token，2.5K 分辨率的截图会直接 400。
        返回 (缩放后图像, scale)；scale = 缩放图 / 原图，之后所有坐标要除以它换回屏幕像素。"""
        if img.width <= self.MAX_IMG_W:
            return img, 1.0
        s = self.MAX_IMG_W / float(img.width)
        return img.resize((self.MAX_IMG_W, max(1, int(img.height * s))), Image.LANCZOS), s

    def _fg_exe_matches(self, app: str) -> bool:
        """前台窗口的 exe 是否就是 app。中文程序名的窗口标题常常是英文（记事本→Notepad），
        光比标题永远对不上；exe 才是不撒谎的那个。"""
        exe = self._foreground_exe()
        if not exe:
            return False
        keys = self._PROC_MAP.get(self._norm_app_name(app)) or [app.lower()]
        return any(AgentSession._exe_matches(exe, k) for k in keys)

    def _dim_background(self, img):
        """把前台窗口以外的区域调暗，逼模型的注意力落在"正在操作的那个窗口"上。

        为什么必须有：真机实测（CU_KEEP_SHOTS 留图验证），记事本明明在前台、
        里面已经有内容，模型却一口咬定"屏幕上是文件资源管理器，记事本未打开"——
        因为背景里资源管理器占了 60% 面积，8B 模型的注意力被它带走了。
        调暗背景后画面上亮着的只有一个窗口，这种张冠李戴会大幅减少。

        坐标一致性：pyautogui 截图和 GetWindowRect 都是物理像素
        （run_task / ui 都先调了 ensure_dpi_aware），直接用即可。
        拿不到前台窗口 / 坐标越界时返回原图，绝不因为这段代码让任务挂掉。
        """
        try:
            import ctypes
            import ctypes.wintypes as wt

            from PIL import Image, ImageEnhance

            u = ctypes.windll.user32
            hwnd = u.GetForegroundWindow()
            if not hwnd:
                return img
            rect = wt.RECT()
            if not u.GetWindowRect(hwnd, ctypes.byref(rect)):
                return img
            l, t, rr, b = rect.left, rect.top, rect.right, rect.bottom
            W, H = img.size
            # 坐标必须大体落在截图里，否则是 DPI 或多屏错位，宁可不动
            if rr - l < 60 or b - t < 60 or rr < 0 or b < 0 or l >= W or t >= H:
                return img
            if rr > W * 1.05 or b > H * 1.05:
                # rect 明显大于截图：说明 img 被外部缩放过（探针/UI 缩略图），
                # 拿物理坐标去贴缩放图必然错位 —— 返回原图，亮区贴错比不贴更糟。
                return img
            l, t = max(0, l), max(0, t)
            rr, b = min(W, rr), min(H, b)
            mask = Image.new("L", (W, H), 70)          # 背景：亮度压到 27%
            mask.paste(255, (l, t, rr, b))             # 前台窗口：保持原亮度
            dark = ImageEnhance.Brightness(img).enhance(0.27)
            return Image.composite(img, dark, mask)
        except Exception:
            return img

    @staticmethod
    def _changed(a, b, thr: float = 0.003, max_w: int = 320) -> bool:
        """轻量画面差异检测：缩到 max_w 宽灰度，比较「明显不同」的像素占比。
        用于兜底校验模型自报的 expect_change（模型说没变化，但实际变了就纠正它）。
        耗时约 20ms，远低于一次模型推理（10s+）。
        max_w=None 用原分辨率：检测一行打字这种小变化必须用它——
        320 宽会把一行文字缩小 8 倍直接抹没（真机踩坑：打了字被判没变化）。"""
        try:
            from PIL import ImageChops
            sa, sb = a.convert("L"), b.convert("L")
            if max_w and sa.width > max_w:
                sa = sa.resize((max_w, max(1, int(sa.height * max_w / sa.width))))
            if max_w and sb.width > max_w:
                sb = sb.resize((max_w, max(1, int(sb.height * max_w / sb.width))))
            if sa.size != sb.size:
                return True
            hist = ImageChops.difference(sa, sb).histogram()
            total = sa.width * sa.height
            changed = sum(hist[12:])  # 灰度差 > 12 视为变了
            return changed / float(total) > thr
        except Exception:
            return True  # 判断不了就当作变了，宁可多截一张

    # ---------- 「青蛙眼」：只看变化的那部分 ----------
    # 灵感是青蛙的视网膜：不动的东西不产生信号。落地方式要小心 —— 直接把画面裁成
    # 一小块给模型是行不通的，它连「我在哪个窗口、这个按钮在界面哪儿」都要靠整屏
    # 判断（真机踩过：背景里一个大窗口就能把它带偏）。所以「只看变化」拆成三件事：
    #   ① 判定：这一帧和上一帧几乎一样 → 整个视觉流水线都可以跳过
    #   ② 裁剪：只有变化区域重跑 OCR（不变区域的元素连名字带坐标直接复用上一帧）
    #   ③ 标注：给模型的图仍是全屏，但把变化区域框出来，用红框引导注意力

    @staticmethod
    def _diff_regions(a, b, thr: int = 28, cell: int = 64, min_px: int = 200,
                      max_regions: int = 6):
        """两帧差分的「变化区域」。返回 (物理像素 bbox 列表, 变化像素占比)。

        尺寸不一致 / 整屏都在变 时返回 ([], ratio>0.35) —— 调用方据此走全屏流程。

        为什么要够快：这一步以前不存在，现在每步都跑，所以只能半分辨率 + 格子聚类。
        全分辨率 numpy 差分约 30ms，半分辨率约 8ms，而 16px 的字缩到 8px 仍能查出差异。
        """
        try:
            if a is None or b is None or a.size != b.size:
                return [], 1.0
            W, H = a.size
            hw, hh = max(1, W // 2), max(1, H // 2)
            ga = a.convert("L").resize((hw, hh), Image.BILINEAR)
            gb = b.convert("L").resize((hw, hh), Image.BILINEAR)
            return AgentSession._diff_half(ga, gb, a.size, thr, cell, min_px, max_regions)
        except Exception:
            return [], 1.0        # 差分本身出错就当全屏变了，宁可多算不可漏看

    @staticmethod
    def _diff_half(ga, gb, size, thr: int = 28, cell: int = 64, min_px: int = 200,
                   max_regions: int = 6):
        """在两张「半分辨率灰度图」上做差分。

        单独抽出来是为了缓存：上一帧的灰度半图算一次就存着，下一步直接拿来比 ——
        否则每步都要把两张 2.5K 图各缩一遍（实测 29ms，其中八成花在缩放上）。
        """
        try:
            import numpy as np
            if ga is None or gb is None or ga.size != gb.size:
                return [], 1.0
            W, H = size
            hw, hh = ga.size
            sa = np.asarray(ga, dtype=np.int16)
            sb = np.asarray(gb, dtype=np.int16)
            mask = np.abs(sa - sb) > thr
            ratio = float(mask.sum()) / float(mask.size)
            if ratio <= 0.0:
                return [], 0.0
            if ratio > 0.35:
                return [], ratio

            gh, gw = max(1, hh // cell), max(1, hw // cell)
            m = mask[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell)
            # 一个格子里超过 0.4% 的像素变了才算「热格」：闪烁的光标只有几个像素，
            # 落在格子里不到这个比例 —— 这正是「看不见不动的东西」要滤掉的噪声
            hot = m.sum(axis=(1, 3)) > (cell * cell * 0.004)

            seen = np.zeros_like(hot)
            boxes = []
            for y in range(gh):
                for x in range(gw):
                    if not hot[y, x] or seen[y, x]:
                        continue
                    stack, seen[y, x] = [(y, x)], True
                    y0 = y1 = y
                    x0 = x1 = x
                    while stack:                       # 8 邻域连通，把相邻热格并成一块
                        cy, cx = stack.pop()
                        y0, y1 = min(y0, cy), max(y1, cy)
                        x0, x1 = min(x0, cx), max(x1, cx)
                        for ny in (cy - 1, cy, cy + 1):
                            for nx in (cx - 1, cx, cx + 1):
                                if 0 <= ny < gh and 0 <= nx < gw \
                                        and hot[ny, nx] and not seen[ny, nx]:
                                    seen[ny, nx] = True
                                    stack.append((ny, nx))
                    # 格子坐标 → 物理像素（×cell 回到半分辨率，再 ×2 回到原图）
                    boxes.append((x0 * cell * 2, y0 * cell * 2,
                                  min(W, (x1 + 1) * cell * 2),
                                  min(H, (y1 + 1) * cell * 2)))
            out = []
            for (cx0, cy0, cx1, cy1) in boxes:
                # 格子只是「粗定位」，真正的框从变化像素本身取 —— 128px 的格子框
                # 拿去裁剪，OCR 要多算一大片空白，而且容易把邻座元素卷进来
                hx0, hy0, hx1, hy1 = cx0 // 2, cy0 // 2, cx1 // 2, cy1 // 2
                sub = mask[hy0:hy1, hx0:hx1]
                if not sub.any():
                    continue
                ys, xs = np.nonzero(sub)
                px = int(len(xs)) * 4          # 半分辨率像素数 → 物理像素数
                if px < min_px:
                    continue
                px0, px1 = hx0 + int(xs.min()), hx0 + int(xs.max()) + 1
                py0, py1 = hy0 + int(ys.min()), hy0 + int(ys.max()) + 1
                out.append((max(0, px0 * 2 - 8), max(0, py0 * 2 - 8),
                            min(W, px1 * 2 + 8), min(H, py1 * 2 + 8)))
            if not out:
                # 全屏的变化加起来都不够一块 —— 光标闪烁级的噪声。按「没变」处理：
                # 整条视觉流水线（YOLO / OCR / 认名）都可以直接复用上一步的结果。
                if ratio * W * H < min_px:
                    return [], 0.0
                return [], ratio
            out.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
            return out[:max_regions], ratio
        except Exception:
            return [], 1.0        # 差分本身出错就当全屏变了，宁可多算不可漏看

    def _frog_eye_diff(self, img):
        """拿「窗口期以内」的上一帧跟当前帧比。

        为什么必须有时间窗口（用户提的 1 秒）：更早的帧不可比。中途可能有弹窗
        出现又消失、页面滚出去又滚回来 —— 那类变化在两帧之间表现为「没变」，
        拿十秒前的帧比却会得出满屏都在变，差分就失去意义了。

        只缓存灰度半图（约 1MB）而不是整帧（2.5K 彩色约 12MB）：省内存，也省掉
        每步重缩上一帧的时间。
        """
        now = time.time()
        prev = getattr(self, "_last_gray", None)
        prev_ts = getattr(self, "_last_frame_ts", 0.0)
        self._last_frame_ts = now
        half = None
        try:
            half = img.convert("L").resize(
                (max(1, img.width // 2), max(1, img.height // 2)), Image.BILINEAR)
        except Exception:
            half = None
        self._last_gray = half
        if prev is None or half is None or (now - prev_ts) > self.frog_eye_window:
            return [], 1.0
        return self._diff_half(prev, half, img.size)

    @staticmethod
    def _rects_overlap(b, r, pad: int = 0) -> bool:
        """元素框（缩放坐标）和变化区域（物理坐标）是否相交。调用方负责换算。"""
        return not (b[2] + pad < r[0] or b[0] - pad > r[2]
                    or b[3] + pad < r[1] or b[1] - pad > r[3])

    @staticmethod
    def _looks_like_popup(regions, W, H) -> bool:
        """变化区像不像「点完冒出来的一块对话框」。

        真机最典型的场景：点「另存为」/「删除」/「设置」→ 屏幕中央弹出一块框，
        背后的网页一动不动。这时变化区就是 1~3 块边界清楚的矩形，
        面积 1%~40% 屏 —— 只 OCR 它、只把它交给模型，又快又不被背后的网页带偏。

        反过来，翻页/滚动/整页跳转产生的是大片变化或贴边的横条，
        那不是弹窗，老老实实跑全屏（硬裁只会把该看的东西裁掉）。
        """
        if not (1 <= len(regions) <= 3):
            return False
        x0 = min(r[0] for r in regions)
        y0 = min(r[1] for r in regions)
        x1 = max(r[2] for r in regions)
        y1 = max(r[3] for r in regions)
        w, h = x1 - x0, y1 - y0
        if w < 150 or h < 60:
            return False                      # 一行字、一个按钮、光标闪一下
        if not (0.008 <= (w * h) / float(W * H) <= 0.42):
            return False
        if w > W * 0.92 and h < H * 0.3:
            return False                      # 整条横带：地址栏/标签栏/工具栏在变
        if h > H * 0.92 and w < W * 0.3:
            return False                      # 整条竖带：侧边栏在变
        return True

    def _frog_eye_plan(self, img, regions, scale, popup_only: bool = False):
        """决定这次 OCR 要不要裁剪。返回 (crop_box 或 None, 可复用的旧元素, 变化区缩放坐标)。

        crop_box 为 None 表示「老老实实跑全屏」—— 三种情况会走到这里：
        变化区域太大（裁了也省不下）、没有上一帧的元素清单（不敢冒险）、没开这个开关。
        popup_only=True 时再加一条：变化区不像弹窗就不裁。
        """
        if not regions or not getattr(self, "_frog_els", None):
            return None, [], []
        W, H = img.size
        if popup_only and not self._looks_like_popup(regions, W, H):
            return None, [], []
        pad = 28
        x0 = max(0, min(r[0] for r in regions) - pad)
        y0 = max(0, min(r[1] for r in regions) - pad)
        x1 = min(W, max(r[2] for r in regions) + pad)
        y1 = min(H, max(r[3] for r in regions) + pad)
        if (x1 - x0) * (y1 - y0) > W * H * 0.45:
            return None, [], []
        # 上一帧里完全落在变化区之外的元素：名字、坐标都是现成的，不用重新认
        keep = []
        for el in self._frog_els:
            try:
                b = [float(v) / scale for v in el["bbox"]]      # 缩放坐标 → 物理像素
            except Exception:
                continue
            if not any(self._rects_overlap(b, r, pad=6) for r in regions):
                keep.append(dict(el, bbox=list(el["bbox"])))
        return (x0, y0, x1, y1), keep, [
            (int(r[0] * scale), int(r[1] * scale), int(r[2] * scale), int(r[3] * scale))
            for r in regions]

    @staticmethod
    def _mark_changes(img, regions, scale, dim: float = 0.0):
        """在给模型看的那张图上，把「刚发生变化」的区域框出来。

        为什么不干脆把不变的部分压暗（`_dim_background` 那套）：压暗是为了压住
        背景里的大窗口，压到 0.27 会让它看不见，但它本来就不该看；这里的「不变
        区域」恰恰是它下一步要点的目标，压暗等于把路标涂掉。所以默认只画框。
        dim>0 时才额外压暗，留给真机对比实验。
        """
        try:
            if not regions:
                return img
            if dim > 0:
                from PIL import ImageEnhance
                mask = Image.new("L", img.size, int(255 * (1 - dim)))
                md = ImageDraw.Draw(mask)
                for (x0, y0, x1, y1) in regions:
                    md.rectangle([int(x0 * scale), int(y0 * scale),
                                  int(x1 * scale), int(y1 * scale)], fill=255)
                img = Image.composite(img, ImageEnhance.Brightness(img).enhance(1 - dim), mask)
            d = ImageDraw.Draw(img)
            for (x0, y0, x1, y1) in regions:
                d.rectangle([int(x0 * scale), int(y0 * scale),
                             int(x1 * scale), int(y1 * scale)], outline=(255, 70, 70), width=3)
            return img
        except Exception:
            return img

    # 窗口标题里夹的不可见字符。真机铁证：Edge 的标题是
    # "… - Microsoft\u200b Edge"（U+200B 零宽空格），于是
    # "microsoft edge" in 标题 永远 False —— 每步都误判"前台不是浏览器"，
    # 白点一次窗口、多花 0.6~2 秒，还把标签页顺序点乱。
    _ZW_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")

    @classmethod
    def _clean_title(cls, s: str) -> str:
        return re.sub(r"\s{2,}", " ", cls._ZW_RE.sub("", s or "")).strip()

    @classmethod
    def _window_info(cls):
        """返回 (前台窗口标题, 顶层可见窗口标题列表)。给模型提供零幻觉的系统状态：
        程序有没有真的打开、前台是谁，看截图猜不如直接读系统。"""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            buf = ctypes.create_unicode_buffer(256)

            fg_h = user32.GetForegroundWindow()
            user32.GetWindowTextW(fg_h, buf, 256)
            fg = cls._clean_title(buf.value)

            titles = []

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            def _cb(hwnd, _):
                if user32.IsWindowVisible(hwnd):
                    user32.GetWindowTextW(hwnd, buf, 256)
                    t = cls._clean_title(buf.value)
                    if t:
                        titles.append(t)
                return True

            user32.EnumWindows(_cb, 0)
            return fg, titles[:20]
        except Exception:
            return "", []

    # 程序名 -> 进程 exe 关键词（大小写不敏感，匹配 exe 名或窗口标题）
    _PROC_MAP = {
        "记事本": ["notepad"], "notepad": ["notepad"],
        "计算器": ["calc"], "calc": ["calc"],
        "浏览器": ["msedge", "chrome", "firefox"], "edge": ["msedge"],
        "chrome": ["chrome"], "firefox": ["firefox"],
        "资源管理器": ["explorer"], "explorer": ["explorer"],
        "终端": ["windowsterminal", "powershell", "cmd"],
        "powershell": ["powershell"], "cmd": ["cmd"],
        "画图": ["mspaint"], "paint": ["mspaint"],
        # Dev-C++（含 Red Panda 续命版）：exe 统一是 devcpp.exe
        "dev-c++": ["devcpp"], "devc++": ["devcpp"], "dev-cpp": ["devcpp"],
        "devcpp": ["devcpp"],
        "微信": ["weixin", "wechat"], "wechat": ["weixin", "wechat"],
    }

    @staticmethod
    def _exe_matches(exe: str, key: str) -> bool:
        """进程文件名匹配：比"程序名"判等的边界，不是任意子串。

        真机翻车：任务「打开计算器」，_PROC_MAP 给的是 "calc"，而窗口标题
        「Lo**calc**UA - 文件资源管理器」里正好含 calc —— 程序认定"计算器已在运行"、
        把资源管理器聚焦了一下就报成功，计算器一次都没开。
        """
        if not exe or not key:
            return False
        stem = exe.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if stem.endswith(".exe"):
            stem = stem[:-4]
        key = key.lower()
        return stem == key or stem.startswith(key)

    @classmethod
    def _title_matches(cls, title: str, key: str) -> bool:
        """窗口标题匹配：按「 - 」切成片段后逐段比，绝不做整串子串匹配。

        "LocalCUA - 文件资源管理器" 里含 "calc"，整串子串匹配会把它当成计算器。
        片段比等：标题「无标题 - Notepad」的片段是 ["无标题","Notepad"]，
        key="notepad" 命中；"localcua"/"文件资源管理器" 都不命中 "calc"。
        """
        key = (key or "").lower().strip()
        if not key:
            return False
        parts = [p.strip().lower() for p in re.split(r"[-–—|]", title or "")]
        return any(p == key or p.startswith(key + " ") or p.startswith(key + "(")
                   for p in parts)

    @classmethod
    def _focus_program(cls, text: str):
        """从任务文本提取程序名，把对应窗口调到前台。成功返回窗口标题，失败返回 None。
        专治「窗口开着但被全屏应用挡住，模型点不到」的场景 —— 程序直调 Win32，零幻觉。"""
        t = (text or "").lower()
        exes = []
        for k, v in cls._PROC_MAP.items():
            if k in t:
                exes.extend(v if isinstance(v, list) else [v])
        if not exes:
            return None
        try:
            import ctypes
            import ctypes.wintypes
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            title_buf = ctypes.create_unicode_buffer(256)
            exe_buf = ctypes.create_unicode_buffer(512)
            found = []

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            def _cb(hwnd, _):
                if not user32.IsWindowVisible(hwnd):
                    return True
                user32.GetWindowTextW(hwnd, title_buf, 256)
                title = title_buf.value.strip()
                pid = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                h = kernel32.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED
                exe = ""
                if h:
                    size = ctypes.c_ulong(512)
                    if kernel32.QueryFullProcessImageNameW(h, 0, exe_buf, ctypes.byref(size)):
                        exe = exe_buf.value.replace("\\", "/").lower()
                    kernel32.CloseHandle(h)
                for e in exes:
                    if cls._exe_matches(exe, e) or (title and cls._title_matches(title, e)):
                        if title:
                            found.append((hwnd, title))
                        break
                return True

            user32.EnumWindows(_cb, 0)
            if not found:
                return None
            hwnd, title = found[0]
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE（最小化则还原）
            # SetForegroundWindow 有前台锁：后台进程调用会被拒绝。
            # 经典绕法：把当前线程附加到前台线程的输入队列，获得"前台身份"后再切换。
            kernel32 = ctypes.windll.kernel32
            fg_hwnd = user32.GetForegroundWindow()
            fg_thread = user32.GetWindowThreadProcessId(fg_hwnd, None)
            my_thread = kernel32.GetCurrentThreadId()
            attached = False
            if fg_thread and fg_thread != my_thread:
                attached = bool(user32.AttachThreadInput(my_thread, fg_thread, True))
            ok = False
            try:
                user32.BringWindowToTop(hwnd)
                ok = bool(user32.SetForegroundWindow(hwnd))
                if not ok:
                    user32.keybd_event(0x12, 0, 0, 0)   # alt down 解锁兜底
                    ok = bool(user32.SetForegroundWindow(hwnd))
                    user32.keybd_event(0x12, 0, 2, 0)   # alt up
            finally:
                if attached:
                    user32.AttachThreadInput(my_thread, fg_thread, False)

            def _is_fg():
                buf2 = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(user32.GetForegroundWindow(), buf2, 256)
                return buf2.value == title

            time.sleep(0.3)
            if not (ok and _is_fg()):
                # 终极兜底：真实鼠标点击窗口中心 —— 真实点击必然使窗口前台化
                rect = ctypes.wintypes.RECT()
                if user32.GetWindowRect(hwnd, ctypes.byref(rect)) and rect.right > rect.left:
                    import pyautogui
                    cx, cy = (rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2
                    pyautogui.click(min(max(cx, 0), user32.GetSystemMetrics(0) - 1),
                                    min(max(cy, 0), user32.GetSystemMetrics(1) - 1))
                    time.sleep(0.4)
            return title if _is_fg() else None
        except Exception:
            return None

    # ---------- 进场确认：青蛙眼该用的地方 ----------
    # 用户原意：不是「每时每刻都裁图」，而是「点完蹦出新窗口那一刻」把它当成
    # 一条确定事实告诉模型。全程不写死任何应用的界面特征：
    #   ① 画面变了多少（帧差分）  ② 现在前台是谁（Win32 标题 + exe）
    # 两件事对所有程序都一样，所以 WorkBuddy / Dev-C++ / 任意没见过的程序都通用。
    @classmethod
    def _app_window_keys(cls, app: str):
        """app 名 -> 一组匹配「窗口标题 / exe 文件名」的关键词。
        关键词只来自任务文字和系统登记，绝不含界面特征，因此不需要为每个应用写模板。"""
        a = (app or "").strip().lower()
        if not a:
            return set()
        keys = {a, a.replace(" ", ""), a.replace("-", ""), a.replace("_", "")}
        for k, v in cls._PROC_MAP.items():
            if k in a or a in k:
                keys.update(v if isinstance(v, list) else [v])
        return {k for k in keys if len(k) >= 2}

    @classmethod
    def _find_app_window(cls, app: str):
        """按 app 名真去枚举可见窗口，返回 (hwnd, 标题, exe) 或 None。

        和 _focus_program 的分工：后者是「从一句文本里找 _PROC_MAP 关键字」，
        程序名没登记时直接 return None，从不枚举窗口。_wait_app_window 以前误用它
        当「按名字找窗口」，于是 WorkBuddy 这类未登记程序明明开着也判「没打开」，
        白烧 40 秒 fallback，再让模型重发 open_app —— 真机一条任务转了 234 秒。
        """
        keys = cls._app_window_keys(app)
        if not keys:
            return None
        try:
            import ctypes
            import ctypes.wintypes as wt
            u, k = ctypes.windll.user32, ctypes.windll.kernel32
            tb = ctypes.create_unicode_buffer(512)
            eb = ctypes.create_unicode_buffer(600)
            found = []

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            def _cb(h, _):
                if not u.IsWindowVisible(h):
                    return True
                u.GetWindowTextW(h, tb, 512)
                title = cls._clean_title(tb.value)
                if not title:
                    return True
                pid = wt.DWORD()
                u.GetWindowThreadProcessId(h, ctypes.byref(pid))
                ph = k.OpenProcess(0x1000, False, pid.value)
                exe = ""
                if ph:
                    n = wt.DWORD(600)
                    if k.QueryFullProcessImageNameW(ph, 0, eb, ctypes.byref(n)):
                        exe = eb.value.rsplit("\\", 1)[-1].lower()
                    k.CloseHandle(ph)
                tl = title.lower().replace(" ", "")
                for key in keys:
                    if key in tl or (exe and key in exe):
                        found.append((h, title, exe))
                        break
                return True

            u.EnumWindows(_cb, 0)
            return found[0] if found else None
        except Exception:
            return None

    @classmethod
    def _activate_window(cls, hwnd, title):
        """把指定窗口提到前台：AttachThreadInput 破前台锁，真点击兜底。"""
        try:
            import ctypes
            import ctypes.wintypes as wt
            u, k = ctypes.windll.user32, ctypes.windll.kernel32
            u.ShowWindow(hwnd, 9)
            fg_hwnd = u.GetForegroundWindow()
            fg_thread = u.GetWindowThreadProcessId(fg_hwnd, None)
            my_thread = k.GetCurrentThreadId()
            attached = bool(fg_thread and fg_thread != my_thread
                            and u.AttachThreadInput(my_thread, fg_thread, True))
            ok = False
            try:
                u.BringWindowToTop(hwnd)
                ok = bool(u.SetForegroundWindow(hwnd))
                if not ok:
                    u.keybd_event(0x12, 0, 0, 0)
                    ok = bool(u.SetForegroundWindow(hwnd))
                    u.keybd_event(0x12, 0, 2, 0)
            finally:
                if attached:
                    u.AttachThreadInput(my_thread, fg_thread, False)

            def _is_fg():
                b = ctypes.create_unicode_buffer(512)
                u.GetWindowTextW(u.GetForegroundWindow(), b, 512)
                return cls._clean_title(b.value) == title

            time.sleep(0.3)
            if not (ok and _is_fg()):
                rect = wt.RECT()
                if u.GetWindowRect(hwnd, ctypes.byref(rect)) and rect.right > rect.left:
                    import pyautogui
                    cx = min(max((rect.left + rect.right) // 2, 0), u.GetSystemMetrics(0) - 1)
                    cy = min(max((rect.top + rect.bottom) // 2, 0), u.GetSystemMetrics(1) - 1)
                    pyautogui.click(cx, cy)
                    time.sleep(0.4)
            return title if _is_fg() else None
        except Exception:
            return None

    def _record_entry(self, app: str, via: str = "系统判定"):
        """把「已经在目标应用里」记成系统事实，下一轮强制注入提示词。

        模型反复发 open_app 的真正原因不是它看不见截图，而是它从没收到过
        「你进去了」这个确定反馈 —— 日志里只有「⚠️ 未检测到窗口」。
        """
        fg, _ = self._window_info()
        hit = self._find_app_window(app)
        self._entered_app = {
            "app": app, "title": (hit[1] if hit else (fg or app)),
            "exe": self._foreground_exe(), "via": via,
        }
        self._entry_step = getattr(self, "step", 0)
        self.long_memory.note_open(
            app, self._entered_app["title"], self._entered_app["exe"])
        self.log(f"[进场] {via}：已进入「{self._entered_app['title']}」"
                 f"（{self._entered_app['exe'] or 'exe 未知'}），下一轮明确告知模型不要再打开")

    def _confirm_entry(self, app: str, before, max_wait: float = 5.0):
        """双击之后：画面是不是蹦出了新窗口？系统说了算，模型没有否决权。

        为什么必须配前台 exe：差分只知道「哪里变了」，不知道「变成了谁」。
        双击坏快捷方式也会弹出错误框，所以还要问系统「现在前台是谁」。
        两条都成立才宣布进场，全程与应用无关。
        """
        if before is None or not getattr(self, "entry_confirm", True):
            return False
        deadline = time.time() + max_wait
        # 次数上限是必要的：测试会把 time.sleep mock 成 no-op，只靠 deadline 会空转
        for _ in range(12):
            if time.time() > deadline:
                break
            time.sleep(0.4)
            try:
                after = self.screenshot_fn()
            except Exception:
                return False
            _regions, ratio = self._diff_regions(before, after)
            # 不要求 regions 非空：大窗口打开时整屏都变，_diff_half 会返回 ([], ratio)，
            # 那时恰恰是最该判「进场」的一次。只要变化量够 + 前台对得上就行。
            if ratio < 0.02:
                continue
            hit = self._find_app_window(app)
            if hit:
                self._activate_window(hit[0], hit[1])
                self._record_entry(app, f"青蛙眼：画面变化 {ratio * 100:.1f}%")
                return True
        return False

    def _observe_entry(self, action, before, after):
        """模型自己发的 double_click 也要有进场反馈（用户原意是「动作级」的钩子）。

        和 _confirm_entry 的分工：那个服务 open_app 模板，知道 app 名；
        这个不知道模型想打开什么，所以判据换成「前台进程有没有换人」——
        同样是 Win32 硬事实，同样与应用无关。用 exe 而不是窗口标题，是因为
        同一应用里双击打开文件标题也会变（不该被当成「进入新应用」）。
        """
        if not getattr(self, "entry_confirm", True) or before is None:
            return
        if action.get("action") not in ("double_click", "open_app"):
            return
        if getattr(self, "_entered_app", None) and \
                getattr(self, "_entry_step", -99) == getattr(self, "step", 0):
            return   # open_app 模板已经确认过了，别覆盖成较粗的信息
        exe0 = getattr(self, "_fg_exe_before", "") or ""
        fg0 = getattr(self, "_fg_before", "") or ""
        for i in range(3):
            shot = after if i == 0 else self.screenshot_fn()
            _regions, ratio = self._diff_regions(before, shot)
            exe1 = self._foreground_exe()
            fg1 = self._window_info()[0]
            if ratio >= 0.02 and exe1 and exe1 != exe0:
                self._entered_app = {
                    "app": fg1 or exe1, "title": fg1 or exe1, "exe": exe1,
                    "via": (f"青蛙眼：画面变化 {ratio * 100:.1f}%，"
                            f"前台「{fg0 or '桌面'}」→「{fg1 or exe1}」"),
                }
                self._entry_step = getattr(self, "step", 0)
                # 关键：动作瞬间那张截图里窗口往往还没刷出来。必须把「真的看到
                # 新窗口」这一帧交给下一轮，否则模型知道「我进去了」却看着一张
                # 空桌面，只能瞎点坐标（真机第 2 步就点了右上角空白处）。
                self._entry_shot = shot
                self._entry_just_confirmed = True
                self.log(f"[进场] {self._entered_app['via']}：动作确实打开了新窗口")
                self.history.append(
                    f"第{self.step}步: ✅ 系统确认你已经进入「{fg1 or exe1}」"
                    "（前台进程已切换、画面出现新窗口）。不要再打开它，直接做下一步。")
                return
            if i < 2:
                time.sleep(0.3)
        return

    # 「回车」会把内容提交到外面的窗口。no_send 只在这些窗口拦 enter，
    # 浏览器地址栏 / 开始菜单的回车不受影响，测试时搜索和开程序照样能跑。
    _CHAT_WINDOW_HINTS = ("wechat", "weixin", "微信", "qq", "dingtalk", "钉钉",
                          "飞书", "lark", "telegram", "discord", "workbuddy")
    _INSTANT_SETTLE_ACTIONS = ("type", "key", "hotkey", "press", "scroll", "move", "wait")

    def _no_send_guard(self, action) -> bool:
        """True = 这是对外发送动作，护栏下不执行（但视为成功，链路继续走完）。"""
        if not self.no_send:
            return False
        a = str((action or {}).get("action") or "").lower()
        if a == "send_message":
            return True
        if a in ("key", "hotkey", "press"):
            k = str(action.get("key") or action.get("keys") or "").lower()
            if "enter" in k:
                fg, _ = self._window_info()
                low = (fg or "").lower()
                exe = (self._foreground_exe() or "").lower()
                return any(h in low or h in exe for h in self._CHAT_WINDOW_HINTS)
        return False

    def _settle_budget(self, action) -> float:
        """观察窗上限：打字/按键/滚动是即时生效的，只采一次；开窗动作才给满 3 秒。"""
        if self.change_settle <= 0 or self.change_settle_mode == "off":
            return 0.0
        a = str((action or {}).get("action") or "").lower()
        if a in self._INSTANT_SETTLE_ACTIONS:
            return min(self.change_settle, 0.5)
        return self.change_settle

    def _out_of_time(self) -> bool:
        return (self.max_seconds > 0
                and getattr(self, "_t_run", None) is not None
                and (time.time() - self._t_run) >= self.max_seconds)

    def _settle_screen(self, before, first, seconds: float = 3.0,
                       stable: bool = True):
        """期待有新画面时，给 UI 渲染一个观察窗，期间用青蛙眼采样。

        真机问题：动作一执行完就截屏，常常截到「窗口边框刚出来、内容还没画出来」
        的半帧，于是系统判「画面没有变化」，模型跟着脑补「程序没装」。

        stable=True 时连续两帧变化量几乎相同就提前返回。真机曲线
        「17.0% → 25.6% → 25.6% ×4」说明画面 1 秒就稳了，固定等满 3 秒
        有 2 秒是白等。stable=False 保留旧行为（固定等满），便于对照。
        """
        try:
            W, H = first.size
            hw, hh = max(1, W // 2), max(1, H // 2)
            g0 = before.convert("L").resize((hw, hh), Image.BILINEAR)
        except Exception:
            g0 = None
        steps = max(1, min(12, int(round(seconds / 0.5))))
        deadline = time.time() + seconds
        ratios = []
        last = first
        early = False
        t_start = time.time()
        for i in range(steps):
            # 测试里 time.sleep 会被 mock 成 no-op：次数上限必须独立存在，
            # 只靠 deadline 会把循环空转到 CPU 打满（之前踩过）。
            if i and time.time() >= deadline:
                break
            time.sleep(min(0.5, max(0.0, deadline - time.time())))
            try:
                last = self.screenshot_fn()
            except Exception:
                break
            if g0 is None:
                continue
            try:
                g = last.convert("L").resize((hw, hh), Image.BILINEAR)
                _regions, ratio = self._diff_half(g0, g, last.size)
                ratios.append(ratio * 100)
            except Exception:
                pass
            # 连续两帧变化量几乎相同 = 渲染收敛，没必要再等。
            # 画面纹丝不动时两帧都是 0.0%，同样立刻结束（省掉 2 秒白等）。
            if stable and len(ratios) >= 2 and abs(ratios[-1] - ratios[-2]) < 0.5:
                early = True
                break
        if not ratios:
            return last, ""
        elapsed = time.time() - t_start
        trend = " → ".join(f"{r:.1f}%" for r in ratios)
        tag = "变稳提前结束" if early else "观察满"
        note = (f"[青蛙眼] 期待新画面：{tag} {elapsed:.1f}s/{seconds:.1f}s"
                f"（{trend}），取最后一帧，画面变化 {ratios[-1]:.1f}%")
        self._last_settle = {"seconds": seconds, "ratios": ratios,
                             "elapsed": elapsed, "early": early}
        return last, note

    @staticmethod
    def _snap_to_element(action: dict, elements, tol: float = 40.0) -> dict:
        """模型自己估的坐标常常差几十像素。吸附规则（按优先级）：
        1) 坐标落在某元素包围盒【内部】→ 吸到其中面积最小的（最内层=实际目标）；
        2) 否则找距离中心 < tol 的最近元素。
        容差不能太大：任务栏相邻图标中心距只有 ~60px，tol=60 会吸到隔壁图标（乱点来源之一）。"""
        if not elements or action.get("x") is None or action.get("y") is None:
            return action
        if action.get("_desc_locked"):  # element_desc 已精确定位，不再吸附
            return action
        x, y = float(action["x"]), float(action["y"])
        inner = []
        for el in elements:
            x1, y1, x2, y2 = el["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                inner.append(((x2 - x1) * (y2 - y1), el))
        if inner:
            inner.sort(key=lambda t: t[0])
            x1, y1, x2, y2 = inner[0][1]["bbox"]
            action["x"], action["y"] = int((x1 + x2) / 2), int((y1 + y2) / 2)
            return action
        best, bestd = None, tol
        for el in elements:
            x1, y1, x2, y2 = el["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            d = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            if d < bestd:
                best, bestd = (cx, cy), d
        if best:
            action["x"], action["y"] = int(best[0]), int(best[1])
        return action

    @staticmethod
    def _norm_element_id(v):
        """把模型给的 element_id 归一成整数/纯字符串。
        真机踩坑：7B 会输出 "element_id": [34]（列表）、"element_id": "[34]"
        （带方括号），老代码直接拿去查表必然查不到，于是报"没有可用坐标"，
        模型只好改猜坐标 —— 一步就把精确坐标优势全丢了。"""
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            v = v[0] if v else None
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float) and v == int(v):
            return int(v)
        s = str(v).strip().strip("[](){} ").strip()
        m = re.fullmatch(r"(?:元素|element[_\s]*id)?\s*[:#]?\s*(\d+)", s, re.I)
        return int(m.group(1)) if m else (s or None)

    def _to_screen_coords(self, action: dict, scale: float) -> dict:
        """模型是按「缩放后的截图」给坐标的，执行前必须换算回真实屏幕像素。
        element_id 也在这里解析成坐标（grounding 存的是截图坐标），再一起放大。"""
        if not action:
            return action
        # element_id 解析必须放在 scale 判断之前：截图没缩放时也要能查表
        if action.get("element_id") is not None and action.get("x") is None:
            eid = self._norm_element_id(action["element_id"])
            pt = None
            if eid is not None:
                try:
                    pt = self.grounding.resolve(eid)
                except Exception:
                    pt = None
            if pt:
                action["x"], action["y"] = pt
                action.pop("element_id", None)
            else:
                action["element_id"] = eid   # 归一后的值留给"无效"提示，别显示成 [34]
        if scale == 1.0:
            return action
        up = 1.0 / scale
        if action.get("x") is not None and action.get("y") is not None:
            action["x"] = int(round(float(action["x"]) * up))
            action["y"] = int(round(float(action["y"]) * up))
        if action.get("action") == "drag":
            for k in ("from", "to"):
                v = action.get(k)
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    action[k] = [int(round(float(v[0]) * up)), int(round(float(v[1]) * up))]
        return action

    # ------------------------- prompt 构造 -------------------------
    @staticmethod
    def _focused_control_center():
        """前台窗口当前键盘焦点控件的矩形中心（物理像素）+ 宽高，查不到返回 None。
        GetGUIThreadInfo(0) 直接返回前台线程的焦点控件句柄，无需附加输入队列。
        用途：「输入框/编辑区」是 YOLO 的天然盲区（不是图标），但 Win32 知道焦点在哪 ——
        模型说"点输入框"时直接查表，不再让它瞎猜坐标。"""
        try:
            import ctypes
            from ctypes import wintypes

            class GUITHREADINFO(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
                            ("hwndActive", wintypes.HWND), ("hwndFocus", wintypes.HWND),
                            ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
                            ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND),
                            ("rcCaret", wintypes.RECT)]

            user32 = ctypes.windll.user32
            gti = GUITHREADINFO()
            gti.cbSize = ctypes.sizeof(GUITHREADINFO)
            if not user32.GetGUIThreadInfo(0, ctypes.byref(gti)):
                return None
            h = gti.hwndFocus or gti.hwndActive
            if not h:
                return None
            rect = wintypes.RECT()
            if not user32.GetWindowRect(h, ctypes.byref(rect)):
                return None
            w, hh = rect.right - rect.left, rect.bottom - rect.top
            if w <= 5 or hh <= 5:
                return None
            return (rect.left + w / 2.0, rect.top + hh / 2.0, w, hh)
        except Exception:
            return None

    def _apply_element_desc(self, action: dict, img_s, scale: float = 1.0) -> dict:
        """模型用自然语言描述要点的目标（element_desc），程序负责精确定位。
        实测这比模型直接给坐标（±30~90px 误差）或选编号（对号入座不稳）都准。"""
        desc = action.get("element_desc")
        if not desc:
            return action
        g = self.grounding
        if not hasattr(g, "resolve_desc"):
            action.pop("element_desc", None)
            return action
        r = g.resolve_desc(str(desc), img_s)
        if r:
            x, y, eid, label, score = r
            # 定位成功就覆盖模型自己给的坐标：VLM 目测坐标误差 ±30~90px，
            # YOLO 包围盒中心是 ±3px。模型只要说清"点哪个"，位置交给程序算。
            action["x"], action["y"] = float(x), float(y)
            action["_desc_locked"] = True  # 定位已足够准，不再做 snap 吸附
            self.log(f"[定位] '{desc}' -> #{eid} '{label}' (分{score}) @({x:.0f},{y:.0f})")
        else:
            action.pop("element_desc", None)
            if action.get("x") is not None:
                # 模型描述的是编辑区/标题栏这类【非图标控件】（视觉检测本来就不覆盖），
                # 但它自己给了坐标 —— 这种坐标往往是它唯一能表达位置的方式，别再作废。
                self.log(f"[定位] '{desc}' 匹配不到元素，沿用模型给的坐标 "
                         f"({action.get('x')},{action.get('y')})")
                action["_miss_desc"] = str(desc)
                return action
            self._desc_miss = str(desc)   # 记下来，下一轮在 user_text 里明确告诉模型"不在屏幕上"
            self.log(f"[定位] '{desc}' 不在当前屏幕上，提示模型先打开它")
            # 程序级判断：目标是已知程序 → 直接转 open_app 模板（不等模型悟）
            # 只在「desc 本身就是在说这个程序」时才转模板。
            # 不能拼上子任务文本再匹配：模型说"记事本窗口中的输入框"时，
            # 子任务"双击打开记事本"里的"记事本"会让程序误判成要开程序（真机踩坑）。
            # 判据：程序名要占 desc 的一半以上 —— "记事本图标"✓，"记事本窗口中的输入框"✗。
            d = str(desc).lower()
            # 开始按钮/开始菜单是【任务栏系统控件】，YOLO 只检窗口内的图标类元素，
            # 永远检不到它；模型自己估的坐标又会偏（真机：为了点它白烧 39 秒，
            # 开始菜单根本没打开，后面所有"搜索程序"全部落空）。
            # win 键和点开始按钮完全等价，且零误差 —— 程序直接接管。
            if any(k in d for k in ("开始按钮", "启动按钮", "开始菜单", "任务栏", "win键")):
                self.log(f"[自动] 目标是开始按钮/开始菜单（系统控件，元素检测不覆盖）"
                         f"→ 程序接管，用 win 键代替点击")
                self._desc_miss = None
                action.clear()
                action["action"] = "key"
                action["key"] = "win"
                action["expect_change"] = True
                return action
            # 「输入框/编辑区」是 YOLO 的天然盲区（不是图标类控件）。但 Windows
            # 一直知道键盘焦点在哪个控件上 —— 直接查表给中心坐标，零猜测。
            # 注意：这一步必须在「转 open_app」之前——"记事本输入框"里的程序名
            # 会占一半长度，先查输入框才不会误判成"要打开记事本"。
            if any(k in d for k in ("输入框", "输入区", "编辑区", "编辑框", "文本框",
                                    "代码区", "代码编辑", "搜索框")):
                c = self._focused_control_center()
                if c:
                    fx, fy, fw, fh = c
                    # 物理像素 -> 缩放截图坐标（和 YOLO 元素同一坐标系，后续统一换算）
                    ix, iy = fx / scale, fy / scale
                    self.log(f"[自动] '{desc}' 是编辑区类目标 → 查得前台焦点控件 "
                             f"@({fx:.0f},{fy:.0f}) {fw:.0f}x{fh:.0f}px，直接点它")
                    self._desc_miss = None
                    action.clear()
                    action["action"] = "click"
                    action["x"], action["y"] = float(ix), float(iy)
                    action["expect_change"] = True
                    return action
                self.log(f"[定位] '{desc}' 是编辑区类目标，但拿不到焦点控件位置，继续常规匹配")
            prog = next((k for k in self._PROC_MAP
                         if k in d and len(k) * 2 >= len(d)), None)
            if prog:
                self.log(f"[自动] 目标命中已知程序「{prog}」→ 转程序级 open_app 模板")
                # 程序接管了，必须把 miss 提示清掉：否则下一轮 user_text 会告诉模型
                # 「目标不在屏幕上，先去点开始按钮」——而程序其实已经把它打开了
                self._desc_miss = None
                action.clear()
                action["action"] = "open_app"
                action["app"] = prog
                action["expect_change"] = True
            else:
                # 没转模板就保留原始描述，后面「无坐标」反馈时用它，别让模型看到 None
                action["_miss_desc"] = str(desc)
        return action

    # ------------------------- 光标感知 -------------------------
    # 背景：Windows 截图（ImageGrab/pyautogui）【不包含鼠标光标】。
    # 模型看截图永远找不到"我现在点在哪"，这是乱点/重复点/打错地方的重要根源。
    # 对策：① 每步动作后把光标停靠到固定角落（状态确定、不产生 hover 干扰）；
    #      ② 送模型的截图上画出光标标记；③ user_text 里给出精确坐标。

    _PARK = (14, 14)   # 停靠点：贴近左上角但不落在角上（pyautogui 的急停保护会拦四角）

    @staticmethod
    def _cursor_pos():
        """当前鼠标的屏幕物理像素坐标；拿不到就返回 None。"""
        try:
            import ctypes
            from ctypes import wintypes
            pt = wintypes.POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                return (int(pt.x), int(pt.y))
        except Exception:
            pass
        return None

    @staticmethod
    def _unstick_cursor():
        """开场把光标从屏幕四角挪开。
        pyautogui 的急停保护检查的是"这次调用那一刻"的坐标：光标在四角就直接抛
        FailSafeException。而程序自己会把光标停靠在左上角附近、任务又可能被超时
        杀在半路 —— 于是下一次任务第一个 pyautogui 调用就崩（真机实测：Win+D
        刚发出就抛错，任务 1 秒结束，日志只有一行 traceback）。
        只在开场清理一次，用户中途把鼠标甩到角落仍然能急停。"""
        try:
            import ctypes
            from ctypes import wintypes
            u = ctypes.windll.user32
            pt = wintypes.POINT()
            if not u.GetCursorPos(ctypes.byref(pt)):
                return
            w, h = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
            for cx, cy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
                if abs(pt.x - cx) <= 8 and abs(pt.y - cy) <= 8:
                    u.SetCursorPos(80, 80)
                    return
        except Exception:
            pass

    @classmethod
    def _park_cursor(cls):
        """动作执行完把鼠标挪到固定停靠点。
        好处：① 下一步截图时 hover 状态（tooltip/按钮高亮）已消失，画面干净；
             ② 光标位置永远已知且固定，模型不用猜。"""
        try:
            import ctypes
            ctypes.windll.user32.SetCursorPos(*cls._PARK)
        except Exception:
            pass

    @staticmethod
    def _mark_cursor(annotated, scale: float):
        """在送模型的截图上画出光标位置（红圈+十字），让模型"看得见"鼠标。"""
        try:
            import ctypes
            from ctypes import wintypes
            from PIL import ImageDraw
            pt = wintypes.POINT()
            if not ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                return
            cx, cy = pt.x * scale, pt.y * scale
            w, h = annotated.size
            if not (0 <= cx < w and 0 <= cy < h):
                return
            r = max(6, int(10 * max(scale, 0.4)))
            d = ImageDraw.Draw(annotated)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 40, 40), width=3)
            d.line([cx - 2 * r, cy, cx + 2 * r, cy], fill=(255, 40, 40), width=2)
            d.line([cx, cy - 2 * r, cx, cy + 2 * r], fill=(255, 40, 40), width=2)
        except Exception:
            pass  # 标记失败不影响主流程

    def _ensure_focus_for_type(self):
        """type 前的最后防线：如果刚打开过程序、但前台不是它，先把窗口调到前台。
        否则文字会敲进不相干的窗口（「能打开记事本却几乎没输入文字」的主因之一）。"""
        app = getattr(self, "_last_opened_app", None)
        if not app:
            return
        fg, _ = self._window_info()
        # 模型给的是中文名「记事本」，窗口标题却常是「无标题 - Notepad」——
        # 只匹配中文名会每次都误判"前台不对"（真机踩坑：打了字还被说没聚焦）
        names = [app.lower()] + [x.lower() for x in self._PROC_MAP.get(app, [])]
        if fg and any(n in fg.lower() for n in names):
            return  # 前台就是它，放心输入
        self.log(f"[焦点] 前台是「{fg or '未知'}」而不是「{app}」-> 先聚焦再输入")
        if self._focus_program(app):
            time.sleep(0.6)

    # 带扩展名的文件名：标题里出现它 = 这个窗口正打开着某个文件
    _DOC_EXT_RE = re.compile(
        r"[^\s\\/:*?\"<>|()\[\]]+\.(?:txt|log|md|py|c|cc|cpp|h|hpp|json|ya?ml|ini|bat|ps1"
        r"|js|ts|html?|css|csv|xml|java|go|rs|sh)\b", re.I)

    @staticmethod
    def _foreground_title() -> str:
        """前台窗口标题（轻量版，不走 EnumWindows）。"""
        try:
            import ctypes
            u = ctypes.windll.user32
            buf = ctypes.create_unicode_buffer(512)
            u.GetWindowTextW(u.GetForegroundWindow(), buf, 512)
            return buf.value.strip()
        except Exception:
            return ""

    def _screen_has_text(self, text: str) -> bool:
        """OCR 看一眼屏幕，确认这段文字真的在上面（输入到底生效没有）。

        只看前 8 个非空白字符：OCR 会漏字、会断行，严格相等永远对不上。
        """
        probe = "".join(str(text).split()).lower()[:8]
        if not probe:
            return False
        g = self.grounding
        if not hasattr(g, "ocr_enrich"):
            return False
        try:
            g.ocr_enrich(self.screenshot_fn(), [], scale=1.0)
            texts = getattr(g, "last_texts", None) or []
        except Exception:
            return False
        joined = "".join("".join(str(x[4]).split()) for x in texts).lower()
        return probe in joined

    def _chat_header_text(self) -> str:
        """OCR 读当前聊天窗口顶部那条窄带（会话名所在位置），返回拼起来的文本。

        实测（微信窗口 1290×971）：顶部 3%~13% 这条带能精确读到「文件传输助手」，
        而且**不含左侧会话列表** —— 这点很关键：列表里也印着同一个名字，
        一起 OCR 进来会让「顶栏校验」永远通过，等于没校验。
        """
        try:
            import ctypes
            import ctypes.wintypes as wt

            if not hasattr(self.grounding, "ocr_enrich"):
                return ""
            u = ctypes.windll.user32
            hwnd = u.GetForegroundWindow()
            if not hwnd:
                return ""
            rect = wt.RECT()
            u.GetWindowRect(hwnd, ctypes.byref(rect))
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w <= 0 or h <= 0:
                return ""
            box = (rect.left + int(w * 0.30), rect.top + int(h * 0.03),
                   rect.right, rect.top + int(h * 0.13))
            img = self.screenshot_fn().crop(box)
            self.grounding.ocr_enrich(img, [], scale=1.0)
            texts = getattr(self.grounding, "last_texts", None) or []
            return " ".join(str(x[4]) for x in texts).strip()
        except Exception as exc:
            self.log(f"[顶栏] 读取失败（{exc}）")
            return ""

    @staticmethod
    def _is_wechat_window(title: str) -> bool:
        """前台是不是微信。标题命中优先；会话名做标题时不带「微信」二字，看 exe。"""
        low = (title or "").lower()
        if "微信" in low or "wechat" in low:
            return True
        if not low:
            return False
        try:
            return "wechat" in (AgentSession._foreground_exe() or "")
        except Exception:
            return False

    # 浏览器窗口标题/进程关键词。用进程名兜底：标签页一多，标题里
    # 只有网页名，早就不一定带 "Edge" 了。
    _BROWSER_MARKS = ("edge", "chrome", "firefox", "浏览器", "brave", "opera")

    @classmethod
    def _is_browser_window(cls, title: str) -> bool:
        low = (title or "").lower()
        if any(m in low for m in cls._BROWSER_MARKS):
            return True
        try:
            _exe = AgentSession._foreground_exe() or ""
            return any(m in _exe for m in ("msedge", "chrome", "firefox", "brave", "opera"))
        except Exception:
            return False

    def _wechat_type_allowed(self):
        """在微信窗口里打字 = 可能发消息。只有「给白名单会话发消息」的任务才放行。"""
        task = getattr(self, "task", "") or ""
        intent = self._parse_message_intent(task)
        if not intent:
            return False, "当前任务不是「给白名单会话发消息」"
        return self._send_target_allowed(intent["app"], intent["target"])

    def _type_guard(self, action=None) -> bool:
        """打字前的第一道闸：绝不把字打进「已经打开着某个文件」的窗口。

        真机事故（6 步 119 秒全废）：任务「打开记事本，然后输入 hello world」，
        模型把整句话敲进了用户本来就开着的 `models_fetch.log - Notepad`，
        标题出现 `*`（已修改）后还连打 3 遍 —— 用户的文件被改了。
        这类损坏不可撤销，所以宁可不打：拦下来，把原因写进历史让模型换目标。

        两道判断，顺序不能颠倒：
        1) 这段文字是不是刚输入过（同一个窗口、同样的字）→ 跳过，绝不重复敲；
        2) 前台窗口是不是正打开着用户的文件 → 拦下（记事本先 Ctrl+N 换空白页）。

        返回 True = 可以输入；False = 已拦下，原因写在 self._type_block_reason。
        """
        self._type_block_reason = ""
        title = self._foreground_title()
        text = str((action or {}).get("text") or "")

        # ---- 0) 微信护栏（最硬的一道）----
        # 允许名单以 config.yaml 的 agent.wechat_allowed 为准；
        # 代码默认值只保留两个非真人会话（见 _WECHAT_ALLOWED）。
        # 在微信窗口里打字几乎等于"发消息"（打完回车就出去了），所以这里的闸
        # 不区分动作类型：只要前台是微信，且任务不是白名单会话发消息，一律不打。
        if self._is_wechat_window(title):
            ok_w, why_w = self._wechat_type_allowed()
            if not ok_w:
                self.log(f"[打字] ⛔ 微信里不自己打字（{why_w}）")
                self._note_fail(
                    f"不要在微信窗口里自己打字发消息（{why_w}）。"
                    "发消息必须走 send_message 动作，由程序按白名单发送。")
                self._type_block_reason = "wechat"
                return False

        # ---- 1) 重复输入：真机上模型打完字不收尾，同一句话敲了 3 遍 ----
        if text:
            dup = any(txt == text and tt == title
                      for (tt, txt) in self._typed_recent[-3:])
            if dup:
                if self._screen_has_text(text):
                    self._type_block_reason = "duplicate"
                    self.log(f"[打字] 「{text}」已经在屏幕上（标题「{title}」），不再重复敲")
                    return False
                # 上次可能根本没打进去（前台没聚焦）→ 允许重打
                self.log(f"[打字] 「{text}」记录里有、但屏幕上没有 -> 允许重新输入")
                self._typed_recent = [x for x in self._typed_recent
                                      if not (x[0] == title and x[1] == text)]
                # 打空的第一大原因：模型只给 type、没先点输入框，焦点压根不在输入处。
                # 浏览器里 ctrl+L 是最稳的「把焦点送回能打字的地方」，比让它再猜一遍
                # 坐标可靠得多（真机：4 步全卡在同一个搜索框上，每次都没落字）。
                if self._is_browser_window(title):
                    try:
                        import pyautogui
                        pyautogui.hotkey("ctrl", "l")
                        time.sleep(0.35)
                        self.log("[打字] 上次没落字，已 ctrl+L 聚焦地址栏再重打")
                    except Exception:
                        pass

        if not title:
            return True  # 读不到标题，不瞎拦

        # ---- 2) 是不是我们自己的输入（刚打完字，标题会变成第一行内容 + *）----
        if self._typed_recent and self._typed_recent[-1][0] == title:
            return True   # 正在往自己刚写出来的内容后面接着写（写诗、写代码）

        m = self._DOC_EXT_RE.search(title)
        modified = title.startswith("*")     # 编辑器的「已修改」标记
        if not m and not modified:
            return True

        fname = m.group(0) if m else ""
        stem = fname.rsplit(".", 1)[0] if fname else ""
        task = (getattr(self, "task", "") or "").lower()
        # 任务里点名了这个文件 → 用户就是要改它，放行
        if fname and (fname.lower() in task or (len(stem) >= 3 and stem.lower() in task)):
            return True
        # 只有 * 号、任务又没在写东西（比如刚开局）→ 是用户没保存的草稿，别碰
        if not fname and self._typed_recent:
            return True

        # 记事本：自动开一个空白窗口，而不是放弃这个动作
        low = title.lower()
        if "记事本" in title or "notepad" in low:
            self.log(f"[打字] 前台是已打开的文件窗口「{title}」-> 先 Ctrl+N 开空白窗口")
            self.executor.execute({"action": "key", "key": "ctrl+n"})
            time.sleep(0.8)
            new_title = self._foreground_title()
            if not self._DOC_EXT_RE.search(new_title):
                self.log(f"[打字] 新窗口「{new_title or '无标题'}」是空白的，可以输入")
                return True
            self.log(f"[打字] Ctrl+N 没切到空白窗口（现在「{new_title}」），本次不输入")
            self._note_fail(f"想打字，但前台一直是已打开的文件窗口「{new_title}」")
            self._type_block_reason = "file"
            return False

        self.log(f"[打字] 前台「{title}」正打开着文件 {fname or '（已修改）'}，"
                 "不是要写的地方 -> 本次不输入")
        self._note_fail(
            f"你打算打字，但前台窗口「{title}」里已经打开着文件 —— "
            "字会改进用户的文件里。必须先把光标放到正确的输入位置"
            "（新建空白窗口，或打开你要写的那个程序/输入框），再输入。")
        self._type_block_reason = "file"
        return False

    @staticmethod
    def _idle_seconds() -> float:
        """距用户最后一次动键盘/鼠标过了多少秒。
        用来避免程序跟人抢鼠标：如果人正在操作，程序该让路，
        而不是把人正在拖的窗口抢走、把人正在打的字打歪。
        """
        import ctypes
        from ctypes import wintypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        li = LASTINPUTINFO()
        li.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(li)):
            return 999.0
        # GetTickCount 是 32 位会回绕，用掩码算差值
        delta = (ctypes.windll.kernel32.GetTickCount() - li.dwTime) & 0xFFFFFFFF
        return delta / 1000.0

    def _wait_until_idle(self, need: float = 1.5, max_wait: float = 20.0) -> bool:
        """用户在操作就等着，直到安静 need 秒。返回是否等到了空闲。

        `agent.wait_idle: false` 可以直接跳过（自动化测试必须用它：否则跑一次
        测试要等真人停手，测试时长完全取决于用户此刻在不在动鼠标）。
        """
        cfg = getattr(self, "_cfg_snapshot", None) or {}
        if (cfg.get("agent") or {}).get("wait_idle") is False:
            return True
        waited = 0.0
        while self._idle_seconds() < need:
            if waited >= max_wait:
                self.log(f"[让路] 人一直在操作（{waited:.0f}s），本次不抢鼠标，稍后重试")
                return False
            time.sleep(0.5)
            waited += 0.5
        return True

    def _wait_screen_change(self, before, timeout: float = 2.5, settle: float = 0.4):
        """截图等待画面出现变化（菜单展开/搜索结果渲染），比固定 sleep 又快又稳。
        固定睡 2 秒的问题是：快机器白等，慢机器截太早（搜索结果还没出来，
        后面去匹配自然匹配不到——真机日志「未匹配到搜索结果」多半是这个）。
        返回变化后的最新截图；一直不变就返回超时前的最后一帧。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.15)
            cur = self.screenshot_fn()
            if self._changed(before, cur):
                time.sleep(settle)          # 再给动画一点时间收尾
                return self.screenshot_fn()
        return self.screenshot_fn()

    def _show_desktop(self, settle: float = 1.0):
        """Win+D 一次性最小化所有窗口，露出桌面。

        比一个个去找窗口右上角的「-」按钮快得多，也不会点错：
        找按钮要点像素，按组合键只要一次键盘事件。
        """
        import pyautogui

        pyautogui.hotkey("win", "d")
        time.sleep(settle)

    _DESKTOP_CLASSES = ("Progman", "WorkerW")

    def _go_home_desktop(self) -> bool:
        """Win+D 回桌面，作为通用循环的统一起点。

        为什么值得单独做：8B 的注意力会被眼前的东西牵走 —— 真机实测（
        「在浏览器里查今天的天气」）它把 WorkBuddy 底部的输入框当成开始菜单的
        搜索框，把「今天天气」打进了一个不相干的输入框，连着试三次被熔断。
        睁眼就是干净桌面，这类误认从源头上就没了。

        注意 Win+D 是 toggle：已经在桌面时再按一次会把所有窗口还原出来，
        所以必须先判断当前是不是桌面。
        """
        import pyautogui

        if self._foreground_class() in self._DESKTOP_CLASSES:
            self.log("[起点] 已经在桌面，跳过 Win+D")
            return False
        self._unstick_cursor()   # 光标在四角时 pyautogui 会直接抛急停异常
        pyautogui.hotkey("win", "d")
        time.sleep(0.9)
        if self._foreground_class() in self._DESKTOP_CLASSES:
            self.log("[起点] 已按 Win+D 回到桌面")
            return True
        self.log("[起点] Win+D 之后前台仍不是桌面，照常继续")
        return False

    _DESKTOP_PS1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "desktop_icons.ps1")

    def _desktop_icons(self):
        """读桌面图标清单（名字 + 物理像素屏幕坐标）。返回 (items, screen_w)。

        桌面本身就是一个 SysListView32 控件，直接问 Windows 要清单：
        实测 0.6 秒拿全，坐标 ±0px。换成让视觉模型在截图上逐个给图标起名，
        十几个图标就是十几次模型调用，真机实测 30 秒起步 —— 演示时这不能忍。
        """
        import subprocess

        # PowerShell 5.1 读 .ps1 时按 ANSI 解码，文件一旦没有 BOM，里面的中文注释
        # 就会把语法吃坏（"当前上下文中不存在名称…"）。每次调用前补一次 BOM，
        # 免得以后谁用别的编辑器改一下这个脚本，功能就整个失效。
        try:
            raw = Path(self._DESKTOP_PS1).read_bytes()
            if not raw.startswith(b"\xef\xbb\xbf"):
                Path(self._DESKTOP_PS1).write_bytes(b"\xef\xbb\xbf" + raw)
        except Exception:
            pass

        try:
            p = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", self._DESKTOP_PS1],
                capture_output=True, timeout=25)
            out = (p.stdout or b"").decode("utf-8", "replace").strip()
        except Exception as e:
            self.log(f"[开程序] 读桌面图标失败：{e}")
            return [], 0
        if not out:
            err = (p.stderr or b"").decode("utf-8", "replace").strip()[:200]
            self.log(f"[开程序] 读桌面图标没有输出：{err}")
            return [], 0
        try:
            data = json.loads(out.splitlines()[-1])
        except Exception as e:
            self.log(f"[开程序] 桌面图标清单解析失败：{e}")
            return [], 0
        if not data.get("ok"):
            self.log(f"[开程序] 桌面图标清单不可用：{data.get('error', '未知')}")
            return [], 0
        return data.get("items", []), data.get("screen_w", 0)

    # 同一个程序在桌面上的叫法和用户嘴里说的常常不一样
    # （"Dev-C++" vs "Red Panda Dev-C++"、"微信" vs "WeChat"），先归一再比
    _APP_ALIAS = {
        "devcpp": "devc++", "redpandadevc++": "devc++", "devc++6.7.5": "devc++",
        "wechat": "微信", "weixin": "微信",
        "notepad": "记事本", "calc": "计算器", "calculator": "计算器",
        "microsoftedge": "edge", "msedge": "edge",
        "vscode": "visualstudiocode",
    }

    @classmethod
    def _norm_app_name(cls, s: str) -> str:
        """快捷方式名 → 可比对的键：去后缀、去空格/横杠/括号、转小写。"""
        t = (s or "").strip().lower()
        for suf in ("- 快捷方式", "-快捷方式", "快捷方式", ".lnk", ".url", ".exe", ".bat"):
            if t.endswith(suf):
                t = t[: -len(suf)]
        t = re.sub(r"[\s\u3000\-_—（）()【】\[\]、,，.。]+", "", t)
        return t

    @classmethod
    def _match_desktop_icon(cls, app: str, items):
        """在桌面图标里找目标。返回 (item, score)；找不到 (None, 0)。

        先精确（归一后相等），再包含（"Dev-C++" ⊂ "Red Panda Dev-C++"）。
        多个包含命中时取名字最短的 —— 那通常是最直接的那个。
        """
        a = cls._norm_app_name(app)
        if not a or not items:
            return None, 0
        a = cls._APP_ALIAS.get(a, a)
        fuzzy = []
        for it in items:
            n = cls._norm_app_name(it.get("name", ""))
            if not n:
                continue
            n2 = cls._APP_ALIAS.get(n, n)
            if n2 == a:
                return it, 100
            if a in n2 or n2 in a:
                fuzzy.append((len(n2), it))
        if fuzzy:
            fuzzy.sort(key=lambda t: t[0])
            return fuzzy[0][1], 80
        return None, 0

    @staticmethod
    def _icon_to_py_coords(hit, screen_w):
        """图标物理像素坐标 → pyautogui 逻辑坐标（本机 125% 缩放下差 1.25 倍）。"""
        try:
            py_w = pyautogui.size()[0]
        except Exception:
            py_w = screen_w or 0
        ratio = (py_w / screen_w) if (screen_w and py_w) else 1.0
        return hit["x"] * ratio, hit["y"] * ratio

    def _open_from_desktop(self, app: str) -> bool:
        """优先路：桌面图标有就直接双击（Win+D 露桌面 → 双击 → 验证窗口）。

        桌面图标名和坐标由 _desktop_icons() 直接向 Windows 要，不经过视觉模型，
        所以这条路的耗时基本只有 Win+D + 程序冷启动，通常 2~3 秒。
        桌面上没有该图标时**不动用户窗口**，直接返回 False 交给开始菜单搜索。
        """
        import pyautogui

        items, screen_w = self._desktop_icons()
        hit, score = self._match_desktop_icon(app, items)
        if not hit:
            self.log(f"[开程序] 桌面上没有「{app}」的图标（桌面共 {len(items)} 项），改走开始菜单搜索")
            return False
        # 桌面被别的窗口挡住时双击会落到挡住它的那个窗口上，所以先 Win+D 露出来
        self._show_desktop(settle=0.6)
        # 基准帧必须在「露桌面之后、双击之前」拍：Win+D 本身会让整屏变化，
        # 拿露桌面之前的帧比会把「窗口全收起来」误判成「蹦出了新窗口」。
        before = None
        if getattr(self, "entry_confirm", True):
            try:
                before = self.screenshot_fn()
            except Exception:
                before = None
        x, y = self._icon_to_py_coords(hit, screen_w)
        self.log(f"[开程序] 桌面图标命中「{hit['name']}」(分{score}) @({x:.0f},{y:.0f})，双击打开")
        pyautogui.doubleClick(x, y)
        # 青蛙眼进场确认：双击后画面出现新窗口 + Win32 前台就是它 → 直接成功，
        # 短路掉后面那串「点开始按钮 / 搜索 / 回车 / 视觉找图标」的 40 秒 fallback。
        if self._confirm_entry(app, before):
            self.log(f"[开程序] ✅ 双击后进场确认成立，跳过开始菜单 fallback")
            return True
        title = self._wait_app_window(app, max_wait=6.0)
        if title:
            self.log(f"[开程序] ✅ 桌面图标打开成功，窗口「{title}」")
            return True
        self.log(f"[开程序] ⚠️ 双击了桌面图标，但没检测到「{app}」窗口")
        return False

    def _open_from_desktop_visual(self, app: str) -> bool:
        """视觉兜底：桌面 ListView 读不出来时，才让模型在截图里找图标（慢）。"""
        import pyautogui

        self.log(f"[开程序] 桌面图标清单读不到，改用视觉在桌面截图里找「{app}」")
        self._show_desktop()
        img = self.screenshot_fn()
        img_s, scale = self._fit_image(img)
        r = None
        if hasattr(self.grounding, "resolve_desc"):
            r = self.grounding.resolve_desc(app, img_s)
        if not r:
            self.log(f"[开程序] 桌面上也没找到「{app}」图标")
            return False
        x, y = r[0] / scale, r[1] / scale
        self.log(f"[开程序] 双击桌面图标「{r[3]}」(分{r[4]}) @({x:.0f},{y:.0f})")
        pyautogui.doubleClick(x, y)
        title = self._wait_app_window(app, max_wait=6.0)
        if title:
            self.log(f"[开程序] ✅ 桌面图标打开成功，窗口「{title}」")
            return True
        self.log(f"[开程序] ⚠️ 双击了桌面图标，但没检测到「{app}」窗口")
        return False

    def _wait_app_window(self, app: str, max_wait: float = 8.0):
        """轮询等目标窗口出现。两条路：登记过的程序走 _focus_program（快、可被测试桩接管），
        没登记的任意程序走 _find_app_window 真去枚举窗口标题 / exe。

        真机踩坑：WorkBuddy 窗口标题就叫「WorkBuddy」且已在前台，但 _focus_program
        只认 _PROC_MAP 里的关键字，未登记的程序名直接返回 None —— 于是判「没打开」，
        白跑 40 秒 fallback，模型再发一次 open_app，转了 234 秒。"""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            t = self._focus_program(app)
            if t:
                return t
            hit = self._find_app_window(app)
            if hit and self._activate_window(hit[0], hit[1]):
                return hit[1]
            time.sleep(0.5)
        return None

    def _open_app(self, app: str) -> bool:
        """打开程序；只要成功，就把「已经进去了」作为系统事实回灌给模型。

        包一层的意义：_open_app_impl 有七八个成功出口（桌面图标 / 开始菜单 /
        enter 兜底 / 视觉兜底 / 慢启动…），漏一个就会让模型继续发 open_app。
        """
        ok = self._open_app_impl(app)
        if ok and not getattr(self, "_entered_app", None):
            self._record_entry(app, "窗口轮询")
        return ok

    def _open_app_impl(self, app: str) -> bool:
        """程序级模板：打开一个程序，模型只发 {"action":"open_app","app":"记事本"}。
        序列：已在运行→聚焦；桌面有图标→双击；否则 点开始按钮 → 敲程序名 → 回车/点它 → 验证窗口。
        专治 7B 决策串联失败：这条高频链路不再依赖模型逐步推理。"""
        import pyautogui
        from agent.actions import _type_unicode
        app = (app or "").strip()
        if not app:
            return False
        self._wait_until_idle()   # 有人正在用电脑就先让路，别抢鼠标

        # 1) 已在运行 -> 直接聚焦
        title = self._focus_program(app)
        if title:
            self.log(f"[开程序] 「{app}」已在运行，已聚焦窗口「{title}」")
            return True

        # 2) 桌面图标优先：Win+D 露桌面 → 直接双击图标。
        #    Win11 的开始菜单搜索要 点开始→敲名→等结果→回车 四步（3~6 秒，还会被
        #    搜索结果的分组标题骗），桌面图标只要 Win+D + 双击两次操作。演示时
        #    「像人一样找到图标点开」也比搜索更像人。
        #    桌面上没有这个图标时这里不动任何窗口，直接往下走搜索。
        if self._open_from_desktop(app):
            return True

        # 3) 点开始按钮。整屏检测时任务栏图标只有 ~10px 高，YOLO 会漏检
        #    → 裁任务栏条带放大二次检测，再用 Win11 几何铁律定位：
        #    开始按钮 = 搜索框（条带里最宽的长条元素）左边相邻的那个图标
        img = self.screenshot_fn()
        w, h = img.size
        bar = img.crop((0, h - 100, w, h)).resize((1280, 100), Image.LANCZOS)
        els, _ = self.grounding.annotate(bar)
        start_btn = None
        if els:
            wide = [e for e in els if (e["bbox"][2] - e["bbox"][0]) > 60]
            if wide:
                search = max(wide, key=lambda e: e["bbox"][2] - e["bbox"][0])
                lefts = [e for e in els
                         if e["bbox"][2] <= search["bbox"][0]
                         and 0 < search["bbox"][0] - e["bbox"][2] < 80]
                if lefts:
                    start_btn = max(lefts, key=lambda e: e["bbox"][2])
        if not start_btn:
            self.log("[开程序] 任务栏条带里没定位到开始按钮，改走视觉在桌面上找")
            return self._open_from_desktop_visual(app)
        bx = (start_btn["bbox"][0] + start_btn["bbox"][2]) / 2 * (w / 1280.0)
        by = (h - 100) + (start_btn["bbox"][1] + start_btn["bbox"][3]) / 2
        self.log(f"[开程序] 点开始按钮 @({bx:.0f},{by:.0f})（任务栏条带检测）")
        # 开始按钮是【开关】：菜单已经开着时再点会把它关掉，接着敲的字就打到桌面上了
        # （真机踩坑：第1步模型自己点开菜单，第2步 open_app 又点一下关上，搜索永远空）
        pyautogui.press("esc")
        time.sleep(0.4)
        pyautogui.click(bx, by)
        img_menu = self._wait_screen_change(img, timeout=2.0)   # 等菜单真的展开

        # 3) 敲程序名（开始菜单搜索框自动获得焦点）
        _type_unicode(app)
        # 等搜索结果渲染出来再匹配：固定睡 2 秒在慢机器上会截太早，
        # 结果列表还没出来 -> 匹配不到 -> 白白退到 enter（真机日志常见）
        img2 = self._wait_screen_change(img_menu, timeout=3.0, settle=0.6)

        # 4a) 快路：开始菜单里回车 = 打开最佳匹配。零模型调用、约 3 秒。
        # 真机教训：视觉找结果项要先给 12~40 个元素逐个起名（每个都是一次模型调用），
        # 光这一步就烧了 97 秒 —— 现场演示时这 1 分半钟是致命的。回车几乎总能开对，
        # 开错了再走下面的慢路。
        pyautogui.press("enter")
        title = self._wait_app_window(app, max_wait=6.0)
        if title:
            self.log(f"[开程序] ✅ 「{app}」已打开（回车快路），窗口「{title}」")
            return True
        self.log("[开程序] 回车没打开，重新搜索后走视觉匹配（会慢一些）")
        # 回车已经把开始菜单关掉了，慢路必须重新开一次搜索，否则截到的是桌面
        pyautogui.press("esc")
        time.sleep(0.4)
        pyautogui.click(bx, by)
        self._wait_screen_change(img, timeout=2.0)
        _type_unicode(app)
        img2 = self._wait_screen_change(img_menu, timeout=3.0, settle=0.6)

        # 4b) 慢路：在搜索结果里找目标应用并点它；找不到就 enter 选第一项兜底
        img2_s, scale2 = self._fit_image(img2)
        r2 = self.grounding.resolve_desc(app, img2_s)
        picked = False
        if r2:
            # 真机踩坑：开始菜单里有个分组标题就叫「应用」，匹配分 1.0，
            # 点它等于点了个标题，程序永远起不来（日志里卡了 5 步）。
            # 判据：元素标签里必须真的含程序名，否则宁可走 enter 兜底。
            label2 = str(r2[3] or "")
            if app in label2 or label2.lower() in app.lower():
                x2, y2 = r2[0] / scale2, r2[1] / scale2
                self.log(f"[开程序] 点搜索结果「{label2}」(分{r2[4]}) @({x2:.0f},{y2:.0f})")
                pyautogui.click(x2, y2)
                picked = True
            else:
                self.log(f"[开程序] 搜索结果「{label2}」不含程序名「{app}」，改用 enter 选第一项")
        if not picked:
            self.log("[开程序] 按 enter 选第一项")
            pyautogui.press("enter")
        time.sleep(0.5)   # 后面 _wait_app_window 会轮询，这里只需让点击生效

        # 5) 验证窗口真的开了（轮询最长 8s，别急着判失败）
        title = self._wait_app_window(app)
        if title:
            self.log(f"[开程序] ✅ 「{app}」已打开，窗口「{title}」")
            return True

        # 5b) 点了没开：可能点在分组标题上、或搜索面板焦点还在搜索框。
        #     再按一次 enter —— 开始菜单的搜索里 enter 永远打开最佳匹配，比再点一次稳。
        self.log("[开程序] 点了没打开，补按 enter 兜底")
        pyautogui.press("enter")
        title = self._wait_app_window(app, max_wait=5.0)
        if title:
            self.log(f"[开程序] ✅ 「{app}」已打开（enter 兜底成功），窗口「{title}」")
            return True
        # 兜底之前再确认一次：走了这么久，可能刚好起来
        title = self._wait_app_window(app, max_wait=2.0)
        if title:
            self.log(f"[开程序] ✅ 「{app}」已打开（慢启动），窗口「{title}」")
            return True
        self.log(f"[开程序] ⚠️ 未检测到「{app}」窗口，改走视觉在桌面上找")
        if self._open_from_desktop_visual(app):
            return True
        # 最后再给一次机会：Win+D 可能把正在启动的窗口藏了起来，等它露头
        title = self._wait_app_window(app, max_wait=3.0)
        if title:
            self.log(f"[开程序] ✅ 「{app}」其实已经起来了，窗口「{title}」")
            return True
        return False

    # 不该被"关闭程序"模板关掉的东西：控制台本身、资源管理器（关了桌面会黑）
    _CLOSE_DENY = ("python", "localcua", "explorer", "cmd", "powershell",
                   "windowsterminal", "conhost", "csrss", "dwm")

    # 无状态应用：关掉不会丢任何用户数据，允许最后一步强制结束。
    # 有文档的程序（记事本/Word/Dev-C++/画图）绝不在此列。
    _CLOSE_STATELESS = ("calculatorapp", "calculator", "systemsettings", "windowscamera",
                        "snippingtool", "yourphone")

    def _is_save_dialog(self) -> bool:
        """前台是不是「要不要保存」这类对话框。

        只看类名 #32770 会误判：Alt+F4 之后前台可能短暂是别的对话框（比如
        输入法、系统提示），真机上就因为这一条把「关闭计算器」判成了"弹保存框"。
        所以还要标题里真的出现保存/确认类字样。
        """
        if self._foreground_class() != "#32770":
            return False
        t = (self._foreground_title() or "").lower()
        keys = ("保存", "另存为", "确认", "是否", "未保存", "save", "confirm", "discard")
        return any(k in t for k in keys)

    def _alive_windows(self, exes):
        """这些 exe 关键词下还有没有可见窗口（过滤掉 _CLOSE_DENY 里的系统进程）。"""
        n = 0
        for e in exes:
            if any(d in e.lower() for d in self._CLOSE_DENY):
                continue
            n += len(self._find_windows(e))
        return n

    def _close_app(self, app: str) -> bool:
        """关闭一个程序：先发 WM_CLOSE，无效再聚焦窗口按 Alt+F4。

        为什么要两条路：Win32 程序（记事本、Dev-C++）收到 WM_CLOSE 就正常退出，
        而且不会碰到内容区；但 UWP 应用（计算器、设置）根本不响应 WM_CLOSE
        —— 真机实测发了 8 秒窗口还在。这类只能用"人所做的那个动作"：聚焦 + Alt+F4。

        遇到保存/确认对话框立刻停手 —— 不替用户决定丢不丢数据。
        """
        import ctypes
        import ctypes.wintypes

        user32 = ctypes.windll.user32
        key = self._norm_app_name(app)
        exes = list(self._PROC_MAP.get(key) or [])
        if not exes:
            exes = [key.lower()]
        self.log(f"[关闭] 目标「{app}」→ 进程关键词 {exes}")
        self._wait_until_idle()   # Alt+F4 那一步会抢焦点，你正在用电脑就先让路

        wins = []
        for e in exes:
            if any(d in e.lower() for d in self._CLOSE_DENY):
                continue
            wins.extend(self._find_windows(e))
        seen, uniq = set(), []
        for hwnd, rect in wins:
            if hwnd in seen:
                continue
            seen.add(hwnd)
            uniq.append((hwnd, rect))
        if not uniq:
            self.log(f"[关闭] 没找到「{app}」的窗口（可能本来就没开）")
            return False

        # ---- 第一步：WM_CLOSE（等价于点右上角 X）----
        for hwnd, _rect in uniq:
            user32.PostMessageW(hwnd, 0x0010, 0, 0)
        for _ in range(3):
            time.sleep(0.4)
            if self._is_save_dialog():
                self.log("[关闭] 程序弹出了保存/确认对话框 —— 不替你决定，已停手")
                return False
            if not self._alive_windows(exes):
                self.log(f"[关闭] ✅ 「{app}」已关闭（WM_CLOSE）")
                if getattr(self, "_last_opened_app", None) == app:
                    self._last_opened_app = None   # 关掉了，防抖记录跟着清掉
                return True

        # ---- 第二步：UWP 不响应 WM_CLOSE，改成聚焦 + Alt+F4 ----
        self.log(f"[关闭] WM_CLOSE 没生效（多半是 UWP 应用），改用「点窗口 + Alt+F4」")
        import pyautogui
        for hwnd, rect in uniq:
            try:
                if not user32.IsWindow(hwnd):
                    continue
                cx = int((rect[0] + rect[2]) / 2)
                cy = int((rect[1] + rect[3]) / 2)
                pyautogui.click(cx, cy)      # 先激活它（Alt+F4 只对前台生效）
                time.sleep(0.5)
                if self._is_save_dialog():
                    self.log("[关闭] 点下去弹出了对话框，停手，交给你处理")
                    return False
                pyautogui.hotkey("alt", "f4")
                time.sleep(1.0)
                if self._is_save_dialog():
                    self.log("[关闭] 弹出了保存/确认对话框 —— 不替你决定，已停手")
                    return False
            except Exception as exc:
                self.log(f"[关闭] 关闭窗口时出错（{exc}）")
                continue
            for _ in range(3):
                time.sleep(0.4)
                if not self._alive_windows(exes):
                    self.log(f"[关闭] ✅ 「{app}」已关闭（Alt+F4）")
                    return True
        self.log(f"[关闭] 「{app}」还有窗口没关掉（可能被最小化到托盘了）")

        # ---- 第三步：无状态 UWP 应用走强制结束 ----
        # 为什么允许强制：计算器/设置这类程序没有任何用户数据可丢；
        # 而 Windows 11 的 UWP 窗口对 WM_CLOSE / SC_CLOSE / Alt+F4 全部无反应，
        # 真机实测三条路都试过，窗口纹丝不动。再加一层白名单兜底，且只对白名单。
        try:
            import subprocess
            u2 = ctypes.windll.user32
            for hwnd, _rect in uniq:
                if not u2.IsWindow(hwnd):
                    continue
                pid = ctypes.wintypes.DWORD()
                u2.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                exe = self._exe_of_pid(pid.value)
                if not any(s in exe for s in self._CLOSE_STATELESS):
                    continue
                self.log(f"[关闭] {exe} 是无状态应用，强制结束（不会丢数据）")
                subprocess.run(["taskkill", "/F", "/PID", str(pid.value)],
                               capture_output=True, timeout=8)
            for _ in range(6):
                time.sleep(0.4)
                if not self._alive_windows(exes):
                    self.log(f"[关闭] ✅ 「{app}」已关闭（强制结束）")
                    return True
        except Exception as exc:
            self.log(f"[关闭] 强制结束这一步出错（{exc}）")
        self.log(f"[关闭] 「{app}」还是没关掉")
        return False

    @staticmethod
    def _exe_of_pid(pid: int) -> str:
        """按 PID 取 exe 文件名（小写）。取不到给空串。"""
        try:
            import ctypes
            import ctypes.wintypes as wt
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x1000, False, int(pid))
            if not h:
                return ""
            buf = ctypes.create_unicode_buffer(600)
            n = wt.DWORD(600)
            ok = k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n))
            k.CloseHandle(h)
            return buf.value.rsplit("\\", 1)[-1].lower() if ok else ""
        except Exception:
            return ""

    # 认得出是"文件"而不是"程序"的扩展名。没有这一条，
    # 「打开桌面上的 报告.txt」会被当成打开一个叫 报告.txt 的程序，必然失败。
    _FILE_EXTS = (".txt", ".md", ".c", ".cpp", ".h", ".py", ".js", ".json", ".csv",
                  ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pdf",
                  ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".mp4", ".mp3", ".wav",
                  ".zip", ".rar", ".7z", ".exe", ".lnk", ".log")
    _FILE_OPEN_RE = re.compile(
        r"^(?:请|帮我|帮忙|麻烦)?\s*(?:打开|查看|显示|用默认程序打开)\s*(?:一下)?\s*"
        r"(.+?)\s*[。.!！]?$")

    @classmethod
    def _parse_file_intent(cls, task: str):
        """「打开 报告.txt」→ 文件名。带已知扩展名才算，避免把程序名当文件。"""
        t = (task or "").strip()
        if not t or len(t) > 60:
            return None
        m = cls._FILE_OPEN_RE.match(t)
        if not m:
            return None
        cand = m.group(1).strip().strip("「」\"'《》")
        if not cand:
            return None
        # 「桌面上的 test.md」→ 去掉中文所有格，只留文件名
        cand = re.split(r"(?:桌面|下载|文档|文件|文件夹|磁盘)?(?:上|里|中)的\s*", cand)[-1].strip()
        if not cand:
            return None
        low = cand.lower()
        if not any(low.endswith(e) for e in cls._FILE_EXTS):
            return None
        return cand

    def _resolve_existing_file(self, name: str):
        """把相对文件名解析成真实路径：绝对路径 → 桌面 → 下载 → 文档 → 当前目录。"""
        cand = (name or "").strip().strip("「」\"'《》")
        if not cand:
            return None
        if os.path.isabs(cand):
            return cand if os.path.exists(cand) else None
        bases = [self._desktop_dir(),
                 os.path.join(os.path.expanduser("~"), "Downloads"),
                 os.path.join(os.path.expanduser("~"), "Documents"),
                 os.getcwd()]
        for b in bases:
            p = os.path.join(b, cand)
            if os.path.exists(p):
                return p
        return None

    def _open_file(self, path: str) -> bool:
        """用默认程序打开一个文件（等价于用户双击它）。"""
        import os
        import subprocess
        self.log(f"[打开] 用默认程序打开 {path}")
        try:
            os.startfile(path)      # noqa: S606  (Windows 上就是"双击"的语义)
        except Exception:
            try:
                subprocess.Popen(["cmd", "/c", "start", "", path])
            except Exception as exc:
                self.log(f"[打开] 打不开（{exc}）")
                return False
        time.sleep(1.8)
        title = self._window_info()[0] or ""
        leaf = os.path.basename(path)
        if leaf and leaf in title:
            self.log(f"[打开] ✅ 已打开「{title}」")
            return True
        self.log(f"[打开] 已发出打开请求，当前前台是「{title}」")
        return True

    # 这些词后面跟的通常不是「程序」，而是文件/网页/菜单，交给通用循环更合适
    _OPEN_STOPWORDS = ("文件", "文件夹", "目录", "网页", "网址", "链接", "图片", "照片",
                       "文档", "窗口", "菜单", "设置", "这里", "它", "那个", "这个")
    # 出现这些词说明是「打开 X 之后还要做事」，不能只做前半句
    _OPEN_COMPOSITE = ("并", "然后", "接着", "再", "输入", "保存", "关闭", "点击",
                       "双击", "发送", "写", "打", "搜", "找")

    @classmethod
    def _parse_open_intent(cls, task: str):
        """「打开/启动/运行 X」且整句只有一个打开动作 —— 直接走 open_app 模板。
        带附加动作（"打开记事本并写..."）不接管，否则后半句会被丢掉。
        返回程序名或 None。"""
        import re
        t = (task or "").strip().strip("。.!！?？")
        if not t or len(t) > 20:
            return None
        m = re.fullmatch(
            r"(?:(?:请|帮我|帮忙|麻烦)\s*){0,3}(?:打开|启动|运行|开启|调出)\s*"
            r"(?:一下)?\s*([^\s，,。；;、]{1,12})", t)
        if not m:
            return None
        app = m.group(1).strip()
        for suffix in ("程序", "软件", "应用"):   # "运行记事本程序" -> "记事本"
            if app.endswith(suffix) and len(app) > len(suffix):
                app = app[: -len(suffix)]
        if not app or len(app) > 6:
            return None                            # 长的多半是一句话，不是一个程序名
        if re.search(r"[A-Za-z]:[\\/]", app):
            return None                            # "打开D:\测试" 是路径不是程序名，交给导航模板
        if any(k in app for k in cls._OPEN_STOPWORDS):
            return None
        if any(k in app for k in cls._OPEN_COMPOSITE):
            return None
        return app

    # ==================== 「打开 X 并输入文字」复合模板 ====================
    # 为什么需要：这类任务不属于任何一个单动作模板（_parse_open_intent 见到
    # 「输入」就主动让路，怕丢后半句），于是一路落进通用循环。真机实测：
    # 「打开记事本，输入三行字：春天来了、花开满树、燕子归来，每行一个」
    # 在通用循环里 5 步 108 秒，模型反复 open_app（前台已经是记事本还发），
    # 同时把「三行字：……每行一个」整句当正文理解错，一个字都没敲对。
    # 动作链其实是纯确定的：开程序 → Ctrl+N 新文档 → 点编辑区 → 逐行敲 → OCR 验收。
    _TYPE_VERB = r"(?:输入|打字|敲入|敲上|敲|写入|录入|填写|填上|写上|写下)"
    _TYPE_RE = re.compile(
        r"^(?:(?:请|帮我|帮忙|麻烦)\s*){0,3}(?:打开|启动|运行|开启|调出)\s*(?:一下)?\s*"
        r"(?P<app>[^\s，,。；;、]{1,14}?)\s*(?:程序|软件|应用)?\s*[，,、]?\s*"
        r"(?:并|并且)?\s*(?:然后|接着|之后|再)?\s*(?:在里面|在它里面|在其中)?\s*"
        + _TYPE_VERB + r"\s*(?:一下)?\s*[：:]?\s*(?P<text>.+)$", re.S)

    # 聊天软件不走这条路：往里打字等于替用户发消息，必须走白名单发消息模板
    _TYPE_APP_DENY = ("微信", "wechat", "weixin", "qq", "钉钉", "飞书")
    # 正文里出现这些词说明是「让它创作」，不是「照抄这几个字」，交给通用循环/写码模板
    _TYPE_GEN_HINT = ("一首", "一篇", "一段诗", "诗", "故事", "文章", "作文", "代码",
                      "程序", "函数", "算法")
    # 正文里出现这些词说明后面还挂着别的动作（"输入第一行然后保存"），只做前半句会丢需求
    _TYPE_TAIL_DENY = ("然后", "接着", "之后", "保存", "发送", "关闭", "删除",
                       "点击", "双击", "粘贴", "复制")

    @classmethod
    def _clean_type_text(cls, raw: str):
        """把「三行字：春天来了、花开满树、燕子归来，每行一个」洗成真正的正文。

        用户说的话里有一大半是"格式说明"（三行字 / 每行一个 / 冒号），
        照原样敲进去就是一堆废话——真机日志里模型就是这么把提示语一起打上去的。
        """
        import re as _re
        s = (raw or "").strip()
        if not s:
            return ""
        # 1) 掐掉「三行字：」「以下内容：」「内容如下：」这类前缀
        s = _re.sub(r"^(?:如下|以下)\s*(?:的)?\s*(?:内容|文字|这些字|信息)?\s*[：:，,]?\s*", "", s)
        s = _re.sub(r"^(?:[一二两三四五六七八九十几\d]+行字|几行字|"
                    r"[一二两三四五六七八九十几\d]+行|一行|"
                    r"[一二两三四五六七八九十几\d]+句(?:话)?|几句话|"
                    r"一段(?:话|文字|内容)?|这些字)\s*[：:，,]?\s*", "", s)
        # 2) 去掉包裹的引号
        s = s.strip().strip("「」“”\"'《》").strip()
        # 3) 拆掉尾巴上的格式说明
        want_lines = bool(_re.search(r"每行(?:一个|一行|一句|一条)|一行一个|分行|换行|另起一行", s))
        s = _re.sub(r"[，,、；;]?\s*(?:每行(?:一个|一行|一句|一条)|一行一个|分行写?|换行|"
                    r"另起一行)[。.!！]?\s*$", "", s)
        s = _re.sub(r"^(?:内容|文字)?\s*(?:是|为)\s*[：:]?\s*", "", s)
        s = s.strip(" ，,。；;、")
        if not s:
            return ""
        # 4) 多段就用换行连起来（用户说「每行一个」时，中文逗号也算分行符）
        sep = r"[、；;，,]" if want_lines else r"[、；;]"
        parts = [p.strip() for p in _re.split(sep, s) if p.strip()]
        if len(parts) > 1 and (want_lines or len(parts) <= 8):
            s = "\n".join(parts)
        return s

    @classmethod
    def _parse_open_and_type_intent(cls, task: str):
        """「打开记事本，输入 XXX」→ {"app","text"}；不是这类任务返回 None。"""
        t = (task or "").strip().strip("。.!！")
        if not t or len(t) > 400:
            return None
        m = cls._TYPE_RE.match(t)
        if not m:
            return None
        app = m.group("app").strip().strip("「」\"'《》")
        for suffix in ("程序", "软件", "应用"):
            if app.endswith(suffix) and len(app) > len(suffix):
                app = app[: -len(suffix)]
        if not app or len(app) > 10:
            return None
        if any(k in app.lower() for k in cls._TYPE_APP_DENY):
            return None                 # 聊天软件：交给发消息模板 + 白名单
        if any(k in app for k in cls._OPEN_STOPWORDS):
            return None
        if any(k in app for k in ("关闭", "保存", "删除", "发送", "双击")):
            return None                 # 程序名里混进了别的动作，说明这不是"打开X再输入"
        text = cls._clean_type_text(m.group("text"))
        if not text or len(text) > 1500:
            return None
        if any(k in text for k in cls._TYPE_GEN_HINT):
            return None                 # 「写一首诗」= 创作，不是打字
        if any(k in text for k in cls._TYPE_TAIL_DENY):
            return None                 # 后面还挂着别的动作，别只做前半句
        return {"app": app, "text": text}

    _CLOSE_RE = re.compile(
        r"^(?:(?:请|帮我|帮忙|麻烦)\s*){0,3}(?:关闭|关掉|关一下|退出|结束)\s*(?:一下)?\s*"
        r"([^\s，,。；;、]{1,12})\s*(?:程序|软件|应用)?\s*[。.!！]?$")

    @classmethod
    def _parse_close_intent(cls, task: str):
        """「关闭记事本」/「关掉计算器」→ 程序名。只认整句，不认"关掉之后再..."。"""
        t = (task or "").strip()
        if not t or len(t) > 20:
            return None
        m = cls._CLOSE_RE.match(t)
        if not m:
            return None
        app = m.group(1).strip()
        if not app or any(k in app for k in ("电脑", "计算机", "系统", "窗口", "页面",
                                             "标签", "文件", "文件夹", "所有")):
            return None   # "关闭窗口"/"关闭所有程序" 不接管
        # 复合任务只做前半句会把后半句丢掉（"关闭记事本然后打开画图"）
        if any(k in app for k in ("然后", "接着", "之后", "并", "再", "，")):
            return None
        return app

    _SEARCH_RE = re.compile(
        r"(?:请|帮我|帮忙|麻烦)?\s*(?:在?浏览器(?:里|中|里面)?|打开浏览器|去浏览器)?\s*(?:去)?\s*"
        r"(?:搜索|搜一下|搜下|搜一搜|搜|"
        # 单独一个「查」也算（「在浏览器里查今天的天气」真机就漏在这）：
        # 但「检查/排查/审查/复查/调/核查」里的查不是搜索，
        # 「查收/查看/查找」是操作本机，都要排掉。负向后顾要盖住整组，
        # 否则「排查一下」会被「查一下」这一支偷偷放进来。
        r"(?<![检排审复调核])(?:查一下|查下|查一查|查询|查(?!收|看|找))|"
        r"百度一下|百度|谷歌一下|谷歌)\s*"
        r"[：:，,]?\s*(.+?)\s*$")

    @classmethod
    def _parse_search_intent(cls, task: str):
        """「打开浏览器搜索 X」/「查一下 X」→ 直接给出搜索词，走浏览器搜索模板。

        为什么要在程序里做：真机跑「打开浏览器搜索今天的天气」，8 步 138 秒没完成 ——
        模型不叫 open_app，而是去任务栏图标上连点两次（同一位置），再在页面上乱点，
        最后也没输出 done。搜索是纯确定性的四步（开浏览器 / ctrl+L / 打字 / 回车），
        交给程序 5 秒内稳过，而且全程还是真人那样在操作界面。
        """
        t = (task or "").strip().strip("。.!！?？")
        if not t or len(t) > 40:
            return None
        m = cls._SEARCH_RE.search(t)
        if not m:
            return None
        query = m.group(1).strip().strip("。.!！?？吧呢啊 ")
        if not query or len(query) > 40:
            return None
        # 搜的是本机文件/路径 → 那是导航或查找，不是网页搜索
        if re.search(r"[A-Za-z]:[\\/]|\\\\", query) or query.startswith(("D:", "C:")):
            return None
        if any(k in query for k in ("文件", "文件夹", "目录", "路径")):
            return None
        if any(k in query for k in cls._OPEN_COMPOSITE):
            return None
        return query

    def _browser_search(self, query: str, kind: str = "搜索", expect: str = "") -> bool:
        """用「人」的方式搜：打开浏览器 → ctrl+L 聚焦地址栏 → 逐字输入 → 回车。

        kind="网址" 时用同一套动作链打开一个域名（输入的是地址不是搜索词），
        expect 给窗口标题里应该出现的片段（通常是域名主体）。
        """
        import pyautogui
        from agent.actions import _type_unicode

        _t0 = time.time()
        if not self._open_app("浏览器"):
            self.log(f"[{kind}] 浏览器没打开，模板放弃")
            return False
        self._gui_home_set()
        time.sleep(0.8)
        pyautogui.hotkey("ctrl", "l")     # 地址栏全选
        time.sleep(0.5)
        _type_unicode(query, interval=0.04)
        time.sleep(0.3)
        pyautogui.press("enter")
        self.log(f"[{kind}] 已在浏览器里{kind}「{query}」"
                 f"（{time.time() - _t0:.1f}s）")
        # 等结果页出来：标题变了 / 画面变了都算成功，避免"提交了但没加载"就报完成
        before, _ = self._window_info()
        self._wait_screen_change(self.screenshot_fn(), timeout=6.0)
        after, _ = self._window_info()
        probe = expect or query[:4]
        if not (after and (probe in (after or "") or after != before)):
            # 有些站点首屏慢（JS 渲染完才改标题），再给 2 秒，别急着报"没确认到"
            time.sleep(2.0)
            after, _ = self._window_info()
        if after and (probe in (after or "") or after != before):
            self.log(f"[{kind}] 前台窗口：{after}")
            return True
        self.log(f"[{kind}] 结果页没确认到，但请求已经提交")
        return True

    # 网址识别：只认常见顶级域，避免把「main.c」「3.14」当成地址
    _URL_RE = re.compile(
        r"((?:https?://)?(?:[\w-]+\.)+(?:com|cn|net|org|io|gov|edu|co|top|xyz|info|me|cc|tv)"
        r"(?:/[^\s，。；;、\"'）】]*)?)", re.I)

    @classmethod
    def _parse_url_intent(cls, task: str):
        """「打开 www.qq.com」/「访问 https://www.baidu.com」→ 返回补全后的 URL。

        为什么要有：让模型自己"在地址栏里敲网址"它经常敲成搜索词，
        或者干脆点收藏夹。打开网址和搜索是同一条确定动作链，只是输的内容不同。
        """
        t = (task or "").strip()
        if not t:
            return None
        if not re.search(r"打开|访问|进入|去|浏览|看看|上", t):
            return None
        m = cls._URL_RE.search(t)
        if not m:
            return None
        url = m.group(1).rstrip("。.,，、")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        host = url.split("//", 1)[-1].split("/", 1)[0]
        # 域名主体（去掉 www.）用于验证窗口标题
        body = host[4:] if host.lower().startswith("www.") else host
        return url, body.split(".")[0]


    _NAV_VERBS = ("打开", "进入", "前往", "导航到", "跳到", "去")
    _NAV_PATH_RE = re.compile(
        r"(?:打开|进入|前往|导航到|跳到|去|在)\s*"
        r"([A-Za-z]):\s*[\\/]?\s*([^，。；;？！?\"'）)】]*)")
    _NAV_DRIVE_RE = re.compile(r"(?:打开|进入|前往|导航到|跳到|去|在)\s*([A-Za-z])\s*盘(?![一-龥A-Za-z0-9])")

    # 常见目录别名：「在桌面新建一个文件夹叫报告」不需要盘符，直接映射到用户目录。
    # 只认「动词 + 目录名」这种明确说法，「桌面上那个文件」不会误命中。
    _NAV_ALIAS = {
        "桌面": "Desktop", "文档": "Documents", "下载": "Downloads",
        "图片": "Pictures", "视频": "Videos", "音乐": "Music",
    }
    _NAV_ALIAS_RE = re.compile(
        r"(?:打开|进入|前往|导航到|跳到|去|在)\s*(桌面|文档|下载|图片|视频|音乐)"
        r"(?:文件夹|目录)?")

    @classmethod
    def _parse_nav_task(cls, task: str):
        """「打开D:\测试，在里面新建一个文件」→ ("D:\\测试", "在里面新建一个文件")。

        为什么要有这个模板：让 7B 在资源管理器里"自己走到某个路径"，
        真机 40 步 579 秒全花在左侧导航栏空点（x 在 20~132 之间猜了 15 次），
        它不知道地址栏可以输入路径，也不知道 Win+E。路径导航是纯确定性操作，
        和 open_app 一个性质：程序做，模型只做剩下真正需要理解的部分。
        """
        t = (task or "").strip()
        if not t:
            return None
        m = cls._NAV_PATH_RE.search(t)
        if not m:
            m2 = cls._NAV_DRIVE_RE.search(t)
            if not m2:
                # 第三档：桌面/文档/下载… 这些目录没有盘符，靠别名映射。
                # 「在桌面新建一个文件夹叫报告」走这里，从几十步压成 1 步。
                m3 = cls._NAV_ALIAS_RE.search(t)
                if not m3:
                    return None
                folder = cls._NAV_ALIAS.get(m3.group(1))
                if not folder:
                    return None
                path = os.path.join(os.path.expanduser("~"), folder)
                remainder = (t[:m3.start()] + t[m3.end():]).strip()
                remainder = remainder.lstrip("里中，。；;、, 的").strip()
                if remainder.startswith(("然后", "再", "接着", "之后")):
                    remainder = remainder[2:].lstrip("，、 ").strip()
                return path, remainder
            path = f"{m2.group(1).upper()}:\\"
            remainder = (t[:m2.start()] + t[m2.end():]).strip()
        else:
            drive, rest = m.group(1).upper(), m.group(2).strip()
            prefix, suffix = t[:m.start()], t[m.end():]
            # 「在D:\测试里新建一个文件」：贪婪的路径段会把"里新建..."吞进路径。
            # 只有「里/中」后面跟动词（新建/创建/放/存）才截断——文件夹名里
            # 也可能有"里"字（D:\阿里云盘），不能一刀切。
            m_cut = re.search(r"[里中](?=新建|创建|放|存)", rest)
            if m_cut:
                suffix = rest[m_cut.start():] + suffix
                rest = rest[:m_cut.start()]
            # "打开D:\测试" 的反斜杠经常被吃掉（"打开D:测试"），兼容两种写法
            rest = rest.strip("\\/").rstrip(".。,， ")
            path = f"{drive}:\\" + rest.replace("/", "\\") if rest else f"{drive}:\\"
            remainder = (prefix + suffix).strip()
        remainder = remainder.lstrip("里，。；;、, ").strip()
        if remainder.startswith(("然后", "再", "接着", "之后")):
            remainder = remainder[2:].lstrip("，、 ").strip()
        return path, remainder

    @staticmethod
    def _parse_message_intent(task: str):
        """从任务文本里解析「给某人发消息」意图，命中则程序模板直通，不让 7B 逐步决策。
        支持的句式（例）：
          在微信里给文件传输助手发送一条消息：你好
          给文件传输助手发：你好
          对文件传输助手说你好
        返回 {"app","target","text"} 或 None。"""
        import re
        t = (task or "").strip()
        if not t:
            return None
        app = next((a for a in ("企业微信", "微信", "QQ", "钉钉", "飞书", "Telegram",
                                "Discord") if a in t), None)
        # 没写明用哪个软件就默认微信（主要场景），不因此放弃整条解析
        app = app or "微信"
        # 消息内容：优先冒号/引号
        m_text = re.search(r"[：:]\s*(.+?)\s*$", t)
        # 会话目标：「给 X 发」「对 X 说」「跟 X 说一声」「告诉 X」
        m_target = re.search(r"给\s*(.+?)\s*(?:发送|发条|发一条|发个|发)", t) \
            or re.search(r"对\s*(.+?)\s*说", t) \
            or re.search(r"向\s*(.+?)\s*(?:发送|发)", t) \
            or re.search(r"(?:跟|和|与)\s*(.+?)\s*(?:说一声|说一下|说句|讲一声|讲一下|聊聊)", t) \
            or re.search(r"(?:告诉|通知|转告)\s*(.+?)\s*(?:[：:，,]|\s|$)", t)
        if not m_target:
            return None
        target = m_target.group(1).strip().strip('「」""\'')
        text = ""
        if m_text:
            text = m_text.group(1).strip()
        else:
            # 「发消息」必须排在单个「发」前面：否则「给示例联系人A发消息说你好」会
            # 从「发」开始捕获成「消息说你好」，把固定词组吃进正文（真机踩过）。
            # 捕获前再吃掉紧随其后的「说/讲/告诉」，剩下的才是真正要说的话。
            m2 = re.search(
                r"(?:说一声|说一下|说句|讲一声|讲一下|告诉|通知|转告|发送|"
                r"发条|发一条|发个|发消息|发|说)"
                r"\s*(?:说|讲|告诉)?\s*[「『\"']?(.+?)[」』\"']?\s*$", t)
            if m2:
                text = m2.group(1).strip()
        text = text.strip('「』『』""\'。.')
        text = clean_message_text(text)
        if not (target and text):
            return None
        # 过滤明显误匹配：目标是任务描述性长句（>20 字）多半不是会话名
        if len(target) > 20 or len(text) > 120:
            return None
        return {"app": app, "target": target, "text": text}

    # 收信人：先抓「发给X」（最自然的口语），再兼容「给X发」「对X说」「告诉X」
    _RECIPIENT_RES = (
        re.compile(r"(?:发送给|传给|发给|发到)\s*([^\s，,。；;：:、]{1,16})"),
        re.compile(r"给\s*(.+?)\s*(?:发送|发条|发一条|发个|发)"),
        re.compile(r"对\s*(.+?)\s*说"),
        re.compile(r"向\s*(.+?)\s*(?:发送|发)"),
        re.compile(r"(?:告诉|通知|转告)\s*(.+?)\s*(?:[：:，,]|\s|$)"),
    )
    _APP_NAMES = ("企业微信", "微信", "QQ", "钉钉", "飞书", "Telegram", "Discord")

    @classmethod
    def _parse_recipient(cls, task: str):
        """抽出 (app, 收信人)。抽不到收信人返回 (app, None)。"""
        t = (task or "").strip()
        app = next((a for a in cls._APP_NAMES if a in t), None) or "微信"
        for rx in cls._RECIPIENT_RES:
            m = rx.search(t)
            if not m:
                continue
            name = m.group(1).strip().strip(_QUOTE_CHARS)
            # 「把答案发给示例联系人A」这类：后面不会再有尾巴；但「发给文件传输助手吧」要削掉语气词
            name = re.sub(r"(吧|啊|呢|哦|呀|一下|一条消息|消息)$", "", name).strip()
            if name and len(name) <= 20:
                return app, name
        return app, None

    @classmethod
    def _parse_calc_send_intent(cls, task: str):
        """「算出 X 再发给 Y」一条龙：算术归程序，发送归已验证的发消息模板。

        判据三条同时成立才接管（缺一条就走通用循环，不硬猜）：
          ① 句子里有计算意图词；② 抠得出合法算式；③ 说得清收信人。
        返回 {"expr","app","target"} 或 None。
        """
        t = (task or "").strip()
        if not t or not any(w in t for w in _CALC_WORDS):
            return None
        expr = _parse_expr(t)
        if not expr:
            return None
        app, target = cls._parse_recipient(t)
        if not target:
            return None
        return {"expr": expr, "app": app, "target": target}

    # ==================== 「打开计算器算一道题」（不算发给谁） ====================
    # 「打开计算器，算128*64」三个判据缺了收信人，原路由直接放走 → 通用循环。
    # 真机教训：通用循环会在记事本里敲"128*64="，计算器一次都没开。

    @classmethod
    def _parse_calc_show_intent(cls, task: str):
        """「算 X」但没说发给谁 → 打开计算器把算式敲进去，结果留在屏幕上。"""
        t = (task or "").strip()
        if not t or not any(w in t for w in _CALC_WORDS):
            return None
        expr = _parse_expr(t)
        if not expr:
            return None
        _app, target = cls._parse_recipient(t)
        if target:
            return None            # 有收信人 → 走 calc_then_send
        return {"expr": expr}

    def _calc_show(self, expr: str) -> bool:
        """程序求值 → 计算器里演一遍 → OCR 验收结果真的出现在屏幕上。"""
        try:
            value = _safe_eval(expr)
        except Exception as exc:
            self.log(f"[计算] 表达式「{expr}」算不出来（{exc}）")
            return False
        shown = _normalize_expr(expr).replace("*", "×").replace("/", "÷").replace("**", "^")
        result = _fmt_number(value)
        self.log(f"[计算] {shown}={result}（程序 AST 求值，不经模型）")
        if not self._show_in_calculator(expr, f"{shown}={result}"):
            return False
        time.sleep(0.8)
        # 验收：OCR 找结果数字（计算器可能带千分位逗号，两种形态都认）
        for probe in {result, f"{int(value):,}" if abs(value - round(value)) < 1e-9 else result}:
            if self._screen_has_text(probe):
                self.log(f"[计算] ✅ 屏幕上读到了结果 {probe}")
                return True
        self.log("[计算] ⚠️ OCR 没认出结果数字（计算器可能被挡住），但算式已敲入")
        return True

    def _show_in_calculator(self, expr: str, body: str) -> bool:
        """把算式真的敲进系统计算器 —— 让用户看得见"它确实算过"。
        纯附加步骤：失败只记日志，不影响已经求出的结果和随后要发的消息。"""
        try:
            if not self._focus_program("计算器"):
                if not self._open_app("计算器"):
                    self.log("[计算器] 打不开系统计算器，跳过展示（结果由程序求得，不受影响）")
                    return False
                self._focus_program("计算器")
            time.sleep(0.5)
            # 先 Esc 清状态：计算器会保留上一次的结果，不清的话这次敲的数字
            # 会接到旧结果后面（真机：128*64 之后算 256*32 得出 25,624,576）
            pyautogui.press("esc")
            time.sleep(0.3)
            # 用虚拟键码写而不是 UNICODE 注入：Win11 计算器是 UWP，虚拟键更稳
            pyautogui.write(_normalize_expr(expr), interval=0.03)
            pyautogui.press("enter")      # 计算器里回车 = 等号
            time.sleep(0.5)
            self.log(f"[计算器] 已打开计算器、键入「{_normalize_expr(expr)}」并回车；"
                     f"屏幕结果应与 {body} 一致")
            return True
        except Exception as exc:
            self.log(f"[计算器] 展示失败（{exc}），不影响结果")
            return False

    def _calc_then_send(self, expr: str, app: str, target: str) -> bool:
        """算 → 在计算器里演一遍 → 发送。

        顺序按用户的原话「先算出来，再发」：算完先把计算器摆到屏幕上（看得见的证据），
        再去发消息。计算器失败不阻塞发送。
        """
        try:
            value = _safe_eval(expr)
        except Exception as exc:
            self.log(f"[计算] 表达式「{expr}」算不出来（{exc}）")
            return False
        shown = _normalize_expr(expr).replace("*", "×").replace("/", "÷").replace("**", "^")
        num = _fmt_number(value)
        body = f"{shown}={num}"
        self.log(f"[计算] {body}（程序 AST 求值，不经模型）")
        self._show_in_calculator(expr, body)
        return self._send_message(app, target, body)

    # ==================== 写代码任务（Dev-C++） ====================
    # 为什么必须是模板：真机任务「打开dev-c++，写一个求15以内阶乘的函数。
    # 命名为"阶乘"保存到桌面」丢进通用循环后，7B 第1步给了 element_id:[34]
    # （列表形式，程序没解析出来），随后在控制台页面同一个点上连点 6 次，
    # Dev-C++ 一次都没打开。这类任务有 4 个确定性子步骤（写码 / 新建 / 输入 / 另存），
    # 模型逐个做必错，交给程序串联、模型只负责"写什么代码"。

    _CODE_APP_RE = re.compile(r"dev[\s\-_]*c\s*\+{1,2}|dev[\s\-_]*cpp|devc\+\+|red\s*panda", re.I)
    _CODE_WORD_RE = re.compile(r"写|编|代码|函数|程序|实现|算法")
    # 「用 C 写一个程序」这种没点名编辑器的说法，也认成写码任务：
    # 现场演示时评委会直接说「写个 C 程序算阶乘」，不会先说"打开 Dev-C++"。
    # 不接管的话就掉进通用循环 —— 而通用循环在这类任务上从来没成功过。
    _CODE_LANG_RE = re.compile(
        r"c\s*语言|c\+\+|(?<![a-z])cpp(?![a-z])|c\s*程序|c\s*代码|"
        r"(?:用|以|使用|通过)\s*c\s*(?:来|写|编写|实现|编)", re.I)
    # 明确点名了别的语言/编辑器就别抢活
    _CODE_OTHER_RE = re.compile(
        r"python|java\b|javascript|pycharm|记事本|写字板|vscode|vs\s*code|word\b", re.I)
    _CODE_NAME_RE = re.compile(
        r"(?:命名为|保存为|文件名为|文件名叫|取名为|名字叫|叫做|叫)\s*"
        r"[「“\"'《]?\s*([^\s，,。；;、」”\"'》]{1,40})")

    @classmethod
    def _parse_code_task(cls, task: str):
        """识别「在 Dev-C++ 里写代码并存成文件」。返回
        {"app","name","requirement","to_desktop"} 或 None（None = 不接管）。"""
        t = (task or "").strip()
        if not t:
            return None
        named_app = bool(cls._CODE_APP_RE.search(t))
        if not named_app:
            # 没点名 Dev-C++：只有"确实在说 C 语言 + 写代码"才接管
            if not cls._CODE_LANG_RE.search(t):
                return None
            if cls._CODE_OTHER_RE.search(t):
                return None
            # 复合任务不接管（"写个冒泡排序发给示例联系人A"= 写码 + 发消息，只做前半句会丢需求）
            if re.search(r"发给|发送|告诉|然后|接着|并\s*发", t):
                return None
        if not cls._CODE_WORD_RE.search(t):
            return None

        # 文件名：命名为「阶乘」/ 保存为 阶乘 / 文件名叫 阶乘
        name = ""
        m = cls._CODE_NAME_RE.search(t)
        if m:
            name = m.group(1).strip("。.，,；;的 ")
            name = re.sub(r"\.(c|cpp|cc|h)$", "", name, flags=re.I)  # 用户写了后缀就先剥掉
        if not name:
            name = "main"          # 没给名字就用 main，别失败

        # 需求正文：掐掉「打开 dev-c++，」前缀和「命名为…保存到桌面」尾巴
        req = t
        req = re.sub(r"^\s*(?:请|帮我|麻烦)?\s*(?:先)?\s*(?:打开|启动|运行|开)\s*"
                     r"dev[\s\-_]*c\s*\+{1,2}[^，,。；;]*[，,。；;:：]?\s*", "", req, flags=re.I)
        req = re.sub(r"^\s*(?:请|帮我|麻烦)?\s*(?:在|用)\s*dev[\s\-_]*c\s*\+{1,2}\s*"
                     r"(?:里|中|上)?\s*[，,。；;:：]?\s*", "", req, flags=re.I)
        if not named_app:
            # 「用C语言写一个程序，求…」→ 只留「求…」，需求越干净生成得越准
            req = re.sub(r"^\s*(?:请|帮我|麻烦)?\s*(?:用|以|使用|通过)?\s*c\s*(?:语言)?\s*"
                         r"(?:来)?\s*(?:写|编写|实现|编)\s*(?:一个|个|一段|段)?\s*"
                         r"(?:c\s*)?(?:程序|代码|函数)?\s*[，,。；;:：]?\s*", "", req, flags=re.I)
            # 「写个C语言程序计算阶乘」= 动词在前、语言在后
            req = re.sub(r"^\s*(?:请|帮我|麻烦)?\s*(?:写|编写|实现|编)\s*(?:一个|个|一段|段)?\s*"
                         r"c\s*(?:语言)?\s*(?:程序|代码|函数)?\s*[，,。；;:：]?\s*", "", req, flags=re.I)
        req = re.sub(r"[，,。；;]?\s*(?:并)?\s*(?:命名为|保存为|文件名为|文件名叫|取名为|名字叫).*$", "", req)
        req = re.sub(r"[，,。；;]?\s*(?:并)?\s*(?:保存|存|放)(?:到|至|在)?\s*[^，,。；;]*[。.]?$", "", req)
        req = req.strip(" ，,。；;")
        if len(req) < 2:
            req = t            # 掐完什么都不剩就用整句，宁可啰嗦也别丢需求

        to_desktop = "桌面" in t
        return {"app": "Dev-C++", "name": name, "requirement": req,
                "to_desktop": to_desktop}

    @staticmethod
    def _clean_code(raw: str) -> str:
        """把模型输出擦成纯代码：去 markdown 围栏、去前后解释、统一换行。"""
        s = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
        fence = re.search(r"```[a-zA-Z+#]*\s*\n(.*?)```", s, re.S)
        if fence:
            s = fence.group(1)
        else:
            s = re.sub(r"^\s*```[a-zA-Z+#]*\s*$", "", s, flags=re.M)
            s = re.sub(r"^\s*```\s*$", "", s, flags=re.M)
        if "#include" in s:
            s = s[s.index("#include"):]          # 围栏外的前言一律砍掉
        lines = [ln.rstrip() for ln in s.split("\n")]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines) + "\n"

    @staticmethod
    def _desktop_dir() -> str:
        d = os.path.join(os.path.expanduser("~"), "Desktop")
        if os.path.isdir(d):
            return d
        d2 = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop")
        return d2 if os.path.isdir(d2) else d

    _GCC_CANDIDATES = (
        r"D:\dev-C++\Dev-Cpp\MinGW64\bin\gcc.exe",
        r"C:\Program Files (x86)\Dev-Cpp\MinGW64\bin\gcc.exe",
        r"C:\Program Files\Dev-Cpp\MinGW64\bin\gcc.exe",
    )
    _DEVCPP_CANDIDATES = (
        r"D:\dev-C++\Dev-Cpp\devcpp.exe",
        r"C:\Program Files (x86)\Dev-Cpp\devcpp.exe",
        r"C:\Program Files\Dev-Cpp\devcpp.exe",
    )

    @classmethod
    def _gcc_path(cls):
        return next((p for p in cls._GCC_CANDIDATES if os.path.exists(p)), None)

    def _compile_c(self, code: str, tag: str):
        """真编译一遍再往下走：模型给的代码有语法错就当场回填让它改，
        绝不让一段编译不过的代码进演示现场。返回 (ok, 报错文本)。"""
        import subprocess
        gcc = self._gcc_path()
        if not gcc:
            self.log("[编译] 没找到 gcc，跳过编译校验（只做代码生成）")
            return True, ""
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "code_build")
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, f"{tag}.c")
        exe = os.path.join(d, f"{tag}.exe")
        with open(src, "w", encoding="utf-8") as f:
            f.write(code)
        env = dict(os.environ)
        # 致命坑：MSYS2 版 MinGW 的 cc1.exe 是子进程，bin 不在 PATH 时它静默退出，
        # gcc 只返回失败、stderr 一个字都没有（看起来像"编译失败但没原因"）
        env["PATH"] = os.path.dirname(gcc) + os.pathsep + env.get("PATH", "")
        try:
            p = subprocess.run([gcc, src, "-std=c11", "-Wall", "-o", exe],
                               capture_output=True, text=True, timeout=45,
                               encoding="utf-8", errors="replace", env=env)
        except Exception as exc:
            self.log(f"[编译] 调用 gcc 失败：{exc}")
            return True, ""
        if p.returncode == 0:
            return True, (p.stderr or "").strip()
        return False, ((p.stderr or "") + (p.stdout or "")).strip()[:800]

    @staticmethod
    def _output_completeness(code: str, out: str):
        """粗查"建了 N 个节点、却只打印 M 个数"这种静默丢数据。

        真机教训：7 个节点的二叉树遍历输出 `4 2 5 1 3`（丢了 6、3、7），
        编译零警告、退出码 0、格式也像模像样 —— 只核对退出码会误判成成功。
        """
        n_alloc = len(re.findall(r"\b(?:malloc|calloc)\s*\(", code or ""))
        if n_alloc == 0:
            return True, ""
        m = len(re.findall(r"-?\d+", out or ""))
        if m < n_alloc:
            return False, (f"代码申请了 {n_alloc} 个节点/对象，却只输出了 {m} 个数字，"
                           f"很可能有节点没接上、或遍历漏了一部分")
        return True, ""

    @staticmethod
    def _output_suspicious(out: str) -> str:
        """抓"只打印一个 0"这种一眼就不对的交付。

        真机教训：百元买百鸡那稿把约束写成 5x+3y+2z==100 又要求 x+y+z==100，
        两式相减得 4x+2y+z=0，非负解只有 (0,0,0)，而它不满足总和 100 —— 零组解。
        程序老老实实输出了 0，编译零警告、退出码 0，只有人看一眼才知道错了。
        """
        lines = [l.strip() for l in (out or "").splitlines() if l.strip()]
        if len(lines) == 1 and re.fullmatch(r"-?0(?:\.0+)?", lines[0]):
            return ("程序只输出一个 0。算法题的结果极少正好是 0，"
                    "通常是约束写矛盾了（无解）或单价/系数抄错。")
        return ""

    def _run_c(self, tag: str, timeout: float = 6.0):
        """把编译产物跑一遍。返回 (退出码, 输出)。

        为什么必须跑：真机那道二叉树题**编译通过、运行却栈溢出崩溃**——
        模型用 malloc 建节点却没把 left/right 置 NULL，叶子节点的子指针是野指针，
        递归踩到垃圾地址直接 0xC00000FD。编译器永远发现不了这种错，只有跑一次才知道。
        """
        import subprocess
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "code_build")
        exe = os.path.join(d, f"{tag}.exe")
        if not os.path.exists(exe):
            return -1, "（没有生成可执行文件）"
        try:
            p = subprocess.run([exe], capture_output=True, text=True, timeout=timeout,
                               encoding="utf-8", errors="replace",
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return -1, "（运行超时，可能有死循环）"
        except Exception as exc:
            return -1, f"（运行失败：{exc}）"
        return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()

    @staticmethod
    def _classic_expect(requirement: str):
        """经典题的独立验算：程序按标准参数把答案算出来，用于给模型交的作业判卷。

        注意这不是替模型答题 —— 代码仍然由模型自己写，这里只提供"阅卷标准"。
        真机教训：百元买百鸡连试四稿，四种错法（约束写矛盾 / 小鸡 1 元 1 只 /
        小鸡=1/3 整数除成 0 / 把只数就地除 3 覆盖掉），每稿都能编译、能跑、
        退出码 0，光看日志根本发现不了。
        """
        t = requirement or ""
        if any(k in t for k in ("百钱买百鸡", "百元买百鸡", "百鸡")):
            sols = []
            for x in range(0, 101):
                for y in range(0, 101):
                    z = 100 - x - y
                    if z < 0 or z % 3:
                        continue
                    if 5 * x + 3 * y + z // 3 == 100:
                        sols.append((x, y, z))
            return "百钱买百鸡", sols
        if "水仙花" in t:
            return "水仙花数", [(n,) for n in range(100, 1000)
                              if sum(int(d) ** 3 for d in str(n)) == n]
        return "", []

    @staticmethod
    def _verify_classic(requirement: str, out: str) -> str:
        name, sols = AgentSession._classic_expect(requirement)
        if not sols:
            return ""
        nums = [int(m) for m in re.findall(r"-?\d+", out or "")]
        missing = []
        for tup in sols:
            found = False
            for i in range(len(nums) - len(tup) + 1):
                if tuple(nums[i:i + len(tup)]) == tup:
                    found = True
                    break
            if not found:
                missing.append(tup)
        if missing:
            return (f"{name}的标准解共 {len(sols)} 组，输出里缺 {missing[:3]}"
                    f"（实际输出前几个数：{nums[:12]}）")
        return ""

    def _plan_vars(self, requirement: str) -> str:
        """先只让它列变量表：变量名 = 含义（单位）。

        为什么这招有效：真机百元买百鸡连错 7 稿，全是同一个病 —— 把"小鸡只数"
        就地除以 3 变成"小鸡花的钱"，之后仍当只数用。直接写代码时它守不住这个区分；
        但让它**先单独列出变量表**，它会老老实实写出"chick_count = 小鸡只数（只）/
        chick_cost = 小鸡花的钱（文）"，再照着表写代码就不会混。
        实测同一道题：直接写 4 次全错，两步法 2 次全对。
        """
        prompt = (
            f"题目：{requirement}\n\n"
            "先不要写代码。只列出程序里需要的变量，每个变量一行，格式：\n"
            "变量名 = 它代表什么（单位：个/只/元/文/次数）\n"
            "要求：\n"
            "1. 变量名必须是 ASCII 英文（如 rooster_count / chick_cost），"
            "说明部分可以用中文。中文变量名在编辑器里会乱码。\n"
            "2. 凡是\"数量\"和\"金额\"有区别的量，必须用两个不同的变量（例如只数 vs 花的钱）。\n"
            "3. 循环变量也列出来。最多 8 个变量，不要凑数。\n"
            "4. 不要写代码，不要解释，只给变量表。\n"
        )
        try:
            return (self.model.chat_coder([{"role": "user", "content": prompt}],
                                          max_tokens=250) or "").strip()
        except Exception:
            return ""

    def _review_code(self, code: str, requirement: str) -> str:
        """交付前做一次只读审查，返回问题描述（没问题返回空串）。

        为什么要这一道：真机百元买百鸡连续三稿都"能跑、退出码 0、解的数量也对"，
        但把 `chicks /= 3` 就地覆盖后仍然当"只数"打印，输出 25 只小鸡（应为 75）。
        这类"变量含义被改掉"的错误编译器不报、运行不崩，只有逐行读才看得出来。
        这里只让它挑错、不让它改代码 —— 免得它把对的改错。
        """
        prompt = (
            "下面是一段 C 代码。只做检查，不要改写代码。逐项核对：\n"
            "1) 有没有变量在循环里被重新赋值后含义变了"
            "（例如把小鸡\"只数\"就地除以 3 变成\"钱数\"，后面还当只数用/打印）；\n"
            "2) 有没有整数除法丢精度（1/3 在 C 里等于 0）；\n"
            "3) printf 里每个变量的实际含义和标签是否一致；\n"
            "4) 每一组输出是否真的满足题目条件（把数字代回去算一遍）。\n"
            "全部没问题就只回答 OK。发现问题就用一句英文指出具体位置和错因。\n\n"
            f"题目：{requirement}\n\n代码：\n{code}"
        )
        try:
            ans = self.model.chat_coder([{"role": "user", "content": prompt}], max_tokens=200)
        except Exception:
            return ""
        ans = (ans or "").strip()
        if not ans:
            return ""
        if re.match(r"^ok[.。!！]?$", ans.strip(), re.I):
            return ""
        return ans.splitlines()[0][:300]

    def _generate_code(self, requirement: str):
        """让模型写 C 代码（纯文本，不看屏幕），然后**真编译 + 真运行**双重验收。

        为什么运行也要验：真机那道「遍历二叉树」编译一次过、运行却 0xC00000FD 栈溢出——
        模型用 malloc 建节点却没把子指针置 NULL，叶子节点的 left/right 是野指针，
        递归踩垃圾地址当场崩。编译器对这类错永远沉默，只有跑一次才暴露。
        """
        sys_msg = (
            "你是 C 语言编程助手。只输出一段完整、可直接编译运行的 C 代码。\n"
            "硬性要求：\n"
            "0. 代码里所有文字（注释、printf 的字符串）必须是 ASCII 英文，一个汉字都不许有。\n"
            "1. 只输出代码本体，不要 markdown 代码围栏，不要任何解释文字。\n"
            "2. C11 + 标准库，#include 齐全，必须有 main 函数。\n"
            "3. 注释和 printf 的字符串一律只用 ASCII 英文（编辑器默认 GBK，中文会乱码）；"
            "变量名也用英文。\n"
            "4. 不要用 scanf 读输入（没人给它喂键盘）；需要数据就在代码里写死演示。\n"
            "5. 这是 Windows/MinGW 环境：long 只有 32 位。可能超过 2^31 的整数"
            "一律用 long long 声明变量，printf 用 %lld，格式符与实参类型必须完全一致。\n"
            "5b. C 的整数除法会截断：1/3 的结果是 0（不是 0.333）。"
            "\"三只小鸡 1 元\" 这类价格绝不能写成 单价 = 1/3；"
            "要在算钱时用 数量/3（先乘总只数再除 3），或用 double 并写成 1.0/3。\n"
            "6. 用 malloc 建树/链表节点时必须立刻把指针字段置 NULL（或用 calloc），"
            "否则叶子节点的子指针是野指针，遍历会无限递归崩溃。\n"
            "7. 每个 malloc 的返回值都要判空。\n"
            "8. 经典题目按下面的标准参数实现，题目没写也必须用这些，绝对不要自己改单价或总量：\n"
            "   - 百钱买百鸡：公鸡 5 元一只，母鸡 3 元一只，小鸡 1 元三只（三只共 1 元），"
            "100 元买 100 只；枚举时小鸡数量必须是 3 的倍数。\n"
            "   - 水仙花数：三位数，各位数字立方和等于自身。\n"
            "   并在代码第一行用英文注释写明采用的规则，"
            "例如：/* 5/rooster, 3/hen, 3 chicks per yuan, 100 for 100 */\n"
            "9. 凡是\"求满足条件的解\"的题，必须把每一组解都逐行打印出来，"
            "最后打印一行 total: N；不要只打印一个聚合值（最大值、和、个数之一）就完事。\n"
            "9b. 禁止用 /= *= += -= 修改循环变量或候选解变量。"
            "\"只数\"和\"钱数\"必须是两个不同的变量：例如 chicks 表示只数，"
            "cost_chicks = chicks / 3 表示这笔小鸡花的钱。打印只数时只能打印 chicks 本身。\n"
            "10. 交付前自查一遍：把约束代进极端取值，确认至少存在一组解。"
            "如果算出 0 组解，说明约束写矛盾了（例如两个等式相减后要求左边为零、右边却不为零），"
            "必须重新核对单价与总量，不要交一个只打印 0 的程序。\n"
        )
        msgs = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": f"题目：{requirement}"}]
        code = None
        for attempt in range(1, 4):
            # 第 2 稿起换打法：先让它单独列变量表，再照着表写。
            # 直接写时它会把"只数"就地改成"钱数"，列了表就守得住。
            if attempt == 2:
                table = self._plan_vars(requirement)
                if table and len(table) < 900:
                    self.log(f"[写码] 先列变量表：\n{table}")
                    msgs = [{"role": "system", "content": sys_msg},
                            {"role": "user", "content":
                             f"题目：{requirement}\n\n下面是已经定好的变量表，"
                             f"请严格按它写代码，不许改变任何变量的含义：\n{table}\n\n"
                             "铁律：每个变量只赋一次值；不许出现 /= += -= *= ；"
                             "一个变量不许有两种含义；printf 打印的变量必须和标签含义一致；"
                             "注释和字符串全用 ASCII 英文。只输出完整代码。"}]
            t0 = time.time()
            try:
                raw = self.model.chat_coder(msgs, max_tokens=1600)
            except Exception as exc:
                self.log(f"[写码] 第 {attempt} 稿模型调用失败（{exc}）")
                return code
            code = self._clean_code(raw)
            self.log(f"[写码] 第 {attempt} 稿 {len(code.splitlines())} 行 / {time.time() - t0:.1f}s")

            ok, err = self._compile_c(code, f"draft{attempt}")
            if not ok:
                first = (err.splitlines() or ["（无输出）"])[0]
                self.log(f"[编译] 第 {attempt} 稿 失败：{first}")
                msgs += [{"role": "assistant", "content": code},
                         {"role": "user", "content":
                          f"编译失败，报错如下：\n{err}\n请修正后只输出完整代码，不要解释。"}]
                continue
            rc, out = self._run_c(f"draft{attempt}")
            if rc != 0:
                self.log(f"[运行] 第 {attempt} 稿 崩溃，退出码 0x{rc & 0xFFFFFFFF:08X}"
                         f"（常见：malloc 后子指针没置 NULL / 数组越界 / 无限递归）")
                msgs += [{"role": "assistant", "content": code},
                         {"role": "user", "content":
                          f"程序编译通过，但运行崩溃，退出码 0x{rc & 0xFFFFFFFF:08X}。"
                          "常见原因：malloc 之后没有把 struct 的指针字段置 NULL、数组越界、"
                          "或递归没有终止条件。请修正后只输出完整代码。"}]
                continue
            full, why_bad = self._output_completeness(code, out)
            if not full:
                self.log(f"[运行] 第 {attempt} 稿 输出疑似不完整：{why_bad}")
                msgs += [{"role": "assistant", "content": code},
                         {"role": "user", "content":
                          f"程序能编译能运行，但输出疑似不完整：{why_bad}。"
                          "请检查每个节点的子指针是否都正确赋值（不要有节点悬空），"
                          "以及遍历是否覆盖了整棵树。只输出完整代码。"}]
                continue
            why_odd = self._output_suspicious(out)
            if why_odd:
                self.log(f"[运行] 第 {attempt} 稿 结果可疑：{why_odd}")
                msgs += [{"role": "assistant", "content": code},
                         {"role": "user", "content":
                          f"程序能跑，但结果可疑：{why_odd} "
                          "请重新核对题意里的数量与单价约束（尤其经典题目不要改系数），"
                          "确认确实有解后再输出，并把每一组解都打印出来。只输出完整代码。"}]
                continue
            head = "\n".join(out.splitlines()[:6])
            self.log(f"[运行] 第 {attempt} 稿 通过" + (f"：\n{head}" if head else ""))

            # 交付前：经典题先按标准答案判卷（比"人眼看"更硬的 ground truth）
            wrong = self._verify_classic(requirement, out)
            if wrong:
                self.log(f"[验算] 第 {attempt} 稿 未通过：{wrong}")
                hint = ("注意：公鸡5元、母鸡3元、小鸡3只1元，100元买100只；"
                        "小鸡只数 z 和它花的钱 z/3 是两个不同的量，别混。"
                        if "百鸡" in requirement else "请按题目标准条件重算。")
                msgs += [{"role": "assistant", "content": code},
                         {"role": "user", "content":
                          f"答案不对：{wrong}\n{hint}"
                          "把每一组解按 x y z 的顺序逐行打印，只输出完整代码。"}]
                continue

            # 交付前最后一道：变量含义/单位/标签一致性审查
            # （经典题已经用标准答案判过卷，不再重复跑——那一步更硬也更便宜）
            issue = "" if self._classic_expect(requirement)[1] else \
                self._review_code(code, requirement)
            if issue:
                self.log(f"[审查] 第 {attempt} 稿 发现问题：{issue}")
                msgs += [{"role": "assistant", "content": code},
                         {"role": "user", "content":
                          f"代码能跑，但审查发现问题：{issue}\n"
                          "请修正后只输出完整代码（注意别把一个变量既当只数又当钱数）。"}]
                continue
            self.log(f"[审查] 第 {attempt} 稿 未发现问题")
            self.code_verified = True
            return code
        self.log("[写码] ⚠️ 三稿都没跑通，仍按最后一稿交付（日志里有报错，可人工看）")
        self.code_verified = False
        return code

    @staticmethod
    def _foreground_class() -> str:
        """前台窗口的类名。Windows 标准对话框（另存为/覆盖确认）都是 #32770。"""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            h = user32.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(h, buf, 64)
            return buf.value
        except Exception:
            return ""

    @staticmethod
    def _fg_menu_open() -> bool:
        """前台线程是否正开着菜单（含右键菜单）。

        菜单开着的时候单击才是正确动作：双击会把菜单项执行两遍
        （"删除"变两次、"新建"开了又关）。所以判双击区域前先排除这种情况。
        """
        try:
            import ctypes
            from ctypes import wintypes

            class GUITHREADINFO(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
                            ("hwndActive", wintypes.HWND), ("hwndFocus", wintypes.HWND),
                            ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
                            ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND),
                            ("rcCaret", wintypes.RECT)]

            gti = GUITHREADINFO()
            gti.cbSize = ctypes.sizeof(GUITHREADINFO)
            if not ctypes.windll.user32.GetGUIThreadInfo(0, ctypes.byref(gti)):
                return False
            return bool(gti.hwndMenuOwner)
        except Exception:
            return False

    @staticmethod
    def _explorer_list_rect():
        """资源管理器文件列表的精确矩形（物理像素）；前台不是资源管理器则返回 None。

        不猜「左边导航栏占多少、顶部命令栏多高」—— Windows 自己知道：
        文件视图就是 SHELLDLL_DefView 这个子窗口，取它的矩形即列表区域。
        按比例切会切错：本机 2560 宽时导航栏只到 300px，而 18% 阈值是 460px，
        正好把「名称」列的开头一起切掉，于是点中文件名也判不出该双击。
        """
        try:
            import ctypes
            import ctypes.wintypes as wt

            user32 = ctypes.windll.user32
            h = user32.GetForegroundWindow()
            if not h:
                return None
            buf = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(h, buf, 64)
            if buf.value not in ("CabinetWClass", "ExploreWClass"):
                return None
            found = []

            def _walk(hw, depth=0):
                if depth > 8 or found:
                    return

                @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
                def _cb(ch, _):
                    if found:
                        return False
                    b = ctypes.create_unicode_buffer(64)
                    user32.GetClassNameW(ch, b, 64)
                    if b.value == "SHELLDLL_DefView":
                        r = wt.RECT()
                        if user32.GetWindowRect(ch, ctypes.byref(r)) \
                                and r.right - r.left > 100 and r.bottom - r.top > 100:
                            found.append((r.left, r.top, r.right, r.bottom))
                            return False
                    _walk(ch, depth + 1)
                    return True

                user32.EnumChildWindows(hw, _cb, 0)

            _walk(h)
            return found[0] if found else None
        except Exception:
            return None

    def _double_click_zone(self, x: float, y: float):
        """(x,y) 落在「必须双击才有效」的区域吗？是则返回区域名，否则 None。

        为什么要在程序里判这一次：7B 明明知道 double_click 这个动作
        （prompt 里写了两遍、执行层也支持），真机上却一次都没用过 ——
        它一贯的做法是「点一下、看画面没变、换个位置再点一下」，
        37 步里同一个 y 点出 7 个不同的 x。这类"单击只选中"的地方
        不能交给它临场判断，直接用几何位置兜住。
        """
        if self._fg_menu_open():
            return None          # 菜单开着，单击才是对的
        rect = self._explorer_list_rect()
        if rect:
            l, t, r, b = rect
            if l <= x <= r and t <= y <= b:
                return "资源管理器文件列表"
            return None
        cls = self._foreground_class()
        if cls in ("Progman", "WorkerW"):
            # 桌面：图标单击只选中，双击才打开；点在空白处双击也没副作用
            return "桌面"
        return None

    def _scroll_point(self):
        """滚轮该在哪个位置滚：优先资源管理器文件列表中心，其次上次点击的位置。

        滚轮只对鼠标下面的窗口生效，而本程序每步结束都把光标停靠左上角
        （14,14）—— 直接滚就会滚到别的窗口里去。
        """
        r = self._explorer_list_rect()
        if r:
            return ((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
        p = getattr(self, "_last_click_pt", None)
        if p:
            return (int(p[0]), int(p[1]))
        return (self.safety.W // 2, self.safety.H // 2)

    def _scroll_to_find(self, desc: str, pages: int = 3):
        """目标不在当前画面里时，滚动翻屏去找；找到返回物理坐标 (x, y)。

        为什么交给程序：7B 在清单里看不到目标就只会原地乱点（真机 37 步在同一处
        点出 7 个不同的 x），它想不起"往下滚一下看看"。滚一屏 + OCR 重扫是
        确定性动作，不该等模型悟。同一个目标只尝试一次，避免每步白烧 10 秒。
        """
        key = _clean_desc(desc)
        if not key or len(key) < 2:
            return None
        tried = getattr(self, "_scroll_tried", None)
        if tried is None:
            self._scroll_tried = tried = set()
        if key in tried:
            return None
        tried.add(key)

        at = self._scroll_point()
        for i in range(pages):
            self.executor.execute({"action": "scroll", "dy": 6, "_at": at})
            time.sleep(0.9)
            img = self.screenshot_fn()
            img_s, scale = self._fit_image(img)
            els, _ = self.grounding.annotate(img_s)
            if hasattr(self.grounding, "ocr_enrich"):
                els = list(els) + self.grounding.ocr_enrich(img, els, scale=scale)
            try:
                self.grounding._last_elements = els   # 后续 element_id 解析要用
            except Exception:
                pass
            self._last_elements = els
            hit = _match_label(els, key)
            if hit:
                x1, y1, x2, y2 = hit["bbox"]
                px, py = (x1 + x2) / 2 / scale, (y1 + y2) / 2 / scale
                self.log(f"[滚动查找] 翻到第 {i + 1} 屏，找到「{hit.get('label')}」"
                         f"@({px:.0f},{py:.0f})")
                return (int(px), int(py))
        self.log(f"[滚动查找] 连翻 {pages} 屏没找到「{key}」")
        return None

    # 这些窗口不能当"任务的家"：开始菜单/搜索面板是打开程序的副产物，
    # 真正要守的是程序自己的窗口。真机事故：计算器开好后前台还停在开始菜单，
    # 家被锚到了开始菜单，计算器一露头就被当成"前台被抢"，连抢 3 次直接失败。
    _HOME_BAD_CLASS = ("Windows.UI.Core.CoreWindow", "XamlExplorerHostIslandWindow",
                       "MultitaskingViewFrame", "ForegroundStaging")

    @classmethod
    def _is_bad_home(cls, hwnd: int) -> bool:
        """开始菜单 / 任务视图 / 搜索面板 —— 不能当锚点。"""
        import ctypes
        u = ctypes.windll.user32
        buf = ctypes.create_unicode_buffer(128)
        u.GetClassNameW(hwnd, buf, 128)
        if buf.value in cls._HOME_BAD_CLASS:
            return True
        tb = ctypes.create_unicode_buffer(128)
        u.GetWindowTextW(hwnd, tb, 128)
        return tb.value.strip() in ("开始", "Start", "搜索", "Search")

    def _gui_home_set(self, settle: float = 1.6):
        """把当前前台窗口记为本任务的"家"：之后每次动作前发现前台被抢就抢回来。

        刚打开程序时前台常常还是开始菜单，所以要等一小会儿，拿到真正的程序窗口
        才锚定（否则程序窗口一露面就被自己的守卫判成"被抢"，任务直接失败）。
        """
        try:
            import ctypes
            u = ctypes.windll.user32
            deadline = time.time() + settle
            h = u.GetForegroundWindow()
            while h and self._is_bad_home(h) and time.time() < deadline:
                time.sleep(0.2)
                h = u.GetForegroundWindow()
            if h and not self._is_bad_home(h):
                self._gui_home = h
        except Exception:
            pass

    @staticmethod
    def _win_pid(hwnd: int) -> int:
        import ctypes
        import ctypes.wintypes as wt

        pid = wt.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return pid.value

    def _guard_foreground(self) -> bool:
        """动作执行前的前台守卫。真机两次翻车是同一个病：
        用户正在用电脑 → 前台被切走 → 程序把点击/打字打进别人的窗口
        （23 行 C 代码进 Edge 地址栏；"新建文件"点进 WorkBuddy 的"新建任务"
        —— 模型看到的截图里那个按钮就叫"新建任务"，它不是乱点是被骗了）。
        返回 False = 本步不许执行（已尝试抢回一次或判定该让路）。
        """
        hwnd = getattr(self, "_gui_home", None)
        if not hwnd:
            return True
        import ctypes
        import ctypes.wintypes as wt

        u = ctypes.windll.user32
        if not u.IsWindow(hwnd):          # 任务窗口已经关了，守卫自然失效
            self._gui_home = None
            return True

        fg = u.GetForegroundWindow()
        if fg == hwnd or self._win_pid(fg) == self._win_pid(hwnd):
            # 同进程 = 任务程序自己的弹出菜单/对话框/新标签，放行
            # （拦了会把刚点开的「新建」菜单当敌人点没）
            self._fg_steal = 0
            return True
        # UWP 应用（计算器/照片/设置）是 ApplicationFrameWindow 套一个 CoreWindow，
        # 焦点从外壳移进内容区时 pid 会变、hwnd 也变，但根窗口是同一个 —— 放行
        try:
            if u.GetAncestor(fg, 2) == u.GetAncestor(hwnd, 2):   # GA_ROOT
                self._fg_steal = 0
                return True
        except Exception:
            pass
        # 前台被抢。SetForegroundWindow 常被系统拒绝，点标题栏一定有效且无副作用
        r = wt.RECT()
        if u.GetWindowRect(hwnd, ctypes.byref(r)) and r.right > r.left:
            if u.IsIconic(hwnd):
                u.ShowWindow(hwnd, 9)
                time.sleep(0.6)
            import pyautogui
            pyautogui.click(r.left + max(150, (r.right - r.left) // 4), r.top + 15)
            time.sleep(0.5)
        if u.GetForegroundWindow() == hwnd or self._win_pid(u.GetForegroundWindow()) == self._win_pid(hwnd):
            self._fg_steal = 0
            self.log("[守卫] 前台被切走了，已把任务窗口点回前台")
            return True
        self._fg_steal = getattr(self, "_fg_steal", 0) + 1
        self.log(f"[守卫] ⚠️ 前台是别的程序（用户在用电脑？），"
                 f"第 {self._fg_steal} 次抢回失败，本步不执行任何动作")
        return False

    _NAV_NEWFILE_RE = re.compile(
        r"新建(?:一个|个)?(?:空白)?(文件夹|文本文档|文件|txt)")
    _NAV_NEWNAME_RE = re.compile(r"(?:名为|命名为|叫|名字是)\s*[「『\"']?([^，。；;」』\"']{1,40})")

    def _new_file_in_explorer(self, kind: str = "文本文档",
                              name: str | None = None) -> bool:
        """资源管理器里新建文件/文件夹的收尾模板。

        真机：7B 能点中工具栏「新建」，但菜单弹出后它不去点「文本文档」，
        反而再点一次「新建」把菜单关掉（开→关→开…直到熔断）。
        菜单选择是纯查表操作：OCR 读到什么点什么，2 秒完事。
        """
        import pyautogui

        name = (name or "").strip().strip("「」\"'《》")   # "叫 LocalCUA测试" → "LocalCUA测试"
        if not self._guard_foreground():
            return False
        if self._foreground_class() not in ("CabinetWClass", "ExploreWClass"):
            self.log(f"[新建] 前台不是资源管理器（{self._window_info()[0]}），模板不适用")
            return False

        if kind == "文件夹":
            pyautogui.hotkey("ctrl", "shift", "n")   # 资源管理器内置快捷键
            time.sleep(1.2)
        else:
            # ① 点工具栏「新建」按钮（OCR/元素清单里按名字找，位置不写死）
            img = self.screenshot_fn()
            img_s, scale = self._fit_image(img)
            els, _ = self.grounding.annotate(img_s)
            els = list(els) + self.grounding.ocr_enrich(img, els, scale=scale)
            self.grounding._last_elements = els
            btn = next((e for e in els if (e.get("label") or "") == "新建"), None)
            if not btn:
                self.log("[新建] 屏幕上没找到「新建」按钮")
                return False
            x1, y1, x2, y2 = btn["bbox"]
            self.executor.execute({"action": "click",
                                   "x": (x1 + x2) / 2 / scale, "y": (y1 + y2) / 2 / scale})
            time.sleep(1.2)

            # ② 弹出的菜单里按名字找「文本文档」并点它
            img2 = self.screenshot_fn()
            img_s2, scale2 = self._fit_image(img2)
            els2, _ = self.grounding.annotate(img_s2)
            els2 = list(els2) + self.grounding.ocr_enrich(img2, els2, scale=scale2)
            self.grounding._last_elements = els2
            item = next((e for e in els2
                         if (e.get("label") or "").strip().startswith(kind)), None)
            if not item:
                pyautogui.press("esc")   # 把菜单收起来，别留个弹层祸害下一步
                self.log(f"[新建] 菜单里没找到「{kind}」项")
                return False
            x1, y1, x2, y2 = item["bbox"]
            self.executor.execute({"action": "click",
                                   "x": (x1 + x2) / 2 / scale2, "y": (y1 + y2) / 2 / scale2})
            time.sleep(1.4)

        # ③ 指定了名字就在重命名态直接敲；没指定保持默认名
        if name:
            from agent.actions import _type_unicode
            _type_unicode(name, interval=0.05)
            time.sleep(0.3)
            pyautogui.press("enter")
            time.sleep(0.6)
        self.log(f"[新建] ✅ 已新建{kind}" + (f"「{name}」" if name else ""))
        return True

    # 用户目录的英文路径 vs 资源管理器里的中文显示名。
    # 不映射的话，"导航到 Desktop" 永远验不过：窗口标题写的是「桌面」，
    # 程序拿 "Desktop" 去比对必然失败，然后白白回退到通用循环（真机 3 步 92 秒）。
    _FOLDER_ALIASES = {
        "desktop": ("桌面",), "documents": ("文档",), "downloads": ("下载",),
        "pictures": ("图片",), "videos": ("视频",), "music": ("音乐",),
    }

    @classmethod
    def _path_leaf_names(cls, path: str):
        """路径末级可能有几个名字（英文名 + 中文显示名）。"""
        p = (path or "").replace("/", "\\").rstrip("\\")
        leaf = p.split("\\")[-1] if p else p
        names = [leaf] if leaf else []
        names.extend(cls._FOLDER_ALIASES.get(leaf.lower(), ()))
        return [n for n in names if n]

    def _navigate_to(self, path: str) -> bool:
        """资源管理器导航模板：Ctrl+L 地址栏输路径回车，失败再用 explorer 直开。

        真机教训：把「打开D:\测试」丢给通用循环，7B 在左侧导航栏连点 15 次
        （x 在 20~132 之间猜），40 步 579 秒都没走到 D 盘——它不知道地址栏
        能输路径。这整条链路是确定性的，不该花一次模型推理。
        """
        import subprocess

        import pyautogui

        from agent.actions import _type_unicode
        win_path = path.replace("/", "\\")
        names = self._path_leaf_names(win_path)

        for attempt in (1, 2):
            # 第 1 次复用当前资源管理器（没有就 Win+E 开一个新的）；
            # 第 2 次重开一个干净窗口再试（上次可能开在了库/主页这种特殊视图）
            if attempt == 2 or self._foreground_class() not in ("CabinetWClass", "ExploreWClass"):
                pyautogui.hotkey("win", "e")
                time.sleep(1.6)
            pyautogui.hotkey("ctrl", "l")       # 地址栏（会全选，输入即覆盖）
            time.sleep(0.5)
            _type_unicode(win_path, interval=0.03)
            time.sleep(0.3)
            pyautogui.press("enter")
            time.sleep(1.8)
            title = self._window_info()[0] or ""
            if names and any(n in title for n in names):
                self.log(f"[导航] ✅ 已打开 {win_path}（窗口「{title}」）")
                return True
            self.log(f"[导航] 第 {attempt} 次输入路径后窗口是「{title}」，没到位")

        self.log("[导航] 地址栏两试未成，改用 explorer 命令直开")
        try:
            subprocess.Popen(["explorer", win_path])
            time.sleep(2.2)
            title = self._window_info()[0] or ""
            if names and any(n in title for n in names):
                self.log(f"[导航] ✅ explorer 直开成功（窗口「{title}」）")
                return True
            self.log(f"[导航] explorer 直开后窗口是「{title}」")
        except Exception as e:
            self.log(f"[导航] explorer 直开失败: {e}")
        return False

    def _ocr_text_near(self, x: float, y: float, scale: float = 1.0):
        """物理坐标 (x,y) 附近 OCR 读到的文字（缩放图坐标比对），没有则 None。"""
        g = getattr(self, "grounding", None)
        texts = getattr(g, "last_texts", None)
        if not texts:
            return None
        sx, sy = x * scale, y * scale
        best, bd = None, 1e18
        for bx1, by1, bx2, by2, t in texts:
            bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if bx1 - 20 <= sx <= bx2 + 20 and by1 - 12 <= sy <= by2 + 14:
                d = abs(bcx - sx) + abs(bcy - sy)
                if d < bd:
                    bd, best = d, t
        return best

    def _prefer_double_click(self, action: dict, scale: float = 1.0) -> dict:
        """把落在「必须双击」区域里的单击改成双击。

        模型只负责"点哪个目标"，"这个目标要不要双击"由位置决定 ——
        资源管理器列表里的文件夹、文件、磁盘，单击永远只是选中。
        但带扩展名的目标（xxx.xlsx / xxx.jpg）是文件不是文件夹：
        升级双击会把别人的文件直接打开（真机：在主文件夹页把
        「新建 Microsoft Excel 工作表.xlsx」双击开了），所以这种只保持单击。
        """
        if not isinstance(action, dict) or action.get("action") != "click":
            return action
        try:
            x = float(action.get("x"))
            y = float(action.get("y"))
        except (TypeError, ValueError):
            return action
        why = self._double_click_zone(x, y)
        if not why:
            return action
        near = self._ocr_text_near(x, y, scale)
        if near and re.search(r"\.[A-Za-z0-9]{1,5}\s*$", near.strip()):
            self.log(f"[双击] 「{near}」是文件不是文件夹，保持单击选中")
            return action
        action["action"] = "double_click"
        action["_dbl_why"] = why
        self.log(f"[双击] 「{why}」里单击只选中不进去 -> 本步改按双击")
        return action

    # Dev-C++ 的编辑器有一堆"贴心"功能，会当场把代码改坏：
    #   BraceComplete=1       敲 { 自动补一个 }  —— 我们再敲 } 就多一个
    #   DoubleQuoteComplete=1 敲 " 自动补一个 "  —— 字符串直接变两份
    #   AutoIndent=1          回车自动缩进      —— 我们敲的行首空格又被加一层
    #   CodeCompletion/Delay=180  弹补全框，回车会"选中候选"而不是换行
    # 真机证据：遍历二叉树那次敲完，文件尾部多出 `} } } };`，正是自动补括号的产物。
    # 做法：动手前把这几项写进 devcpp.ini（它只在启动时读，所以要重启才生效）。
    _DEVCPP_QUIET = {
        "Editor": {"AutoIndent": "0", "AddIndent": "0",
                   "BraceComplete": "0", "ParentheseComplete": "0",
                   "CommentComplete": "0", "ArrayComplete": "0",
                   "SingleQuoteComplete": "0", "DoubleQuoteComplete": "0",
                   "CompleteSymbols": "0", "DeleteSymbolPairs": "0",
                   "ShowFunctionTip": "0", "AutoCheckSyntax": "0",
                   # 启动时自动恢复上次打开的文件会害死人：Ctrl+N 之后焦点可能还在
                   # 上次那个有文件名的标签页上，Ctrl+S 就静默覆盖它、不弹另存为。
                   # （这个键在 ini 里属于 [Editor] 段，不在 [Options] 段）
                   "LoadLastOpens": "0"},
        "CodeCompletion": {"Enabled": "0", "ShowCompletionWhileInput": "0",
                           "ShowCodeIns": "0"},
    }

    @staticmethod
    def _devcpp_ini_path() -> str:
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "Dev-Cpp", "devcpp.ini")

    def _patch_devcpp_ini(self) -> int:
        """关掉会改坏代码的编辑器功能，返回改了几项。配置是 GBK，别用 utf-8 读写。"""
        path = self._devcpp_ini_path()
        if not os.path.exists(path):
            self.log("[编辑器] 没找到 devcpp.ini（没启动过？），跳过设置修正")
            return 0
        try:
            with open(path, encoding="gbk", errors="replace") as f:
                text = f.read()
        except Exception as exc:
            self.log(f"[编辑器] 读配置失败：{exc}")
            return 0
        bak = path + ".agent.bak"
        if not os.path.exists(bak):
            try:
                with open(bak, "w", encoding="gbk", errors="replace") as f:
                    f.write(text)
            except Exception:
                pass
        out, sec, changed = [], "", 0
        for line in text.splitlines():
            st = line.strip()
            if st.startswith("[") and st.endswith("]"):
                sec = st[1:-1]
                out.append(line)
                continue
            if "=" in st and sec in self._DEVCPP_QUIET:
                k, v = st.split("=", 1)
                want = self._DEVCPP_QUIET[sec].get(k.strip())
                if want is not None and v.strip() != want:
                    out.append(f"{k.strip()}={want}")
                    changed += 1
                    continue
            out.append(line)
        if changed:
            try:
                with open(path, "w", encoding="gbk", errors="replace") as f:
                    f.write("\n".join(out) + "\n")
            except Exception as exc:
                self.log(f"[编辑器] 写配置失败：{exc}")
                return 0
            self.log(f"[编辑器] 已关闭自动补全/自动括号等 {changed} 项"
                     f"（防止花括号被补成两份、引号变两份）；原配置备份在 devcpp.ini.agent.bak")
        else:
            self.log("[编辑器] 自动补全等已是关闭状态")
        return changed

    def _devcpp_running(self) -> bool:
        return self._window_rect("devcpp") is not None

    def _close_devcpp(self, wait: float = 6.0) -> bool:
        """让 Dev-C++ 正常退出（它退出时会把内存里的设置写回 ini，
        所以必须"先关、再改配置、再开"，改配置才留得住）。"""
        import ctypes
        user32 = ctypes.windll.user32
        wins = self._find_windows("devcpp")
        if not wins:
            return True
        for hwnd, _ in wins:
            user32.PostMessageW(hwnd, 0x0010, 0, 0)   # WM_CLOSE
        deadline = time.time() + wait
        while time.time() < deadline:
            time.sleep(0.4)
            if self._is_save_dialog():
                self.log("[编辑器] Dev-C++ 弹出了保存提示（有未保存的文件），"
                         "为了不弄丢你的东西，这里停手，请手动关掉它再重试。")
                try:
                    import pyautogui as _pg
                    _pg.press("esc")     # 取消关闭，保持原样
                except Exception:
                    pass
                return False
            if not self._devcpp_running():
                time.sleep(0.8)            # 等它把 ini 写完
                return True
        self.log("[编辑器] Dev-C++ 没能自己退出")
        return False

    def _clear_editor(self):
        """Ctrl+A 全选 + Delete 清空编辑区（重敲前用，避免叠加）。"""
        import pyautogui
        pyautogui.hotkey("ctrl", "a")
        time.sleep(0.2)
        pyautogui.press("delete")
        time.sleep(0.2)
        self.log("[编辑区] 已清空，准备重敲")

    @staticmethod
    def _foreground_exe() -> str:
        """前台窗口的 exe 文件名（小写）。拿不到返回 ""。

        敲字前必须知道"现在是谁在前面"：SendInput 是把按键送给**前台窗口**的，
        前台一旦不是目标程序，整段代码会打进别的窗口（真机踩过：23 行代码
        全进了 Edge 的地址栏，变成一串搜索词）。
        """
        try:
            import ctypes
            import ctypes.wintypes as wt

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return ""
            pid = wt.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            h = kernel32.OpenProcess(0x1000, False, pid.value)   # QUERY_LIMITED_INFORMATION
            if not h:
                return ""
            buf = ctypes.create_unicode_buffer(600)
            n = wt.DWORD(600)
            ok = kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n))
            kernel32.CloseHandle(h)
            return buf.value.rsplit("\\", 1)[-1].lower() if ok else ""
        except Exception:
            return ""

    def _ensure_front(self, exe_key: str, app: str, tries: int = 2) -> bool:
        """确保前台是目标程序；被抢走就点窗口把它抢回来。

        SetForegroundWindow 在非前台进程里常被系统直接拒绝（只闪任务栏），
        而"点一下窗口"永远有效，所以这里用点击。
        """
        import pyautogui

        if exe_key in self._foreground_exe():
            return True
        for _ in range(tries):
            t = self._window_rect(exe_key)
            if not t:
                return False
            pyautogui.click(int((t[0] + t[2]) / 2), int((t[1] + t[3]) / 2))
            time.sleep(0.4)
            if exe_key in self._foreground_exe():
                self.log(f"[焦点] {app} 的窗口被抢走了，已点回前台")
                return True
        return False

    # ==================== 「打开 X 并输入文字」的执行部分 ====================

    @classmethod
    def _exe_keys(cls, app: str):
        """程序名 → 可能的进程名列表（用于判断"前台是不是它"）。"""
        v = cls._PROC_MAP.get(cls._norm_app_name(app))
        if v:
            return [k.lower() for k in (v if isinstance(v, (list, tuple)) else [v])]
        n = cls._norm_app_name(app)
        return [n] if n else []

    def _front_ok(self, keys) -> bool:
        exe = self._foreground_exe()
        if not exe:
            return False
        return any(AgentSession._exe_matches(exe, k) for k in keys)

    def _ensure_front_keys(self, keys, app: str, tries: int = 3) -> bool:
        """确保前台属于 keys 里的任一进程；被抢走就点【标题栏】抢回来。

        为什么点标题栏而不是 _ensure_front 的窗口中心：中心是编辑区，
        点一下就把光标挪到正文中间，接下来敲的字全插到中间去了。
        标题栏同样能把窗口提到最前，而且不动光标、不改任何内容。
        """
        import pyautogui

        if self._front_ok(keys):
            return True
        for _ in range(tries):
            rect = None
            for k in keys:
                rect = self._window_rect(k)
                if rect:
                    break
            if not rect:
                return False
            pyautogui.click(int((rect[0] + rect[2]) / 2), int(rect[1]) + 12)
            time.sleep(0.35)
            if self._front_ok(keys):
                self.log(f"[焦点] 「{app}」被抢到后台，已点标题栏抢回前台")
                return True
        return False

    # 支持 Ctrl+N「新建一个空白文档」的程序：先新建再打字，绝不往用户已有内容后面追加
    _NEW_DOC_APPS = ("notepad", "记事本", "devcpp", "dev-c++", "devc++", "wordpad",
                     "写字板", "word", "sublime", "notepad++", "vscode",
                     "visualstudiocode", "mspaint", "画图")

    def _open_and_type(self, app: str, text: str) -> bool:
        """程序级模板：打开某程序 → 新建空白文档 → 点编辑区 → 逐行敲 → OCR 验收。

        真机为什么必须有这条：「打开记事本，输入三行字：…每行一个」落到通用循环，
        5 步 108 秒没写完，模型反复 open_app 而前台早就是记事本。
        这里整条链路零推理，约 8~12 秒完成，而且全程是真人式操作
        （点开始菜单写名字、点编辑区、一个字一个字敲）。
        """
        import pyautogui

        from agent.actions import _type_unicode

        text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        if not text.strip():
            self.log("[输入] 正文是空的，不执行")
            return False
        lines = text.split("\n")

        self._wait_until_idle()          # 有人正在用电脑就先让路
        if not self._open_app(app):
            self.log(f"[输入] ❌ 打不开「{app}」")
            return False
        self._last_opened_app = app
        time.sleep(0.5)

        keys = self._exe_keys(app)
        if not keys or not self._ensure_front_keys(keys, app):
            self.log(f"[输入] ❌ 前台不是「{app}」（现在是 "
                     f"「{self._foreground_exe() or '未知'}」），拒绝敲入："
                     "字会打进别的窗口")
            return False

        # 1) 新建空白文档：连着上次的内容往下追加会出错，也不该动用户已写的东西
        if any(k in app.lower() for k in self._NEW_DOC_APPS):
            img0 = self.screenshot_fn()
            pyautogui.hotkey("ctrl", "n")
            self._wait_screen_change(img0, timeout=2.0, settle=0.3)
            time.sleep(0.4)
            if not self._ensure_front_keys(keys, app):
                self.log("[输入] ❌ Ctrl+N 之后前台变了，拒绝敲入")
                return False
        title = self._window_info()[0] or ""
        self.log(f"[输入] 窗口「{title}」← 将要往这里敲 {len(lines)} 行 / {len(text)} 字")

        # 2) 点编辑区：优先用 Win32 的"焦点控件"矩形（编辑区对视觉模型是盲区），
        #    拿不到就按窗口几何估算，绝不让模型猜坐标
        c = self._focused_control_center()
        clicked = None
        if c and c[2] > 80 and c[3] > 80:
            clicked = (int(c[0]), int(c[1]))
        else:
            for k in keys:
                rect = self._window_rect(k)
                if rect:
                    clicked = (int((rect[0] + rect[2]) / 2),
                               int(rect[1] + (rect[3] - rect[1]) * 0.55))
                    break
        if clicked:
            self.log(f"[输入] 点编辑区 @({clicked[0]},{clicked[1]})")
            pyautogui.click(clicked[0], clicked[1])
            time.sleep(0.3)
            # Ctrl+End = 跳到文末：万一新建没生效，也只会追加在末尾，不会插到正文中间
            pyautogui.hotkey("ctrl", "end")
            time.sleep(0.15)
        else:
            self.log("[输入] 拿不到编辑区位置，直接敲（焦点通常已经在编辑区里）")

        before = self.screenshot_fn()

        # 3) 逐行敲；每行之前确认前台还是它
        t0 = time.time()
        for idx, ln in enumerate(lines):
            if idx and not self._front_ok(keys):
                if not self._ensure_front_keys(keys, app):
                    self.log(f"[输入] ⚠️ 敲到第 {idx + 1}/{len(lines)} 行时焦点被抢走，"
                             f"抢不回来（现在前台「{self._foreground_exe() or '未知'}」），立即停止")
                    return False
            if ln:
                _type_unicode(ln, interval=0.02)
            if idx < len(lines) - 1:
                pyautogui.press("enter")
        self.log(f"[输入] 已敲入 {len(lines)} 行 / {time.time() - t0:.1f}s")

        # 4) 验收：OCR 读第一行；OCR 不可用/认不出时退回像素差异。
        #    绝不自报成功——"看起来完成了"和"字真的在屏幕上"是两件事。
        time.sleep(0.5)
        probe = lines[0].strip()[:6]
        ok_ocr = bool(probe) and self._screen_has_text(probe)
        if ok_ocr:
            self.log(f"[输入] ✅ 验收通过：屏幕上读到了「{probe}」")
            return True
        after = self.screenshot_fn()
        if AgentSession._changed(before, after, thr=0.002, max_w=None):
            self.log("[输入] ✅ 画面已变化（OCR 没认出来，但像素确认字上屏了）")
            return True
        self.log("[输入] ❌ 验收失败：屏幕上没读到这段文字，可能没打进编辑区")
        self._note_fail(f"想在「{app}」里输入「{probe}…」，但字没有出现在编辑区里。"
                        "先点一下编辑区内部再输入，不要点标题栏或工具栏。")
        return False


    def _type_code_in_devcpp(self, code: str, new_file: bool = True) -> bool:
        """在 Dev-C++ 里：Ctrl+N → 点编辑区 → 逐行敲。返回是否完整敲完。

        全程盯着前台窗口：不是 Dev-C++ 就先抢回来，抢不回来立刻停手 ——
        宁可不敲，也不能把代码打进别人的窗口。
        """
        import pyautogui
        from agent.actions import _type_unicode

        # 先确认前台是 Dev-C++，再按 Ctrl+N（否则 Ctrl+N 会触发别的程序的"新建"）
        if not self._ensure_front("devcpp", "Dev-C++"):
            self.log(f"[编辑区] ❌ 前台不是 Dev-C++（现在是「{self._foreground_exe() or '未知'}」），"
                     "拒绝敲入：代码会打进别的窗口")
            return False
        if new_file:
            pyautogui.hotkey("ctrl", "n")
            time.sleep(0.9)
            if not self._ensure_front("devcpp", "Dev-C++"):
                self.log("[编辑区] ❌ Ctrl+N 之后前台变了，拒绝敲入")
                return False
        fg, _ = self._window_info()
        self.log(f"[编辑区] 当前窗口「{fg}」")   # 确认敲进的是新建的空白文件，不是上次那个
        c = self._focused_control_center()
        if c and c[2] > 60 and c[3] > 60:
            self.log(f"[编辑区] 焦点控件 {c[2]:.0f}x{c[3]:.0f}px @({c[0]:.0f},{c[1]:.0f})")
            pyautogui.click(int(c[0]), int(c[1]))
        else:
            self.log("[编辑区] 拿不到焦点控件，按窗口中心点一下兜底")
            t = self._window_rect("devcpp")
            if t:
                pyautogui.click(int((t[0] + t[2]) / 2), int((t[1] + t[3]) / 2))
        time.sleep(0.4)

        lines = code.split("\n")
        t0 = time.time()
        for idx, ln in enumerate(lines):
            # 每行敲之前确认前台还是 Dev-C++（有人动鼠标/别的程序弹窗就切走了）
            if idx and "devcpp" not in self._foreground_exe():
                if not self._ensure_front("devcpp", "Dev-C++"):
                    self.log(f"[输入] ⚠️ 敲到第 {idx}/{len(lines)} 行时焦点被抢走，"
                             f"抢不回来（现在前台「{self._foreground_exe() or '未知'}」），立即停止")
                    return False
            if ln.strip():
                _type_unicode(ln, interval=0.012)   # Scintilla 是原生控件，快喂不丢字
            pyautogui.press("enter")
        self.log(f"[输入] 已逐行敲入 {len(lines)} 行 / {time.time() - t0:.1f}s")
        if "devcpp" not in self._foreground_exe():
            self.log("[输入] ⚠️ 敲完时前台已经不是 Dev-C++，后面的保存动作跳过")
            return False
        return True

    @staticmethod
    def _find_windows(exe_key: str):
        """按 exe 关键词找出所有可见顶层窗口：[(hwnd, (l,t,r,b))]。"""
        try:
            import ctypes
            import ctypes.wintypes as wt
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            found = []

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            def _cb(hwnd, _):
                if not user32.IsWindowVisible(hwnd):
                    return True
                pid = wt.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                h = kernel32.OpenProcess(0x1000, False, pid.value)
                exe = ""
                if h:
                    buf = ctypes.create_unicode_buffer(512)
                    size = wt.DWORD(512)
                    if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                        exe = buf.value.lower()
                    kernel32.CloseHandle(h)
                if AgentSession._exe_matches(exe, exe_key):
                    r = wt.RECT()
                    if user32.GetWindowRect(hwnd, ctypes.byref(r)):
                        found.append((hwnd, (r.left, r.top, r.right, r.bottom)))
                return True

            user32.EnumWindows(_cb, 0)
            return found
        except Exception:
            return []

    @staticmethod
    def _window_rect(exe_key: str):
        """按 exe 关键词取窗口矩形（物理像素），找不到返回 None。"""
        try:
            import ctypes
            import ctypes.wintypes as wt
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            found = []

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            def _cb(hwnd, _):
                if not user32.IsWindowVisible(hwnd):
                    return True
                pid = wt.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                h = kernel32.OpenProcess(0x1000, False, pid.value)
                exe = ""
                if h:
                    buf = ctypes.create_unicode_buffer(512)
                    size = wt.DWORD(512)
                    if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                        exe = buf.value.lower()
                    kernel32.CloseHandle(h)
                if AgentSession._exe_matches(exe, exe_key):
                    r = wt.RECT()
                    if user32.GetWindowRect(hwnd, ctypes.byref(r)):
                        found.append((r.left, r.top, r.right, r.bottom))
                return True

            user32.EnumWindows(_cb, 0)
            return found[0] if found else None
        except Exception:
            return None

    def _save_in_devcpp(self, filename: str) -> bool:
        """Ctrl+S 保存：弹「另存为」就敲文件名回车；弹覆盖确认就再回车。"""
        import pyautogui
        from agent.actions import _type_unicode

        pyautogui.hotkey("ctrl", "s")
        time.sleep(1.0)
        if self._foreground_class() != "#32770":
            fg, _ = self._window_info()
            self.log(f"[保存] 没弹另存为对话框，当前窗口「{fg}」，先看文件是否落地")
        else:
            self.log(f"[保存] 另存为对话框已弹出，输入文件名「{filename}」")
            _type_unicode(filename)
            time.sleep(0.3)
            pyautogui.press("enter")
            time.sleep(0.9)
            # 同名文件已存在 -> "是否覆盖" 再确认一次
            if self._is_save_dialog():
                self.log("[保存] 覆盖确认对话框，回车确认")
                pyautogui.press("enter")
                time.sleep(0.7)
        return self._foreground_class() != "#32770"

    def _write_code_task(self, job: dict) -> bool:
        """完整链路：模型写码 → 真编译 → 开 Dev-C++ → 新建 → 逐行敲 → 另存到桌面。"""
        name, req = job["name"], job["requirement"]
        out_dir = self._desktop_dir() if job.get("to_desktop") else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "code_build")
        target = os.path.join(out_dir, f"{name}.c")

        self.log(f"[写码] 题目：{req}｜文件：{name}.c｜位置：{'桌面' if job.get('to_desktop') else out_dir}")
        code = self._generate_code(req)
        if not code:
            return False

        # 1) 先让编辑器"闭嘴"：自动补全/自动括号会当场改坏代码，
        #    而这些开关只在启动时读 ini —— 所以先关进程、改配置、再打开。
        if self._devcpp_running():
            self.log("[编辑器] Dev-C++ 正在运行，先关闭它（配置只在启动时生效）")
            if not self._close_devcpp():
                self.log("[编辑器] 关闭失败，直接写代码的话会被自动补全改坏，先停手。")
                return False
        self._patch_devcpp_ini()

        # 2) 打开 Dev-C++（开始菜单优先，失败直接启动已知路径）
        if not self._focus_program("Dev-C++"):
            if not self._open_app("Dev-C++"):
                exe = next((p for p in self._DEVCPP_CANDIDATES if os.path.exists(p)), None)
                if not exe:
                    self.log("[开程序] 找不到 devcpp.exe，写代码任务无法继续")
                    return False
                self.log(f"[开程序] 开始菜单没搜到，直接启动 {exe}")
                import subprocess
                subprocess.Popen([exe])
                if not self._wait_app_window("Dev-C++", max_wait=12.0):
                    self.log("[开程序] ❌ Dev-C++ 没起来")
                    return False
        self._focus_program("Dev-C++")

        # 3) 新建 + 输入
        typed = self._type_code_in_devcpp(code)
        if not typed:
            # 焦点被抢走（用户在操作电脑 / 别的程序弹窗）：这里绝不能再发 Ctrl+S，
            # 那会把"另存为"发到别人的窗口里。直接写盘保证有交付物，编译验证照旧跑。
            self.log("[写码] ⚠️ GUI 敲入没走完，改为直接写文件保证有交付物"
                     "（Dev-C++ 里不会出现这段代码）")
            try:
                os.makedirs(out_dir, exist_ok=True)
                with open(target, "w", encoding="utf-8", newline="\n") as f:
                    f.write(code)
                self.ok_saved_by = "disk"
            except Exception as exc:
                self.log(f"[写码] 写盘也失败：{exc}")
                return False
            self._final_check(target)
            return True

        # 4) 保存（先试裸文件名；不行再试绝对路径）
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(target):
            # 同名文件先备份再覆盖：不偷偷删用户的东西，也不让"是否覆盖"对话框卡住流程
            try:
                os.replace(target, target + ".bak")
                self.log(f"[保存] 桌面已有 {name}.c，已备份为 {name}.c.bak")
            except Exception:
                pass
        self._save_in_devcpp(f"{name}.c")
        if not os.path.exists(target):
            self.log("[保存] 裸文件名没落到桌面，改用绝对路径再存一次")
            self._save_in_devcpp(target)

        # 5) 校验：文件在不在、内容对不对；不对就把编辑区清干净重敲一遍
        norm = lambda s: re.sub(r"[ \t\r\n]+", "", s or "")
        if not os.path.exists(target):
            self.log("[保存] ❌ 桌面上没找到文件，用程序直接写一份"
                     "（保证交付，但这不是 GUI 输入的成果）")
            try:
                with open(target, "w", encoding="utf-8", newline="\n") as f:
                    f.write(code)
            except Exception as exc:
                self.log(f"[保存] 直接写盘也失败了：{exc}")
                return False
            self.ok_saved_by = "fallback"
        else:
            self.ok_saved_by = "gui"
            got = open(target, encoding="utf-8", errors="replace").read()
            if norm(got) != norm(code):
                self.log("[保存] ⚠️ 编辑区内容与原稿不一致（自动补全/吃字符），"
                         "清空重敲一遍")
                self._clear_editor()
                self._type_code_in_devcpp(code, new_file=False)
                self._save_in_devcpp(target)
                got = open(target, encoding="utf-8", errors="replace").read() \
                    if os.path.exists(target) else ""
            if norm(got) == norm(code):
                self.log(f"[保存] ✅ 编辑区内容与模型原稿一致（{len(code.splitlines())} 行）")
            else:
                # 编辑区还是坏的：只在磁盘上落一份正确的，并明确告诉用户别保存缓冲区
                with open(target, "w", encoding="utf-8", newline="\n") as f:
                    f.write(code)
                self.ok_saved_by = "disk"
                self.log("[保存] ⚠️ 重敲后仍不一致。磁盘上已是最新正确版本，"
                         "但 Dev-C++ 编辑区里是坏文本 —— 请不要在 Dev-C++ 里按保存/编译，"
                         "否则会用坏文本覆盖磁盘。建议关掉该窗口（保存提示选「否」）"
                         "后重新打开文件。")

        # 6) 成品双重验收：编译 + 运行
        return self._final_check(target)

    def _final_check(self, target: str) -> bool:
        """成品验收：编译 + 运行 + 输出完整性 / 可疑值 / 经典题判卷。

        GUI 敲入和"直接写盘"两条路都走这里，标准保持一致。
        """
        ok, err = self._compile_c(open(target, encoding="utf-8", errors="replace").read(), "final")
        self.log(f"[编译] 成品 {'通过' if ok else '失败'}" + ("" if ok else f"：{err.splitlines()[0]}"))
        if ok:
            rc, out = self._run_c("final")
            if rc == 0:
                head = "\n".join(out.splitlines()[:8])
                if head:
                    self.log(f"[运行] 输出：\n{head}")
                full, why_bad = self._output_completeness(
                    open(target, encoding="utf-8", errors="replace").read(), out)
                if not full:
                    self.log(f"[运行] ⚠️ 输出疑似不完整：{why_bad}（请人工核对结果）")
                why_odd = self._output_suspicious(out)
                if why_odd:
                    self.log(f"[运行] ⚠️ 结果可疑：{why_odd}")
                wrong = self._verify_classic(getattr(self, "task", "") or "", out)
                if wrong:
                    self.log(f"[验算] ❌ 成品未通过：{wrong}")
                    ok = False
            else:
                self.log(f"[运行] ⚠️ 成品运行异常退出 0x{rc & 0xFFFFFFFF:08X}"
                         f"（0xC00000FD=栈溢出）")
                ok = False
        if getattr(self, "code_verified", True) is False:
            self.log(f"❌ 代码已保存到 {target}，但**答案未通过验算**，请勿直接拿去演示")
            if ok:
                self.log("[验算] 建议：换一道题，或把这道题的输入输出样例告诉它重写")
            return False
        self.log(f"✅ 已保存到 {target}")
        return ok


    _ROUTE_HINT = re.compile(r"发|说|告诉|消息|通知|转告|回复|发给|传话")

    # ==================== 微信发送白名单 ====================
    # 代码里的保守默认值：只有 config 没给 wechat_allowed 时才生效。
    # 默认拒绝策略：误发给真人不可撤销，只有 config 显式列出的会话才放行。
    # 会话名只放在使用者自己的配置里，
    # 不写进代码 —— 打包分发时代码仍是最小权限。
    # 想放开改 config.yaml 的 agent.wechat_allowed。
    _WECHAT_ALLOWED = ("文件传输助手", "示例联系人A")

    @classmethod
    def _allowed_targets(cls):
        """白名单取值：优先 config，其次类常量。"""
        cfg = getattr(cls, "_cfg_snapshot", None) or {}
        v = None
        try:
            v = (cfg.get("agent") or {}).get("wechat_allowed")
        except Exception:
            v = None
        if isinstance(v, (list, tuple)) and v:
            return tuple(str(x).strip() for x in v if str(x).strip())
        return cls._WECHAT_ALLOWED

    @classmethod
    def _send_target_allowed(cls, app: str, target: str):
        """聊天软件发消息的白名单闸。返回 (ok, reason)。

        - 匹配用「互相包含」而不是相等：搜索结果标签常带尾巴
          （「文件传输助手（2）」未读数、「示例联系人A - 最后一条消息…」），
          严格相等会把合法目标也拒掉，人就会去关这个闸。
        - 太短的 target（1 个字）不做包含匹配，避免「王」命中一堆人。
        """
        if "微信" not in (app or ""):
            return True, ""
        t = (target or "").strip().strip(_QUOTE_CHARS).strip()
        if not t:
            return False, "收信人为空，不知道要发给谁"
        if len(t) == 1:
            return False, f"收信人「{t}」只有 1 个字，太模糊，不发送"
        for ok_name in cls._allowed_targets():
            if ok_name in t or (len(t) >= 2 and t in ok_name):
                return True, ""
        return False, (f"微信会话「{t}」不在白名单内（当前只允许："
                       f"{'、'.join(cls._allowed_targets())}）")

    def _route_task(self, task: str):
        """把任务路由到程序模板：模型只做「分类 + 抽参数」，动作永远归程序。

        路径：正则先匹配（零成本、精确）；不中但任务里确实有「发消息」语义时，
        再让 7B 判断一次 —— 这样「跟小王说一声明天开会」这种口语也能走稳定流程。
        目前只路由发消息：它收益最高（模板已验证），风险最低（发错可撤回）。
        返回 {"app","target","text"} 或 None（None 表示交给通用循环）。
        """
        intent = self._parse_message_intent(task)
        if intent:
            ok, why = self._send_target_allowed(intent["app"], intent["target"])
            if not ok:
                self.log(f"[路由] ⛔ 已拦截：{why}")
                return {"blocked": why}
            self.log("[路由] 正则命中，直接走发消息模板")
            return intent
        if not self._ROUTE_HINT.search(task):
            return None   # 不像发消息，不浪费一次 10s 推理
        prompt = (
            "你是任务参数抽取器。判断下面这条任务是不是「给某人/某个会话发一条消息」。\n"
            '如果是，只输出 JSON：{"ok":true,"app":"微信","target":"收信人","text":"消息内容"}\n'
            '如果不是，只输出：{"ok":false}\n'
            "规则：app 只能是 微信/QQ/钉钉/飞书 之一，判断不出就用「微信」；"
            "target 只填人名或会话名，不要带「给」「发」「说」等动词；"
            "text 是用户要传达的原话，不要改写、不要加解释。\n"
            f"任务：{task}"
        )
        try:
            raw = self.model.chat([{"role": "user", "content": prompt}])
            s, e = raw.find("{"), raw.rfind("}")
            data = json.loads(raw[s:e + 1]) if s >= 0 and e > s else {}
        except Exception as exc:
            self.log(f"[路由] 模型分类失败（{exc}），交给通用循环")
            return None
        if not isinstance(data, dict) or not data.get("ok"):
            return None
        app = str(data.get("app") or "微信").strip() or "微信"
        target = str(data.get("target") or "").strip()
        text = str(data.get("text") or "").strip()
        if not target or not text or len(target) > 20 or len(text) > 120:
            self.log(f"[路由] 参数不合理（target={target!r} text={text!r}），交给通用循环")
            return None
        ok, why = self._send_target_allowed(app, target)
        if not ok:
            self.log(f"[路由] ⛔ 已拦截：{why}")
            return {"blocked": why}
        self.log(f"[路由] 7B 判定为发消息 → {app} / {target}：{text}")
        return {"app": app, "target": target, "text": text}

    def _input_candidates(self, rect, limit: int = 3):
        """定位「聊天输入区」，返回候选点击坐标（程序自己测，不依赖硬编码比例）。

        所有聊天软件的输入框都满足同一组几何特征：在窗口下半部、宽度大、高度扁。
        用它给 YOLO 检出的元素打分，比写死 w*0.61 稳：窗口缩放、侧栏宽度不同、
        软件改版，都能重新找对。检测失败时退回几组几何估算，交给上层逐个试错。
        """
        w = max(1, rect.right - rect.left)
        h = max(1, rect.bottom - rect.top)
        cands = []
        try:
            from agent.yolo_grounding import detect_elements

            img = self.screenshot_fn()
            img_s, scale = self._fit_image(img)
            scored = []
            for d in detect_elements(img_s, conf=0.25):
                x1, y1, x2, y2 = (v / scale for v in d[:4])
                bw, bh = x2 - x1, y2 - y1
                if x2 < rect.left or x1 > rect.right or y2 < rect.top or y1 > rect.bottom:
                    continue  # 不在当前窗口内
                if y1 < rect.top + h * 0.45:
                    continue  # 输入框不会在上半部
                if bw < w * 0.30 or bh < 10 or bh > h * 0.40:
                    continue  # 太窄或太高，不是输入条
                score = bw / w - (bh / h) * 1.2 + ((y1 - rect.top) / h) * 0.4
                scored.append((score, (x1 + x2) / 2.0, (y1 + y2) / 2.0, bw, bh))
            scored.sort(key=lambda t: -t[0])
            for s, cx, cy, bw, bh in scored[:limit]:
                cands.append((cx, cy, f"视觉检测 {bw:.0f}x{bh:.0f}px 分{s:.2f}"))
        except Exception as e:
            self.log(f"[定位] 视觉检测不可用（{e}），退回几何估算")
        for px, py, tag in ((0.61, 0.90, "61%/90%"), (0.55, 0.88, "55%/88%"),
                            (0.50, 0.92, "50%/92%")):
            cands.append((rect.left + w * px, rect.top + h * py, f"几何估算 {tag}"))
        return cands

    def _send_message(self, app: str, target: str, text: str) -> bool:
        """程序级模板：在聊天软件里给某个会话发消息。
        模型只发 {"action":"send_message","app":"微信","target":"文件传输助手","text":"你好"}。
        序列：聚焦程序 → Ctrl+F 搜会话 → 敲会话名 → 点匹配结果（或 enter 兜底）
              → 定位聊天输入区（视觉检测优先，退几何估算，逐个候选试错）
              → 清残留 → 敲内容 → 自验证是否吃进 → 回车 → 自验证是否发出。
        """
        import ctypes
        import ctypes.wintypes

        import pyautogui

        from agent.actions import _type_unicode
        app, target, text = (app or "").strip(), (target or "").strip(), str(text or "")
        text = clean_message_text(text)      # 去引号、折叠双份标点
        target = target.strip(_QUOTE_CHARS)
        if not (app and target and text):
            self.log("[发消息] 参数不完整（app/target/text 都必填）")
            return False
        # 白名单闸（第一道，也是最后一道）：模板层再校一次，防模型在通用循环里
        # 直接发 {"action":"send_message"} 绕过路由层。
        allow, why = self._send_target_allowed(app, target)
        if not allow:
            self.log(f"[发消息] ⛔ 已拦截，未发送：{why}")
            return False
        # 测试护栏：所有发消息模板（含路由层的「计算并发送」）都从这里走。
        # 返回 True 让链路继续，验证的是流程而不是真的把消息推出去。
        if self.no_send:
            self.log(f"[NO-SEND] ⛔ 已拦截发送 → 会话「{target}」正文「{text}」"
                     "（护栏开启，未对外发消息）")
            return True
        self.log(f"[发消息] 将要发送 → 会话「{target}」正文「{text}」")
        self._wait_until_idle()   # 有人正在用电脑就先让路
        if not self._focus_program(app):
            # 窗口不在前台（常见：被关到托盘）→ 自己把它叫回来，别直接放弃
            self.log(f"[发消息] 「{app}」窗口不在前台，尝试自动打开它")
            if not self._open_app(app):
                self.log(f"[发消息] 打不开「{app}」，请先手动打开窗口再重试")
                return False
        time.sleep(0.8)

        # 1) Ctrl+F 搜索会话（主流聊天软件通用）
        img0 = self.screenshot_fn()
        pyautogui.hotkey("ctrl", "f")
        self._wait_screen_change(img0, timeout=1.5)
        _type_unicode(target)
        # 等结果列表渲染完再匹配：截太早会匹配不到，白白放弃点击退回 enter
        img = self._wait_screen_change(img0, timeout=3.0, settle=0.6)

        # 2) 在搜索结果里点目标会话；匹配不到就 enter 选第一项
        r = None
        if hasattr(self.grounding, "resolve_desc"):
            img_s, scale = self._fit_image(img)
            r = self.grounding.resolve_desc(target, img_s)
        if r:
            # 防误点：标签必须真含会话名，且不能是「包含 X 的聊天记录」这类分组行
            label = str(r[3] or "")
            clean = ("聊天记录" not in label) and ("聊天文件" not in label)
            if (target in label or label in target) and clean:
                x, y = r[0] / scale, r[1] / scale
                self.log(f"[发消息] 点搜索结果「{label}」(分{r[4]}) @({x:.0f},{y:.0f})")
                pyautogui.click(x, y)
            else:
                self.log(f"[发消息] 结果「{label}」不是会话条目，改按 enter")
                pyautogui.press("enter")
        else:
            self.log("[发消息] 未匹配到搜索结果，按 enter 选第一项")
            pyautogui.press("enter")
        time.sleep(1.2)

        # 2.5) 发送前的最后一道校验：顶栏会话名必须是目标。
        # 为什么必须有：搜索结果没匹配上时会按 enter 选第一项，万一第一项是别人的会话，
        # 接下来打的字 + 回车就发给真人了，撤不回来。这里宁可判失败也不发。
        header = self._chat_header_text()
        if header:
            hit = (target in header) or (header in target) or (
                len(target) >= 2 and target[:2] in header)
            if not hit:
                self.log(f"[发消息] ⛔ 顶栏是「{header}」，不是「{target}」—— "
                         "很可能进错了会话，停止发送（一个字都没发）")
                self._note_fail(
                    f"想给「{target}」发消息，但进入的会话是「{header}」。"
                    "搜索后要点中名字完全一致的那个会话，不能按 enter 随便选。")
                return False
            self.log(f"[发消息] 顶栏确认「{header}」= 目标会话 ✅")
        else:
            self.log("[发消息] 顶栏会话名 OCR 读不到，按原逻辑继续（存在发错风险）")

        # 3) 定位输入区并输入：视觉检测优先，逐个候选试错，每个都自验证
        u = ctypes.windll.user32
        hwnd = u.GetForegroundWindow()
        rect = ctypes.wintypes.RECT()
        u.GetWindowRect(hwnd, ctypes.byref(rect))
        w, h = rect.right - rect.left, rect.bottom - rect.top
        typed_ok, box, last = False, None, None
        for idx, (ix, iy, why) in enumerate(self._input_candidates(rect), 1):
            ix, iy = int(ix), int(iy)
            self.log(f"[发消息] 输入区候选 #{idx} @({ix},{iy}) ← {why}")
            pyautogui.click(ix, iy)
            time.sleep(0.5)
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("delete")
            time.sleep(0.2)
            box = (max(0, ix - int(w * 0.30)), max(0, iy - int(h * 0.12)),
                   min(self.safety.W, ix + int(w * 0.30)),
                   min(self.safety.H, iy + int(h * 0.12)))
            before = self.screenshot_fn().crop(box)
            _type_unicode(text)
            time.sleep(0.5)
            last = self.screenshot_fn().crop(box)
            # 真机 bug：给一个「，」发出去是「，，」。
            # 根因不是输入法：单个标点在 2560x1600 上只占约 100 像素，
            # 默认缩略图差分 (thr=0.003,max_w=320) 检不出，连 type 分支用的
            # thr=0.0001,max_w=None 也检不出（实测要 0.00001）。
            # 原来一判「没吃进」就换候选再敲一遍 → 一个标点变两个。
            # 现在用原分辨率极低阈值；并且正文只敲一次，检不出宁可放弃，绝不重敲。
            if self._changed(before, last, thr=0.00001, max_w=None):
                typed_ok = True
                self.log(f"[发消息] ✅ 文字已进输入框（候选 #{idx} 命中）")
                break
            self.log("[发消息] ⚠️ 正文已敲一次但差分无变化，放弃发送"
                     "（绝不重敲，避免一个标点变两个）")
            break

        # 4) 回车发送；用「有文字」和「发送后」两张图比，确认发出去了
        if not typed_ok:
            self.log("[发消息] ❌ 没找到输入区，文字没打进去，已放弃（未按回车，避免误发）")
            return False
        pyautogui.press("enter")
        time.sleep(1.2)
        # 发送后输入区清空也可能只是一个小标点的差别，同样得用极低阈值
        sent = self._changed(last, self.screenshot_fn().crop(box),
                             thr=0.00001, max_w=None)
        self.log("[发消息] ✅ 已发送（输入区已清空）" if sent
                 else "[发消息] ⚠️ 已回车，但输入区画面没变化，请核对是否发出")
        return True

    def _system_prompt(self, img_w=None, img_h=None) -> str:
        W, H = self.safety.W, self.safety.H
        iw, ih = (img_w or W), (img_h or H)
        return (
            "你是一个运行在 Windows 上的「电脑操作智能体」。你会看到当前屏幕截图和用户的任务，"
            "判断接下来该做什么，并按下面的【输出格式】只输出一个 JSON 对象（不要 markdown、不要解释）。\n"
            f"你看到的这张截图分辨率：宽 {iw} 像素，高 {ih} 像素。所有坐标都按这张截图的像素给，"
            f"相对图片左上角（x 范围 0~{iw - 1}，y 范围 0~{ih - 1}）。\n"
            "【输出格式 · 硬性要求】每次都必须输出这三个部分，一个都不能少：\n"
            '  {"plan":"用一句话说明这一步要干什么","actions":[{"action":"...",...}]}\n'
            # plan 原来要求写满 3 句。实测每步 90~110 token，按 21 token/s 就是
            # 4~5 秒生成时间 —— 而同一个任务 17 步里它全在复述同一句话
            # （"还没调字体/还是没选中文字"），信息量为零。压成一句 25 字，
            # 生成时间砍一半以上，决策信息一点没少。
            "  · plan —— 必填。一句话、不超过 25 字，说清「这一步做什么」；有失败原因就先说原因。\n"
            "      例：「上一步点击没生效，改按 Esc 再点右上角齿轮」。禁止照抄任务名、禁止复述截图。\n"
            "  · actions —— 必填，是数组，按顺序放接下来要做的动作：\n"
            "      下一步要点了才知道点哪 → 只放 1 个动作；\n"
            "      接下来几步都不需要看屏幕（按键→打字→回车、点一个位置→打字→回车、打开程序→按键）\n"
            "      → 一次放 2~4 个，系统会一口气做完，中间不截图。\n"
            "      例：{\"plan\":\"调出地址栏输路径再回车\",\"actions\":[{\"action\":\"key\",\"key\":\"ctrl+l\"},"
            "{\"action\":\"type\",\"text\":\"D:\\\\测试\"},{\"action\":\"key\",\"key\":\"enter\"}]}\n"
            "  · 硬规矩：一个数组最多 1 个点击，且必须是第一个；点击之后只能跟 type/key/wait/scroll。\n"
            "    点开菜单/文件夹/下拉框之后，里面【刚出现】的项绝不能放进同一个数组（位置要看了才知道）。\n"
            "允许的动作（每个动作对象的 action 字段只能选一个）：\n"
            '- open_app     : 打开程序 {"action":"open_app","app":"记事本"}  ← 打开任何程序首选这个，'
            "系统会自动完成「点开始→搜索→点开→验证」整条链路，最可靠\n"
            '- search_web   : 在浏览器里搜一下 {"action":"search_web","query":"今天的天气"}  ← '
            "凡任务涉及「搜索 / 查一下 / 查资料」，首选这个：系统会打开浏览器、聚焦地址栏、"
            "逐字输入、回车，整条链路一次做完。不要自己去找地址栏点\n"
            '- send_message  : 在聊天软件里发消息 {"action":"send_message","app":"微信",'
            '"target":"文件传输助手","text":"你好"}  ← 系统自动完成「聚焦→搜索会话→点开→输入→发送」'
            "整条链路；凡任务涉及给某人发消息，首选这个，不要自己逐步点\n"
            '- close_app    : 关闭程序 {"action":"close_app","app":"记事本"}  ← '
            "系统会走正常的关闭流程（遇到「要不要保存」会停下来问你）\n"
            '- click        : 单击    优先 {"action":"click","element_id":元素清单里的编号}；'
            '清单里没有的目标才用 {"action":"click","element_desc":"目标描述"} 或 {"action":"click","x":整数,"y":整数}\n'
            '- double_click : 双击    ← 打开文件夹、打开文件、启动桌面/列表里的程序图标，'
            "必须用这个（单击只会选中，不会进去）\n"
            '- right_click  : 右键    同上，优先 element_id\n'
            '- type         : 输入文字 {"action":"type","text":"要输入的文字，支持中文"}\n'
            '- key          : 组合键  {"action":"key","key":"enter 或 ctrl+c 或 win+r"}\n'
            '- scroll       : 滚动    {"action":"scroll","dy":6}  （dy>0 向下、dy<0 向上，'
            "一屏约 6；横向用 dx）\n"
            '- drag         : 拖拽    {"action":"drag","from":[x,y],"to":[x,y]}\n'
            '- wait         : 等待    {"action":"wait","ms":整数}\n'
            '- done         : 完成    {"action":"done","result":"一句话说明完成什么"}\n'
            '- ask          : 澄清    {"action":"ask","question":"向用户提的问题"}\n'
            "规则：1)必须按上面的【输出格式】给 plan + actions；绝不允许猜位置；2)完成整体任务输出 done；"
            "3)不要轻易 ask —— 只要你还能做点什么（比如先打开开始菜单搜索），就直接做，"
            "只有任务本身信息严重缺失、且没有任何可执行动作时才 ask；"
            "4)坐标必须落在屏幕内尽量点中目标中心；"
            "5)长任务分几段做时，做完一段直接接着做下一段，"
            "别回头重做已经完成的段落（真机上出现过查完天气又从头搜一遍）；"
            "6)不要重复上一步已经做过的动作 —— 先看截图确认它是否生效，生效了就往下走；"
            "7)若连续两次点不中/没反应，换个方式（例如改用键盘：win 键搜索、tab 切换）；"
                "8)【鼠标优先】像人一样用鼠标操作屏幕：打开程序可以点开始菜单里的图标，"
                "也可以点任务栏图标；窗口按钮、菜单、缩放、拖拽、右键都直接用鼠标。"
                "键盘（win 搜索/tab/enter/esc）是备选，不要只用键盘绕过鼠标操作。\n"
                "9)【找不到就别硬点】若截图上根本没有你要点的东西（比如记事本图标不在屏幕上），"
                "不要硬点别的图标凑数：先点开始按钮打开开始菜单再找图标；"
                "如果程序已经在窗口列表里，点它的任务栏图标即可。\n"
                "9b)【双击还是单击】桌面图标、文件资源管理器里的文件夹和文件、"
                "「此电脑」里的磁盘和目录 —— 这些必须用 double_click 才会打开/进入，"
                "单击只是选中（图标变蓝但画面不变）。开始菜单/任务栏/对话框里的按钮、"
                "菜单项则是单击。\n"
                "    判断依据：如果点完只是名字变蓝、状态栏出现「选中 1 个项目」，"
                "说明【目标点对了、动作错了】—— 下一步对【同一个编号/同一位置】用 double_click，"
                "绝对不要换个位置再单击一次。\n"
                "9c)【找不到就往下翻】元素清单只有屏幕上【当前可见】的东西。"
                "清单里没有目标 ≠ 屏幕上没有它 —— 长列表（文件夹、网页、聊天记录）里"
                "它很可能被滚到视野外了。这时候的正确动作是 {\"action\":\"scroll\",\"dy\":6} 往下翻一屏，"
                "看新截图再判断，而不是在原处换几个坐标反复点。只有在滚动之后仍然没有，才认为它不存在。\n"
                "10)【画面预判】每个动作都要带 \"expect_change\" 字段，先自己判断「这一步执行后，"
                "屏幕上会不会出现需要重新观察的新内容」：\n"
                "   会（弹窗/菜单/新页面/列表刷新/焦点明显切换/输入了文字）-> \"expect_change\": true\n"
                "   不会（点击已选中的项、纯等待、点击输入框内部、鼠标微调、重复点击同一位置）-> "
                "\"expect_change\": false\n"
                "   填 false 时系统会复用当前截图直接继续，省一次截屏；填错也没关系，系统会兜底校验。\n"
                "11)【达成即停】任务目标已经在屏幕上达成了（搜索结果已经显示出来、要输入的"
                "文字已经在输入框里、窗口已经打开且内容正确）—— 立刻输出 "
                '{"action":"done","result":"一句话说明"}，不要再多点任何一个东西。'
                "多余的一步点击经常把刚做好的事情弄没。\n"
                "12)【前台已对，直接做事】任务点名了某个程序，而【前台窗口】已经是它 —— "
                "它已经准备好接收操作了：直接点输入区、直接 type，"
                "**不要再发 open_app、不要再点它的标题栏/任务栏/空白处**。\n"
            "13)【捷径优先，别瞎找】你有两个比「一层层点菜单」快得多的手段，先想到它们：\n"
            "    ① 搜索框：开始菜单直接打字找程序、资源管理器地址栏 ctrl+l 直接输路径、"
            "应用里顶部的搜索框直接搜目标名字 —— 找东西一律优先用搜索，不要靠翻列表；\n"
            "    ② 快捷键：新建文件夹 ctrl+shift+n、重命名 F2、全选 ctrl+a、复制 ctrl+c、"
            "粘贴 ctrl+v、撤销 ctrl+z、聚焦地址栏 ctrl+l、切窗口 alt+tab、"
            "打开文件资源管理器 win+e、保存 ctrl+s、关闭当前窗口 alt+f4、跳到文末 ctrl+end。\n"
            "14)【不知道从哪进，就右键】Windows 里「对某个东西还能做什么」几乎都收在右键菜单里。"
            "当你不确定某件事该怎么开始（比如怎么转发一条消息、怎么解压、怎么改属性），"
            "正确做法是先在目标上右键看一眼，而不是在界面上乱试图标。"
            "例如：转发聊天消息 = 在消息上右键 → 在菜单里找含「转发」的项 → 选会话 → 发送；"
            "要一次转发多条 = 右键 → 多选 → 勾选要发的几条 → 转发。\n"
            "    这类菜单项在点开之前不存在于元素清单里，所以必须分两步做：\n"
            "    先右键把它点开，看过新截图里的清单，再点其中那一项。\n"
            "15)【说不上来的需求，先落到界面动作上】如果任务提到的功能你不知道在哪"
            "（转发、导出、收藏、设为…），先想「它属于哪个对象」——是某条消息、某个文件、"
            "还是某个窗口 —— 然后对这个对象右键。找不到再考虑应用顶部的「…」或齿轮图标。\n"
            "16)【打字前必须先点中输入框】type 只会把字打给「当前有焦点的那个地方」。"
            "不先点一下，字会掉进网页空白处或干脆丢失，而且画面几乎不变，你会误以为"
            "「打了但没生效」而反复重打（真机就卡死在这里）。所以：要输入时，这一串动作"
            "必须是 click(输入框) → type(内容) → key(enter)。浏览器里没有明确输入框时，"
            "改用 key ctrl+l 聚焦地址栏再 type —— 这比猜坐标可靠得多。\n"
            "【元素定位】点击类动作【首选】用 \"element_desc\" 给出目标的一句话描述，"
            "描述里尽量带上位置线索（如：任务栏最左边的开始按钮、右上角的最小化按钮、"
            "左侧边栏顶部的新建任务）——系统会用屏幕元素检测+文字匹配帮你精确点中，"
            "比你自己算 x/y 准得多。只有描述匹配不到时才回退给 x/y 坐标。\n"
            "【编号硬规矩】element_id 只能用下面清单里【已经出现】的编号。"
            "清单里没有的编号（比如清单最大是 80 你给了 377）一律无效，那一步等于白做。"
            "要点的东西不在清单里 → 改用 element_desc 描述它，或先滚动/打开它。\n"
        )

    # 送模型的元素清单行数上限。真机教训：80 行清单 = 约 2000 token，
    # 叠上 system 规则和 6 条历史，prompt 冲到 8033/8192 —— Ollama 那次
    # prefill 从 3.6s 飙到 20s+（整条请求 29 秒），只剩 159 token 余量。
    # 而且对 8B 来说选项从 80 个降到 48 个，挑目标反而更准。
    _ELEMENT_LIST_MAX = 48

    def _element_block(self, elements) -> str:
        if not elements:
            return ""
        # 之前 yolo_caption 模式不给清单，模型只能对着截图目测坐标（±90px 误差）。
        # 现在把「编号 + 名字 + 中心」列出来：7B 干"按名字选编号"这种语言任务很可靠，
        # 干"目测像素坐标"很不可靠 —— 把目测从链路里挤出去，就是准确率本身。
        lines = ["可交互元素清单（已检测到的可点目标；点击时优先发 "
                 "{\"action\":\"click\",\"element_id\":编号}，由系统精确定位，不要自己估坐标）："]
        named = [el for el in elements if (el.get("label") or "").strip()]
        _cap = self._ELEMENT_LIST_MAX
        for el in (named or elements)[:_cap]:
            x1, y1, x2, y2 = el["bbox"]
            name = (el.get("label") or "").strip() or "（未识别）"
            lines.append(f"[{el['id']}] {name} ({(x1 + x2) // 2},{(y1 + y2) // 2}) "
                         f"{int(x2 - x1)}x{int(y2 - y1)}")
        rest = len(elements) - min(_cap, len(named or elements))
        if rest > 0:
            lines.append(f"（另有 {rest} 个元素没能识别出名字，未列出；"
                         "清单里确实没有的目标，才用 element_desc 或 x/y 坐标）")
        return "\n".join(lines)

    def _build_user_text(self, task, elements) -> str:
        current = self.planner.current() or "整体任务"
        # prompt 逼近上下文上限时主动瘦身。history 是唯一可以砍的大块：
        # 任务不能丢、元素清单每步都要、失败记录正要用，只有"上上步干了什么"会过期。
        # 不砍的代价是硬撞上限：真机 8033/8192 那次 prefill 从 3.6s 飙到 20s+。
        _st0 = getattr(self.model, "last_stats", None) or {}
        _ctx0 = int(getattr(self.model, "num_ctx", 8192) or 8192)
        if _st0.get("prompt", 0) > 0.88 * _ctx0 and len(self.history) > 3:
            self.log(f"[上下文] 上一步 {_st0['prompt']}/{_ctx0} 已接近上限，"
                     "砍掉旧历史（只留任务和前两步）防 prefill 变慢")
            self.history = self.history[:1] + self.history[-2:]
            self._fail_log = (getattr(self, "_fail_log", None) or [])[-3:]
        # 历史每条截到 120 字：一条里塞着完整 plan 和系统注释时会长到 300+ 字，
        # 6 条就是 2000 token，而模型真正用到的只是「第几步做了什么」。
        _hist = [h if len(h) <= 120 else h[:117] + "…"
                 for h in self.history[-self.history_window:]]
        recent = "\n".join(_hist)
        pad = self.scratch.recent(6)
        pad_note = ("【你自己的草稿：最近真的做过的事】\n" + pad
                    + "（这些是已经做过的，不要重复；如果目标已经达成，直接输出 done 或做下一件事。）\n"
                    ) if pad else ""
        block = self._element_block(elements)
        note = ""
        miss = getattr(self, "_desc_miss", None)
        if miss:
            self._desc_miss = None
            fg_miss, _ = self._window_info()
            if fg_miss:
                # 目标程序已经开着 → miss 的真实原因是「那不是图标类控件」。
                # 还提示"先打开它"会把模型引进 open_app 死循环（真机踩坑）。
                note = (
                    f"❌ 你要点的「{miss}」没匹配到任何可交互元素。原因：视觉检测只能找到"
                    "【图标/按钮类控件】，窗口本身、标题栏、大面积文字输入区都不在元素清单里——"
                    "但这不代表它不存在。\n"
                    f"注意：当前前台窗口就是「{fg_miss}」。如果你要操作的就是它内部，"
                    "正确做法是：① 直接给 x/y 坐标，点截图上你亲眼看到的位置；"
                    "② 或者输入文字直接用 type（前台窗口自动接收键盘输入）。禁止再发 open_app。\n"
                )
            else:
                note = (
                    f"❌ 你刚才要点的「{miss}」不在当前屏幕上 —— 系统检测了全部可交互元素，"
                    "没有匹配项（描述也对不上）。先别再找它，它还没被打开。\n"
                    "正确做法：① 先点任务栏最左下角的开始按钮"
                    "（{\"action\":\"click\",\"element_desc\":\"任务栏最左边的开始按钮\",\"expect_change\":true}）"
                    "打开开始菜单；② 再输入程序名搜索或直接点菜单里的图标；③ 程序打开后才能操作它。\n"
                )
        if getattr(self, "_reused_screen", False):
            reuse_note = ("（注意：上一步动作预期不会产生新内容，因此这张截图与上一步相同，"
                          "画面本来就没变，不要因为「看起来没变化」就重复同一个动作；"
                          "如果目标已经达成请继续下一步或输出 done。）\n")
        else:
            reuse_note = ""
        if getattr(self, "_repeat_count", 0) >= 1:
            reuse_note += (
                f"⚠️ 你已经连续 {self._repeat_count} 次做几乎相同的动作且画面没有进展。"
                "必须换一种做法：改用键盘（win 搜索 / tab / enter / esc），"
                "或先按 esc 关掉弹层，或直接输出 done。严禁再重复同一个动作。\n"
            )
        if getattr(self, "_expect_failed", False):
            sig = getattr(self, "_last_action_sig", None)
            act = (sig[0] if sig else None) or "动作"
            reuse_note += (
                f"❌ 上一步的 {act} 你预期屏幕会变化，但画面纹丝不动 —— "
                "这个动作【没有生效】，很可能没点中/没聚焦/打在了没有输入焦点的地方。"
                "严禁原样重试。请先分析失败原因，再换一种做法（例如改用键盘）。\n"
            )

        # 【画面变化】图上红框是什么意思，必须解释。8B 会把任何标注当成界面的一部分
        # 去点（真机：它点过编号圆点）。顺便把「没框的地方＝已经看过」告诉它，
        # 省得它每步都把整屏重新描述一遍 —— 那些描述正是推理耗时的大头。
        chg = getattr(self, "_chg_regions", None) or []
        if chg and getattr(self, "frog_eye_model", "outline") != "off":
            chg_note = (f"【本步画面变化】图上有 {len(chg)} 个红框，标的是「与上一步相比"
                        "发生变化」的区域 —— 它们是标注，不是界面元素，不要点红框本身。"
                        "没被框住的部分和上一步完全一样、你已经看过了，不用重新描述；"
                        "注意力放在红框里，再判断任务是不是已经完成。\n")
        else:
            chg_note = ""
        fg, titles = self._window_info()
        win_note = ""
        if titles:
            win_note = ("【系统窗口状态】前台窗口（正在接收键盘输入的唯一窗口）：" + (fg or "未知") + "；"
                        "其他打开的顶层窗口：" + " | ".join(t for t in titles[:10] if t != fg) + "。\n"
                        "（这是系统真实状态，比截图可靠。要点：① 只有【前台窗口】才能接收键盘输入，"
                        "打字前必须让目标窗口成为前台；② 你要打开的程序若已在列表里，"
                        "说明它已经打开——不要再发 open_app，直接在这个窗口里做任务的下一步；\n"
                        "③ 截图上可能同时出现多个窗口：亮着的大块是前台窗口，暗下去的是背景。"
                        "**判断是否已经打开、永远以前台窗口标题为准，不要看背景**——"
                        "背景里的大窗口不是你要操作的对象。\n"
                        "⚠️ 不要用 alt+tab —— 它切到的是「上一个窗口」，不是你要的那个；"
                        "真机上模型连按两次 alt+tab 把自己切回了浏览器，白烧 4 步 230 秒。\n")
        mouse = self._cursor_pos()
        parked = mouse == self._PARK
        if mouse:
            mouse_note = (f"【鼠标状态】当前位于 ({mouse[0]},{mouse[1]})"
                          + ("（系统已把它停靠在屏幕左上角，截图上左上角的红圈十字就是它，"
                             "它不会遮挡任何界面内容）。\n" if parked else "（截图上的红圈十字标记就是它）。\n"))
        else:
            mouse_note = ""
        # 按前台窗口类型给一条"这个窗口里怎么点才有用"的硬规矩。
        # 只写进 system prompt 不够：7B 每步都会忘，贴着任务再讲一次才认。
        cls_now = self._foreground_class()
        if cls_now in ("CabinetWClass", "ExploreWClass"):
            dbl_note = ("【本窗口操作规矩】前台是文件资源管理器："
                        "① 列表里的文件夹、文件、磁盘必须 double_click 才会进入/打开，"
                        "单击只会变蓝（状态栏出现「选中 N 个项目」）；"
                        "② 要新建文件：点窗口左上角工具栏的「新建」按钮，"
                        "在弹出的菜单里选「文本文档」（别在列表里乱点）；"
                        "③ 要跳到其他路径：按 ctrl+l，地址栏会选中，直接打字输入路径再回车。\n")
        elif cls_now in ("Progman", "WorkerW"):
            dbl_note = ("【本窗口操作规矩】前台是桌面：桌面图标必须 double_click 才会打开，"
                        "单击只是选中。\n")
        else:
            dbl_note = ""
        hint = getattr(self, "_circuit_hint", "")
        if hint:
            self._circuit_hint = ""
        # 【复盘】上一轮失败后单独问出来的原因，比模型自己写的 plan 具体得多
        reflect = getattr(self, "_reflect_hint", "")
        if reflect:
            self._reflect_hint = ""
            reflect = (f"【复盘（你自己上一步失败后分析出来的，这一步要照着改）】\n{reflect}\n")
        # 【负经验】只给"已完成"是不够的：模型看不到自己失败过什么，
        # 就会一遍遍重试同一条死路（真机：同一区域点了 7 次、同一句话敲了 15 次）。
        fails = getattr(self, "_fail_log", None) or []
        fail_note = ""
        if fails:
            fail_note = ("【已经试过、并且失败了的做法 —— 不要再重复】\n"
                         + "\n".join(f"- {f}" for f in fails[-6:])
                         + "\n（同一件事失败两次以上，就必须换【工具级】做法："
                         "用 open_app 打开程序、用快捷键盘 ctrl+l / win / alt+tab，"
                         "而不是在屏幕上继续找位置。宁可直接输出 done 也别硬凑。）\n")
        # 【跨任务经验】只给极短一段：上下文只有 8192，撞到 8033 那次
        # prefill 从 3.6s 飙到 20s+，长期记忆绝不能变成新的负担。
        mem_note = self.long_memory.hint(task)
        # 【进场确认】动作生效的硬反馈。模型反复发 open_app 不是因为看不见截图，
        # 而是系统从没明确告诉它「你进去了」—— 之前日志里只有「⚠️ 未检测到窗口」。
        entry = getattr(self, "_entered_app", None)
        # 只在「刚进场」的这一两步里强制回灌：后面的任务可能要打开另一个程序，
        # 一直挂着「禁止 open_app」会把正常的第二步也堵死。
        if entry and getattr(self, "step", 0) - getattr(self, "_entry_step", -99) <= 1:
            entry_note = (
                f"✅【系统确认：你已经在「{entry['title']}」里了】"
                f"（{entry['via']}；前台进程 {entry['exe'] or '未知'}）。"
                "这是 Win32 读到的真实窗口状态 + 画面变化，不是猜的。\n"
                "【禁止再发 open_app / 再双击桌面图标 / 再搜一次程序名】——"
                "重复打开只会把你带回起点，白烧几十秒。"
                "直接在这个应用里做任务的下一步（例如新建对话、选模型、输入问题、点发送）。\n"
            )
        else:
            entry_note = ""
        return (
            f"任务：{task}\n"
            f"当前子任务：{current}\n"
            f"已完成步骤回顾：\n{recent}\n"
            + win_note
            + entry_note
            + chg_note
            + mouse_note
            + dbl_note
            + mem_note
            + pad_note
            + note
            + hint
            + reflect
            + fail_note
            + reuse_note
            + (block + "\n" if block else "")
            + '请按【输出格式】给出 plan + actions（只输出 JSON）：{"plan":"…","actions":[{…}]}\n'
            "plan 一句话、不超过 25 字，说清这一步做什么（有失败原因先说原因），"
        )

    def _trace(self, action, note="", before=None, result=""):
        """写一条轨迹，带上本步耗时和模型的自报思路。

        为什么单独包一层：这些字段是事后分析（tools/analyze_runs.py）的唯一依据。
        有没有它们，直接决定"下一步该优化什么"是拍脑袋还是有据可依。
        轨迹写失败绝不能影响任务本身，所以整段吞异常。
        """
        try:
            if self.memory.path is None:
                return
            t0 = (getattr(self, "_t_step_start", None)
                  or getattr(self, "_t_run", None) or time.time())
            self.memory.add(
                self.step, action, note, before, None,
                elapsed=max(0.0, time.time() - t0),
                think=getattr(self, "_plan_now", ""),
                result=result)
        except Exception:
            pass

    def _note_fail(self, text: str):
        """记一条「这招已经试过、没用」。

        模型的坏习惯是每一步都重新猜一遍，把失败过的做法摆在它眼前，
        比在症状上（改坐标、换措辞）修修补补有效得多。
        """
        log = getattr(self, "_fail_log", None)
        if log is None:
            log = self._fail_log = []
        t = str(text or "").strip()
        if not t or t in log:
            return
        log.append(t)
        del log[:-8]     # 只留最近 8 条，别把上下文撑满
        self.long_memory.note_fail(t)   # 同一条弯路，下次开机别再走

    def _reflect(self, why: str) -> str:
        """失败之后单独问一次「为什么」，把结论变成下一步的思考。

        为什么不让它在 plan 里自己想：plan 是"打算"，7B 每步都把它抄成同一句
        （真机 17 步里从头到尾都是"打开微信并进入示例联系人B的聊天记录"）。
        单独一次调用、只把失败现象摆在它面前，它才说得出一句具体原因。
        """
        try:
            recent = "\n".join(self.history[-4:])
            msg = [
                {"role": "system", "content":
                 "你是 Windows 电脑操作智能体的复盘模块。只输出两句话，中文，"
                 "不要 JSON、不要编号、不要说客套话。"},
                {"role": "user", "content":
                 f"任务：{self.task}\n最近做过的事：\n{recent}\n"
                 f"刚刚的失败现象：{why}\n"
                 "请说两点：①为什么会这样（一句）；②下一步换成什么做法"
                 "（一句，要具体到工具或按键，比如 open_app / win 键 / ctrl+l / alt+tab）。"},
            ]
            txt = (self.model.chat(msg) or "").strip()
        except Exception as e:
            self.log(f"[复盘] 调用失败：{e}")
            return ""
        txt = " ".join(txt.replace("**", "").split())
        return txt[:200]

    def _reflect_note(self, why: str):
        """把复盘结论写进日志（用户看得到）和历史（下一步模型看得到）。"""
        if not getattr(self, "reflect_on_fail", True):
            return
        if getattr(self, "_reflecting", False):
            return
        self._reflecting = True
        try:
            txt = self._reflect(why)
        finally:
            self._reflecting = False
        if not txt:
            return
        self.log(f"[思考] {txt}")
        self.history.append(f"第{self.step}步 复盘：{txt}")
        self._reflect_hint = txt

    # 任务里点名要用的程序（判断"该不该由程序直接打开"用）
    _KNOWN_APPS = ("企业微信", "微信", "钉钉", "飞书", "浏览器", "Edge", "Chrome",
                   "记事本", "计算器", "资源管理器", "Dev-C++", "终端", "QQ")

    def _app_in_task(self):
        t = (self.task or "").lower()
        for a in self._KNOWN_APPS:
            if a.lower() in t:
                return a
        return None

    def _dismiss_blocking_dialog(self) -> bool:
        """前台是个标准对话框、而任务根本没提保存/打开 → 是残留的挡路弹窗。

        为什么必须有：真机实测，屏幕上残留一个「另存为」对话框，
        模型对着它连发 3 步 open_app（它以为"另存为 = 记事本没打开"），
        4 步 88 秒一事无成。对话框不是任务的一部分，就该先关掉。

        返回 True 表示关掉了一个挡路对话框（本步动作跳过，重新看屏幕）。
        """
        if not getattr(self, "dialog_guard", True):
            return False
        if self._foreground_class() != "#32770":
            return False
        title = self._foreground_title() or ""
        task = (getattr(self, "task", "") or "").lower()
        if any(k in task for k in ("保存", "另存", "导出", "对话框", "确认", "弹出")):
            return False       # 任务本来就要操作对话框，别捣乱
        self.log(f"[清障] 前台是任务无关的对话框「{title}」，按 Esc 取消")
        self.executor.execute({"action": "key", "key": "esc"})
        time.sleep(0.8)
        self._note_fail("屏幕上有残留的对话框挡路（已按 Esc 取消）。"
                        "如果再遇到，先点对话框上的关闭/取消，不要绕开它做别的事。")
        return True

    def _goal_marker(self) -> str:
        """从任务里抠出"应该出现在屏幕上的东西"：引号内容优先，其次冒号后面。

        「在记事本里输入'完成'」→ 完成
        「打开记事本，输入三行字：苹果、香蕉、橘子」→ 苹果、香蕉、橘子
        没有明确目标物的任务返回空串（不做上屏判定）。
        """
        import re
        t = getattr(self, "task", "") or ""
        ms = re.findall(r"[「\"“']([^「」\"“”']{2,24})[」\"”']", t)
        if ms:
            return ms[-1].strip()
        m = re.search(r"[：:]\s*([^\s。；;]{2,24})", t)
        if m:
            return m.group(1).strip()
        return ""

    # 任务里点名「在哪干活」的程序 -> _PROC_MAP 里的标准名。
    # 命中就在通用循环前先把它打开：省掉模型推理「我要先打开浏览器」那一整步
    # （实测一步 = 一次 10~15 秒的看图 + 一次执行，20 秒上下）。
    # 顺序：长的先比，英文加词边界（"edge" 不能吃 "knowledge"）。
    _PREOPEN_MARKS = (
        ("dev-c++", ("dev-c++", "devc++", "dev-cpp")),
        ("资源管理器", ("资源管理器", "文件管理器")),
        ("计算器", ("计算器",)),
        ("记事本", ("记事本", "notepad")),
        ("浏览器", ("浏览器", "edge", "chrome", "火狐", "firefox")),
        ("微信", ("微信", "wechat")),
        ("画图", ("画图", "mspaint")),
        ("终端", ("终端", "powershell", "命令行")),
    )
    _PREOPEN_ASCII = frozenset(("edge", "chrome", "firefox", "notepad",
                                "wechat", "mspaint", "powershell", "dev-c++",
                                "devc++", "dev-cpp"))

    def _preopen_target_app(self) -> None:
        """任务里说了「在浏览器里/用记事本」→ 先把那个程序打开再让模型看屏幕。

        为什么值：通用循环的第一步几乎总是「打开目标程序」，这一步的决策
        没有信息含量，却要付一次完整的截图+推理。模板任务在上游就 return 了，
        走到这里的一定要进通用循环，所以这次预打开不会白做。
        """
        t = (getattr(self, "task", "") or "").lower()
        if not t:
            return
        for app, marks in self._PREOPEN_MARKS:
            for m in marks:
                if m in self._PREOPEN_ASCII:
                    hit = re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(m), t)
                else:
                    hit = m in t
                if not hit:
                    continue
                self.log(f"[起点] 任务点名了「{app}」，先打开它，省掉第一步决策")
                if self._open_app(app):
                    self._last_opened_app = app
                    self._gui_home_set()
                return

    def _take_over_desktop_click(self, action: dict) -> dict:
        """点在桌面空白处、任务里却点名了某个已知程序 → 程序直接把程序打开。

        真机最严重的一次：模型以为微信图标在桌面，在 (56,802)(1282,919)(241,885)…
        七八个位置反复单击/双击空桌面，17 步 275 秒一个窗口都没打开。
        "这个程序在不在桌面上"根本不需要模型猜 —— 桌面图标清单是毫秒级可读的。
        """
        if action.get("action") not in ("click", "double_click"):
            return action
        if self._foreground_class() not in ("Progman", "WorkerW"):
            return action
        x, y = action.get("x"), action.get("y")
        if x is None or y is None:
            return action
        items, _w = self._desktop_icons()
        if not items:
            return action
        for it in items:                    # 点在真图标上：照原样点，别抢
            try:
                if abs(x - float(it.get("x", -9999))) < 60 \
                        and abs(y - float(it.get("y", -9999))) < 60:
                    return action
            except (TypeError, ValueError):
                continue
        app = self._app_in_task()
        if not app:
            return action
        self.log(f"[接管] ({x:.0f},{y:.0f}) 是桌面空白处，任务要的「{app}」不在那儿"
                 f" → 程序直接打开，不再让你猜图标位置")
        self._note_fail(f"在桌面空白处点过，那里没有「{app}」的图标")
        self.history.append(
            f"第{self.step}步: 你点的 ({x:.0f},{y:.0f}) 是桌面空白区域，那里没有图标。"
            f"系统已经直接用系统能力打开「{app}」了，不要再点桌面。")
        action.clear()
        action.update({"action": "open_app", "app": app})
        return action

    def _take_over_app_icon_click(self, action: dict) -> dict:
        """"点任务栏/桌面上的 XX 图标" → 直接 open_app(XX)。

        真机（微信 7 步 449 秒）：模型说「先通过任务栏图标打开微信」，
        定位匹配到电池图标、点了两次没反应，白烧 3 步 + 一次 124 秒。
        "把已经装在机器上的程序打开"这件事不需要模型猜图标在哪 ——
        open_app 里已经有 已在运行→聚焦 / 桌面图标→双击 / 开始菜单搜索 三条确定性路径。
        """
        if action.get("action") not in ("click", "double_click", "right_click"):
            return action
        desc = str(action.get("element_desc") or action.get("_miss_desc") or "")
        if not desc or "图标" not in desc:
            return action
        # 「微信窗口里的发送图标」不是要打开微信，别抢
        if not any(k in desc for k in ("任务栏", "桌面", "状态栏", "托盘")) and len(desc) > 12:
            return action
        app = self._known_app_in_text(desc)
        if not app:
            return action
        # 目标程序已经在前台 → 不是"打开它"，别抢（可能真的是要拖它的图标）
        if self._app_is_foreground(app):
            return action
        self.log(f"[接管] 「{desc}」= 打开「{app}」，直接走 open_app（不猜图标坐标）")
        action.clear()
        action.update({"action": "open_app", "app": app})
        return action

    @classmethod
    def _known_app_in_text(cls, text: str):
        """从一句话里认出已知程序名（取最长命中，避免"文件资源管理器"被"资源管理器"抢）。"""
        t = (text or "").lower()
        hit = ""
        for name in cls._PROC_MAP:
            if len(name) >= 2 and name.lower() in t and len(name) > len(hit):
                hit = name
        return hit or None

    def _app_is_foreground(self, app: str) -> bool:
        title, _ = self._window_info()
        names = [app.lower()] + [x.lower() for x in self._PROC_MAP.get(app, [])]
        return bool(title) and any(n in title.lower() for n in names)

    @staticmethod
    def _describe_action(action) -> str:
        """把动作翻译成一句人话，喂回历史，避免模型反复做同一步。"""
        a = action.get("action")
        if a == "open_app":
            app = action.get("app")
            return (f"已打开程序「{app}」并确认窗口出现" if action.get("_open_ok")
                    else f"尝试打开程序「{app}」但未确认成功（可重试一次或换方式）")
        if a == "send_message":
            return (f"已给「{action.get('target')}」发送消息「{action.get('text')}」"
                    "（系统自动完成：搜索会话→点开→输入→回车发送）")
        if a == "search_web":
            return (f"已在浏览器里搜索「{action.get('query')}」并打开结果页"
                    if action.get("_search_ok")
                    else f"尝试搜索「{action.get('query')}」但没确认到结果页")
        if a in ("click", "double_click", "right_click"):
            desc = action.get("element_desc")
            if desc:
                return f"已{'双击' if a == 'double_click' else ('右键' if a == 'right_click' else '单击')}「{desc}」（系统已精确匹配到该元素）"
            return f"已{'双击' if a == 'double_click' else ('右键' if a == 'right_click' else '单击')} ({action.get('x')},{action.get('y')})"
        if a == "type":
            return f"已输入文字「{action.get('text', '')}」（输入框现在有这个内容了，别重复输入）"
        if a == "key":
            return f"已按按键 {action.get('key')}"
        if a == "scroll":
            return f"已滚动 ({action.get('dx', 0)},{action.get('dy', 0)})"
        if a == "drag":
            return f"已从 {action.get('from')} 拖到 {action.get('to')}"
        if a == "wait":
            return f"已等待 {action.get('ms')} ms"
        return f"已执行 {a}"

    # ------------------------- 解析动作 -------------------------
    @staticmethod
    def _repair_json(t: str) -> str:
        """修本地小模型常见的 JSON 手滑：key 没引号、单引号、尾逗号、中文引号。"""
        import re
        s = t.strip()
        s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        s = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', s)  # y":  ->  "y"
        s = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)"\s*:', r'\1"\2":', s)  # y":115 这种缺左引号
        s = re.sub(r",\s*([}\]])", r"\1", s)                                    # 尾逗号
        return s

    @staticmethod
    def _escape_bad_backslashes(t: str) -> str:
        r"""把 JSON 里非法的反斜杠转义补成合法的。

        小模型写 Windows 路径时几乎总是给 "D:\测试" —— \测 不是合法 JSON 转义，
        json.loads 直接失败，整步作废（真机上表现为「模型返回无法解析，重试」）。
        只动真正非法的那些（\测、\用、\t 之外的野字符），合法转义原样保留。
        """
        import re
        return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", t)

    @classmethod
    def _try_parse(cls, s: str):
        fixed = cls._repair_json(s)
        esc = cls._escape_bad_backslashes
        for cand in (s, fixed, esc(fixed), esc(s)):
            try:
                d = json.loads(cand)
            except Exception:
                continue
            if isinstance(d, dict) and ("action" in d or isinstance(d.get("actions"), list)):
                return d
        return None

    @classmethod
    def parse_action(cls, text: str):
        t = (text or "").strip()
        if "```" in t:  # 去掉 markdown 代码块围栏
            for part in t.split("```"):
                p = part.strip()
                if p.lower().startswith("json"):
                    p = p[4:]
                act = cls._try_parse(p.strip())
                if act:
                    return act
        act = cls._try_parse(t)
        if act:
            return act
        # 从每个 "{" 起找配对的 "}"，逐个试（应对前后夹着一堆解释文字的情况）
        for i, ch in enumerate(t):
            if ch != "{":
                continue
            for j in range(len(t) - 1, i, -1):
                if t[j] == "}":
                    act = cls._try_parse(t[i : j + 1])
                    if act:
                        return act
        return None

    # ------------------------- 动作序列（一次看屏幕，连做几个动作） -------------------------
    _BATCH_KB = ("type", "key", "wait", "scroll")            # 不需要看屏幕就能接着做
    _BATCH_PTR = ("click", "double_click", "right_click")     # 会改变画面，只能排第一个

    # 修饰键：模型经常把 ctrl+shift+n 拆成三个独立 key 动作。
    # 分开发送等于"按一下 ctrl、松手、按一下 shift、松手、按一下 n" —— 什么都不会发生。
    _MODIFIER_KEYS = {"ctrl", "control", "shift", "alt", "win", "cmd", "meta"}

    @classmethod
    def _merge_modifier_keys(cls, actions):
        """把「ctrl, shift, n」三条合并回「ctrl+shift+n」。

        真机证据：模型想按 Ctrl+Shift+N 新建文件夹，给了三个 key 动作，
        程序老老实实按了三次单键，资源管理器毫无反应，白烧一步。
        """
        out, mods = [], []
        for a in (actions or []):
            if not isinstance(a, dict):
                continue
            if str(a.get("action") or "") == "key":
                k = str(a.get("key") or "").strip().lower()
                if k in cls._MODIFIER_KEYS:
                    mods.append(k)
                    continue
                if mods and k:
                    a = dict(a)
                    a["key"] = "+".join(mods + [k])
                    a["_merged"] = True
                    mods = []
            else:
                mods = []      # 中间插了别的动作，孤立的修饰键没有意义，丢掉
            out.append(a)
        return out

    def _plan_batch(self, actions):
        """把模型给的 actions 数组裁成「真能连做」的一段。

        为什么必须裁：数组里第 2 个点击用的是第 1 个动作之前的截图坐标。
        点开菜单/文件夹之后，界面已经不是那张图了 —— 拿旧坐标点新界面
        正是真机里那些"乱点"的来源。所以只允许「最多一次点击，后面全是键盘」。
        """
        out = []
        ptr_used = False
        for a in self._merge_modifier_keys(actions)[:6]:
            if not isinstance(a, dict):
                continue
            name = str(a.get("action") or "").strip()
            if not name:
                continue
            if name in ("done", "ask"):
                break                      # 收尾动作必须单独一步，不能塞进序列里
            if name not in self._BATCH_KB + self._BATCH_PTR + ("open_app",):
                break
            if ptr_used and name not in self._BATCH_KB:
                self.log(f"[序列] 「{name}」要看一眼屏幕才知道点哪，序列到此为止")
                break
            if name in self._BATCH_PTR or name == "open_app":
                ptr_used = True
            out.append(a)
        return out

    def _run_batch(self, actions, img_s, scale, elements):
        """连续执行一串动作，中间不截图。返回 (status, executed)。

        status: OK=做完了该重新看屏幕 / DONE / ASK / STOP=一个都没做（交回单步逻辑）
        """
        planned = self._plan_batch(actions)
        if not planned:
            return "STOP", 0
        if len(planned) < 2:
            # 唯一例外：模型把组合键拆成几条单键，合并后只剩一条。
            # 这时必须执行合并结果（ctrl+shift+n），不能退回原始的单键动作。
            if not planned[0].get("_merged"):
                return "STOP", 0
            self.log(f"[序列] 几条单键合并成组合键「{planned[0].get('key')}」")

        # 同一串动作反复做 = 卡死。序列比单动作快，但也更容易掩盖"原地打转"。
        # 签名只看「做什么」，不看「点在哪」：模型打转时每轮都会在页面上换个像素位置
        # 点搜索框，把坐标拼进签名就等于给每次重复发新号，永远数不到 3 次
        # （真机 18~20 步：同一串「点搜索框→输二道区→回车」连做，一次都没被拦住）。
        # 代价：连续 scroll/按键这类天然可重复的动作要更高的阈值，见 _REPEAT_HARD。
        sig = " > ".join(
            "%s:%s" % (
                str(a.get("action") or ""),
                str(a.get("element_desc") or a.get("text") or a.get("key")
                    or a.get("app") or a.get("url") or "")[:40],
            )
            for a in planned
        )[:400]
        self._batch_sigs = getattr(self, "_batch_sigs", {})
        n_sig = self._batch_sigs.get(sig, 0) + 1
        self._batch_sigs[sig] = n_sig
        # 只读型动作（滚动/翻页/快捷键）重复点很正常，别误杀
        _read_only = all(str(a.get("action")) in ("scroll", "key", "wait", "screenshot")
                         for a in planned)
        _hard = 5 if _read_only else 3
        if n_sig >= _hard:
            self.log(f"⛔ [卡死] 同一串 {len(planned)} 个动作已连做 {n_sig} 次仍没完成任务，"
                     "判定卡住，主动终止（再跑下去只是烧时间）。")
            self._stuck_reason = ("模型反复执行同一串动作、画面无进展：" + sig[:120])
            return "ABORT", 0
        if n_sig >= 2:
            # 第二次就敲一记：让它自己换路，别等第三次被程序掐掉
            self.history.append(
                f"⚠️ 你刚才已经原样做过这串动作（{sig[:80]}），但画面没有变成你要的样子。"
                "再来一遍不会有不同结果：要么换工具级做法（open_app / 地址栏直接输网址），"
                "要么先确认真正卡住的环节（比如这个词搜不到结果，就换一个更宽的词）。")

        self.log(f"[序列] 一次连做 {len(planned)} 个动作（中间不重新截图）")
        _t0 = time.time()
        executed = 0
        done = False
        for i, act in enumerate(planned, 1):
            if self.stop_event.is_set():
                self.log("⛔ 已停止。")
                break
            name = str(act.get("action") or "").strip()

            # 第 1 个指针动作走和单步完全一样的定位链路（element_desc→元素→屏幕坐标）
            if name in self._BATCH_PTR:
                act = self._apply_element_desc(act, img_s, scale)
                name = str(act.get("action") or "").strip()   # 可能被程序改写成 key（如"开始按钮"）
                if name in self._BATCH_PTR:
                    if not act.get("_desc_locked"):
                        act = self._snap_to_element(act, elements)
                    act = self._to_screen_coords(act, scale)
                    act = self._prefer_double_click(act, scale)
                    act.pop("_desc_locked", None)
                    if act.get("x") is None:
                        # 第一个就定位失败时别把整串作废：_plan_batch 已经保证后面的动作
                        # 不需要重新看屏幕（纯键盘/纯输入），丢掉它继续做，比重新截图+
                        # 再问一遍模型便宜得多（真机：这一下白烧 28.5 秒，后面那句
                        # ctrl+a 明明是可以做的）。
                        if not executed and i < len(planned):
                            self.log(f"[序列] 第{i}个动作没定位到，跳过它，继续做后面的")
                            self.history.append(
                                f"第{self.step}步: 你要点的「"
                                f"{act.get('element_desc') or '那个东西'}」屏幕上找不到，"
                                "系统跳过了这一步，后面的动作照常执行。")
                            continue
                        self.log(f"[序列] 第{i}个动作没定位到，序列中止，改重新看屏幕")
                        break

            if name in ("click", "double_click", "right_click", "type"):
                if not self._guard_foreground():
                    self.log("[序列] 前台被别的程序抢走，序列中止")
                    break

            if name == "open_app":
                app = str(act.get("app") or "")
                if not app:
                    break
                # 与单步同款的防抖：还要确认前台标题真的含程序名，否则死循环
                fg_b = self._window_info()[0] or ""
                if app == getattr(self, "_last_opened_app", None) \
                        and AgentSession._focus_program(app) \
                        and (app in fg_b or AgentSession._title_matches(fg_b, app)
                             or self._fg_exe_matches(app)):
                    self.log(f"[序列] 「{app}」已在前台，跳过这一步")
                    continue
                if not self._open_app(app):
                    self.log(f"[序列] 打开「{app}」失败，序列中止")
                    break
                self._last_opened_app = app
                self._gui_home_set()
                self._desc_miss = None
            else:
                if name == "type" and self.type_guard_enabled:
                    if not self._type_guard(act):
                        self.log("[序列] 打字被拦下（前台是已打开的文件窗口），序列中止")
                        break
                    self._ensure_focus_for_type()
                if name == "scroll":
                    act["_at"] = self._scroll_point()
                res = self.executor.execute(act)
                if name == "type":
                    self._typed_recent.append(
                        (self._foreground_title(), str(act.get("text") or "")))
                    del self._typed_recent[:-6]
                if res is False:
                    done = True
                    break
                if res == "ASK":
                    return "ASK", executed
                if name in self._BATCH_PTR and act.get("x") is not None:
                    self._last_click_pt = (act["x"], act["y"])

            executed += 1
            self._park_cursor()
            did = self._describe_action(act)
            self.history.append(f"第{self.step}步.{i}: {did}")
            self.scratch.note(self.step, did, "已执行")
            self.log(f"[序列] 第{self.step}.{i} 个动作完成（这一步共 {time.time() - _t0:.1f}s）")
            if name != "type":
                time.sleep(0.25)   # 给窗口一点处理时间：回车/切焦点后立刻下一步会丢键

        return ("DONE" if done else "OK"), executed

    # ------------------------- 主循环 -------------------------
    def run(self, task: str, reply: str = None):
        """执行任务。reply 不为 None 表示这是模型 ask 后的用户回复（resume）。"""
        if reply is None:
            self._t_run = time.time()
            self._t_prev_step = None
            self.memory.start(task)
            self.scratch.start(task)   # 每个任务一张新草稿纸，绝不和上次混
            self.task = task
            self.history = [f"用户任务：{task}"]
            self.step = 0
            self._repeat_count = 0
            self._last_action_sig = None
            self._sig_counts = {}       # 整场任务里每个动作被执行的累计次数（卡死判定）
            self._expect_failed = False
            self._last_opened_app = None
            self._gui_home = None       # 任务锚定的窗口：动作前发现前台被抢就抢回
            self._fg_steal = 0
            self._desc_miss = None      # 上个任务的「目标不在屏幕上」不能污染新任务
            self._fail_log = []         # 本任务里已经失败过的做法，每轮都摆给模型看
            self._batch_sigs = {}       # 同一串动作连做了几次（卡死判定）
            self._stuck_reason = None   # 判卡死时给用户看的一句话原因
            self._entered_app = None    # 「已经进了某应用」只在本任务、刚进场的两步内有效
            self._idle_steps = 0        # 连续几步动作没让画面变化（收尾判定）
            self._goal_seen = False     # 目标文字是否已确认上屏（确认过就催模型收尾）

            # 路由 0.1：「打开浏览器搜索 X」/「查一下 X」—— 真人式的四步，程序保证十来秒完成。
            # 真机教训：这活儿交给通用循环，8 步 138 秒没完成，还没输出 done。
            search_q = self._parse_search_intent(task)
            if search_q:
                self.log(f"[意图] 网页搜索任务 → 搜「{search_q}」")
                _t_s = time.time()
                ok = self._browser_search(search_q)
                self.log(f"[耗时] 搜索模板 {time.time() - _t_s:.1f}s")
                if ok:
                    self._last_opened_app = "浏览器"
                    self._gui_home_set()
                    self.history.append(f"已在浏览器里搜索「{search_q}」")
                    self._trace({"action": "template", "name": "search_web",
                                 "query": search_q}, "1 步完成", result="done")
                    self.log("✅ 搜索完成。")
                    return "DONE"
                self.log("❌ 搜索模板没走通，交给通用循环自己找。")

            # 路由 0.15：「打开 www.xxx.com」—— 和搜索是同一套动作链，只是输地址。
            # 交给通用循环的代价：模型会把它当搜索词敲进搜索引擎，或者去点收藏夹。
            url_hit = self._parse_url_intent(task)
            if url_hit:
                _url, _host = url_hit
                self.log(f"[意图] 打开网址 → {_url}")
                _t_u = time.time()
                ok = self._browser_search(_url, kind="访问", expect=_host.lower())
                self.log(f"[耗时] 打开网址模板 {time.time() - _t_u:.1f}s")
                if ok:
                    self._last_opened_app = "浏览器"
                    self._gui_home_set()
                    self.history.append(f"已在浏览器打开 {_url}")
                    self._trace({"action": "template", "name": "open_url",
                                 "url": _url}, "1 步完成", result="done")
                    self.log("✅ 已打开网址。")
                    return "DONE"
                self.log("❌ 打开网址模板没走通，交给通用循环自己找。")

            # 路由 0.18：「打开某个文件」—— 用默认程序打开，等价于双击。
            # 交给通用循环的代价：模型会去资源管理器里翻目录、双击猜位置，
            # 而且它分不清"打开文件"和"打开程序"（会把 报告.txt 当成程序名去开始菜单搜）。
            file_name = self._parse_file_intent(task)
            if file_name:
                fpath = self._resolve_existing_file(file_name)
                if fpath:
                    self.log(f"[意图] 打开文件任务 → {fpath}")
                    _t_f = time.time()
                    ok = self._open_file(fpath)
                    self.log(f"[耗时] 打开文件模板 {time.time() - _t_f:.1f}s")
                    if ok:
                        self._trace({"action": "template", "name": "open_file",
                                     "path": fpath}, "1 步完成", result="done")
                        return "DONE"
                    self.log("❌ 打开文件模板没走通，交给通用循环自己找。")
                else:
                    self.log(f"[意图] 认得出是文件「{file_name}」，但在桌面/下载/文档里没找到，"
                             "交给通用循环自己找。")

            # 路由 0.05：「打开某程序并输入这几行字」—— 复合模板（开程序 → Ctrl+N →
            # 点编辑区 → 逐行敲 → OCR 验收）。必须排在纯 open_app 之前：后者见到
            # 「输入」会主动让路，这个任务就会掉进通用循环（真机 5 步 108 秒没写完）。
            type_job = self._parse_open_and_type_intent(task)
            if type_job:
                _t_ty = time.time()
                ok = self._open_and_type(type_job["app"], type_job["text"])
                self.log(f"[耗时] 输入文字模板 {time.time() - _t_ty:.1f}s")
                if ok:
                    self._last_opened_app = type_job["app"]
                    self._gui_home_set()
                    self.history.append(
                        f"已在「{type_job['app']}」里输入：{type_job['text'][:40]}")
                    self._trace({"action": "template", "name": "open_and_type",
                                 "app": type_job["app"]}, "1 步完成", result="done")
                    self.log("✅ 已输入完成。")
                    return "DONE"
                # 不退回通用循环：字可能已经进去一半，再敲一遍会变成双份
                self.log("❌ 输入文字模板没走通，停在模板层（避免把同样的字再敲一遍）")
                return "FAILED"

            # 路由 0：纯「打开某程序」—— 直接走 open_app 模板，不拆解、不逐步决策。
            # 真机教训：让 7B 拆解「打开记事本」会拆成 Win7 时代的老路径
            # 「点击开始菜单 → 选择所有程序 → 选择附件 → 选择记事本」，
            # 而 Win11 根本没有「所有程序/附件」这两级。模型从此一路在找不存在的
            # 东西，把「打开记事本」走成 15 步乱点。这一条把它变成 1 步。
            open_app_name = self._parse_open_intent(task)
            if open_app_name:
                self.log(f"[意图] 纯打开程序任务「{open_app_name}」→ 直接走 open_app 模板")
                _t_oa = time.time()
                ok = self._open_app(open_app_name)
                self.log(f"[耗时] 打开程序模板 {time.time() - _t_oa:.1f}s")
                if ok:
                    self._last_opened_app = open_app_name
                    self._gui_home_set()
                    self.history.append(f"已打开「{open_app_name}」")
                    self._trace({"action": "template", "name": "open_app",
                                 "app": open_app_name}, "1 步完成", result="done")
                    self.log(f"✅ 已打开「{open_app_name}」。")
                    return "DONE"
                self.log(f"❌ 模板没能打开「{open_app_name}」，改走通用循环自己找。")

            # 路由 0.2：「关闭某程序」—— 发 WM_CLOSE，等于替用户点 X。
            # 交给通用循环的代价：模型会去窗口右上角估坐标点，点偏就点到别的地方，
            # 而且遇到「要保存吗」的对话框它会乱按（真机把用户改了一半的文件丢了）。
            close_app = self._parse_close_intent(task)
            if close_app:
                self.log(f"[意图] 关闭程序任务「{close_app}」→ 直接发 WM_CLOSE")
                _t_cl = time.time()
                ok = self._close_app(close_app)
                self.log(f"[耗时] 关闭程序模板 {time.time() - _t_cl:.1f}s")
                if ok:
                    self._trace({"action": "template", "name": "close_app",
                                 "app": close_app}, "1 步完成", result="done")
                    self.log(f"✅ 已关闭「{close_app}」。")
                    return "DONE"
                self.log("❌ 关闭程序模板没走通（没开 / 弹了保存对话框）。")
                return "FAILED"

            # 路由 0.5：「先算 X 再发给 Y」—— 7B 做不了两步任务（真机 21 步 303 秒
            # 全在微信里反复敲 "123*123"，计算器一次都没打开）。
            # 算术是确定性运算：程序求值；发送走已验证的发消息模板。全程零推理。
            calc = self._parse_calc_send_intent(task)
            if calc:
                ok_c, why_c = self._send_target_allowed(calc["app"], calc["target"])
                if not ok_c:
                    self.log(f"⛔ 这个任务不执行：{why_c}")
                    self.log("   （白名单在 config.yaml 的 agent.wechat_allowed 里改）")
                    return "BLOCKED"
                self.log(f"[意图] 计算并发送：{calc['expr']} → 会话「{calc['target']}」"
                         f"（{calc['app']}）")
                _t_calc = time.time()
                ok = self._calc_then_send(calc["expr"], calc["app"], calc["target"])
                self.log(f"[耗时] 计算发送模板 {time.time() - _t_calc:.1f}s")
                self.history.append(
                    f"已把 {calc['expr']} 的结果发给「{calc['target']}」"
                    + ("（程序确认已发出）" if ok else "（程序判定没发出去）"))
                if ok:
                    self._trace({"action": "template", "name": "calc_then_send"},
                                "1 步完成", result="done")
                    self.log("✅ 模板任务完成。")
                    return "DONE"
                self.log("❌ 计算到了，但消息没能发出去（多因目标程序没打开/在托盘）。"
                         "请先把微信窗口打开再重试。")
                return "FAILED"

            # 路由 0.55：「算一道题」但没说发给谁 —— 打开计算器把结果留在屏幕上。
            # 必须在 calc_then_send 之后判（有收信人时那边优先）。
            calc_show = self._parse_calc_show_intent(task)
            if calc_show:
                self.log(f"[意图] 计算任务（不发送）→ {calc_show['expr']}")
                _t_cs = time.time()
                ok = self._calc_show(calc_show["expr"])
                self.log(f"[耗时] 计算器模板 {time.time() - _t_cs:.1f}s")
                self.history.append(f"已用计算器算出 {calc_show['expr']}"
                                    + ("，结果已显示在屏幕上" if ok else ""))
                if ok:
                    self._last_opened_app = "计算器"
                    self._gui_home_set()
                    self._trace({"action": "template", "name": "calc_show",
                                 "expr": calc_show["expr"]}, "1 步完成", result="done")
                    self.log("✅ 计算完成。")
                    return "DONE"
                self.log("❌ 计算器模板没走通，交给通用循环自己找。")

            # 路由 0.6：「在 Dev-C++ 里写代码并存成文件」—— 4 个确定性子步骤
            # （生成 / 新建 / 逐行输入 / 另存），模型只该负责「写什么代码」。
            # 真机教训：这类任务丢进通用循环，7B 第1步给 element_id:[34]（列表形式，
            # 程序没解析出来），随后在控制台页面同一坐标连点 6 次，Dev-C++ 从未打开。
            code_job = self._parse_code_task(task)
            if code_job:
                self.log(f"[意图] 写代码任务 → {code_job['app']}，文件「{code_job['name']}.c」")
                _t_code = time.time()
                ok = self._write_code_task(code_job)
                self.log(f"[耗时] 写代码模板 {time.time() - _t_code:.1f}s")
                if ok:
                    self._trace({"action": "template", "name": "write_code",
                                 "file": code_job["name"]}, "1 步完成", result="done")
                    self.log("✅ 模板任务完成。")
                    return "DONE"
                self.log("❌ 写代码任务未完成，看上面哪一步断的（生成/打开/输入/保存/编译）。")
                return "FAILED"

            # 路由 0.7：「打开/进入 某个路径」—— 资源管理器导航模板。
            # 真机教训：这活儿丢给通用循环，7B 在左栏空点 40 步 579 秒都没到 D 盘。
            # 导航是纯确定性操作；导航完的剩余目标（如"新建一个文件"）继续走通用循环。
            nav = self._parse_nav_task(task)
            if nav:
                nav_path, nav_rest = nav
                self.log(f"[意图] 打开路径任务 → {nav_path}"
                         + (f"，然后：{nav_rest}" if nav_rest else ""))
                _t_nav = time.time()
                ok = self._navigate_to(nav_path)
                self.log(f"[耗时] 导航模板 {time.time() - _t_nav:.1f}s")
                if ok:
                    self._gui_home_set()
                    m_new = self._NAV_NEWFILE_RE.search(nav_rest or "")
                    if m_new:
                        # 导航到了地方，剩余目标就是「新建文件/文件夹」→ 模板直接收尾
                        kind = ("文本文档" if m_new.group(1) in ("文件", "txt")
                                else m_new.group(1))
                        m_name = self._NAV_NEWNAME_RE.search(nav_rest or "")
                        self.log(f"[意图] 在 {nav_path} 里新建{kind}"
                                 + (f"，命名「{m_name.group(1)}」" if m_name else ""))
                        _t_nf = time.time()
                        ok2 = self._new_file_in_explorer(
                            kind, m_name.group(1) if m_name else None)
                        self.log(f"[耗时] 新建文件模板 {time.time() - _t_nf:.1f}s")
                        if ok2:
                            self._trace({"action": "template", "name": "nav_newfile",
                                         "path": nav_path}, "1 步完成", result="done")
                            self.log("✅ 模板任务完成。")
                            return "DONE"
                        self.log("❌ 新建模板没走通，剩余目标交给通用循环。")
                        task = nav_rest
                        self.task = nav_rest
                        self.history.append(
                            f"程序已在 {nav_path} 里尝试新建{kind}但可能没成功。"
                            f"请继续完成：{nav_rest}")
                    elif nav_rest:
                        task = nav_rest
                        self.task = nav_rest
                        self.history.append(
                            f"程序已在资源管理器打开 {nav_path}。继续完成剩余目标：{nav_rest}")
                    else:
                        self.history.append(f"程序已在资源管理器打开 {nav_path}")
                        self._trace({"action": "template", "name": "navigate",
                                     "path": nav_path}, "1 步完成", result="done")
                        self.log("✅ 已打开目标文件夹。")
                        return "DONE"
                else:
                    self.log("❌ 导航模板没打开目标文件夹，交给通用循环自己想办法。")

            # 路由 1：能落到程序模板的任务直接走稳定流程，不让 7B 逐步点像素。
            # 先正则（零成本），正则不中再让 7B 只做「分类+抽参数」，动作仍归程序。
            intent = self._route_task(task)
            if intent and intent.get("blocked"):
                self.log(f"⛔ 这个任务不执行：{intent['blocked']}")
                self.log("   （白名单在 config.yaml 的 agent.wechat_allowed 里改；"
                         f"当前允许：{'、'.join(self._allowed_targets())}）")
                return "BLOCKED"
            if intent:
                self.log(f"[意图] 识别为发消息任务：app={intent['app']} "
                         f"target={intent['target']} text={intent['text']}")
                _t_sm = time.time()
                ok = self._send_message(intent["app"], intent["target"], intent["text"])
                self.log(f"[耗时] 发消息模板 {time.time() - _t_sm:.1f}s")
                self.history.append(
                    f"已给「{intent['target']}」发送消息「{intent['text']}」"
                    + ("（发送成功）" if ok else "（程序判定没有发出去）"))
                if ok:
                    self._trace({"action": "template", "name": "send_message",
                                 "target": intent["target"]}, "1 步完成", result="done")
                    self.log("✅ 模板任务完成。")
                    return "DONE"
                # 假成功比失败更糟：明明没发出去还报完成，用户会被误导
                self.log("❌ 模板任务失败：程序没能完成发消息"
                         "（常见原因：目标程序没打开或最小化到了托盘）。"
                         "请先把程序窗口打开，再重试。")
                return "FAILED"

            self.subtasks = self.planner.decompose(task)
            self._pending = None
            self._last_elements = None
            self._last_annotated = None
            self.log(f"[规划] 拆成 {len(self.subtasks)} 步：{self.subtasks}")
        else:
            self.history.append(f"用户回复：{reply}")
            self.log(f"[你] {reply}")

        # 只有首轮才需要统一起点。模板任务在上面已经 return 了，能走到这里的一定要
        # 进通用循环，所以这次 Win+D 不会白付；用户中途插话（reply 非空）时不打扰。
        if reply is None:
            self._unstick_cursor()
        if self.home_desktop and reply is None:
            self._go_home_desktop()
        if getattr(self, "preopen_target", True) and reply is None:
            self._preopen_target_app()

        try:
            while self.step < self.max_steps:
                # 卡死判定只抓「同一串动作重复」，抓不住「步步不同却没进展」。
                # 评测现场必须有硬上限，否则评委只能盯着屏幕干等好几分钟。
                if self._out_of_time():
                    last = (self.history[-1][:100] if self.history else "无")
                    self.log(f"⏱ 已达总时长上限 {self.max_seconds:.0f}s（第 {self.step} 步），"
                             f"主动停止。最近一步：{last}")
                    return "TIMEOUT"
                self.step += 1
                _t_step = time.time()
                # 上一步的耗时在下一步开始时结算：continue / return 分支都能覆盖到
                if getattr(self, "_t_prev_step", None) is not None:
                    _bd = getattr(self, "_bd", {})
                    _parts = " ".join(f"{k} {v:.1f}" for k, v in _bd.items() if v >= 0.05)
                    self.log(f"[耗时] 第{self.step - 1}步 共 {_t_step - self._t_prev_step:.1f}s"
                             + (f"（{_parts}）" if _parts else ""))
                self._t_prev_step = _t_step
                self._t_step_start = _t_step     # 本步耗时（写进轨迹供事后分析）
                self._plan_now = ""              # 本步模型自报思路（同上）
                self._bd = {"截屏": 0.0}
                # 残留对话框清障：挡在前面就什么都做不了，先按 Esc 关掉
                if self._dismiss_blocking_dialog():
                    self._pending = None      # 画面变了，下一轮必须重新截图
                    continue
                if self.stop_event.is_set():
                    self.log("⛔ 已停止。")
                    return "STOPPED"

                before_path = ""
                self._reused_screen = False
                _t_shot = time.time()
                if self._pending is not None:
                    # 上一步已经截好的图（或判定无变化而沿用的旧图），不再重新截屏
                    img, same_screen = self._pending
                    self._pending = None
                    self._reused_screen = same_screen
                else:
                    img = self.screenshot_fn()
                    before_path = self.memory.save_image(img, f"step_{self.step:03d}_before.png")

                # ---- 青蛙眼：先跟上一帧比，看这一步实际动了哪块 ----
                # 不做这个也能跑，但整条视觉流水线是「每步无差别全屏重算」：
                # 点一下输入框、按个快捷键，画面其实只变了一个光标，也要重读 87 段
                # 文字、重认 40 个元素。差分让这些重复劳动直接消失。
                _chg, _chg_ratio = [], 0.0
                self._chg_regions = []
                if self.frog_eye:
                    _t_df = time.time()
                    _chg, _chg_ratio = self._frog_eye_diff(img)
                    self._bd["差分"] = time.time() - _t_df
                    self._chg_regions = _chg
                    if not self._reused_screen and _chg_ratio == 0.0 \
                            and self._last_annotated is not None:
                        # 像素级确认过：变化量在光标闪烁/时钟跳格这个量级，等于没动
                        self._reused_screen = True
                        self.log("[青蛙眼] 画面实质没变（只有光标级噪声），整条流水线复用")
                    elif _chg:
                        self.log(f"[青蛙眼] 变化 {len(_chg)} 块、占屏 {_chg_ratio * 100:.1f}%")

                # 缩放后再送模型（原图 2.5K 会撑爆 7B 模型的 4096 上下文 -> HTTP 400）
                img_s, scale = self._fit_image(self._dim_background(img))
                # 排查用：CU_KEEP_SHOTS=1 时把「模型实际看到的这张图」留下来。
                # 踩过的坑：模型一口咬定"屏幕上是文件资源管理器"，可前台明明是记事本，
                # 没有这张图就只能靠猜是模型幻觉还是程序送错了图。
                if os.environ.get("CU_KEEP_SHOTS"):
                    try:
                        self.memory.save_image(img_s, f"step_{self.step:03d}_model.png")
                    except Exception:
                        pass
                _t_det = time.time()
                if self._reused_screen and self._last_annotated is not None:
                    # 同一张画面，元素检测结果直接复用，省一次 YOLO 推理
                    elements, annotated = self._last_elements, self._last_annotated
                    self.log(f"[元素] 画面未变，复用上次检测的 {len(elements)} 个元素")
                else:
                    elements, annotated = self.grounding.annotate(img_s.copy())
                    self._last_elements, self._last_annotated = elements, annotated
                    self.log(f"[元素] 检出 {len(elements)} 个可交互元素")
                self._bd["检测"] = time.time() - _t_det
                # 给元素补名字：只有「编号+坐标」的清单模型认不出目标，
                # 会退化成目测截图小字，认不清就编编号/猜坐标（真机：同一处点出 5 个 x）
                _ocr_new = []
                # 画面没变 → 元素清单就是上次那一份（已含 OCR 补出来的文字元素），
                # 再 OCR 一遍结果必然一模一样。真机实测这一跑 2.0 秒，而整步模型
                # 决策才 5 秒 —— 白占三成时间。复用时直接跳过。
                if self._reused_screen and self._last_annotated is not None:
                    self.log("[元素] 画面未变，OCR 结果复用（省一次全屏识别）")
                    self._bd["OCR"] = 0.0
                elif hasattr(self.grounding, "ocr_enrich"):
                    _t_ocr = time.time()
                    _crop, _keep, _regs = (None, [], [])
                    _popup_only = (self.frog_eye_model == "popup")
                    if self.frog_eye and _chg:
                        _crop, _keep, _regs = self._frog_eye_plan(
                            img, _chg, scale, popup_only=_popup_only)
                        if _popup_only:
                            self.log("[青蛙眼] 变化区是弹窗形状 -> 只读这一块"
                                     if _crop else "[青蛙眼] 不是弹窗 -> 走全屏")
                    if _crop and _keep:
                        # 只对变化的那一块读字。不变区域的旧元素（名字、坐标都现成）
                        # 原样捡回来 —— 这才是「青蛙眼」真正省时间的地方：不用重认。
                        _inner = [el for el in elements
                                  if any(self._rects_overlap(el["bbox"], r, pad=8)
                                         for r in _regs)]
                        _prev_texts = [
                            t for t in (getattr(self.grounding, "last_texts", None) or [])
                            if not any(self._rects_overlap(
                                [t[0] / scale, t[1] / scale, t[2] / scale, t[3] / scale], r)
                                for r in _chg)]
                        _ocr_new = self.grounding.ocr_enrich(
                            img.crop(_crop), _inner, scale=scale,
                            origin=_crop[:2], extra_texts=_prev_texts)
                        elements = _keep + _inner + _ocr_new
                        for _i, _el in enumerate(elements, 1):
                            _el["id"] = _i          # 旧元素的 id 会和新的撞车，重新编号
                        annotated = self.grounding._draw(img_s.copy(), elements)
                        self.log(f"[青蛙眼] OCR 只跑变化区 "
                                 f"{_crop[2] - _crop[0]}x{_crop[3] - _crop[1]}，"
                                 f"复用旧元素 {len(_keep)} 个、新补文字 {len(_ocr_new)} 个"
                                 f"（{time.time() - _t_ocr:.1f}s）")
                    else:
                        _ocr_new = self.grounding.ocr_enrich(img, elements, scale=scale)
                        if _ocr_new:
                            elements = elements + _ocr_new
                            annotated = self.grounding._draw(img_s.copy(), elements)
                    self.grounding._last_elements = elements   # element_id 解析用
                    self._last_elements, self._last_annotated = elements, annotated
                    if not (_crop and _keep):
                        self.log(f"[元素] OCR 补文字元素 {len(_ocr_new)} 个"
                                 f"（{time.time() - _t_ocr:.1f}s）")
                    self._bd["OCR"] = time.time() - _t_ocr
                if hasattr(self.grounding, "ensure_captions_batch"):
                    _t_cap = time.time()
                    # 认名一次调用有 4~5 秒的图像编码固定开销，生成时间随元素数线性涨。
                    # 真机实测每步 4.5~9.5 秒 —— 推理才 10 秒，它吃掉三分之一，
                    # 而模型照样点错位置。所以：只认前 20 个，并且按面积从大到小排
                    # （按钮/菜单项比装饰性小点大，小点本来就该用坐标点）。
                    _cap_pool = sorted(
                        elements[:40],
                        key=lambda e: -((e["bbox"][2] - e["bbox"][0]) *
                                        (e["bbox"][3] - e["bbox"][1])))
                    self.grounding.ensure_captions_batch(img_s, _cap_pool, max_n=20)
                    _named = sum(1 for el in elements[:40] if el.get("label"))
                    self.log(f"[元素] 已认名 {_named}/{min(40, len(elements))} 个"
                             f"（7B 补 {time.time() - _t_cap:.1f}s）")
                    self._bd["认名"] = time.time() - _t_cap
                # 把本步的变化区域框出来。图仍然是全屏 —— 方位感不能丢 —— 只是用红框
                # 指出「刚才动的就是这儿」。8B 的注意力在图上，不在文字说明里。
                annotated = annotated.copy()      # 复用分支拿到的是缓存对象，别画脏它
                if self.frog_eye and self.frog_eye_model != "off" and _chg:
                    annotated = self._mark_changes(
                        annotated, _chg, scale,
                        dim=0.45 if self.frog_eye_model == "dim" else 0.0)
                self._mark_cursor(annotated, scale)   # 截图上画出光标（Windows 截图不含光标）
                # 存下这一帧的最终清单（含认名结果）：下一帧的不变区域直接拿来用。
                # 必须放在认名之后 —— 存早了，复用回来的还是一批没名字的框。
                self._frog_els = [{**el, "bbox": list(el["bbox"])} for el in elements]
                system = self._system_prompt(*annotated.size)

                user_text = self._build_user_text(self.task, elements)
                content = [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(annotated)}},
                ]
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ]

                self.log(f"—— 第 {self.step} 步，思考中…")
                if os.environ.get("CU_DEBUG_PROMPT"):
                    self.log(f"[提示词] {user_text[:160].replace(chr(10), ' / ')}")
                _t_model0 = time.time()
                try:
                    raw = self.model.chat(messages)
                except Exception as e:
                    self.log(f"[错误] 模型调用失败: {e}\n请确认 Ollama/后端已启动且模型名正确。")
                    return "ERROR"
                self.log(f"[模型] 推理 {time.time() - _t_model0:.1f}s")
                self._bd["推理"] = time.time() - _t_model0
                # prompt 规模摆到台面上：它是"同一任务时而 10 秒时而 30 秒"的唯一解释
                _st = getattr(self.model, "last_stats", None) or {}
                _ctx = int(getattr(self.model, "num_ctx", 8192) or 8192)
                if _st.get("prompt"):
                    _ratio = _st["prompt"] / max(1, _ctx)
                    self.log(f"[上下文] {_st['prompt']}/{_ctx} token"
                             f"（输出 {_st.get('gen', 0)}）"
                             + ("  ⚠️ 逼近上限，prefill 会突然变慢"
                                if _ratio > 0.85 else ""))

                action = self.parse_action(raw)
                if not action or ("action" not in action and not action.get("actions")):
                    self.log(f"[无法解析] 模型返回：{raw[:200]}")
                    self.history.append(f"第{self.step}步: 模型返回无法解析，重试")
                    time.sleep(1)
                    continue

                # 模型自报思路（可选 plan 字段）：既给用户看，也让下一步的它记得自己想干什么
                _plan = str(action.pop("plan", "") or "").strip()
                if _plan:
                    self._plan_now = _plan
                    self.log(f"[思考] {_plan}")
                    self.history.append(f"第{self.step}步 思路：{_plan}")

                # 一次给了一串「不需要重新看屏幕」的动作 → 连做，整串只算一步。
                # 这是"步数变少"的主手段：一次贵重的看图，换回好几个动作。
                if isinstance(action.get("actions"), list):
                    acts = [a for a in action.pop("actions")
                            if isinstance(a, dict) and a.get("action")]
                    if action.get("action"):          # 兼容 {"action":..., "actions":[...]} 的手滑
                        acts.insert(0, action)
                    st, n = self._run_batch(acts, img_s, scale, elements)
                    if st == "ABORT":
                        self.log("⛔ 任务因原地打转终止。")
                        return "STUCK"
                    if st in ("OK", "DONE") and n:
                        self._trace({"actions": acts}, f"连做 {n} 个动作",
                                    before_path, result="batch")
                        if st == "DONE":
                            self.log("✅ 任务结束。")
                            return "DONE"
                        self._pending = (self.screenshot_fn(), False)
                        continue
                    if st == "ASK":
                        self.log("（模型在等你回复，回复后继续）")
                        return "NEED_REPLY"
                    if st == "OK" and not n:
                        # 程序判定这串动作在原地打转，一个都不做——逼它换工具级做法
                        self._pending = (self.screenshot_fn(), False)
                        continue
                    if not acts:
                        self.history.append(f"第{self.step}步: 你给的 actions 是空的，请给动作。")
                        self._pending = (self.screenshot_fn(), False)
                        continue
                    # 这串连不了（比如第 2 个动作还在点新出现的菜单）→ 只执行第一个，其余作废
                    if len(acts) > 1:
                        self.log(f"[序列] 后 {len(acts) - 1} 个动作要看屏幕才知道怎么做，"
                                 "先只做第一个")
                    action = acts[0]
                    # 同一个动作在序列里重复出现 = 模型在打转，提前给一句硬提醒
                    if len(acts) > 1 and acts[1].get("action") in ("click", "double_click"):
                        self.history.append(
                            f"第{self.step}步: 你想连点「{acts[1].get('element_desc') or acts[1].get('x')}」，"
                            "但它的位置要等上一个动作做完才知道，系统没执行它。"
                            "先看这张新截图，再决定点哪里。")

                # 小模型偶尔给出 {"actions": "..."} 这类形状不对的东西：当解析失败处理，别硬执行
                if "action" not in action:
                    self.log(f"[无法解析] 结构不对：{json.dumps(action, ensure_ascii=False)[:160]}")
                    self.history.append(
                        f"第{self.step}步: 你的 JSON 结构不对。要么给单个动作 "
                        '{"action":"click",...}，要么给 {"actions":[{...},{...}]}。')
                    self._pending = (self.screenshot_fn(), False)
                    continue

                action = self._apply_element_desc(action, img_s, scale)
                action = self._snap_to_element(action, elements)
                action = self._to_screen_coords(action, scale)
                action.pop("_desc_locked", None)
                # 坐标已落地，判断要不要双击必须在这之后：桌面/资源管理器列表里
                # 的条目单击只选中，模型真机上从没自己用过 double_click
                action = self._prefer_double_click(action, scale)
                action = self._take_over_desktop_click(action)   # 点桌面空白处 → 程序直接开程序
                action = self._take_over_app_icon_click(action)  # 点"XX 图标" → 直接 open_app
                self.log(f"[决策] {json.dumps(action, ensure_ascii=False)}")

                # 点击类动作必须有可用坐标；desc 匹配不到/element_id 无效就明确告诉模型，别让它一直循环
                if action.get("action") in ("click", "double_click", "right_click", "drag") \
                        and action.get("x") is None and elements:
                    tgt = (action.get("element_desc") or action.get("_miss_desc")
                           or "（模型没说要点哪，这是它自己的错）")
                    # 目标可能只是被滚出视野了。先让程序自己往下翻几屏找 ——
                    # 7B 不会想到"滚动"这一步，它会原地换坐标乱点（真机 37 步 7 个 x）
                    found = None
                    if action.get("action") == "click" and self._explorer_list_rect():
                        found = self._scroll_to_find(str(tgt))
                    if found:
                        action["x"], action["y"] = found
                        action = self._prefer_double_click(action, scale)
                        self.log(f"[决策] {json.dumps(action, ensure_ascii=False)}")
                    else:
                        ids = ",".join(str(e["id"]) for e in elements[:20]) or "（空）"
                        self.log(f"[无效] 目标'{tgt}'没有可用坐标")
                        self._note_fail(
                            f"想点「{tgt}」，但屏幕上找不到它（元素清单里没有，描述也匹配不上）")
                        self._reflect_note(f"要点的「{tgt}」在屏幕上找不到，什么也没做")
                        self.history.append(
                            f"第{self.step}步: 你要点的「{tgt}」匹配不到任何已检测元素。三种可能："
                            "① 它被滚出了视野（长列表里很常见）；② 它真的不在这台机器上；"
                            "③ 它是窗口/标题栏/大面积输入区这类【非图标控件】，"
                            "视觉检测本来就不覆盖（不代表它不存在）。\n"
                            "正确做法：① 先 {'\"action\":\"scroll\",\"dy\":6} 往下翻一屏再看；"
                            "② 或者直接给 x/y 坐标，点截图上你亲眼看到的位置"
                            f"（可用元素 id 仅供参考: {ids[:60]}…）。\n"
                        )
                        continue

                # 安全闸门：高风险操作需人工确认
                if self.safety.is_risky(action):
                    self.log("⚠️ [安全] 检测到高风险操作，等待确认…")
                    if not self.confirm(action):
                        self.log("⛔ 用户拒绝执行，本步跳过。")
                        self.history.append(f"第{self.step}步: 用户拒绝高风险操作")
                        continue

                # 执行前停顿，给人干预时间
                if self.step_pause > 0:
                    time.sleep(self.step_pause)
                self._fg_before = self._window_info()[0]   # 动作前的前台标题（type 验证用）
                self._fg_exe_before = self._foreground_exe()  # 动作前前台进程（进场判定用）
                self._entry_shot = None
                self._entry_just_confirmed = False

                # 程序级模板：算术（模型只给表达式，结果由 AST 求值，避免它心算出错）
                if action.get("action") == "calc":
                    expr = str(action.get("expr") or action.get("text") or "")
                    try:
                        val = _safe_eval(expr)
                        line = f"{_normalize_expr(expr)}={_fmt_number(val)}"
                        self.log(f"[计算] {line}（程序求值）")
                        self.history.append(
                            f"第{self.step}步: 程序算出 {line}。这个结果可以直接用 "
                            f"type 输入，或作为消息正文发给目标。不要再自己心算。")
                    except Exception as exc:
                        self.log(f"[计算] 「{expr}」算不出来（{exc}）")
                        self.history.append(f"第{self.step}步: 算式「{expr}」不合法（{exc}），换一条。")
                    self._pending = (img, True)
                    continue

                # 程序级模板：发消息（模型只给参数，链路由程序可靠执行）
                if action.get("action") == "send_message":
                    action["_sent_ok"] = self._send_message(
                        str(action.get("app", "")),
                        str(action.get("target", "")),
                        str(action.get("text", "")),
                    )
                    cont = True
                # 程序级模板：打开程序（模型只发一条指令，序列由程序可靠执行）
                # 程序级模板：网页搜索（开浏览器→ctrl+L→打字→回车，整条链路由程序保证）
                elif action.get("action") == "search_web":
                    q = str(action.get("query") or action.get("text") or "").strip()
                    action["_search_ok"] = self._browser_search(q) if q else False
                    if action["_search_ok"]:
                        self._last_opened_app = "浏览器"
                        self._gui_home_set()
                    cont = True
                elif action.get("action") == "close_app":
                    app_name = str(action.get("app", ""))
                    if not app_name:
                        self.history.append("第{self.step}步: close_app 缺少 app 参数。")
                        self._pending = (img, True)
                        continue
                    action["_close_ok"] = self._close_app(app_name)
                    self.history.append(
                        f"第{self.step}步: close_app「{app_name}」"
                        + ("已完成" if action["_close_ok"] else "没关掉（没开/弹了保存框）"))
                    cont = True
                elif action.get("action") == "open_app":
                    app_name = str(action.get("app", ""))
                    # 防抖的三重条件：① 刚打开过；② _focus_program 切得动；③ 前台标题真的含程序名。
                    # 少了 ③ 会死循环：_focus_program 自报"已聚焦"，可实际画面还是别的窗口，
                    # 模型看到的和程序说的对不上，就一直发 open_app 被一直拒（真机 5 步 108 秒一事无成）。
                    fg_title_now = self._window_info()[0] or ""
                    if app_name == getattr(self, "_last_opened_app", None) \
                            and AgentSession._focus_program(app_name) \
                            and (app_name in fg_title_now
                                 or AgentSession._title_matches(fg_title_now, app_name)
                                 or self._fg_exe_matches(app_name)):
                        # 防抖：这个程序刚打开过且窗口还在，禁止无限重开
                        self.log(f"[防抖] 「{app_name}」已在前台，拒绝重复 open_app")
                        self.history.append(
                            f"第{self.step}步: 系统提示——「{app_name}」已经打开并在前台，"
                            "禁止再发 open_app！它窗口内部的东西（输入区/菜单/文字）不是图标类控件，"
                            "元素清单里没有——想点就给 x/y 坐标（点截图上你亲眼看到的位置），"
                            "想打字就直接 type。"
                        )
                        self._pending = (self.screenshot_fn(), False)
                        continue
                    action["_open_ok"] = self._open_app(app_name)
                    if action["_open_ok"]:
                        self._last_opened_app = app_name
                        self._gui_home_set()
                        # 程序已确认打开，之前积压的「目标不在屏幕上」提示全部作废——
                        # 不清掉的话下一轮会误导模型去点开始按钮（真机踩过的坑）
                        self._desc_miss = None
                    cont = True  # 继续走正常的画面判定/反馈流程
                else:
                    # 前台守卫：点/拖/打字落进别人的窗口是事故（真机两次），先守后动
                    if action.get("action") in ("click", "double_click", "right_click",
                                                "drag", "type") \
                            and not self._guard_foreground():
                        if getattr(self, "_fg_steal", 0) >= 2:
                            self.log("⛔ [守卫] 你正在使用电脑，程序让出鼠标键盘，任务中止。"
                                     "等电脑空闲后重新下达任务即可。")
                            return "FAILED"
                        self.history.append(
                            f"第{self.step}步: 你的动作没有执行——执行前发现前台窗口被切走了"
                            "（用户在用电脑）。程序已尝试把任务窗口点回前台，"
                            "请先看新截图确认窗口状态，再继续原目标。")
                        self._pending = (self.screenshot_fn(), False)
                        continue
                    if action.get("action") == "type" and self.type_guard_enabled:
                        if not self._type_guard(action):
                            self._pending = (self.screenshot_fn(), False)
                            if self._type_block_reason == "duplicate":
                                # 真机：打完字不收尾，同一句话又敲一遍。文字已确认在屏幕上，
                                # 直接判完成 —— 否则 6 步全在原地重复敲同一句
                                self.log("[收尾] 文字已在屏幕上，判定任务完成")
                                self.history.append(
                                    f"第{self.step}步: 「{action.get('text', '')}」"
                                    "已经输入并在屏幕上显示，任务完成")
                                return "DONE"
                            self.history.append(
                                f"第{self.step}步: 打字被拦下 —— 前台窗口正打开着一个文件，"
                                "字会改进那个文件里。先换到正确的输入位置再输入。")
                            continue
                    if action.get("action") == "type":
                        self._ensure_focus_for_type()
                    if action.get("action") == "scroll":
                        # 滚轮只对鼠标下面的窗口生效，光标此刻停在左上角，
                        # 不给位置就会滚到别人的窗口里
                        action["_at"] = self._scroll_point()
                    cont = self.executor.execute(action)
                    if action.get("action") == "type":
                        # 记下「这个窗口里现在有这段字」：下一步模型重复同一句时，
                        # 靠它 + OCR 判定"已经输入过了"，直接收尾而不是再敲一遍
                        self._typed_recent.append(
                            (self._foreground_title(), str(action.get("text") or "")))
                        del self._typed_recent[:-6]
                    if action.get("action") in ("click", "double_click", "right_click") \
                            and action.get("x") is not None:
                        # 记下这次点在哪：滚动/后续动作可能需要一个"在这附近操作"的锚点
                        self._last_click_pt = (action["x"], action["y"])
                self.history.append(f"第{self.step}步: {self._describe_action(action)}")
                self._park_cursor()  # 动作完鼠标停靠左上角：状态确定、hover 干扰消失
                did = self._describe_action(action)
                if action.get("_sent_ok") is False or action.get("_open_ok") is False:
                    self.scratch.note(self.step, did, "失败（程序判定没生效）")
                else:
                    self.scratch.note(self.step, did, "已执行")

                # 死循环保护：连续相同动作计数（配合 user_text 里的强提醒）
                # 坐标按 64px 粗网格取整：7B 空点时每次都差几个像素
                # （真机 40 步里同一处 x=77/78/80/36/91…猜了 15 次），精确匹配永远抓不住
                _ax, _ay = action.get("x"), action.get("y")
                # 单击/双击/右键都算「在同一处动手」。真机上模型会在 (56,802) 上一会儿
                # 单击一会儿双击（位置几乎不变），按动作名分开计数就永远抓不住它：
                # 17 步 275 秒里同一小片桌面被点了 5 次，计数却分别停在 3 和 2。
                _act = action.get("action")
                _act_key = ("pointer" if _act in ("click", "double_click", "right_click")
                            else _act)
                sig = (_act_key,
                       None if _ax is None else int(_ax) // 64,
                       None if _ay is None else int(_ay) // 64,
                       action.get("key"), action.get("text"), action.get("element_desc"))
                # 累计计数：熔断会把 _repeat_count 清零，但"同一个动作整场做了 5 次"
                # 这件事本身就该判死刑。真机：type "123*123" 一共敲了 15 次，
                # 每次都被熔断"救"回来 3 次，于是一路空转到 21 步 303 秒。
                self._sig_counts = getattr(self, "_sig_counts", {})
                self._sig_counts[sig] = self._sig_counts.get(sig, 0) + 1
                if self._sig_counts[sig] == 3 and (_ax is not None or _ay is not None):
                    self.history.append(
                        "⚠️ 你已经在屏幕同一小区域（±64px）点过 3 次了，画面没有按预期变化"
                        "—— 这里没有你要的东西。禁止再在这一带微调坐标，必须换方法："
                        "换目标、换窗口、用快捷键或先滚动。")
                    self._note_fail(
                        f"在同一小区域（±64px）已经点过 3 次，什么都没发生 —— 那里没有目标")
                if self._sig_counts[sig] >= 5:
                    self.log(f"⛔ [卡死] 同一动作（{action.get('action')} "
                             f"{action.get('text') or action.get('element_desc') or ''}）"
                             f"已执行 {self._sig_counts[sig]} 次仍未完成任务，判定卡住，主动终止。")
                    self.log("   建议：把任务拆小（例如先「打开计算器」，再「给某人发消息」），"
                             "或确认目标窗口是否真的打开了。")
                    return "FAILED"
                if sig == getattr(self, "_last_action_sig", None):
                    self._repeat_count = getattr(self, "_repeat_count", 0) + 1
                else:
                    self._repeat_count = 0
                    self._last_action_sig = sig

                # 熔断：同一动作重复 3 次，模型已救不回来，程序直接接管：
                # esc 清弹层 -> 若目标程序窗口存在，直接把它调到前台
                if self._repeat_count >= 2:
                    self.log(f"[熔断] 第 {self._repeat_count + 1} 次重复同一动作，程序接管：按 esc")
                    self._reflect_note(
                        f"同一个动作 {self._repeat_count + 1} 次重复执行，画面一直没有变化")
                    try:
                        self.executor.execute({"action": "key", "key": "esc"})
                    except Exception:
                        pass
                    self._repeat_count = 0
                    self._last_action_sig = None
                    self._expect_failed = False
                    focused = self._focus_program(self.planner.current() or self.task)
                    if focused:
                        self.log(f"[熔断] 已把窗口「{focused}」调到前台")
                        self._circuit_hint = (
                            f"🚨 系统提示：你的输入一直没有生效，原因是输入焦点不在目标程序里。"
                            f"系统已把窗口「{focused}」调到前台并获得焦点。"
                            "现在可以直接对这个窗口输入文字或点击它的元素，不要再做无关操作。\n"
                        )
                    else:
                        self._circuit_hint = (
                            "🚨 系统强制提示：同一个动作已重复多次全部无效。最常见原因是"
                            "【输入焦点不在目标程序里】。请先用 {\"action\":\"click\","
                            "\"element_desc\":\"目标程序窗口的编辑区域\"} 点一下目标窗口内部让它获得焦点"
                            "（或用 open_app 重新聚焦目标窗口），等截图确认焦点已切换后再输入。不要用 alt+tab。"
                            "如果目标程序根本没打开，重新用 win 键搜索打开。这一步做完前禁止再输入文字。\n"
                        )
                    self._pending = (self.screenshot_fn(), False)

                if cont is False:  # done
                    self._trace(action, "done", before_path, result="done")
                    if self.planner.subtasks and not self.planner.is_complete():
                        self.planner.mark_done()
                    self.log("✅ 任务结束。")
                    return "DONE"

                if cont == "ASK":  # 需要用户澄清
                    self._trace(action, "ask", before_path, result="ask")
                    self.log("（模型在等你回复，回复后继续）")
                    return "NEED_REPLY"

                # ---- 动作后：模型预判「这次会不会出现新内容」，据此决定是否重截 ----
                expect_new = action.get("expect_change", True)
                if isinstance(expect_new, str):
                    expect_new = expect_new.strip().lower() not in ("false", "0", "no", "否", "不会")
                else:
                    expect_new = bool(expect_new)
                after = self.screenshot_fn()
                # 「期待有新画面」时不秒截：给渲染一个观察窗，期间青蛙眼采样。
                # 双击后立刻截屏常截到半渲染的窗口 —— 这才是日志里
                # 「预期有新内容 -> 但画面没有变化」的真实原因。
                # 预算按动作类型给（打字/按键只采一次），且变稳就提前结束。
                budget = self._settle_budget(action) if expect_new else 0.0
                if budget > 0:
                    after, settle_note = self._settle_screen(
                        img, after, budget, stable=(self.change_settle_mode == "stable"))
                    if settle_note:
                        self.log(settle_note)
                # 青蛙眼的动作级用法：双击 / open_app 之后系统自己确认「进没进新窗口」，
                # 不交给模型判断（真机：它把「双击没生效」脑补成「WorkBuddy 未安装」）。
                self._observe_entry(action, img, after)
                if getattr(self, "_entry_shot", None) is not None:
                    after = self._entry_shot   # 换上「窗口真的出现了」那一帧
                if expect_new is False:
                    if self.verify_screen_change and self._changed(img, after):
                        self._pending = (after, False)  # 预判错了，画面确实变了
                        self._expect_failed = False
                        self.log("[画面] 预判无新内容，但实际有变化 -> 用新截图")
                    else:
                        self._pending = (img, True)     # 同一画面，下一步连元素检测一起复用
                        self._expect_failed = False
                        self.log("[画面] 无新内容 -> 复用上一张截图，不重新判断")
                else:
                    if action.get("action") == "type":
                        # 一行文字只占整屏 ~0.07%，且缩放检测会把小字抹没——
                        # 双信号：标题变化（Win11 记事本编辑后带 *）或原分辨率低阈值像素差
                        fg_after, _ = self._window_info()
                        changed = (fg_after != getattr(self, "_fg_before", None)
                                   or self._changed(img, after, thr=0.0001, max_w=None))
                        if changed:
                            # 明确告诉模型「输入已成功」，否则它会不放心再打一遍
                            # （真机踩坑：记事本里连打两遍「你好」才停）
                            self.history.append(
                                f"第{self.step}步: ✅ 输入已确认进入窗口"
                                f"（前台标题「{getattr(self, '_fg_before', '')}」→「{fg_after}」）。"
                                "文字已经在屏幕上了，禁止重复输入相同内容。")
                    else:
                        changed = self._changed(img, after)
                    self._expect_failed = not changed   # 预期变化但画面纹丝不动 = 动作大概率失败
                    if self._expect_failed:
                        self._note_fail(
                            f"{self._describe_action(action)} —— 做了，但屏幕没有任何变化（没生效）")
                        self._reflect_note(f"{json.dumps(action, ensure_ascii=False)[:120]}"
                                           " 执行了，但屏幕没有任何变化")
                        if action.get("action") == "double_click":
                            self.history.append(
                                f"第{self.step}步: 双击后没有新窗口、前台也没换人。"
                                "这不代表程序没装 —— 多半点到了标题栏文字而不是图标。"
                                "下一步用 open_app 交给系统打开，别再猜「未安装/路径错误」。")
                    self._pending = (after, False)      # 新画面，省掉下一轮开头那次截屏
                    self.log("[画面] 预期有新内容 -> 已重新截图"
                             + ("" if changed else "（但画面没有变化）"))

                # 输入成功后立刻确认"目标物上屏了没有"：上屏了就明说"可以收尾了"。
                # 8B 模型做完事不收尾是常态（真机：打完字后又连点 3 步），
                # 提示词里写了"达成即停"它也不守，只能在程序层把事实摆到它眼前。
                if not self._expect_failed and str(action.get("action")) == "type" \
                        and not getattr(self, "_goal_seen", False):
                    goal = self._goal_marker()
                    if goal and self._screen_has_text(goal):
                        self._goal_seen = True
                        self.history.append(
                            f"第{self.step}步: 目标内容「{goal}」已经出现在屏幕上。"
                            "如果任务只是要求输入/展示这些内容，你下一步必须输出 done，"
                            "不要再点击或输入任何东西。")

                # 目标上屏后模型还在空转（连续 2 个动作画面纹丝不动）→ 程序替它收尾。
                # 8B 模型做完事不喊 done 是常态；让它一直空转只会浪费步数、还可能点坏东西。
                if getattr(self, "_goal_seen", False):
                    if self._expect_failed:
                        self._goal_idle = getattr(self, "_goal_idle", 0) + 1
                        if self._goal_idle >= 2:
                            self.log("[收尾] 目标内容早已上屏，之后的动作都没让画面变化"
                                     " —— 判定任务完成")
                            return "DONE"
                    else:
                        self._goal_idle = 0

                # 第二条收尾线：连续 3 步动作做完，画面纹丝不动。
                # 模型这时候多半在做「最后一哆嗦」——再点一下确认、再把同一句话输一遍，
                # 再问它也只会多花 10 秒换一句「完成」。注意它和卡死判定不是一回事：
                # 卡死看的是「同一个动作重复」，这里看的是「三次不同的尝试都白做」。
                if self._expect_failed and self.step >= 3:
                    self._idle_steps = getattr(self, "_idle_steps", 0) + 1
                    if self._idle_steps >= 3:
                        self.log("[收尾] 连续 3 步动作都没让画面发生任何变化"
                                 " —— 能做的已经做完，判定任务完成")
                        return "DONE"
                else:
                    self._idle_steps = 0

                note = ""
                if self.validator.enabled:
                    ok, note = self.validator.verify(
                        json.dumps(action, ensure_ascii=False), img, after
                    )
                    self.log(f"[校验] {'通过' if ok else '未通过'}：{note}")
                    if ok:
                        self.planner.mark_done()
                        self.log(f"[进度] 子任务 {self.planner.idx}/{len(self.planner.subtasks)}")
                    else:
                        self.planner.replan(self.task, note)
                self._trace(action, note, before_path,
                            result=("nochange" if self._expect_failed else "ok"))

            self.log(f"⏱ 已达最大步数 {self.max_steps}，自动停止。可继续下达指令。")
            return "MAX_STEPS"
        except Exception:
            # 任何未预料的异常都必须留下现场。真机出现过只显示
            # 「❌ list index out of range」、没有任何行号的情况——等于没线索。
            import traceback
            _tb = traceback.format_exc(limit=8).strip()
            self.log("[崩溃] 程序内部异常，任务中断：\n        "
                     + _tb.replace("\n", "\n        "))
            return "ERROR"
        finally:
            try:
                self.long_memory.save()   # 跨任务经验只在任务结束时写一次盘
            except Exception:
                pass
            # 最后一步没有「下一步」来结算，这里补上；再给个总耗时
            if getattr(self, "_t_prev_step", None) is not None:
                self.log(f"[耗时] 第{self.step}步 共 {time.time() - self._t_prev_step:.1f}s")
            if getattr(self, "_t_run", None) is not None:
                self.log(f"[耗时] 本次任务合计 {time.time() - self._t_run:.1f}s"
                         f"（共 {self.step} 步）")
            _n = self.memory.cleanup_images()
            if _n:
                self.log(f"[清理] 任务结束，已删除 {_n} 张截图（文字轨迹保留）")
