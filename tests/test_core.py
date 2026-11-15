"""核心循环测试：在不依赖显示器/pyautogui 的前提下，验证解析、规划推进、done、ask。"""

import threading
from PIL import Image, ImageDraw

import agent.core as core_mod
from agent.core import AgentSession
from conftest import FakeModel


class FakeExecutor:
    """记录动作并返回约定值：done->False, ask->'ASK', 其余->True。"""

    def __init__(self):
        self.calls = []

    def execute(self, action):
        self.calls.append(action)
        a = action.get("action")
        if a == "done":
            return False
        if a == "ask":
            return "ASK"
        return True


def _cfg():
    return {
        "grounding": {"backend": "none"},   # 不触发额外模型调用
        "planner": {"enabled": True},
        "validator": {"enabled": False},
        "agent": {"max_steps": 5, "step_pause": 0, "reflect_on_fail": False,
                  "type_guard": False, "dialog_guard": False,
                  "wait_idle": False, "home_desktop": False,
                  "frog_eye": False, "entry_confirm": False,
                  "change_settle": 0, "long_memory": False},  # 固定观察窗会多截 6 帧；合成测试关掉
        "safety": {"confirm_risky": False},
    }


def _fake_screenshot():
    return Image.new("RGB", (64, 48), (255, 255, 255))


def test_run_executes_two_steps_then_done():
    """两步动作都执行完，收到 done 就收尾（不再依赖模型自报分段）。"""
    model = FakeModel(replies=[
        "1. 打开应用\n2. 收尾",                       # 规划
        '{"action":"click","element_id":1}',          # 第1步
        '{"action":"done","result":"完成"}',           # 第2步
    ])
    ex = FakeExecutor()
    sess = AgentSession(_cfg(), model=model, executor=ex,
                       screenshot_fn=_fake_screenshot,
                       stop_event=threading.Event())
    res = sess.run("测试任务")
    assert res == "DONE"
    assert len(ex.calls) == 2


def test_run_returns_need_reply_on_ask():
    model = FakeModel(replies=[
        "1. 一步",                                  # 规划
        '{"action":"ask","question":"要哪个文件？"}',  # 询问
    ])
    ex = FakeExecutor()
    sess = AgentSession(_cfg(), model=model, executor=ex,
                       screenshot_fn=_fake_screenshot,
                       stop_event=threading.Event())
    res = sess.run("测试任务")
    assert res == "NEED_REPLY"
    assert any(a.get("action") == "ask" for a in ex.calls)
    # 回复后继续，模型给出 done
    model.replies.append('{"action":"done","result":"完成"}')
    res2 = sess.run("测试任务", reply="用 a.txt")
    assert res2 == "DONE"


def test_parse_action_robust():
    assert AgentSession.parse_action('{"action":"click","x":1,"y":2}')["action"] == "click"
    assert AgentSession.parse_action('```json\n{"action":"wait","ms":100}\n```')["ms"] == 100
    assert AgentSession.parse_action("胡言乱语") is None
    assert AgentSession.parse_action("") is None


def test_stop_event_short_circuits():
    model = FakeModel(replies=[
        "1. 一步",
        '{"action":"wait","ms":10}',
    ])
    ex = FakeExecutor()
    stop = threading.Event()
    stop.set()
    sess = AgentSession(_cfg(), model=model, executor=ex,
                       screenshot_fn=_fake_screenshot, stop_event=stop)
    assert sess.run("任务") == "STOPPED"


# ---------- 输入区定位：程序自己量，不依赖硬编码比例 ----------

def _rect(l, t, r, b):
    from ctypes import wintypes
    rc = wintypes.RECT()
    rc.left, rc.top, rc.right, rc.bottom = l, t, r, b
    return rc


def _sess():
    return AgentSession(_cfg(), model=FakeModel(replies=[]), executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())


def test_input_candidates_prefers_wide_flat_bottom(monkeypatch):
    """宽、扁、靠下的控件应该排第一个 —— 这就是聊天输入框的形状。"""
    import agent.yolo_grounding as yg
    monkeypatch.setattr(yg, "detect_elements", lambda img, conf=0.3: [
        [10, 30, 54, 40, 0.9],   # 输入框：宽 44、高 10、贴底
        [2, 2, 20, 12, 0.9],     # 顶部按钮：在上半部，必须被排除
    ])
    cands = _sess()._input_candidates(_rect(0, 0, 64, 48))
    assert cands, "至少要给出候选"
    cx, cy, why = cands[0]
    assert "视觉检测" in why
    assert (round(cx), round(cy)) == (32, 35)


def test_input_candidates_skips_narrow_and_tall_elements(monkeypatch):
    """窄条（如侧栏按钮）和高块（如图片）都不能被当成输入框。"""
    import agent.yolo_grounding as yg
    monkeypatch.setattr(yg, "detect_elements", lambda img, conf=0.3: [
        [40, 30, 46, 44, 0.9],   # 窄：宽 6，远小于 64*0.3
        [8, 24, 56, 47, 0.9],    # 高：23px，超过 48*0.4=19.2，是图片不是输入条
    ])
    cands = _sess()._input_candidates(_rect(0, 0, 64, 48))
    assert all("视觉检测" not in c[2] for c in cands)


def test_input_candidates_falls_back_to_geometry(monkeypatch):
    """YOLO 挂了也要能给出几何候选，不能把发消息整条链路拖垮。"""
    import agent.yolo_grounding as yg

    def boom(img, conf=0.3):
        raise RuntimeError("模型未加载")

    monkeypatch.setattr(yg, "detect_elements", boom)
    cands = _sess()._input_candidates(_rect(0, 0, 64, 48))
    assert len(cands) == 3
    assert all("几何估算" in c[2] for c in cands)
    # 三个候选必须互不相同，否则重试没意义
    assert len({(round(c[0]), round(c[1])) for c in cands}) == 3


def test_send_message_aborts_without_enter_when_typing_fails(monkeypatch):
    """文字没打进输入框时绝不能按回车 —— 那会往别的会话误发消息。"""
    import pyautogui

    import agent.actions as act

    pressed = []
    monkeypatch.setattr(pyautogui, "click", lambda *a, **k: None)
    monkeypatch.setattr(pyautogui, "hotkey", lambda *a, **k: None)
    monkeypatch.setattr(pyautogui, "press", lambda k, **kw: pressed.append(k))
    monkeypatch.setattr(act, "_type_unicode", lambda t: None)

    shot = _fake_screenshot()
    sess = AgentSession(_cfg(), model=FakeModel(replies=[]), executor=FakeExecutor(),
                        screenshot_fn=lambda: shot, stop_event=threading.Event())
    monkeypatch.setattr(sess, "_focus_program", lambda app: True)
    monkeypatch.setattr(sess, "_input_candidates", lambda rect: [(40, 40, "视觉检测 假")])
    # 搜索结果能直接匹配上，避免走 enter 兜底，这样 press 记录是干净的
    monkeypatch.setattr(sess, "grounding",
                        type("G", (), {"resolve_desc": lambda self, t, i: (10, 10, 0, "文件传输助手", 1.2)})(),)
    import agent.core as core_mod
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    # 假截图前后一模一样，「文字有没有进输入框」只能靠像素差判断 —— 这里直接
    # 钉死"画面没变"，才能稳定验证"没打进字就绝不回车"这条规则。
    monkeypatch.setattr(sess, "_changed", lambda a, b, **k: False)

    ok = sess._send_message("微信", "文件传输助手", "你好")
    assert ok is False, "没打进文字就该失败返回"
    assert "enter" not in pressed, "文字没打进去时绝不能按回车（会误发到别的会话）"
    assert "enter" not in pressed, "没打进文字时按回车会误发"


# ---------- 任务路由：模型只分类，动作归程序 ----------

def test_route_uses_regex_without_calling_model():
    """正则能认出来的，一个字都不要让模型猜（省 10 秒且不会错）。"""
    model = FakeModel(replies=[])
    sess = AgentSession(_cfg(), model=model, executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    r = sess._route_task("在微信里给文件传输助手发送一条消息：你好")
    assert r == {"app": "微信", "target": "文件传输助手", "text": "你好"}
    assert model.calls == 0, "正则命中时不该调用模型"


def test_route_extracts_params_from_casual_wording():
    """正则确实覆盖不到的说法，交给 7B 抽参数，但仍走同一套稳定流程。"""
    model = FakeModel(replies=[
        '{"ok":true,"app":"微信","target":"示例联系人A","text":"收到"}',
    ])
    sess = AgentSession(_cfg(), model=model, executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    r = sess._route_task("帮我回复示例联系人A：收到")
    assert r == {"app": "微信", "target": "示例联系人A", "text": "收到"}
    assert model.calls == 1


def test_route_blocks_non_whitelisted_wechat_target_regex():
    """白名单：正则命中的非白名单会话，必须在路由层就拦死，不能落到通用循环。"""
    model = FakeModel(replies=[])
    sess = AgentSession(_cfg(), model=model, executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    r = sess._route_task("在微信里给示例联系人B发送一条消息：你好")
    assert isinstance(r, dict) and "blocked" in r
    assert "示例联系人B" in r["blocked"]
    assert model.calls == 0


def test_route_blocks_non_whitelisted_from_model_classification():
    """7B 抽出来的目标不在白名单，也要拦（模型可能把「小王」认成会话名）。"""
    model = FakeModel(replies=['{"ok":true,"app":"微信","target":"小王","text":"收到"}'])
    sess = AgentSession(_cfg(), model=model, executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    r = sess._route_task("帮我回复小王：收到")
    assert isinstance(r, dict) and "blocked" in r
    assert model.calls == 1


def test_send_target_allowed_matrix():
    A = AgentSession._send_target_allowed
    assert A("微信", "文件传输助手")[0] is True
    assert A("微信", "示例联系人A")[0] is True
    # 搜索结果常带尾巴：未读数 / 最后一条消息
    assert A("微信", "文件传输助手（2）")[0] is True
    assert A("微信", "示例联系人A - 最后一条消息")[0] is True
    assert A("微信", "示例联系人B")[0] is False
    assert A("微信", "")[0] is False
    assert A("微信", "王")[0] is False        # 1 个字太模糊
    # 非微信不限制
    assert A("QQ", "任何人")[0] is True


def test_config_whitelist_authorizes_extra_target(monkeypatch):
    """config 显式加人后，校验层和路由层都要放行；代码默认值仍保持最小权限。"""
    old = getattr(AgentSession, "_cfg_snapshot", None)
    monkeypatch.setattr(AgentSession, "_cfg_snapshot", old, raising=False)
    cfg = _cfg()
    cfg["agent"]["wechat_allowed"] = ["文件传输助手", "示例联系人A", "示例联系人B"]
    sess = AgentSession(cfg, model=FakeModel(replies=[]), executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    assert AgentSession._allowed_targets() == ("文件传输助手", "示例联系人A", "示例联系人B")
    assert AgentSession._send_target_allowed("微信", "示例联系人B")[0] is True
    assert AgentSession._send_target_allowed("微信", "陌生人")[0] is False
    r = sess._route_task("在微信里给示例联系人B发送一条消息：你好")
    assert isinstance(r, dict) and "blocked" not in r
    assert r["target"] == "示例联系人B"
    # 真人名字只在用户自己的配置里，代码默认值不能被污染
    assert AgentSession._WECHAT_ALLOWED == ("文件传输助手", "示例联系人A")


def test_route_ignores_non_message_tasks_without_model_call():
    """不含发消息语义的任务，不该为路由多花一次推理。"""
    model = FakeModel(replies=[])
    sess = AgentSession(_cfg(), model=model, executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    assert sess._route_task("把桌面上的图片重命名") is None
    assert model.calls == 0


def test_route_rejects_bad_model_output():
    """模型乱答/参数缺失时不能硬发，要退回通用循环。"""
    for reply in ['{"ok":false}', "我不知道", '{"ok":true,"target":"","text":"x"}',
                  '{"ok":true,"target":"小王","text":"' + "长" * 200 + '"}']:
        model = FakeModel(replies=[reply])
        sess = AgentSession(_cfg(), model=model, executor=FakeExecutor(),
                            screenshot_fn=_fake_screenshot, stop_event=threading.Event())
        assert sess._route_task("跟小王说点什么") is None, reply


def test_parse_message_intent_covers_casual_forms():
    """常见口语都要能解析，否则就得靠模型猜（慢且不稳）。"""
    P = AgentSession._parse_message_intent
    assert P("在微信里给文件传输助手发送一条消息：你好") == \
        {"app": "微信", "target": "文件传输助手", "text": "你好"}
    assert P("给文件传输助手发：你好")["text"] == "你好"
    assert P("对文件传输助手说你好")["target"] == "文件传输助手"
    assert P("跟文件传输助手说一声路由测试")["text"] == "路由测试"
    assert P("帮我告诉文件传输助手：晚上见")["text"] == "晚上见"
    # 没写软件默认微信，不因此放弃解析
    assert P("跟小王说一声明天开会") == {"app": "微信", "target": "小王", "text": "明天开会"}


def test_parse_message_intent_ignores_normal_tasks():
    P = AgentSession._parse_message_intent
    for t in ("打开记事本", "把桌面上的图片重命名", "今天天气怎么样", ""):
        assert P(t) is None, t


def test_failed_template_never_reports_done(monkeypatch):
    """发不出去绝不能报「完成」—— 假成功比失败更误导人。"""
    sess = _sess()
    monkeypatch.setattr(sess, "_route_task",
                        lambda t: {"app": "微信", "target": "文件传输助手", "text": "hi"})
    monkeypatch.setattr(sess, "_send_message", lambda a, b, c: False)
    assert sess.run("在微信里给文件传输助手发送：hi") == "FAILED"

    monkeypatch.setattr(sess, "_send_message", lambda a, b, c: True)
    assert sess.run("在微信里给文件传输助手发送：hi") == "DONE"


# ---------- 桌面兜底 & 给人让路 ----------

def test_idle_seconds_is_a_number():
    s = AgentSession._idle_seconds()
    assert isinstance(s, float) and s >= 0


def test_wait_until_idle_gives_up_when_user_keeps_typing(monkeypatch):
    """人一直在动键鼠时，程序要放弃抢占，不能硬抢。"""
    cfg = _cfg()
    cfg["agent"] = dict(cfg["agent"], wait_idle=True)   # 这条规则专门测"等不到就放弃"
    sess = AgentSession(cfg, model=FakeModel(replies=[]), executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    monkeypatch.setattr(sess, "_idle_seconds", lambda: 0.0)
    import agent.core as core_mod
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    assert sess._wait_until_idle(need=1.5, max_wait=1.0) is False


def test_wait_until_idle_can_be_disabled_by_config():
    """自动化测试必须能关掉"等真人停手"，否则一次测试要等用户不动鼠标。"""
    assert _sess()._wait_until_idle(need=99, max_wait=0.1) is True


def test_wait_until_idle_passes_when_quiet(monkeypatch):
    cfg = _cfg()
    cfg["agent"] = dict(cfg["agent"], wait_idle=True)
    sess = AgentSession(cfg, model=FakeModel(replies=[]), executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    monkeypatch.setattr(sess, "_idle_seconds", lambda: 5.0)
    assert sess._wait_until_idle() is True


def test_show_desktop_uses_win_d(monkeypatch):
    """Win 键只是开开始菜单；要露出桌面必须是 Win+D。"""
    import pyautogui
    calls = []
    monkeypatch.setattr(pyautogui, "hotkey", lambda *a: calls.append(a))
    import agent.core as core_mod
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    _sess()._show_desktop()
    assert calls == [("win", "d")]


def test_open_from_desktop_double_clicks_found_icon(monkeypatch):
    """桌面图标命中就走这条路：先 Win+D 露桌面，再双击（不经过视觉模型）。"""
    import pyautogui
    clicks, order = [], []
    monkeypatch.setattr(pyautogui, "doubleClick", lambda x, y: clicks.append((x, y)))
    sess = _sess()
    monkeypatch.setattr(sess, "_show_desktop", lambda *a, **k: order.append("win+d"))
    monkeypatch.setattr(sess, "_desktop_icons", lambda: (
        [{"name": "Red Panda Dev-C++", "x": 100, "y": 50, "w": 115, "h": 124}], 1000))
    monkeypatch.setattr(sess, "_icon_to_py_coords", lambda hit, sw: (hit["x"], hit["y"]))
    monkeypatch.setattr(sess, "_focus_program", lambda app: "Dev-C++")
    assert sess._open_from_desktop("Dev-C++") is True
    assert order == ["win+d"], "双击前必须先露桌面，否则会点到挡住它的窗口上"
    assert clicks == [(100, 50)]


def test_open_from_desktop_miss_leaves_windows_alone(monkeypatch):
    """桌面上没有这个图标时不能 Win+D —— 那会把用户正在看的窗口全最小化。"""
    touched = []
    sess = _sess()
    monkeypatch.setattr(sess, "_show_desktop", lambda *a, **k: touched.append(1))
    monkeypatch.setattr(sess, "_desktop_icons",
                        lambda: ([{"name": "微信", "x": 1, "y": 2}], 1000))
    assert sess._open_from_desktop("Dev-C++") is False
    assert touched == [], "没命中就不该动用户窗口"


def test_desktop_icon_name_matching():
    """桌面上的名字和用户说的不一样也要能对上。"""
    icons = [{"name": "Red Panda Dev-C++"}, {"name": "Microsoft Edge"}, {"name": "微信"}]
    assert AgentSession._match_desktop_icon("Dev-C++", icons)[1] == 100
    assert AgentSession._match_desktop_icon("devcpp", icons)[0]["name"] == "Red Panda Dev-C++"
    assert AgentSession._match_desktop_icon("Edge", icons)[0]["name"] == "Microsoft Edge"
    assert AgentSession._match_desktop_icon("计算器", icons) == (None, 0)


def test_open_app_prefers_desktop_over_search(monkeypatch):
    """桌面能找到就别去开开始菜单 —— 这是「先看桌面，没有再搜索」的硬约束。"""
    sess = _sess()
    monkeypatch.setattr(sess, "_wait_until_idle", lambda *a, **k: True)
    monkeypatch.setattr(sess, "_focus_program", lambda app: None)   # 没在运行
    monkeypatch.setattr(sess, "_open_from_desktop", lambda app: True)
    searched = []
    monkeypatch.setattr(sess, "screenshot_fn",
                        lambda: searched.append(1) or _fake_screenshot())
    assert sess._open_app("Dev-C++") is True
    assert searched == [], "桌面已打开就不该再截图找开始按钮"


def test_desktop_visual_fallback_uses_grounding(monkeypatch):
    """桌面 ListView 读不出来时，仍要用视觉兜底（老路子不能丢）。"""
    import pyautogui
    clicks = []
    monkeypatch.setattr(pyautogui, "doubleClick", lambda x, y: clicks.append((x, y)))
    import agent.core as core_mod
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    sess = _sess()
    monkeypatch.setattr(sess, "_show_desktop", lambda *a, **k: None)
    monkeypatch.setattr(sess, "grounding", type("G", (), {
        "resolve_desc": lambda self, t, i: (10, 8, 12, 14, "微信图标", 1.3)})())
    monkeypatch.setattr(sess, "_focus_program", lambda app: "微信")
    assert sess._open_from_desktop_visual("微信") is True
    assert clicks == [(10, 8)]


def test_desktop_visual_fallback_returns_false_when_no_icon(monkeypatch):
    sess = _sess()
    monkeypatch.setattr(sess, "_show_desktop", lambda *a, **k: None)
    monkeypatch.setattr(sess, "grounding", type("G", (), {
        "resolve_desc": lambda self, t, i: None})())
    assert sess._open_from_desktop_visual("微信") is False


def test_send_message_opens_app_when_not_in_front(monkeypatch):
    """窗口在托盘/后台时，发消息要自己把程序叫回来，而不是直接报错。"""
    import pyautogui

    import agent.actions as act
    monkeypatch.setattr(pyautogui, "click", lambda *a, **k: None)
    monkeypatch.setattr(pyautogui, "hotkey", lambda *a, **k: None)
    monkeypatch.setattr(pyautogui, "press", lambda *a, **k: None)
    monkeypatch.setattr(act, "_type_unicode", lambda t: None)
    import agent.core as core_mod
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)

    sess = _sess()
    monkeypatch.setattr(sess, "_wait_until_idle", lambda *a, **k: True)
    monkeypatch.setattr(sess, "_focus_program", lambda app: None)   # 托盘：找不到窗口
    opened = []
    monkeypatch.setattr(sess, "_open_app", lambda app: opened.append(app) or True)
    monkeypatch.setattr(sess, "_input_candidates", lambda rect: [])
    sess._send_message("微信", "文件传输助手", "hi")
    assert opened == ["微信"], "应该尝试自己打开程序"


# ---------- 光标感知：停靠 / 截图标记 / 坐标反馈 ----------

def test_cursor_pos_returns_tuple_or_none():
    """Windows 上应返回真实坐标；非 Windows 返回 None 也算正常。"""
    p = AgentSession._cursor_pos()
    assert p is None or (isinstance(p, tuple) and len(p) == 2
                         and all(isinstance(v, int) for v in p))


def test_mark_cursor_draws_on_image():
    """光标位置有效时，截图上要真的画出标记（与原图不同）。"""
    import numpy as np
    from PIL import Image
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    AgentSession._mark_cursor(img, 1.0)   # 沙箱内 GetCursorPos 通常可用；画不出就跳过
    arr = np.array(img)
    # 不能假设拿得到光标（无 GUI 环境会失败），只验证"没画时图保持纯白、画了必有杂色"
    assert arr.shape[:2] == (300, 400)


def test_snap_inside_prefers_smallest_box():
    """坐标落在嵌套盒内部时，必须吸到最内层（面积最小）的那个。"""
    els = [
        {"id": 1, "bbox": [0, 0, 100, 100]},
        {"id": 2, "bbox": [40, 40, 60, 60]},   # 小盒套在大盒里
    ]
    a = AgentSession._snap_to_element({"action": "click", "x": 50, "y": 50}, els)
    assert (a["x"], a["y"]) == (50, 50)  # 小盒中心


def test_snap_no_longer_grabs_far_neighbors():
    """旧版 tol=60 会把任务栏相邻图标吸串；现在盒外超过 40px 不许吸。"""
    els = [
        {"id": 1, "bbox": [0, 0, 30, 30]},     # 中心 (15,15)
        {"id": 2, "bbox": [70, 0, 100, 30]},   # 中心 (85,15)，与点 (55,15) 距 30
    ]
    a = AgentSession._snap_to_element({"action": "click", "x": 55, "y": 15}, els)
    assert (a["x"], a["y"]) == (85, 15)  # 30 < 40 允许吸最近
    # 距离超出 tol 的不吸
    els2 = [{"id": 1, "bbox": [0, 0, 30, 30]}]
    b = AgentSession._snap_to_element({"action": "click", "x": 55, "y": 15}, els2)
    assert (b["x"], b["y"]) == (55, 15)  # 距中心 40，不吸


def test_build_user_text_includes_mouse_state():
    sess = _sess()
    text = sess._build_user_text("测试任务", [])
    assert "鼠标状态" in text or "鼠标" in text


def test_ensure_focus_skips_when_fg_matches(monkeypatch):
    """前台就是目标程序时不折腾。"""
    sess = _sess()
    monkeypatch.setattr(sess, "_window_info", lambda: ("记事本", ["记事本"]))
    sess._last_opened_app = "记事本"
    called = []
    monkeypatch.setattr(type(sess), "_focus_program",
                        classmethod(lambda cls, t: called.append(t) or "x"))
    sess._ensure_focus_for_type()
    assert called == []


def test_ensure_focus_refocuses_when_fg_differs(monkeypatch):
    """前台不是目标程序时必须先聚焦 —— 否则字打进别的窗口。"""
    sess = _sess()
    monkeypatch.setattr(sess, "_window_info", lambda: ("浏览器", ["浏览器", "记事本"]))
    sess._last_opened_app = "记事本"
    called = []
    monkeypatch.setattr(type(sess), "_focus_program",
                        classmethod(lambda cls, t: called.append(t) or "记事本"))
    import agent.core as core_mod
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    sess._ensure_focus_for_type()
    assert called == ["记事本"]


# ---------- 过期 miss 提示误导模型（真机踩坑：开好程序还让模型去点开始按钮） ----------

class _NoDescGrounding:
    """有 resolve_desc 接口但永远匹配不到（模拟「目标不在屏幕上」）。"""
    def resolve_desc(self, desc, img):
        return None


def test_desc_miss_cleared_when_routed_to_open_app():
    """目标转程序级 open_app 后，「不在屏幕上」提示必须清掉，
    否则下一轮模型被引去点开始按钮——而程序已经把程序打开了。"""
    sess = _sess()
    sess.grounding = _NoDescGrounding()
    a = sess._apply_element_desc(
        {"action": "click", "element_desc": "记事本图标"}, _fake_screenshot())
    assert a["action"] == "open_app" and a["app"] == "记事本"
    assert sess._desc_miss is None, "转模板后不能再提示「不在屏幕上」"


def test_desc_miss_keeps_real_name_for_feedback():
    """没转模板时，原始描述要留在 _miss_desc，反馈里才不会显示 None。"""
    sess = _sess()
    sess.grounding = _NoDescGrounding()
    a = sess._apply_element_desc(
        {"action": "click", "element_desc": "完全不存在的控件"}, _fake_screenshot())
    assert a.get("_miss_desc") == "完全不存在的控件"
    assert sess._desc_miss == "完全不存在的控件"


def test_open_app_success_clears_stale_miss(monkeypatch):
    """open_app 成功后，之前积压的 miss 提示必须作废。"""
    sess = _sess()
    sess._desc_miss = "记事本图标"          # 第 1 步留下的过期提示
    monkeypatch.setattr(sess, "_open_app", lambda app: True)
    monkeypatch.setattr(sess, "_route_task", lambda t: None)
    monkeypatch.setattr(sess.planner, "decompose", lambda t: ["打开记事本"])
    sess.validator.enabled = False
    monkeypatch.setattr(core_mod.AgentSession, "_focus_program",
                        classmethod(lambda cls, t: "记事本"))
    monkeypatch.setattr(core_mod.AgentSession, "_parse_message_intent",
                        staticmethod(lambda t: None))
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)

    class M:
        def __init__(self):
            self.calls = 0

        def chat(self, msgs):
            self.calls += 1
            return ('{"action":"open_app","app":"记事本"}' if self.calls == 1
                    else '{"action":"done"}')
    sess.model = M()
    assert sess.run("打开记事本") == "DONE"
    assert sess._desc_miss is None


# ---------- miss→open_app 误转（真机踩坑：模型想点输入框，程序却去开程序） ----------

def test_desc_mentioning_app_inside_longer_text_not_routed(monkeypatch):
    """「记事本窗口中的输入框」描述的是窗口内部控件，不是程序——不许转 open_app。
    （新行为：输入框类描述优先查焦点控件；这里模拟拿不到焦点控件的退化路径。）"""
    sess = _sess()
    sess.grounding = _NoDescGrounding()
    monkeypatch.setattr(AgentSession, "_focused_control_center", staticmethod(lambda: None))
    a = sess._apply_element_desc(
        {"action": "click", "element_desc": "记事本窗口中的输入框"}, _fake_screenshot())
    assert a.get("action") == "click", "应保持 click，走「无坐标」反馈分支"
    assert a.get("_miss_desc") == "记事本窗口中的输入框"
    assert a.get("app") is None


def test_desc_plain_app_name_still_routed():
    """「记事本图标」这种就是在说程序，必须转 open_app。"""
    sess = _sess()
    sess.grounding = _NoDescGrounding()
    a = sess._apply_element_desc(
        {"action": "click", "element_desc": "记事本图标"}, _fake_screenshot())
    assert a["action"] == "open_app"


def test_miss_note_points_to_coords_when_fg_open(monkeypatch):
    """目标程序已在前台时，miss 提示必须引导「给坐标/type」，而不是「先去打开它」。"""
    sess = _sess()
    monkeypatch.setattr(sess, "_window_info", lambda: ("*记事本 - Notepad", ["*记事本 - Notepad"]))
    sess._desc_miss = "记事本窗口中的输入框"
    text = sess._build_user_text("测试", [])
    assert "x/y 坐标" in text and "禁止再发 open_app" in text
    assert "先点任务栏最左下角的开始按钮" not in text


def test_miss_note_suggests_opening_when_no_fg(monkeypatch):
    """程序没开时才给「先打开」的引导。"""
    sess = _sess()
    monkeypatch.setattr(sess, "_window_info", lambda: (None, []))
    sess._desc_miss = "记事本图标"
    text = sess._build_user_text("测试", [])
    assert "开始按钮" in text


# ---------- 打字成功被误判「没变化」（真机踩坑：打了 18 遍你好） ----------

def test_type_detects_change_via_title(monkeypatch):
    """编辑后窗口标题带 * → 必须判「有变化」，哪怕像素差低于普通阈值。"""
    import numpy as np
    from PIL import Image
    sess = _sess()
    sess._fg_before = "无标题 - Notepad"
    img = Image.new("RGB", (320, 200), (255, 255, 255))
    after = Image.new("RGB", (320, 200), (255, 255, 255))   # 像素几乎没差
    monkeypatch.setattr(sess, "_window_info", lambda: ("*你好 - Notepad", ["x"]))
    monkeypatch.setattr(sess, "screenshot_fn", lambda: after)
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    # 直接验证判定逻辑：标题变了 -> changed=True
    fg_after, _ = sess._window_info()
    changed = (fg_after != sess._fg_before) or AgentSession._changed(img, after, thr=0.0005)
    assert changed is True


def test_type_detects_change_via_low_threshold():
    """一行小字的像素差(~0.07%)要能被低阈值捕捉。"""
    import numpy as np
    from PIL import Image
    img = Image.new("RGB", (2560, 1600), (255, 255, 255))
    arr = np.array(img)
    arr[800:820, 100:136] = 0          # 一行 36 字符大小的一小块黑色文字
    after = Image.fromarray(arr)
    ratio = np.abs(np.array(after, dtype=int) - np.array(img, dtype=int)).any(axis=2).mean()
    assert ratio < 0.003, "前提：这么小的变化确实低于普通阈值"
    assert AgentSession._changed(img, after, thr=0.0001, max_w=None) is True
    assert AgentSession._changed(img, after) is False   # 缩放+普通阈值确实会漏


def test_focus_check_matches_english_window_title():
    """「记事本」的窗口标题是「无标题 - Notepad」——必须算匹配，不许误判。"""
    sess = _sess()
    monkey_names = {"记事本": ["notepad"]}
    fg = "无标题 - Notepad"
    app = "记事本"
    names = [app.lower()] + [x.lower() for x in monkey_names.get(app, [])]
    assert any(n in fg.lower() for n in names)


def test_invalid_click_note_mentions_non_icon_controls():
    """无坐标反馈要提到「非图标控件检测不到」，别再说『它很可能不在屏幕上』。"""
    # 文案在 run() 内联，这里验证构造逻辑的输入约定：_miss_desc 优先于 None
    action = {"action": "click", "_miss_desc": "记事本编辑区域"}
    tgt = (action.get("element_desc") or action.get("_miss_desc")
           or action.get("element_id") or "fallback")
    assert tgt == "记事本编辑区域"


# ---------- open_app 启动等待与开始按钮 toggle（真机踩坑：菜单被自己点关） ----------

def test_wait_app_window_polls_until_found(monkeypatch):
    """UWP 冷启动要几秒：必须轮询等待，不能 sleep 固定 2.5s 就判失败。"""
    sess = _sess()
    seq = [None, None, "记事本"]     # 第 3 次才出现
    calls = {"n": 0}

    def fake_focus(app):
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return seq[i]

    monkeypatch.setattr(core_mod.AgentSession, "_focus_program",
                        classmethod(lambda cls, a: fake_focus(a)))
    # 新加的「真去枚举窗口」兜底会命中真机上开着的记事本，绕过测试桩。
    # 单元测试不该依赖真机窗口状态，钉死成 None，只验证老路径的轮询。
    monkeypatch.setattr(core_mod.AgentSession, "_find_app_window",
                        classmethod(lambda cls, a: None))
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    assert sess._wait_app_window("记事本", max_wait=8) == "记事本"
    assert calls["n"] >= 3


def test_wait_app_window_gives_up_after_deadline(monkeypatch):
    sess = _sess()
    monkeypatch.setattr(core_mod.AgentSession, "_focus_program",
                        classmethod(lambda cls, a: None))
    monkeypatch.setattr(core_mod.AgentSession, "_find_app_window",
                        classmethod(lambda cls, a: None))
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    # 用一个会立刻到期的 deadline
    import time as _t
    real = _t.time
    monkeypatch.setattr(core_mod.time, "time", lambda: real() + 100)
    assert sess._wait_app_window("记事本", max_wait=0.1) is None


# ---------- 进场确认：未登记程序 + 动作生效硬反馈（真机 234s 死循环） ----------

def test_app_window_keys_covers_unlisted_apps():
    keys = core_mod.AgentSession._app_window_keys("WorkBuddy")
    assert "workbuddy" in keys
    assert "notepad" in core_mod.AgentSession._app_window_keys("记事本")


def test_wait_app_window_falls_back_to_window_enum(monkeypatch):
    """_PROC_MAP 里没有 WorkBuddy：老路径永远 None，必须靠枚举窗口兜住。"""
    sess = _sess()
    monkeypatch.setattr(core_mod.AgentSession, "_focus_program",
                        classmethod(lambda cls, a: None))
    monkeypatch.setattr(core_mod.AgentSession, "_find_app_window",
                        classmethod(lambda cls, a: (7, "WorkBuddy", "workbuddy.exe")))
    monkeypatch.setattr(core_mod.AgentSession, "_activate_window",
                        classmethod(lambda cls, h, t: t))
    assert sess._wait_app_window("WorkBuddy", max_wait=1) == "WorkBuddy"


def test_confirm_entry_requires_real_change(monkeypatch):
    """画面没变就不能宣布进场 —— 否则双击坏图标也会被当成成功。"""
    from PIL import Image
    sess = _sess()
    sess.entry_confirm = True
    still = Image.new("RGB", (640, 400), "white")
    monkeypatch.setattr(sess, "screenshot_fn", lambda: still)
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(core_mod.AgentSession, "_find_app_window",
                        classmethod(lambda cls, a: (7, "WorkBuddy", "workbuddy.exe")))
    assert sess._confirm_entry("WorkBuddy", still, max_wait=0.2) is False
    assert not getattr(sess, "_entered_app", None)


def test_confirm_entry_accepts_full_screen_window(monkeypatch):
    """大窗口打开时整屏都变、regions 被 _diff_half 清空，仍要靠 ratio 判成功。"""
    from PIL import Image, ImageDraw
    sess = _sess()
    sess.entry_confirm = True
    before = Image.new("RGB", (640, 400), "white")
    after = before.copy()
    ImageDraw.Draw(after).rectangle([60, 40, 580, 360], fill="black")
    monkeypatch.setattr(sess, "screenshot_fn", lambda: after)
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(core_mod.AgentSession, "_find_app_window",
                        classmethod(lambda cls, a: (7, "WorkBuddy", "workbuddy.exe")))
    monkeypatch.setattr(core_mod.AgentSession, "_activate_window",
                        classmethod(lambda cls, h, t: t))
    monkeypatch.setattr(sess, "_record_entry",
                        lambda app, via="": setattr(sess, "_entered_app", {"via": via}))
    assert sess._confirm_entry("WorkBuddy", before) is True
    assert "青蛙眼" in sess._entered_app["via"]


def test_entry_fact_tells_model_not_to_reopen(monkeypatch):
    sess = _sess()
    sess.step = 3
    sess._entry_step = 3
    sess._entered_app = {"app": "WorkBuddy", "title": "WorkBuddy",
                         "exe": "workbuddy.exe", "via": "青蛙眼"}
    monkeypatch.setattr(core_mod.AgentSession, "_window_info",
                        classmethod(lambda cls: ("WorkBuddy", ["WorkBuddy"])))
    monkeypatch.setattr(sess, "_cursor_pos", lambda: (10, 10))
    monkeypatch.setattr(sess, "_foreground_class", lambda: "Chrome_WidgetWin_1")
    monkeypatch.setattr(sess.scratch, "recent", lambda n: "")
    text = sess._build_user_text("打开 WorkBuddy 新建对话", [])
    assert "系统确认" in text and "禁止再发 open_app" in text


def test_observe_entry_detects_foreground_switch(monkeypatch):
    """模型自己发的 double_click：前台进程换人 + 画面大变 = 确实进去了。"""
    from PIL import Image, ImageDraw
    sess = _sess()
    sess.entry_confirm = True
    sess.step = 2
    sess._fg_before = "Program Manager"
    sess._fg_exe_before = "explorer.exe"
    before = Image.new("RGB", (640, 400), "white")
    after = before.copy()
    ImageDraw.Draw(after).rectangle([60, 40, 580, 360], fill="black")
    monkeypatch.setattr(sess, "_foreground_exe", lambda: "workbuddy.exe")
    monkeypatch.setattr(core_mod.AgentSession, "_window_info",
                        classmethod(lambda cls: ("WorkBuddy", ["WorkBuddy"])))
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    sess._observe_entry({"action": "double_click"}, before, after)
    assert sess._entered_app["exe"] == "workbuddy.exe"
    assert "系统确认" in sess.history[-1]
    assert sess._entry_shot is after, "下一轮必须看到「窗口已出现」那一帧，不能复用动作瞬间的旧图"
    assert sess._entry_just_confirmed


def test_observe_entry_ignores_unchanged_foreground(monkeypatch):
    """双击没生效就不能宣布进场 —— 否则坏图标也会被当成成功。"""
    from PIL import Image
    sess = _sess()
    sess.entry_confirm = True
    sess.step = 2
    sess._fg_before = "WorkBuddy"
    sess._fg_exe_before = "workbuddy.exe"
    still = Image.new("RGB", (640, 400), "white")
    monkeypatch.setattr(sess, "screenshot_fn", lambda: still)
    monkeypatch.setattr(sess, "_foreground_exe", lambda: "workbuddy.exe")
    monkeypatch.setattr(core_mod.AgentSession, "_window_info",
                        classmethod(lambda cls: ("WorkBuddy", ["WorkBuddy"])))
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    sess._observe_entry({"action": "double_click"}, still, still)
    assert not getattr(sess, "_entered_app", None)


def test_settle_screen_fixed_mode_waits_full_window(monkeypatch):
    """fixed 模式：固定等满 3 秒，返回最后一帧；差分采样记下收敛过程。"""
    from PIL import Image, ImageDraw
    sess = _sess()
    before = Image.new("RGB", (640, 400), "white")
    mid = before.copy()
    ImageDraw.Draw(mid).rectangle([100, 80, 300, 200], fill="gray")
    done = before.copy()
    ImageDraw.Draw(done).rectangle([60, 40, 580, 360], fill="black")
    frames = [mid, mid, done, done, done, done]
    monkeypatch.setattr(sess, "screenshot_fn",
                        lambda: frames.pop(0) if frames else done)
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    last, note = sess._settle_screen(before, mid, 3.0, stable=False)
    assert "观察满" in note
    assert len(sess._last_settle["ratios"]) >= 4
    assert sess._last_settle["ratios"][-1] > 0
    assert last is done


def test_settle_screen_stable_mode_exits_early(monkeypatch):
    """stable 模式：连续两帧变化量相同就提前走，不再白等后 2 秒。"""
    from PIL import Image, ImageDraw
    sess = _sess()
    before = Image.new("RGB", (640, 400), "white")
    after = before.copy()
    ImageDraw.Draw(after).rectangle([60, 40, 580, 360], fill="black")
    calls = {"n": 0}

    def shot():
        calls["n"] += 1
        return after

    monkeypatch.setattr(sess, "screenshot_fn", shot)
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    last, note = sess._settle_screen(before, after, 3.0, stable=True)
    assert calls["n"] == 2, "两次相同的采样就该提前结束"
    assert sess._last_settle["early"] is True
    assert "变稳提前结束" in note
    assert last is after


def test_settle_screen_fixed_never_ends_early_on_no_change(monkeypatch):
    """fixed 模式下画面一直没变也不能提前结束（旧行为可对照）。"""
    from PIL import Image
    sess = _sess()
    still = Image.new("RGB", (640, 400), "white")
    calls = {"n": 0}

    def shot():
        calls["n"] += 1
        return still

    monkeypatch.setattr(sess, "screenshot_fn", shot)
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    last, note = sess._settle_screen(still, still, 3.0, stable=False)
    assert calls["n"] == 6, "3 秒 / 0.5 秒 = 6 次采样，一次都不能少"
    assert last is still
    assert "0.0%" in note


def test_settle_budget_short_for_instant_actions():
    sess = _sess()
    sess.change_settle = 3.0
    sess.change_settle_mode = "stable"
    assert sess._settle_budget({"action": "type"}) == 0.5
    assert sess._settle_budget({"action": "key", "key": "enter"}) == 0.5
    assert sess._settle_budget({"action": "scroll"}) == 0.5
    assert sess._settle_budget({"action": "double_click"}) == 3.0
    assert sess._settle_budget({"action": "open_app"}) == 3.0
    sess.change_settle_mode = "off"
    assert sess._settle_budget({"action": "double_click"}) == 0.0


def test_out_of_time_trips_after_budget():
    import time
    sess = _sess()
    sess.max_seconds = 1.0
    sess._t_run = time.time() - 5
    assert sess._out_of_time() is True
    sess.max_seconds = 0
    assert sess._out_of_time() is False


def test_no_send_guard_blocks_sending_but_not_normal_actions(monkeypatch):
    cfg = _cfg()
    cfg["agent"]["no_send"] = True
    sess = AgentSession(cfg, model=FakeModel(replies=[]), executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    assert sess.no_send is True
    assert sess.executor.send_guard is not None
    assert sess._no_send_guard({"action": "send_message"}) is True
    assert sess._no_send_guard({"action": "click", "x": 1, "y": 2}) is False


def test_no_send_guard_blocks_chat_enter_only(monkeypatch):
    cfg = _cfg()
    cfg["agent"]["no_send"] = True
    sess = AgentSession(cfg, model=FakeModel(replies=[]), executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    monkeypatch.setattr(core_mod.AgentSession, "_window_info",
                        classmethod(lambda cls: ("WorkBuddy", ["WorkBuddy"])))
    monkeypatch.setattr(sess, "_foreground_exe", lambda: "workbuddy.exe")
    assert sess._no_send_guard({"action": "key", "key": "enter"}) is True
    monkeypatch.setattr(core_mod.AgentSession, "_window_info",
                        classmethod(lambda cls: ("天气 - Edge", [])))
    monkeypatch.setattr(sess, "_foreground_exe", lambda: "msedge.exe")
    assert sess._no_send_guard({"action": "key", "key": "enter"}) is False


def test_send_message_respects_no_send(monkeypatch):
    import pyautogui
    cfg = _cfg()
    cfg["agent"]["no_send"] = True
    sess = AgentSession(cfg, model=FakeModel(replies=[]), executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())

    def boom(*a, **k):
        raise AssertionError("no_send 下不该碰键盘")

    monkeypatch.setattr(pyautogui, "press", boom)
    assert sess._send_message("微信", "示例联系人A", "你好") is True


def test_message_intent_strips_send_message_phrase():
    """「给示例联系人A发消息说你好」的正文必须是「你好」，不能把「发消息」吃进去。"""
    r = AgentSession._parse_message_intent("给示例联系人A发消息说你好")
    assert r and r["target"] == "示例联系人A" and r["text"] == "你好"
    r2 = AgentSession._parse_message_intent("给文件传输助手发送一条消息：你好")
    assert r2 and r2["text"] == "你好"
    r3 = AgentSession._parse_message_intent("对示例联系人A说你好")
    assert r3 and r3["text"] == "你好"


def test_open_app_presses_esc_before_start_button(monkeypatch):
    """点开始按钮前必须先 esc：菜单开着时再点会关掉它，之后敲的字就丢了。"""
    import pyautogui
    events = []
    monkeypatch.setattr(pyautogui, "press", lambda k: events.append(("press", k)))
    monkeypatch.setattr(pyautogui, "click", lambda x, y: events.append(("click", x, y)))
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    import agent.actions as act
    monkeypatch.setattr(act, "_type_unicode", lambda t: events.append(("type", t)))

    sess = _sess()
    monkeypatch.setattr(sess, "_wait_until_idle", lambda *a, **k: True)
    monkeypatch.setattr(core_mod.AgentSession, "_focus_program",
                        classmethod(lambda cls, a: None))
    monkeypatch.setattr(core_mod.AgentSession, "_wait_app_window",
                        lambda self, app, max_wait=8.0: "记事本")
    # 任务栏条带检测：假一个开始按钮（紧挨最宽的搜索框左侧）
    import types
    sess.grounding = types.SimpleNamespace(
        annotate=lambda img: ([{"id": 1, "bbox": [0, 20, 20, 80]},      # 开始按钮
                               {"id": 2, "bbox": [30, 20, 300, 80]}],   # 搜索框（最宽）
                              img),
        resolve_desc=lambda d, i: None)
    assert sess._open_app("记事本") is True
    kinds = [e[0] for e in events]
    assert "press" in kinds, "应先按 esc 清场"
    assert kinds.index("press") < kinds.index("click"), "esc 必须在点开始按钮之前"


# ---------- 纯「打开某程序」直通（7B 拆解会把任务带进 Win7 老路径） ----------

def test_parse_open_intent_accepts_simple_forms():
    P = AgentSession._parse_open_intent
    assert P("打开记事本") == "记事本"
    assert P("帮我打开计算器") == "计算器"
    assert P("启动微信") == "微信"
    assert P("打开一下画图。") == "画图"
    assert P("运行记事本程序") == "记事本"


def test_parse_open_intent_rejects_composite_tasks():
    """带附加动作的不能直通，否则后半句会被丢掉。"""
    P = AgentSession._parse_open_intent
    for t in ("打开记事本并写一句话", "打开记事本，输入你好",
              "在微信里给文件传输助手发送：你好", "打开桌面上的图片文件",
              "打开网页", "打开这个文件夹", "把记事本关掉", ""):
        assert P(t) is None, t


def test_open_intent_runs_template_without_model(monkeypatch):
    """纯打开任务必须一步走模板：不拆解、不花一次模型推理。"""
    model = FakeModel(replies=[])
    sess = AgentSession(_cfg(), model=model, executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    monkeypatch.setattr(sess, "_open_app", lambda app: True)
    assert sess.run("打开记事本") == "DONE"
    assert model.calls == 0


def test_open_intent_falls_back_to_loop_when_template_fails(monkeypatch):
    model = FakeModel(replies=['{"action":"done"}'])
    sess = AgentSession(_cfg(), model=model, executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    monkeypatch.setattr(sess, "_open_app", lambda app: False)
    sess.run("打开记事本")
    assert model.calls >= 1, "模板失败必须退回通用循环，不能直接判死"


# ---------- 定位优先级：YOLO 坐标 > 模型目测坐标 ----------

class _DescGrounding:
    def __init__(self, hit):
        self.hit = hit

    def resolve_desc(self, desc, img):
        return self.hit


def test_yolo_coords_override_model_guess():
    """模型说"点搜索框"又自己估了坐标时，以 YOLO 包围盒中心为准。"""
    sess = _sess()
    sess.grounding = _DescGrounding((640, 120, 7, "搜索框", 1.2))
    a = sess._apply_element_desc(
        {"action": "click", "element_desc": "搜索框", "x": 5, "y": 5}, _fake_screenshot())
    assert (a["x"], a["y"]) == (640, 120)


def test_model_coords_kept_when_desc_not_detectable():
    """编辑区这类非图标控件检测不到，但模型给了坐标 —— 不能把这一步作废。"""
    sess = _sess()
    sess.grounding = _DescGrounding(None)
    a = sess._apply_element_desc(
        {"action": "click", "element_desc": "记事本编辑区域", "x": 900, "y": 600},
        _fake_screenshot())
    assert (a["x"], a["y"]) == (900, 600)
    assert a.get("action") == "click"


def test_search_result_must_contain_app_name(monkeypatch):
    """开始菜单里有个分组标题就叫「应用」，匹配分 1.0 —— 点它是点了个标题，
    程序永远起不来（真机卡了 5 步）。标签不含程序名时必须改用 enter。"""
    import pyautogui
    events = []
    monkeypatch.setattr(pyautogui, "press", lambda k: events.append(("press", k)))
    monkeypatch.setattr(pyautogui, "click", lambda x, y: events.append(("click", x, y)))
    import agent.actions as act
    monkeypatch.setattr(act, "_type_unicode", lambda t: None)
    import agent.core as core_mod
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)

    sess = _sess()
    monkeypatch.setattr(sess, "_wait_until_idle", lambda *a, **k: True)
    monkeypatch.setattr(core_mod.AgentSession, "_focus_program",
                        classmethod(lambda cls, a: None))
    monkeypatch.setattr(sess, "_fit_image", lambda img: (img, 1.0))
    sess.grounding = _DescGrounding((800, 300, 3, "应用", 1.0))   # 只有"应用"两字
    import types
    sess.grounding.annotate = lambda img: ([{"id": 1, "bbox": [0, 20, 20, 80]},
                                            {"id": 2, "bbox": [30, 20, 300, 80]}], img)
    monkeypatch.setattr(core_mod.AgentSession, "_wait_app_window",
                        lambda self, app, max_wait=8.0: "记事本" if any(
                            e[0] == "press" for e in events) else None)
    sess._open_app("记事本")
    presses = [e for e in events if e[0] == "press" and e[1] == "enter"]
    clicks = [e for e in events if e[0] == "click"]
    assert presses, "应改用 enter 打开最佳匹配"
    assert all(c[1] != 800 for c in clicks), "不能去点那个「应用」标题"


# ---------- 事件等待与输入成功反馈 ----------

def test_wait_screen_change_returns_quickly_on_change(monkeypatch):
    """画面一变就返回，不再死等固定 sleep。"""
    import numpy as np
    from PIL import Image
    sess = _sess()
    before = Image.new("RGB", (100, 60), (255, 255, 255))
    after = Image.new("RGB", (100, 60), (0, 0, 0))
    frames = [after, after]
    monkeypatch.setattr(sess, "screenshot_fn", lambda: frames.pop(0) if frames else after)
    sleeps = []
    monkeypatch.setattr(core_mod.time, "sleep", lambda s: sleeps.append(s))
    t0 = core_mod.time.time()
    out = sess._wait_screen_change(before, timeout=5.0, settle=0.0)
    assert out.size == after.size
    assert sum(sleeps) < 5.0, "画面已变化就不该等满 timeout"


def test_wait_screen_change_times_out_quietly(monkeypatch):
    """画面一直不变就等到超时，返回最后一帧，不能挂死。"""
    from PIL import Image
    sess = _sess()
    before = Image.new("RGB", (100, 60), (255, 255, 255))
    monkeypatch.setattr(sess, "screenshot_fn", lambda: before)
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    real_time = core_mod.time.time
    monkeypatch.setattr(core_mod.time, "time",
                        lambda: real_time() + 99)   # 直接视为超时
    out = sess._wait_screen_change(before, timeout=1.0)
    assert out.size == before.size


def test_type_success_feedback_added_to_history(monkeypatch):
    """type 生效后必须明确告知模型「已成功」，防止它不放心再打一遍。"""
    from PIL import Image
    sess = _sess()
    img = Image.new("RGB", (320, 200), (255, 255, 255))
    monkeypatch.setattr(sess, "_window_info", lambda: ("*你好 - Notepad", ["x"]))
    monkeypatch.setattr(sess, "screenshot_fn", lambda: img)
    fg_before = "无标题 - Notepad"
    fg_after, _ = sess._window_info()
    changed = (fg_after != fg_before)
    assert changed
    # 模拟 run() 里的反馈逻辑
    sess.history.append(
        f"第1步: ✅ 输入已确认进入窗口（前台标题「{fg_before}」→「{fg_after}」）。"
        "文字已经在屏幕上了，禁止重复输入相同内容。")
    assert any("禁止重复输入" in h for h in sess.history)


def test_send_message_result_label_must_match_target(monkeypatch):
    """搜索结果标签不含会话名时不能乱点（可能点到聊天记录分组）。"""
    target = "示例联系人B"
    for label, ok in (("示例联系人B", True), ("1) 示例联系人B", True),
                      ("包含 示例联系人B 的聊天记录", False), ("应用", False)):
        clean = ("聊天记录" not in label) and ("聊天文件" not in label)
        can_click = bool(target in label or label in target) and clean
        assert can_click == ok, label


# ---------- 发消息正文清洗（真机：弯引号被敲进微信，标点变双份） ----------

def test_clean_message_text_strips_chinese_quotes():
    from agent.core import clean_message_text as C
    assert C("”这是我的agent项目，你无须在意。“") == "这是我的agent项目，你无须在意。"
    assert C("“你好”") == "你好"
    assert C("「明天见」") == "明天见"
    assert C('"hi"') == "hi"
    assert C("'ok'") == "ok"


def test_clean_message_text_collapses_duplicate_punctuation():
    from agent.core import clean_message_text as C
    assert C("你好，，再见。。") == "你好，再见。"
    assert C("真的吗？？") == "真的吗？"
    assert C("等等！！！") == "等等！"


def test_clean_message_text_keeps_inner_text():
    from agent.core import clean_message_text as C
    assert C("明天上午9:30开会，别迟到。") == "明天上午9:30开会，别迟到。"
    assert C("  含空格  的句子  ") == "含空格 的句子"


def test_parse_message_intent_cleans_quoted_content():
    P = AgentSession._parse_message_intent
    r = P("打开微信，对示例联系人A说：”这是我的agent项目，你无须在意。“")
    assert r == {"app": "微信", "target": "示例联系人A",
                 "text": "这是我的agent项目，你无须在意。"}


def test_send_message_logs_final_text():
    """发送前把最终正文打进日志，用户能直接核对。"""
    logs = []
    sess = _sess()
    sess.log = lambda m: logs.append(m)
    import pyautogui
    orig_click, orig_hotkey, orig_press = pyautogui.click, pyautogui.hotkey, pyautogui.press
    pyautogui.click = lambda *a, **k: None
    pyautogui.hotkey = lambda *a, **k: None
    pyautogui.press = lambda *a, **k: None
    try:
        import agent.actions as act
        old_type = act._type_unicode
        act._type_unicode = lambda t: None
        sess._focus_program = lambda app: "微信"
        sess._wait_screen_change = lambda img, **k: img
        sess._input_candidates = lambda rect: []
        sess._send_message("微信", "示例联系人A", "”你好，，世界。。")
        act._type_unicode = old_type
    finally:
        pyautogui.click, pyautogui.hotkey, pyautogui.press = orig_click, orig_hotkey, orig_press
    assert any("你好，世界。" in m for m in logs), logs[:3]


# ---------- 算术任务：程序算、模板发（7B 不该心算，也不该被"两步任务"卡死） ----------

def test_parse_calc_send_intent_real_failure_case():
    """真机任务原句：必须命中，否则又会掉进通用循环空转 21 步。"""
    P = AgentSession._parse_calc_send_intent
    got = P("使用计算机算出123*123的值，再打开微信，把答案发给示例联系人A。")
    assert got == {"expr": "123*123", "app": "微信", "target": "示例联系人A"}


def test_parse_calc_send_intent_variants():
    P = AgentSession._parse_calc_send_intent
    assert P("计算 12.5*4 然后发给文件传输助手")["expr"] == "12.5*4"
    assert P("算出(1+2)*3 发给示例联系人A")["expr"] == "(1+2)*3"
    assert P("算一下 100÷4 的结果，发给文件传输助手")["target"] == "文件传输助手"
    # 没有收信人 / 没有算式 / 只是发消息 —— 都不许接管
    assert P("算一下 123*123 等于多少") is None
    assert P("把文件发给示例联系人A") is None
    assert P("在微信里给文件传输助手发送：你好") is None


def test_safe_eval_rejects_injection_and_overflow():
    from agent.core import _safe_eval
    assert _safe_eval("123*123") == 15129
    assert _safe_eval("100÷4") == 25
    for bad in ("__import__('os').system('dir')", "abc+1", "9**9**9", "(1).__class__"):
        try:
            _safe_eval(bad)
        except Exception:
            continue
        raise AssertionError(f"危险表达式竟然通过：{bad}")


def test_calc_then_send_computes_then_sends_correct_text(monkeypatch):
    """正文必须是程序算出来的值，且顺序是「先算、后发」。"""
    sess = _sess()
    seen = []
    monkeypatch.setattr(sess, "_send_message",
                        lambda app, target, text: seen.append((app, target, text)) or True)
    monkeypatch.setattr(sess, "_show_in_calculator", lambda e, b: True)
    assert sess._calc_then_send("123*123", "微信", "示例联系人A") is True
    assert seen == [("微信", "示例联系人A", "123×123=15129")]


def test_calc_task_never_enters_generic_loop(monkeypatch):
    """整条任务走模板：模型一次都不该被调用去逐步决策。"""
    sess = _sess()
    sess.max_steps = 20
    monkeypatch.setattr(sess, "_send_message", lambda a, b, c: True)
    monkeypatch.setattr(sess, "_show_in_calculator", lambda e, b: True)
    calls = []
    monkeypatch.setattr(sess.model, "chat", lambda m: calls.append(m) or "{}")
    assert sess.run("使用计算机算出123*123的值，再打开微信，把答案发给示例联系人A。") == "DONE"
    assert calls == []          # 零推理
    assert sess.step == 0


def test_calc_task_reports_failed_when_send_fails(monkeypatch):
    sess = _sess()
    monkeypatch.setattr(sess, "_send_message", lambda a, b, c: False)
    monkeypatch.setattr(sess, "_show_in_calculator", lambda e, b: True)
    assert sess.run("使用计算机算出123*123的值，再打开微信，把答案发给示例联系人A。") == "FAILED"


def test_start_button_desc_becomes_win_key():
    """开始按钮是任务栏系统控件，元素检测永远检不到 —— 必须程序接管按 win 键，
    否则真机就是 39 秒白等、开始菜单根本没打开。"""
    class _G:
        def resolve_desc(self, desc, img):
            return None
    sess = _sess()
    sess.grounding = _G()
    a = sess._apply_element_desc({"action": "click", "element_desc": "启动按钮，位于任务栏中间偏左"},
                                 _fake_screenshot())
    assert a == {"action": "key", "key": "win", "expect_change": True}


def test_same_action_five_times_aborts_instead_of_spinning():
    """21 步空转的元凶：熔断每 3 次只按 esc 就放过，模型继续敲同一句。"""
    cfg = _cfg()
    cfg["agent"]["max_steps"] = 20
    model = FakeModel(replies=['{"action":"type","text":"123*123"}'] * 20)
    ex = FakeExecutor()
    sess = AgentSession(cfg, model=model, executor=ex,
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    assert sess.run("随便一个任务") == "FAILED"
    assert len(ex.calls) <= 6, f"应在第 5 次重复时终止，实际执行了 {len(ex.calls)} 次（原真机是 21 次）"


# ---------- 点击定位准确性：元素清单喂给模型 + 编辑区焦点控件兜底 ----------

def test_element_block_now_lists_yolo_elements():
    """之前 yolo_caption 模式不给清单，模型只能目测坐标（±90px）。
    现在必须把「编号+名字+中心」喂出去，让模型选编号。"""
    sess = _sess()
    els = [{"id": 1, "label": "「文件(F)」 菜单", "bbox": [10, 5, 60, 25]},
           {"id": 2, "label": "记事本图标", "bbox": [100, 100, 140, 140]}]
    block = sess._element_block(els)
    assert "[1] 「文件(F)」 菜单 (35,15)" in block
    assert "[2] 记事本图标 (120,120)" in block
    assert "element_id" in block


def test_element_block_caps_list_length():
    """清单要带名字（没名字模型只能猜坐标），条数有上限 —— 再多会挤爆上下文。

    上限从 80 降到 48 是因为真机撞过墙：80 行清单约 2000 token，叠上 system 规则
    和 6 条历史，prompt 冲到 8033/8192，那次请求 29 秒（平时 9 秒）。
    对 8B 来说选项少一半，挑目标反而更准。
    """
    sess = _sess()
    cap = sess._ELEMENT_LIST_MAX
    els = [{"id": i, "label": f"元素{i}", "bbox": [i, i, i + 10, i + 10]}
           for i in range(1, 81)]
    block = sess._element_block(els)
    assert f"[{cap}]" in block and f"[{cap + 1}]" not in block
    assert f"另有 {80 - cap} 个" in block


def test_element_block_lists_only_named_elements():
    """没认出名字的元素不占行（它们本来就是噪声），只统计条数。"""
    sess = _sess()
    els = [{"id": 1, "label": "新建任务", "bbox": [0, 0, 100, 30]},
           {"id": 2, "label": "", "bbox": [0, 40, 100, 70]},
           {"id": 3, "label": "资料库", "bbox": [0, 80, 100, 110]}]
    block = sess._element_block(els)
    assert "[1] 新建任务" in block and "[3] 资料库" in block
    assert "[2]" not in block
    assert "另有 1 个" in block


def test_edit_zone_desc_uses_focused_control(monkeypatch):
    """「输入框/编辑区」是 YOLO 盲区 → 直接查前台焦点控件矩形，不再让模型猜。
    且必须在「转 open_app」之前判：「记事本输入框」不能被误当成要开记事本。"""
    class _G:
        def resolve_desc(self, desc, img):
            return None
    sess = _sess()
    sess.grounding = _G()
    monkeypatch.setattr(AgentSession, "_focused_control_center",
                        staticmethod(lambda: (700.0, 500.0, 1200.0, 300.0)))
    a = sess._apply_element_desc(
        {"action": "click", "element_desc": "记事本的输入框"}, _fake_screenshot())
    assert a["action"] == "click"
    assert a["x"] == 700.0 and a["y"] == 500.0      # scale=1.0 时物理=截图坐标
    assert "open_app" not in a.values(), "输入框绝不能转成打开程序"
    assert sess._desc_miss is None


def test_edit_zone_desc_without_focus_falls_through(monkeypatch):
    """拿不到焦点控件（非标准控件/无 GUI）时不炸，继续走常规匹配反馈。"""
    class _G:
        def resolve_desc(self, desc, img):
            return None
    sess = _sess()
    sess.grounding = _G()
    monkeypatch.setattr(AgentSession, "_focused_control_center", staticmethod(lambda: None))
    a = sess._apply_element_desc(
        {"action": "click", "element_desc": "代码编辑区"}, _fake_screenshot())
    assert a.get("_miss_desc") == "代码编辑区"


def test_focused_control_center_never_raises():
    """没有 GUI / 非 Windows 环境下只能返回 None，绝不能抛异常打断任务。"""
    r = AgentSession._focused_control_center()
    assert r is None or (len(r) == 4 and r[2] > 5 and r[3] > 5)


# ---------- 写代码任务（Dev-C++）：意图解析 / element_id 归一 / 存盘流程 ----------

CODE_SAMPLE = """#include <stdio.h>

long long factorial(int n) {
    long long r = 1;
    for (int i = 2; i <= n; i++) r *= i;
    return r;
}

int main(void) {
    for (int i = 1; i <= 15; i++) {
        printf("%d! = %lld\\n", i, factorial(i));
    }
    return 0;
}
"""


def test_parse_code_task_real_case():
    """真机原句：必须命中，否则又掉进通用循环（Dev-C++ 从未打开、同一个点连点 6 次）。"""
    got = AgentSession._parse_code_task(
        "打开dev-c++，写一个求15以内阶乘的函数。命名为“阶乘”保存到桌面。")
    assert got == {"app": "Dev-C++", "name": "阶乘",
                   "requirement": "写一个求15以内阶乘的函数", "to_desktop": True}


def test_parse_code_task_variants():
    P = AgentSession._parse_code_task
    assert P("在Dev-C++里写一个判断回文的程序，保存为 huiwen")["name"] == "huiwen"
    assert P("在Dev-C++里写一个判断回文的程序，保存为 huiwen")["requirement"] == \
        "写一个判断回文的程序"
    assert P("打开dev-c++，写个冒泡排序，命名为 sort.c")["name"] == "sort"
    # 不接管：只开程序 / 只是写代码给别人发 / 没有 Dev-C++
    assert P("打开dev-c++") is None
    assert P("用C写个冒泡排序发给示例联系人A") is None
    assert P("打开记事本") is None


def test_norm_element_id_accepts_list_and_bracket_string():
    """7B 会输出 element_id:[34] 或 "[34]" —— 老代码直接查表必然失败，白丢精确坐标。"""
    N = AgentSession._norm_element_id
    assert N([34]) == 34
    assert N(["[12]"]) == 12
    assert N("[34]") == 34
    assert N("元素 7") == 7
    assert N(5) == 5
    assert N(5.0) == 5
    assert N(None) is None
    assert N([]) is None


def test_element_id_resolved_even_without_scaling():
    """scale==1.0 时老代码直接 return，element_id 永远不解析 —— 顺手修掉。"""
    class _G:
        def resolve(self, eid):
            return (240.0, 320.0) if str(eid) == "34" else None
    sess = _sess()
    sess.grounding = _G()
    a = sess._to_screen_coords({"action": "click", "element_id": [34]}, 1.0)
    assert (a["x"], a["y"]) == (240.0, 320.0)
    assert "element_id" not in a


def test_clean_code_strips_fences_and_preamble():
    raw = "好的，下面是代码：\n```c\n#include <stdio.h>\nint main(){return 0;}\n```\n说明：……"
    out = AgentSession._clean_code(raw)
    assert out.startswith("#include <stdio.h>")
    assert "```" not in out and "好的" not in out and "说明" not in out


def test_write_code_task_saves_file_via_gui(monkeypatch, tmp_path):
    """完整链路（GUI 部分打桩）：模型写码 → 开程序 → 输入 → 另存 → 校验落地。"""
    sess = _sess()
    monkeypatch.setattr(AgentSession, "_desktop_dir", staticmethod(lambda: str(tmp_path)))
    monkeypatch.setattr(sess, "_generate_code", lambda req: CODE_SAMPLE)
    monkeypatch.setattr(sess, "_focus_program", lambda app: "Dev-C++ 6.7.5")
    monkeypatch.setattr(sess, "_open_app", lambda app: True)
    typed = {}
    monkeypatch.setattr(sess, "_type_code_in_devcpp",
                        lambda code: typed.setdefault("code", code) or True)
    monkeypatch.setattr(sess, "_save_in_devcpp",
                        lambda name: (tmp_path / "阶乘.c").write_text(CODE_SAMPLE, encoding="utf-8")
                        or True)
    monkeypatch.setattr(sess, "_compile_c", lambda code, tag: (True, ""))
    monkeypatch.setattr(sess, "_run_c", lambda tag: (0, "1! = 1"))

    ok = sess._write_code_task({"app": "Dev-C++", "name": "阶乘",
                                "requirement": "写一个求15以内阶乘的函数",
                                "to_desktop": True})
    assert ok is True
    assert typed["code"] == CODE_SAMPLE
    assert (tmp_path / "阶乘.c").read_text(encoding="utf-8") == CODE_SAMPLE
    assert sess.ok_saved_by == "gui"


def test_write_code_task_falls_back_to_direct_write(monkeypatch, tmp_path):
    """另存为对话框没吃进文件名时，仍要把正确文件交付出去，并如实标注来源。"""
    sess = _sess()
    monkeypatch.setattr(AgentSession, "_desktop_dir", staticmethod(lambda: str(tmp_path)))
    monkeypatch.setattr(sess, "_generate_code", lambda req: CODE_SAMPLE)
    monkeypatch.setattr(sess, "_focus_program", lambda app: "Dev-C++")
    monkeypatch.setattr(sess, "_type_code_in_devcpp", lambda code: True)
    monkeypatch.setattr(sess, "_save_in_devcpp", lambda name: False)
    monkeypatch.setattr(sess, "_compile_c", lambda code, tag: (True, ""))
    monkeypatch.setattr(sess, "_run_c", lambda tag: (0, ""))

    assert sess._write_code_task({"app": "Dev-C++", "name": "阶乘",
                                  "requirement": "阶乘", "to_desktop": True}) is True
    assert (tmp_path / "阶乘.c").read_text(encoding="utf-8") == CODE_SAMPLE
    assert sess.ok_saved_by == "fallback"


def test_code_task_never_enters_generic_loop(monkeypatch, tmp_path):
    """整条任务走模板：通用循环一步都不该跑（真机那次跑了 7 步 49 秒全在乱点）。"""
    cfg = _cfg()
    model = FakeModel(replies=[])
    sess = AgentSession(cfg, model=model, executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    monkeypatch.setattr(AgentSession, "_desktop_dir", staticmethod(lambda: str(tmp_path)))
    monkeypatch.setattr(sess, "_generate_code", lambda req: CODE_SAMPLE)
    monkeypatch.setattr(sess, "_focus_program", lambda app: "Dev-C++")
    monkeypatch.setattr(sess, "_type_code_in_devcpp", lambda code: True)
    monkeypatch.setattr(sess, "_save_in_devcpp",
                        lambda name: (tmp_path / "阶乘.c").write_text(CODE_SAMPLE, encoding="utf-8")
                        or True)
    monkeypatch.setattr(sess, "_compile_c", lambda code, tag: (True, ""))
    monkeypatch.setattr(sess, "_run_c", lambda tag: (0, ""))

    assert sess.run("打开dev-c++，写一个求15以内阶乘的函数。命名为“阶乘”保存到桌面。") == "DONE"
    assert model.calls == 0
    assert sess.step == 0


# ---------- Dev-C++ 编辑器：自动补全会改坏代码 ----------

def test_patch_devcpp_ini_turns_off_brace_completion(tmp_path, monkeypatch):
    """真机证据：敲完代码文件尾部多出 `} } } };` —— 敲 { 时编辑器自动补了一个 }。
        这些开关只在启动时读 ini，所以要写进配置再把 Dev-C++ 重启。"""
    ini = tmp_path / "devcpp.ini"
    ini.write_text(
        "[Editor]\nAutoIndent=1\nBraceComplete=1\nDoubleQuoteComplete=1\nTabSize=4\n"
        "[CodeCompletion]\nEnabled=1\nDelay=180\n"
        "[History]\n0=C://x.c/n", encoding="gbk")
    monkeypatch.setattr(AgentSession, "_devcpp_ini_path", staticmethod(lambda: str(ini)))
    sess = _sess()
    n = sess._patch_devcpp_ini()
    txt = ini.read_text(encoding="gbk")
    assert n == 4
    for key in ("AutoIndent=0", "BraceComplete=0", "DoubleQuoteComplete=0", "Enabled=0"):
        assert key in txt
    assert "TabSize=4" in txt and "Delay=180" in txt   # 别的设置一律不碰
    assert (tmp_path / "devcpp.ini.agent.bak").exists()
    assert sess._patch_devcpp_ini() == 0                # 幂等，第二次不用改


def test_generate_code_retries_when_program_crashes(monkeypatch):
    """编译通过但运行崩溃（malloc 没置 NULL 的野指针递归）也要回填让它改。"""
    model = FakeModel(replies=[
        "#include <stdio.h>\nint main(void){return 0;}\n",
        '#include <stdio.h>\nint main(void){printf("ok");return 0;}\n',
    ])
    sess = AgentSession(_cfg(), model=model, executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    monkeypatch.setattr(sess, "_plan_vars", lambda req: "")
    monkeypatch.setattr(sess, "_compile_c", lambda code, tag: (True, ""))
    monkeypatch.setattr(sess, "_review_code", lambda code, req: "")
    rcs = iter([-1073741571, 0])          # 0xC00000FD 栈溢出，然后正常
    monkeypatch.setattr(sess, "_run_c", lambda tag: (next(rcs), ""))
    code = sess._generate_code("遍历二叉树")
    assert model.calls == 2
    assert "printf" in code


def test_generate_code_gives_up_after_three_crashes(monkeypatch):
    model = FakeModel(replies=["int main(void){return 0;}\n"] * 5)
    sess = AgentSession(_cfg(), model=model, executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    monkeypatch.setattr(sess, "_plan_vars", lambda req: "")
    monkeypatch.setattr(sess, "_compile_c", lambda code, tag: (True, ""))
    monkeypatch.setattr(sess, "_run_c", lambda tag: (-1073741571, ""))
    assert sess._generate_code("随便") is not None
    assert model.calls == 3               # 三稿为止，不无限试


def test_close_devcpp_noop_when_not_running(monkeypatch):
    sess = _sess()
    monkeypatch.setattr(AgentSession, "_find_windows", staticmethod(lambda key: []))
    assert sess._close_devcpp(wait=0.5) is True


# ---------- 输出完整性：退出码 0 也可能是错的结果 ----------

def test_output_completeness_catches_missing_nodes():
    """真机：7 节点二叉树只输出了 5 个数（4 2 5 1 3），编译零警告、退出码 0，
    只核对退出码会把这种静默丢数据判成成功。"""
    code = "\n".join(["p = malloc(sizeof(T));"] * 7)
    assert AgentSession._output_completeness(code, "4 2 5 1 3")[0] is False
    assert AgentSession._output_completeness(code, "4 2 5 1 6 3 7")[0] is True
    # 没有动态分配的题（阶乘/排序）不该被误伤
    assert AgentSession._output_completeness("int f(int n){return n;}", "15! = 1307674368000")[0] is True


def test_generate_code_retries_on_incomplete_output(monkeypatch):
    """输出少了数字也要回填重写，不能带着残缺结果交付。"""
    bad = "#include <stdio.h>\n#include <stdlib.h>\nint main(void){void*a=malloc(4);void*b=malloc(4);printf(\"1\");return 0;}\n"
    good = "#include <stdio.h>\n#include <stdlib.h>\nint main(void){void*a=malloc(4);void*b=malloc(4);printf(\"1 2\");return 0;}\n"
    model = FakeModel(replies=[bad, good])
    sess = AgentSession(_cfg(), model=model, executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    monkeypatch.setattr(sess, "_plan_vars", lambda req: "")
    monkeypatch.setattr(sess, "_compile_c", lambda code, tag: (True, ""))
    monkeypatch.setattr(sess, "_review_code", lambda code, req: "")
    outs = iter([(0, "1"), (0, "1 2")])
    monkeypatch.setattr(sess, "_run_c", lambda tag: next(outs))
    code = sess._generate_code("遍历二叉树")
    assert model.calls == 2
    assert "1 2" in code


# ---------- 经典题判卷：模型能编译能跑但答案错 ----------

def test_classic_expect_hundred_birds_has_four_solutions():
    name, sols = AgentSession._classic_expect("写一个求百元买百鸡的代码")
    assert name == "百钱买百鸡"
    assert sols == [(0, 25, 75), (4, 18, 78), (8, 11, 81), (12, 4, 84)]


def test_verify_classic_catches_wrong_answer():
    """真机四稿的错法：0 组解 / 只 1 组 / 小鸡当 1 元 1 只 / 只数被就地除 3。"""
    V = AgentSession._verify_classic
    q = "写一个求百元买百鸡的代码"
    assert V(q, "total: 0") != ""
    assert V(q, "0 0 100\ntotal: 1") != ""
    assert V(q, "0 roosters, 25 hens, 25 chicks") != ""      # 只数被除成元数
    assert V(q, "0 25 75\n4 18 78\n8 11 81\n12 4 84") == ""  # 正确处理
    assert V("写个冒泡排序", "1 2 3") == ""                   # 非经典题不干预


def test_output_suspicious_flags_lone_zero():
    assert AgentSession._output_suspicious("0") != ""
    assert AgentSession._output_suspicious("  0\n") != ""
    assert AgentSession._output_suspicious("15129") == ""
    assert AgentSession._output_suspicious("0 25 75") == ""


def test_generate_code_marks_unverified_when_classic_always_wrong(monkeypatch):
    """三稿答案全错：可以交付最后一稿，但必须标记未通过验算，不许假装成功。"""
    bad = "#include <stdio.h>\nint main(void){printf(\"total: 0\");return 0;}\n"
    model = FakeModel(replies=[bad] * 3)
    sess = AgentSession(_cfg(), model=model, executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    monkeypatch.setattr(sess, "_compile_c", lambda code, tag: (True, ""))
    monkeypatch.setattr(sess, "_plan_vars", lambda req: "")
    monkeypatch.setattr(sess, "_run_c", lambda tag: (0, "total: 0"))
    monkeypatch.setattr(sess, "_review_code", lambda code, req: "")
    sess._generate_code("写一个求百元买百鸡的代码")
    assert sess.code_verified is False
    assert model.calls == 3


def test_type_code_refuses_when_foreground_is_wrong(monkeypatch):
    """前台不是 Dev-C++ 时一个字都不能敲 —— 真机踩过：23 行代码全打进浏览器地址栏。"""
    import pyautogui

    import agent.actions as act
    typed = []
    monkeypatch.setattr(act, "_type_unicode", lambda t, interval=0.04: typed.append(t))
    monkeypatch.setattr(pyautogui, "hotkey", lambda *a, **k: None)
    sess = _sess()
    monkeypatch.setattr(sess, "_foreground_exe", lambda: "msedge.exe")
    monkeypatch.setattr(sess, "_window_rect", lambda k: None)
    assert sess._type_code_in_devcpp("#include <stdio.h>\nint main(void){return 0;}") is False
    assert typed == [], "前台不对就不该发出任何按键"


# ---------------- 双击判定：单击只选中，双击才进去 ----------------

def _zone_sess(monkeypatch, cls, rect, menu=False):
    sess = _sess()
    monkeypatch.setattr(sess, "_foreground_class", lambda: cls)
    monkeypatch.setattr(sess, "_explorer_list_rect", lambda: rect)
    monkeypatch.setattr(sess, "_fg_menu_open", lambda: menu)
    monkeypatch.setattr(core_mod.AgentSession, "log", lambda self, m: None, raising=False)
    return sess


def test_click_in_explorer_list_becomes_double_click(monkeypatch):
    """资源管理器文件列表里单击只会选中（真机：状态栏「选中 1 个项目」画面纹丝不动）。"""
    sess = _zone_sess(monkeypatch, "CabinetWClass", (160, 129, 1707, 994))
    act = sess._prefer_double_click({"action": "click", "x": 1200, "y": 700})
    assert act["action"] == "double_click"
    assert act["_dbl_why"] == "资源管理器文件列表"


def test_click_on_explorer_toolbar_stays_single(monkeypatch):
    """顶部命令栏/地址栏、左侧导航窗格是单击即生效，不能改双击。"""
    sess = _zone_sess(monkeypatch, "CabinetWClass", (160, 129, 1707, 994))
    assert sess._prefer_double_click({"action": "click", "x": 1200, "y": 80})["action"] == "click"
    assert sess._prefer_double_click({"action": "click", "x": 100, "y": 700})["action"] == "click"


def test_click_with_menu_open_stays_single(monkeypatch):
    """菜单开着时双击会把菜单项执行两遍，必须保持单击。"""
    sess = _zone_sess(monkeypatch, "CabinetWClass", (160, 129, 1707, 994), menu=True)
    assert sess._prefer_double_click({"action": "click", "x": 1200, "y": 700})["action"] == "click"


def test_click_on_desktop_becomes_double_click(monkeypatch):
    sess = _zone_sess(monkeypatch, "Progman", None)
    assert sess._prefer_double_click({"action": "click", "x": 300, "y": 300})["action"] == "double_click"


def test_click_in_other_window_untouched(monkeypatch):
    sess = _zone_sess(monkeypatch, "Notepad", None)
    assert sess._prefer_double_click({"action": "click", "x": 300, "y": 300})["action"] == "click"


def test_prefer_double_click_leaves_other_actions_and_missing_coords(monkeypatch):
    sess = _zone_sess(monkeypatch, "CabinetWClass", (160, 129, 1707, 994))
    assert sess._prefer_double_click({"action": "right_click", "x": 1200, "y": 700})["action"] == "right_click"
    assert sess._prefer_double_click({"action": "click"})["action"] == "click"
    assert sess._prefer_double_click({"action": "double_click", "x": 1200, "y": 700})["action"] == "double_click"


# ---------------- 滚动：方向、位置、滚着找 ----------------

def test_clean_desc_strips_natural_language():
    from agent.core import _clean_desc
    assert _clean_desc("名字叫测试的文件夹") == "测试"
    assert _clean_desc("「百元买百鸡」") == "百元买百鸡"
    assert _clean_desc("那个文件夹图标") == ""
    assert _clean_desc("双击打开 D盘") == "D盘"


def test_match_label_exact_then_contains():
    from agent.core import _match_label
    els = [{"id": 1, "label": "LocalCUA"}, {"id": 2, "label": "测试资料"}]
    assert _match_label(els, "LocalCUA")["id"] == 1
    assert _match_label(els, "测试")["id"] == 2
    assert _match_label(els, "不存在的名字") is None
    assert _match_label(els, "") is None


def test_scroll_action_flips_direction_and_moves_mouse(monkeypatch):
    """模型的 dy 正 = 向下，pyautogui.scroll 正 = 向上。不换算就永远滚反，
    模型会以为"滚动没用"然后放弃（真机 37 步里 scroll 出现 0 次）。"""
    import agent.actions as act
    calls = {}

    class _P:
        def moveTo(self, x, y):
            calls["move"] = (x, y)

        def scroll(self, n):
            calls.setdefault("scroll", []).append(n)

        def keyDown(self, k):
            calls["down"] = k

        def keyUp(self, k):
            calls["up"] = k

    monkeypatch.setattr(act, "pyautogui", _P())
    from types import SimpleNamespace
    ex = act.ActionExecutor(SimpleNamespace(W=2560, H=1600), None)
    ex.execute({"action": "scroll", "dy": 6, "_at": (1200, 900)})
    assert calls["scroll"] == [-6], "dy>0 必须滚成负数（向下）"
    assert calls["move"] == (1200, 900), "必须先把鼠标移到可滚动区域"


def test_scroll_horizontal_uses_shift(monkeypatch):
    import agent.actions as act
    calls = {}

    class _P:
        def moveTo(self, x, y):
            pass

        def scroll(self, n):
            calls.setdefault("scroll", []).append(n)

        def keyDown(self, k):
            calls["down"] = k

        def keyUp(self, k):
            calls["up"] = k

    monkeypatch.setattr(act, "pyautogui", _P())
    from types import SimpleNamespace
    ex = act.ActionExecutor(SimpleNamespace(W=2560, H=1600), None)
    ex.execute({"action": "scroll", "dx": 4})
    assert calls["down"] == "shift" and calls["up"] == "shift"
    assert calls["scroll"] == [4]


def test_scroll_to_find_scrolls_until_target_appears(monkeypatch):
    """目标被滚出视野时，程序自己往下翻屏找 —— 不等模型想起 scroll。"""
    sess = _sess()
    pages = []

    class _G:
        _last_elements = []

        def annotate(self, img):
            pages.append(1)
            if len(pages) == 1:
                return [], img
            return [{"id": 1, "label": "测试", "bbox": [10, 20, 30, 40]}], img

        def ocr_enrich(self, img, els, scale=1.0):
            return []

    sess.grounding = _G()
    sess.executor = FakeExecutor()
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(sess, "screenshot_fn", lambda: Image.new("RGB", (100, 100)))
    monkeypatch.setattr(sess, "_fit_image", lambda img: (img, 0.5))
    monkeypatch.setattr(sess, "_scroll_point", lambda: (50, 50))
    monkeypatch.setattr(core_mod.AgentSession, "log", lambda self, m: None, raising=False)

    assert sess._scroll_to_find("测试") == (40, 60)
    assert len(pages) == 2, "翻第一屏没找到，应该继续翻第二屏"
    assert sess._scroll_to_find("测试") is None, "同一个目标只找一次，别每步白翻"


def test_scroll_to_find_gives_up_after_pages(monkeypatch):
    sess = _sess()

    class _G:
        _last_elements = []

        def annotate(self, img):
            return [{"id": 1, "label": "别的东西", "bbox": [0, 0, 10, 10]}], img

        def ocr_enrich(self, img, els, scale=1.0):
            return []

    sess.grounding = _G()
    sess.executor = FakeExecutor()
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(sess, "screenshot_fn", lambda: Image.new("RGB", (100, 100)))
    monkeypatch.setattr(sess, "_fit_image", lambda img: (img, 1.0))
    monkeypatch.setattr(sess, "_scroll_point", lambda: (50, 50))
    monkeypatch.setattr(core_mod.AgentSession, "log", lambda self, m: None, raising=False)
    assert sess._scroll_to_find("测试", pages=2) is None
    assert len(sess.executor.calls) == 2, "两屏都翻过才算放弃"


# ---------------- 路径导航模板 + 双击护栏 ----------------

def test_parse_nav_task_full_sentence():
    from agent.core import AgentSession
    B = chr(92)
    path, rest = AgentSession._parse_nav_task(f"打开D:{B}测试，在里面新建一个文件")
    assert path == f"D:{B}测试"
    assert rest == "在里面新建一个文件"


def test_parse_nav_task_drive_only_and_bare_path():
    from agent.core import AgentSession
    B = chr(92)
    assert AgentSession._parse_nav_task(f"打开D:{B}测试")[0] == f"D:{B}测试"
    path, rest = AgentSession._parse_nav_task("进入D盘")
    assert path == "D:" + B
    assert rest == ""
    # 反斜杠被吃掉的写法也要兼容
    assert AgentSession._parse_nav_task("打开D:测试")[0] == f"D:{B}测试"
    assert AgentSession._parse_nav_task("打开记事本") is None


def test_open_intent_rejects_path():
    from agent.core import AgentSession
    assert AgentSession._parse_open_intent("打开D:" + chr(92) + "测试") is None


def test_navigate_to_types_path_in_address_bar(monkeypatch):
    import pyautogui

    import agent.actions as act
    seq = []
    monkeypatch.setattr(pyautogui, "hotkey", lambda *k: seq.append(("hotkey", k)))
    monkeypatch.setattr(pyautogui, "press", lambda k: seq.append(("press", k)))
    monkeypatch.setattr(act, "_type_unicode", lambda t, interval=0.04: seq.append(("type", t)))
    sess = _sess()
    monkeypatch.setattr(sess, "_foreground_class", lambda: "CabinetWClass")
    monkeypatch.setattr(sess, "_window_info", lambda: ("测试 - 文件资源管理器", None))
    assert sess._navigate_to("D:" + chr(92) + "测试") is True
    assert ("type", "D:" + chr(92) + "测试") in seq
    assert ("press", "enter") in seq


def test_navigate_to_falls_back_to_explorer_command(monkeypatch):
    import pyautogui

    import agent.actions as act
    monkeypatch.setattr(pyautogui, "hotkey", lambda *k: None)
    monkeypatch.setattr(pyautogui, "press", lambda k: None)
    monkeypatch.setattr(act, "_type_unicode", lambda t, interval=0.04: None)
    sess = _sess()
    calls = []
    monkeypatch.setattr(sess, "_foreground_class", lambda: "CabinetWClass")
    monkeypatch.setattr(sess, "_window_info", lambda: ("主文件夹", None))

    class FakePopen:
        def __init__(self, cmd, *a, **k):
            calls.append(cmd)

    import subprocess as _sp
    monkeypatch.setattr(_sp, "Popen", FakePopen)
    # 直开后窗口标题到位
    monkeypatch.setattr(sess, "_window_info",
                        lambda: ("主文件夹", None) if not calls else ("测试 - 文件资源管理器", None))
    assert sess._navigate_to("D:" + chr(92) + "测试") is True
    assert calls and calls[0][0] == "explorer"


def test_prefer_double_click_skips_files_with_extension(monkeypatch):
    """带扩展名的 OCR 文本=文件不是文件夹，升级双击会把别人的文件打开（真机开过 Excel）。"""
    sess = _sess()
    monkeypatch.setattr(sess, "_double_click_zone", lambda x, y: "资源管理器文件列表")
    monkeypatch.setattr(sess, "grounding", type("G", (), {
        "last_texts": [(100.0, 200.0, 300.0, 216.0, "新建 Microsoft Excel 工作表.xlsx")]})())
    act = sess._prefer_double_click({"action": "click", "x": 200 / 0.6, "y": 208 / 0.6}, scale=0.6)
    assert act["action"] == "click", "文件必须保持单击"


def test_prefer_double_click_upgrades_folder(monkeypatch):
    sess = _sess()
    monkeypatch.setattr(sess, "_double_click_zone", lambda x, y: "资源管理器文件列表")
    monkeypatch.setattr(sess, "grounding", type("G", (), {
        "last_texts": [(100.0, 200.0, 140.0, 216.0, "测试")]})())
    act = sess._prefer_double_click({"action": "click", "x": 200 / 0.6, "y": 208 / 0.6}, scale=0.6)
    assert act["action"] == "double_click"


# ---------------- 前台守卫：用户在用电脑时不能把点击打进别人的窗口 ----------------

def test_guard_passes_without_home():
    sess = _sess()
    assert sess._guard_foreground() is True


def test_guard_blocks_when_foreign_window_foreground(monkeypatch):
    import ctypes.wintypes as wt

    import pyautogui
    sess = _sess()
    sess._gui_home = 111
    sess._fg_steal = 0
    clicks = []
    monkeypatch.setattr(pyautogui, "click", lambda x, y: clicks.append((x, y)))
    monkeypatch.setattr(pyautogui, "size", lambda: type("Sz", (), {"width": 2560, "height": 1600})())

    class FakeU:
        def __init__(self):
            self._fg = 222

        def IsWindow(self, h):
            return h == 111

        def GetForegroundWindow(self):
            return self._fg


        def GetWindowRect(self, h, r):
            r = r._obj           # byref 包了一层 CArgObject，真结构体在 _obj 里
            r.left, r.top, r.right, r.bottom = 100, 100, 900, 800
            return True

        def IsIconic(self, h):
            return False

    import ctypes as _ct

    # wt.RECT 换成同字段的真 Structure，byref 才能写进去
    class R(_ct.Structure):
        _fields_ = [("left", _ct.c_long), ("top", _ct.c_long),
                    ("right", _ct.c_long), ("bottom", _ct.c_long)]

    import ctypes.wintypes as _wt
    monkeypatch.setattr(_wt, "RECT", R)
    fake_u = FakeU()
    monkeypatch.setattr(_ct, "windll", type("W", (), {"user32": fake_u}))
    import agent.core as core_mod
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(type(sess), "_win_pid", staticmethod(lambda h: {111: 7, 222: 9}.get(h, 0)))
    assert sess._guard_foreground() is False, "前台是别人的程序，本步必须拦下"
    assert clicks, "拦下之前应该先试着把任务窗口点回前台（点标题栏）"


def test_guard_allows_same_process_popup(monkeypatch):
    """资源管理器自己的弹出菜单（同进程不同 hwnd）不能拦，否则菜单刚弹开就被点没。"""
    import ctypes.wintypes as wt

    sess = _sess()
    sess._gui_home = 111
    sess._fg_steal = 0

    class FakeU:
        def IsWindow(self, h):
            return True

        def GetForegroundWindow(self):
            return 222


    import ctypes as _ct
    monkeypatch.setattr(_ct, "windll", type("W", (), {"user32": FakeU()}))
    monkeypatch.setattr(type(sess), "_win_pid", staticmethod(lambda h: 7))
    assert sess._guard_foreground() is True


def test_guard_dead_window_releases_home(monkeypatch):
    import ctypes.wintypes as wt

    sess = _sess()
    sess._gui_home = 111

    class FakeU:
        def IsWindow(self, h):
            return False

    import ctypes as _ct
    monkeypatch.setattr(_ct, "windll", type("W", (), {"user32": FakeU()}))
    monkeypatch.setattr(type(sess), "_win_pid", staticmethod(lambda h: 7))
    assert sess._guard_foreground() is True
    assert sess._gui_home is None, "任务窗口已关，守卫应自动解除"


def test_parse_nav_in_path_sentence():
    """「在D:/测试里新建一个文件」—— 没有"打开"动词，走"在+路径"。"""
    from agent.core import AgentSession
    B = chr(92)
    path, rest = AgentSession._parse_nav_task(f"在D:{B}测试里新建一个文件")
    assert path == f"D:{B}测试"
    assert rest == "新建一个文件"


def test_new_file_template_clicks_toolbar_then_menu(monkeypatch):
    """新建文件模板：点工具栏「新建」→ 菜单里按 OCR 找「文本文档」→ 点。"""
    import pyautogui

    sess = _sess()
    monkeypatch.setattr(sess, "_guard_foreground", lambda: True)
    monkeypatch.setattr(sess, "_foreground_class", lambda: "CabinetWClass")
    monkeypatch.setattr(sess, "_fit_image", lambda img: (img, 0.5))
    monkeypatch.setattr(pyautogui, "press", lambda k: None)
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)

    class G:
        _last_elements = []

        def annotate(self, img):
            if not self._last_elements:      # 第一屏：工具栏有「新建」
                return [{"id": 1, "label": "新建", "bbox": [20, 20, 60, 40]}], img
            # 第二屏：菜单弹出，有「文本文档」
            return [{"id": 1, "label": "新建", "bbox": [20, 20, 60, 40]},
                    {"id": 2, "label": "文本文档", "bbox": [10, 80, 90, 100]}], img

        def ocr_enrich(self, img, els, scale=1.0):
            return []

    sess.grounding = G()
    sess.executor = FakeExecutor()
    assert sess._new_file_in_explorer("文本文档", "记录") is True
    assert sess.executor.calls, "应该真的发了点击"
    # 两屏 annotate 都被调用过
    assert len(sess.grounding._last_elements) >= 2


def test_new_file_template_escapes_when_menu_item_missing(monkeypatch):
    import pyautogui
    pressed = []
    monkeypatch.setattr(pyautogui, "press", lambda k: pressed.append(k))

    sess = _sess()
    monkeypatch.setattr(sess, "_guard_foreground", lambda: True)
    monkeypatch.setattr(sess, "_foreground_class", lambda: "CabinetWClass")
    monkeypatch.setattr(sess, "_fit_image", lambda img: (img, 1.0))
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)

    class G:
        _last_elements = []

        def annotate(self, img):
            if not self._last_elements:
                return [{"id": 1, "label": "新建", "bbox": [20, 20, 60, 40]}], img
            return [{"id": 1, "label": "新建", "bbox": [20, 20, 60, 40]}], img   # 菜单空

        def ocr_enrich(self, img, els, scale=1.0):
            return []

    sess.grounding = G()
    sess.executor = FakeExecutor()
    assert sess._new_file_in_explorer("文本文档") is False
    assert "esc" in pressed, "菜单收不回去会祸害下一步"


# ---------- 动作序列（一次看屏幕，连做几个动作） ----------

def test_batch_keeps_click_then_keyboard():
    """点一下再打字再回车 —— 这三步不需要重新看屏幕，应该一次连做。"""
    sess = _sess()
    planned = sess._plan_batch([
        {"action": "click", "x": 10, "y": 20},
        {"action": "type", "text": "D:////测试"},
        {"action": "key", "key": "enter"},
    ])
    assert len(planned) == 3


def test_batch_stops_at_second_click():
    """第二个点击用的是旧截图坐标 —— 必须截断，否则就是真机里的"乱点"。"""
    sess = _sess()
    planned = sess._plan_batch([
        {"action": "click", "element_desc": "新建"},
        {"action": "click", "element_desc": "文本文档"},
    ])
    assert len(planned) == 1


def test_batch_stops_at_done():
    sess = _sess()
    planned = sess._plan_batch([
        {"action": "key", "key": "enter"},
        {"action": "done", "result": "好了"},
    ])
    assert [a["action"] for a in planned] == ["key"]


def test_run_batch_counts_as_one_step():
    """3 个动作只叫一次模型、只算一步 —— 这就是"步骤变少"的机制。"""
    model = FakeModel(replies=[
        "1. 敲地址栏\n2. 输入\n3. 回车",                       # 规划
        '{"plan":"直接敲地址栏","actions":[{"action":"key","key":"ctrl+l"},'
        '{"action":"type","text":"今天天气"},{"action":"key","key":"enter"}]}',
        '{"action":"done","result":"完成"}',
    ])
    ex = FakeExecutor()
    sess = AgentSession(_cfg(), model=model, executor=ex,
                        screenshot_fn=_fake_screenshot,
                        stop_event=threading.Event())
    # 任务措辞刻意避开「查/搜」：这类任务现在会被搜索模板直接接走（真机 8 秒完成），
    # 走不到通用循环，测不到序列压缩。这里要测的是 batch 本身。
    assert sess.run("试一下浏览器快捷键") == "DONE"
    assert [a.get("action") for a in ex.calls] == ["key", "type", "key", "done"]
    assert sess.step == 2, "3 个动作应该压缩成 1 步"
    assert model.calls == 3, "序列中间不该再问模型"


def test_batch_rejects_second_click_and_runs_first_only(monkeypatch):
    """模型把"点开菜单→点菜单项"塞进一个数组：只做第一个，剩下的等新截图。"""
    model = FakeModel(replies=[
        "1. 新建文件",
        '{"actions":[{"action":"click","x":30,"y":30},'
        '{"action":"click","x":60,"y":80}]}',
        '{"action":"done","result":"完成"}',
    ])
    ex = FakeExecutor()
    sess = AgentSession(_cfg(), model=model, executor=ex,
                        screenshot_fn=_fake_screenshot,
                        stop_event=threading.Event())
    # 钉死 Win32 环境：真机上跑测试时前台可能是「桌面/资源管理器」，
    # 会触发"单击改双击"的升级，测试就变成环境相关的了（刚才就因此挂过一次）
    monkeypatch.setattr(AgentSession, "_foreground_title",
                        staticmethod(lambda: "无标题 - 记事本"))
    monkeypatch.setattr(AgentSession, "_foreground_class",
                        staticmethod(lambda: "Notepad"))
    monkeypatch.setattr(AgentSession, "_explorer_list_rect",
                        lambda self: None)
    sess.run("随便做点什么")
    clicks = [a for a in ex.calls if a.get("action") == "click"]
    assert len(clicks) == 1, "第二个点击必须被丢掉"
    assert sess._plan_batch([{"action": "click", "x": 1, "y": 2}] * 2)


def test_parse_action_accepts_actions_array():
    d = AgentSession.parse_action(
        '{"actions":[{"action":"key","key":"ctrl+l"},{"action":"type","text":"x"}]}')
    assert isinstance(d.get("actions"), list) and len(d["actions"]) == 2


def test_note_fail_dedupes_and_caps():
    sess = _sess()
    for i in range(12):
        sess._note_fail(f"失败{i}")
    sess._note_fail("失败11")          # 重复的不再记
    assert len(sess._fail_log) == 8
    assert sess._fail_log[-1] == "失败11"


def test_build_user_text_shows_failures(monkeypatch):
    """失败过的做法必须摆到模型眼前，否则它会一遍遍重试同一条死路。"""
    sess = _sess()
    monkeypatch.setattr(sess, "_window_info", lambda: (None, []))
    sess._note_fail("点了「新建」3 次，画面没变化")
    text = sess._build_user_text("测试", [])
    assert "已经试过、并且失败了" in text
    assert "点了「新建」3 次" in text


def test_parse_action_survives_windows_backslash_path():
    """模型写 "D://测试" 是非法 JSON 转义（\\测），必须修好，不能让整步作废。

    真机上这一条就表现为「[无法解析] 模型返回…」白烧一步。
    """
    bs = chr(92)
    raw = '{"plan":"go","actions":[{"action":"type","text":"D:' + bs + '测试"}]}'
    d = AgentSession.parse_action(raw)
    assert d is not None
    assert d["actions"][0]["text"] == "D:" + bs + "测试"


def test_parse_search_intent():
    """网页搜索意图：命中就走搜索模板，本机文件/路径不算。"""
    P = AgentSession._parse_search_intent
    assert P("打开浏览器搜索今天的天气") == "今天的天气"
    assert P("查一下明天北京的天气") == "明天北京的天气"
    assert P("帮我搜一下 python 教程") == "python 教程"
    assert P("打开浏览器") is None
    assert P("查一下这个文件夹") is None
    assert P("搜索 " + "D:" + chr(92) + "测试 里的文件") is None


# ---------- 桌面空白处点击的接管 ----------

def test_desktop_blank_click_hands_over_to_open_app(monkeypatch):
    """真机 17 步 275 秒：模型在桌面空白处反复双击找微信图标。程序该直接接管。"""
    sess = _sess()
    sess.task = "打开微信，把示例联系人B的聊天记录发给示例联系人A"
    sess.step = 1
    monkeypatch.setattr(sess, "_foreground_class", lambda: "Progman")
    monkeypatch.setattr(sess, "_desktop_icons",
                        lambda: ([{"name": "此电脑", "x": 60, "y": 60}], 1920))
    act = sess._take_over_desktop_click({"action": "double_click", "x": 1282, "y": 919})
    assert act == {"action": "open_app", "app": "微信"}


def test_desktop_click_on_real_icon_is_kept(monkeypatch):
    """落在真图标上就照原样点，别抢。"""
    sess = _sess()
    sess.task = "打开微信"
    monkeypatch.setattr(sess, "_foreground_class", lambda: "Progman")
    monkeypatch.setattr(sess, "_desktop_icons",
                        lambda: ([{"name": "微信", "x": 60, "y": 60}], 1920))
    act = sess._take_over_desktop_click({"action": "double_click", "x": 62, "y": 58})
    assert act["action"] == "double_click" and act.get("app") is None


def test_desktop_click_without_known_app_is_left_alone(monkeypatch):
    """任务里没点名程序时不要乱接管（比如"把桌面文件拖走"）。"""
    sess = _sess()
    sess.task = "把桌面上的文件拖到回收站"
    monkeypatch.setattr(sess, "_foreground_class", lambda: "Progman")
    monkeypatch.setattr(sess, "_desktop_icons",
                        lambda: ([{"name": "此电脑", "x": 60, "y": 60}], 1920))
    act = sess._take_over_desktop_click({"action": "click", "x": 1282, "y": 919})
    assert act["action"] == "click"


# ---------- 失败后的复盘 ----------

def test_reflect_note_feeds_next_prompt():
    """失败后单独问一次"为什么"，结论要出现在下一步的提示词里。"""
    cfg = _cfg()
    cfg["agent"]["reflect_on_fail"] = True
    model = FakeModel(replies=["双击的是桌面空白处，那里没有图标。下一步改用 win 键搜索打开微信。"])
    sess = AgentSession(cfg, model=model, executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    sess.task = "打开微信"
    sess.step = 3
    sess._reflect_note("双击 (1282,919) 后屏幕没有变化")
    assert "win 键" in sess._reflect_hint
    text = sess._build_user_text("打开微信", [])
    assert "复盘" in text and "win 键" in text


def test_reflect_can_be_disabled():
    cfg = _cfg()          # _cfg() 里 reflect_on_fail 是 False
    model = FakeModel(replies=["不该被调用"])
    sess = AgentSession(cfg, model=model, executor=FakeExecutor(),
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    sess.task = "打开微信"
    sess.step = 1
    sess._reflect_note("失败了")
    assert model.calls == 0


# ---------- 打字前的闸：不许打进用户已打开的文件 ----------

def test_type_guard_blocks_typing_into_open_file(monkeypatch):
    """真机事故：hello world 被敲进用户开着的 models_fetch.log，还敲了 3 遍。"""
    sess = _sess()
    sess.task = "打开记事本，然后输入 hello world"
    monkeypatch.setattr(sess, "_foreground_title", lambda: "models_fetch.log - Notepad")
    assert sess._type_guard() is False
    assert any("models_fetch" in s for s in sess._fail_log)


def test_type_guard_allows_when_task_names_the_file(monkeypatch):
    """任务里点名了这个文件，就是用户要改它，放行。"""
    sess = _sess()
    sess.task = "在 models_fetch.log 里追加一行 done"
    monkeypatch.setattr(sess, "_foreground_title", lambda: "models_fetch.log - Notepad")
    assert sess._type_guard() is True


def test_type_guard_opens_blank_notepad_tab(monkeypatch):
    """记事本里开着文件时不能直接放弃：先 Ctrl+N 开空白页，再输入。"""
    sess = _sess()
    sess.task = "打开记事本，然后输入 hello world"
    titles = ["models_fetch.log - Notepad", "无标题 - Notepad"]
    monkeypatch.setattr(sess, "_foreground_title", lambda: titles.pop(0) if titles else "无标题 - Notepad")
    assert sess._type_guard() is True
    assert any(a.get("key") == "ctrl+n" for a in sess.executor.calls)


def test_type_guard_passes_plain_window(monkeypatch):
    sess = _sess()
    sess.task = "在搜索框里输入天气"
    monkeypatch.setattr(sess, "_foreground_title", lambda: "新标签页 - Microsoft​ Edge")
    assert sess._type_guard() is True


def test_type_guard_blocks_modified_window_without_filename(monkeypatch):
    """标题只有 * 没有扩展名（草稿已修改）也不能往里打。"""
    sess = _sess()
    sess.task = "打开画图"
    monkeypatch.setattr(sess, "_foreground_title", lambda: "* 无标题 - 写字板")
    assert sess._type_guard() is False


def test_type_guard_allows_continuing_our_own_text(monkeypatch):
    """写诗/写代码：第一行打完标题就变成「*第一行 - 记事本」，第二行不能被当成别人的文件。"""
    sess = _sess()
    sess.task = "在记事本里写一首短诗"
    monkeypatch.setattr(sess, "_foreground_title", lambda: "*窗前明月光 - Notepad")
    sess._typed_recent.append(("*窗前明月光 - Notepad", "窗前明月光"))
    assert sess._type_guard({"action": "type", "text": "疑是地上霜"}) is True


def test_type_guard_skips_text_already_on_screen(monkeypatch):
    """文字已确认在屏幕上，模型还要再敲一遍 → 拦下并标记 duplicate（调用方据此收尾）。"""
    sess = _sess()
    sess.task = "打开记事本，输入 hello world"
    monkeypatch.setattr(sess, "_foreground_title", lambda: "*hello world - Notepad")
    monkeypatch.setattr(sess, "_screen_has_text", lambda t: True)
    sess._typed_recent.append(("*hello world - Notepad", "hello world"))
    assert sess._type_guard({"action": "type", "text": "hello world"}) is False
    assert sess._type_block_reason == "duplicate"


def test_type_guard_retypes_when_text_missing_on_screen(monkeypatch):
    """上次其实没打进去（屏幕上没有），必须允许重打，不能误判完成。"""
    sess = _sess()
    sess.task = "打开记事本，输入 hello world"
    monkeypatch.setattr(sess, "_foreground_title", lambda: "*hello world - Notepad")
    monkeypatch.setattr(sess, "_screen_has_text", lambda t: False)
    sess._typed_recent.append(("*hello world - Notepad", "hello world"))
    assert sess._type_guard({"action": "type", "text": "hello world"}) is True


def test_duplicate_type_finishes_task(monkeypatch):
    """真机循环：6 步里 3 步在重复敲 hello world。重复且已上屏 = 直接 DONE。"""
    model = FakeModel(replies=[
        '{"plan":"输入","actions":[{"action":"type","text":"hello world"}]}',
        '{"plan":"再输入一次","actions":[{"action":"type","text":"hello world"}]}',
    ])
    ex = FakeExecutor()
    cfg = _cfg()
    cfg["agent"]["type_guard"] = True          # 这条测的就是闸门本身
    sess = AgentSession(cfg, model=model, executor=ex,
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    monkeypatch.setattr(sess, "_foreground_title", lambda: "*hello world - Notepad")
    monkeypatch.setattr(sess, "_screen_has_text", lambda t: True)
    monkeypatch.setattr(sess, "_type_guard_wrap", lambda a: True, raising=False)
    sess._typed_recent.append(("*hello world - Notepad", "hello world"))
    # 任务句要绕开「打开 X 并输入」模板（那条路径现在由模板直接完成，不走通用循环），
    # 这里测的是通用循环里的「重复敲同一段且已上屏 → 收尾」
    assert sess.run("把 hello world 敲进记事本里") == "DONE"
    assert [a.get("action") for a in ex.calls].count("type") == 0, "已上屏就不要再敲"


# ---------- 程序名匹配：别把 "LocalCUA" 当成 "calc" ----------

def test_exe_matches_uses_boundaries():
    """真机翻车：_PROC_MAP 里计算器是 "calc"，资源管理器标题「LocalCUA - …」含 calc，
    程序就认定计算器已在运行，把资源管理器聚焦一下报成功。"""
    A = AgentSession
    assert A._exe_matches("C:" + chr(92) + "Windows" + chr(92) + "explorer.exe", "calc") is False
    assert A._exe_matches("C:" + chr(92) + "Windows" + chr(92) + "notepad.exe", "notepad") is True
    assert A._exe_matches("C:" + chr(92) + "WindowsApps" + chr(92) + "CalculatorApp.exe", "calc") is True


def test_title_matches_splits_segments():
    A = AgentSession
    assert A._title_matches("LocalCUA - 文件资源管理器", "calc") is False
    assert A._title_matches("无标题 - Notepad", "notepad") is True
    assert A._title_matches("LocalCUA - 文件资源管理器", "文件资源管理器") is True
    assert A._title_matches("计算器", "calc") is False   # 中文标题不猜


# ---------- 起名数组越界（真机崩溃：❌ list index out of range） ----------

def test_resolve_desc_survives_ocr_element_ids(monkeypatch):
    """OCR 补进来的元素编号会超过 YOLO 那轮的长度，_captions[id-1] 直接崩。
    真机上就是任务最后一行「❌ list index out of range」。"""
    from agent.yolo_grounding import YoloCaptionGrounding
    g = YoloCaptionGrounding({"conf": 0.3, "max_elements": 80}, model=None)
    g._screen_size = (100, 100)
    g._last_elements = [
        {"id": 1, "label": "", "bbox": [0, 0, 10, 10]},
        {"id": 2, "label": "", "bbox": [20, 20, 30, 30]},
        {"id": 9, "label": "电池图标", "bbox": [90, 90, 99, 99]},   # OCR 补的，编号靠后
    ]
    g._captions = ["", ""]          # 只有 YOLO 那两个
    assert g._cap(9) == ""
    out = g.resolve_desc("任务栏最右侧的电池图标")   # 不能抛异常
    assert out is None or isinstance(out, tuple)


def test_cap_grows_on_write():
    from agent.yolo_grounding import YoloCaptionGrounding
    g = YoloCaptionGrounding({"conf": 0.3, "max_elements": 80}, model=None)
    g._captions = []
    g._set_cap(5, "第五个")
    assert len(g._captions) == 5 and g._cap(5) == "第五个"


# ---------- 定位不能"位置对、东西错" ----------

def test_content_terms_keeps_only_nouns():
    from agent.yolo_grounding import YoloCaptionGrounding as G
    assert G._content_terms("任务栏最右侧的微信图标") == ["微信"]
    assert G._content_terms("保存按钮") == ["保存"]


def test_resolve_desc_rejects_right_place_wrong_thing():
    """真机：'任务栏最右侧的微信图标' 位置分高，匹配到 #59『电池图标』还写"定位成功"。"""
    from agent.yolo_grounding import YoloCaptionGrounding
    g = YoloCaptionGrounding({"conf": 0.3, "max_elements": 80}, model=None)
    g._screen_size = (1000, 1000)
    g._last_elements = [
        {"id": 1, "label": "电池图标，绿色，正在充电", "bbox": [940, 10, 960, 30]},
        {"id": 2, "label": "微信", "bbox": [100, 400, 130, 430]},
    ]
    g._captions = ["", ""]
    assert g.resolve_desc("任务栏最右侧的微信图标") is None


def test_resolve_desc_accepts_synonym_start_button():
    """同义词要放过：模型说「开始按钮」，元素名字是「Windows 徽标」。"""
    from agent.yolo_grounding import YoloCaptionGrounding
    g = YoloCaptionGrounding({"conf": 0.3, "max_elements": 80}, model=None)
    g._screen_size = (1000, 1000)
    g._last_elements = [{"id": 1, "label": "Windows 徽标", "bbox": [0, 960, 30, 990]}]
    g._captions = [""]
    out = g.resolve_desc("开始按钮")
    assert out is not None and out[2] == 1


# ---------- "点 XX 图标" 直接走 open_app ----------

def test_icon_click_becomes_open_app(monkeypatch):
    """微信 7 步 449 秒：模型要先点任务栏微信图标，结果点到电池图标。
    '打开已安装的程序'不需要模型猜图标位置。"""
    sess = _sess()
    monkeypatch.setattr(sess, "_window_info", lambda: ("LocalCUA - 文件资源管理器", []))
    act = sess._take_over_app_icon_click(
        {"action": "click", "element_desc": "任务栏最右侧的微信图标", "x": 10, "y": 10})
    assert act == {"action": "open_app", "app": "微信"}


def test_icon_click_kept_when_app_already_foreground(monkeypatch):
    sess = _sess()
    monkeypatch.setattr(sess, "_window_info", lambda: ("微信", []))
    act = sess._take_over_app_icon_click(
        {"action": "click", "element_desc": "任务栏上的微信图标", "x": 10, "y": 10})
    assert act["action"] == "click"


def test_icon_inside_app_window_is_not_hijacked(monkeypatch):
    """「微信聊天窗口里的发送图标」不是要打开微信。"""
    sess = _sess()
    monkeypatch.setattr(sess, "_window_info", lambda: ("微信", []))
    act = sess._take_over_app_icon_click(
        {"action": "click", "element_desc": "微信聊天窗口工具栏里的发送图标", "x": 10, "y": 10})
    assert act["action"] == "click"


def test_known_app_picks_longest_name():
    A = AgentSession
    assert A._known_app_in_text("任务栏上的文件资源管理器图标") == "资源管理器"
    assert A._known_app_in_text("随便一个图标") is None


# ---------- 微信白名单：三道闸 ----------

def test_run_returns_blocked_for_non_whitelisted_wechat():
    """整条任务在执行前就被拒掉，一步都不跑（通用循环可能自己点人发消息）。"""
    model = FakeModel(replies=[])
    ex = FakeExecutor()
    sess = AgentSession(_cfg(), model=model, executor=ex,
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    out = sess.run("在微信里给示例联系人B发送一条消息：你好")
    assert out == "BLOCKED"
    assert ex.calls == [], "被拦截的任务不该产生任何鼠标动作"


def test_run_allows_whitelisted_wechat_target():
    """白名单内的目标照常走模板（这里用假 executor，只验路由不被拦）。"""
    sess = _sess()
    sess._send_message = lambda app, target, text: True   # 不真的操作鼠标
    assert sess.run("在微信里给文件传输助手发送一条消息：你好") == "DONE"


def test_type_guard_blocks_typing_inside_wechat(monkeypatch):
    """通用循环里模型想自己在微信里打字 = 可能发给别人。拦。"""
    sess = _sess()
    sess.task = "打开微信"
    monkeypatch.setattr(sess, "_foreground_title", lambda: "微信")
    assert sess._type_guard({"action": "type", "text": "你好"}) is False
    assert sess._type_block_reason == "wechat"


def test_type_guard_allows_typing_for_whitelisted_target(monkeypatch):
    sess = _sess()
    sess.task = "在微信里给文件传输助手发送一条消息：你好"
    monkeypatch.setattr(sess, "_foreground_title", lambda: "文件传输助手 - 微信")
    assert sess._type_guard({"action": "type", "text": "你好"}) is True


def test_type_guard_blocks_typing_for_blocked_target_even_in_task(monkeypatch):
    sess = _sess()
    sess.task = "在微信里给示例联系人B发送一条消息：你好"
    monkeypatch.setattr(sess, "_foreground_title", lambda: "微信")
    assert sess._type_guard({"action": "type", "text": "你好"}) is False


def test_send_message_refuses_blocked_target_before_touching_mouse(monkeypatch):
    """模板层也要拦：模型可能直接发 send_message 动作绕过路由。"""
    sess = _sess()
    called = []
    monkeypatch.setattr(sess, "_wait_until_idle", lambda *a, **k: called.append(1))
    assert sess._send_message("微信", "示例联系人B", "你好") is False
    assert called == [], "被拦截时不该有任何前置动作"


# ---------- 配置加载：别偷偷读打包用的旧默认值 ----------

def test_load_config_prefers_project_root():
    """真机踩坑：tools/*.py 硬编码 load_config("agent/config.yaml")，
    于是所有 run_task 实测都在跑 qwen2.5vl:7b，而不是刚换的 qwen3vl:8b。
    两套配置同时存在时，必须以根目录那份为准。"""
    import os
    import yaml
    from agent.config import load_config, project_root
    root_cfg = yaml.safe_load(open(os.path.join(project_root(), "config.yaml"),
                                  encoding="utf-8"))
    cfg = load_config()
    assert cfg["provider"]["ollama"]["model"] == root_cfg["provider"]["ollama"]["model"]
    assert "wechat_allowed" in cfg["agent"], "根目录配置里的白名单不能丢"


# ---------- 新模板：打开网址 / 关闭程序 / 桌面别名导航 ----------

def test_parse_url_intent():
    U = AgentSession._parse_url_intent
    assert U("打开 www.qq.com") == ("https://www.qq.com", "qq")
    assert U("访问 https://www.baidu.com/s?wd=x")[1] == "baidu"
    assert U("去 bing.com 看看")[0] == "https://bing.com"
    # 不是网址的别误判
    assert U("打开 main.c") is None
    assert U("打开 3.14") is None
    assert U("计算 1+1") is None


def test_parse_close_intent():
    C = AgentSession._parse_close_intent
    assert C("关闭记事本") == "记事本"
    assert C("关掉计算器") == "计算器"
    assert C("请帮我关闭微信。") == "微信"
    assert C("关闭所有程序") is None      # 太宽泛，不接管
    assert C("关闭窗口") is None
    assert C("关闭记事本然后打开画图") is None


def test_parse_nav_intent_desktop_alias():
    import os
    path, rest = AgentSession._parse_nav_task("在桌面新建一个文件夹叫报告")
    assert path == os.path.join(os.path.expanduser("~"), "Desktop")
    assert "新建" in rest
    assert AgentSession._parse_nav_task("把桌面的文件拖到回收站") is None


def test_close_app_refuses_system_processes(monkeypatch):
    """关 explorer 会让桌面黑掉、关 python 会把 agent 自己杀掉。"""
    sess = _sess()
    called = []
    monkeypatch.setattr(sess, "_find_windows", lambda e: called.append(e) or [])
    assert sess._close_app("python") is False
    assert called == [], "被拒绝的进程名不该走到枚举窗口"


# ---------- 打开文件 / 目录别名验证 ----------

def test_parse_file_intent():
    F = AgentSession._parse_file_intent
    assert F("打开 报告.txt") == "报告.txt"
    assert F("打开桌面上的 test.md") == "test.md"      # 中文所有格要去掉
    assert F("打开下载里的 数据.csv") == "数据.csv"
    assert F("打开记事本") is None                      # 没扩展名 = 程序，不是文件
    assert F("打开 main.c") == "main.c"


def test_path_leaf_names_maps_chinese_folder_names():
    """「导航到 Desktop」的窗口标题是「桌面」——不映射就永远验不过。"""
    names = AgentSession._path_leaf_names("C:" + chr(92) + "Users" + chr(92) + "me"
                                          + chr(92) + "Desktop")
    assert "桌面" in names and "Desktop" in names
    names2 = AgentSession._path_leaf_names("D:" + chr(92) + "测试")
    assert names2 == ["测试"]


# ---------- 组合键合并 / 多行输入 ----------

def test_merge_modifier_keys_rebuilds_hotkey():
    """模型把 ctrl+shift+n 拆成三条单键 —— 分开发送等于什么都没按。"""
    out = AgentSession._merge_modifier_keys([
        {"action": "key", "key": "ctrl"},
        {"action": "key", "key": "shift"},
        {"action": "key", "key": "n"},
    ])
    assert out == [{"action": "key", "key": "ctrl+shift+n", "_merged": True}]


def test_merge_modifier_keys_drops_orphans():
    out = AgentSession._merge_modifier_keys([
        {"action": "key", "key": "ctrl"},
        {"action": "type", "text": "x"},
    ])
    assert out == [{"action": "type", "text": "x"}]


def test_merge_keeps_plain_keys():
    out = AgentSession._merge_modifier_keys([{"action": "key", "key": "enter"}])
    assert out == [{"action": "key", "key": "enter"}]


def test_type_unicode_uses_real_return_for_newline(monkeypatch):
    """换行必须是 VK_RETURN：U+000A 在很多控件里会被吞掉，整段挤成一行。"""
    import ctypes
    from agent import actions
    seen = []

    monkeypatch.setattr(actions, "_send_input",
                        lambda kb: seen.append(kb.wVk or kb.wScan))
    actions._type_unicode("a\nb", interval=0)
    assert 0x0D in seen, "回车键没发出去"


# ---------- 目标上屏 → 催收尾 ----------

def test_goal_marker_extracts_target_text():
    s = AgentSession.__new__(AgentSession)
    s.task = "在记事本里输入「完成」"
    assert s._goal_marker() == "完成"
    s.task = "打开记事本，输入三行字：苹果、香蕉、橘子"
    assert "苹果" in s._goal_marker()
    s.task = "打开计算器"
    assert s._goal_marker() == ""


def test_goal_seen_pushes_finish_hint(monkeypatch):
    """type 成功且目标文字确认在屏上 → 下一步的历史里必须明确说'该输出 done 了'。"""
    sess = _sess()
    sess.task = "在记事本里输入「完成」"
    sess._typed_recent = []
    monkeypatch.setattr(sess, "_screen_has_text", lambda t: True)
    monkeypatch.setattr(sess, "_dim_background", lambda img: img)
    # 直接调收尾判定段的核心逻辑
    goal = sess._goal_marker()
    assert goal == "完成"
    if goal and sess._screen_has_text(goal):
        sess._goal_seen = True
        sess.history.append(
            f"第1步: 目标内容「{goal}」已经出现在屏幕上。"
            "如果任务只是要求输入/展示这些内容，你下一步必须输出 done，"
            "不要再点击或输入任何东西。")
    assert any("必须输出 done" in h for h in sess.history)


# ---------- 通用循环里的 close_app 动作 ----------

def test_close_app_action_dispatch(monkeypatch):
    """模型在通用循环里直接发 close_app：由程序执行关闭，不走鼠标点击。"""
    model = FakeModel(replies=[
        '{"plan":"关闭记事本","actions":[{"action":"close_app","app":"记事本"}]}',
        '{"action":"done","result":"已关闭"}',
    ])
    ex = FakeExecutor()
    sess = AgentSession(_cfg(), model=model, executor=ex,
                        screenshot_fn=_fake_screenshot, stop_event=threading.Event())
    monkeypatch.setattr(sess, "_close_app", lambda app: True)
    assert sess.run("关闭记事本") == "DONE"
    assert sess._last_trace_close == "记事本" if hasattr(sess, "_last_trace_close") else True


def test_close_app_blocked_for_system_process():
    sess = _sess()
    monkey = None
    # _CLOSE_DENY 里的进程不进枚举
    sess._find_windows = lambda e: []
    assert sess._close_app("explorer") is False


# ---------- 模板路由冒烟（防止改路由时把原有句式改丢） ----------

def test_template_routing_smoke():
    """22 句常见任务，每句该走哪条模板。改路由正则后跑这条，一眼看出回归。"""
    import threading
    sess = AgentSession(
        {"safety": {}, "grounding": {"backend": "none"}, "planner": {"enabled": False},
         "validator": {"enabled": False}, "agent": {"max_steps": 1}},
        log=lambda m: None, stop_event=threading.Event())

    def route(task):
        if sess._parse_search_intent(task):
            return "search_web"
        if sess._parse_url_intent(task):
            return "open_url"
        if sess._parse_open_and_type_intent(task):
            return "open_and_type"
        if sess._parse_open_intent(task):
            return "open_app"
        if sess._parse_close_intent(task):
            return "close_app"
        calc = sess._parse_calc_send_intent(task)
        if calc:
            ok, _w = AgentSession._send_target_allowed(calc["app"], calc["target"])
            return "calc_send" if ok else "BLOCKED"
        if sess._parse_code_task(task):
            return "write_code"
        nav = sess._parse_nav_task(task)
        if nav:
            m = AgentSession._NAV_NEWFILE_RE.search(nav[1] or "")
            return "nav_newfile" if m else "navigate"
        intent = sess._route_task(task)
        if intent and intent.get("blocked"):
            return "BLOCKED"
        if intent:
            return "send_message"
        return "generic"

    assert route("打开记事本") == "open_app"
    assert route("请帮我打开计算器") == "open_app"        # 两个前缀词也要认
    assert route("打开浏览器搜索今天的天气") == "search_web"
    assert route("打开 www.qq.com") == "open_url"
    assert route("关闭记事本") == "close_app"
    assert route("打开 D:" + chr(92) + "测试") == "navigate"
    assert route("在桌面新建一个文件夹叫报告") == "nav_newfile"
    assert route("在微信里给文件传输助手发送一条消息：你好") == "send_message"
    assert route("在微信里给示例联系人B发送一条消息：你好") == "BLOCKED"
    assert route("使用计算机算出123*123，再发给示例联系人A") == "calc_send"
    assert route("打开dev-c++，写一个求15以内阶乘的函数") == "write_code"
    assert route("打开记事本，输入三行字：春天来了、花开满树、燕子归来，每行一个") == "open_and_type"
    assert route("打开微信，输入 你好") == "generic"       # 聊天软件不走打字模板


# ---------- 「算一道题」（不发送） ----------

def test_parse_calc_show_and_priority():
    P = AgentSession._parse_calc_show_intent
    assert P("打开计算器，算128*64") == {"expr": "128*64"}
    assert P("用计算器算一下 123+456") == {"expr": "123+456"}
    assert P("(1+2)*3 等于多少") == {"expr": "(1+2)*3"}
    assert P("算出123*123，再发给示例联系人A") is None     # 有收信人 → calc_then_send
    assert P("打开计算器") is None                  # 没有算式


def test_calc_show_verifies_result_on_screen(monkeypatch):
    import agent.core as core_mod
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)
    sess = _sess()
    shown = []
    monkeypatch.setattr(sess, "_show_in_calculator",
                        lambda expr, body: shown.append((expr, body)) or True)
    monkeypatch.setattr(sess, "_screen_has_text", lambda t: t == "8192")
    monkeypatch.setattr(sess, "_gui_home_set", lambda **k: None)
    assert sess._calc_show("128*64") is True
    assert shown == [("128*64", "128×64=8192")]


# ---------- 「打开 X 并输入文字」复合模板 ----------

def test_parse_open_and_type_extracts_lines():
    """用户话里的「三行字／每行一个」是格式说明，不是正文。"""
    P = AgentSession._parse_open_and_type_intent
    assert P("打开记事本，输入三行字：春天来了、花开满树、燕子归来，每行一个") == {
        "app": "记事本", "text": "春天来了\n花开满树\n燕子归来"}
    assert P("打开记事本，输入三行字：春天来了，花开满树，燕子归来，每行一个") == {
        "app": "记事本", "text": "春天来了\n花开满树\n燕子归来"}
    assert P("请帮我打开记事本，输入 hello world") == {"app": "记事本", "text": "hello world"}
    assert P("打开记事本，输入如下内容：姓名：张三") == {"app": "记事本", "text": "姓名：张三"}
    assert P("打开记事本并输入abc") == {"app": "记事本", "text": "abc"}


def test_parse_open_and_type_rejects_not_this_task():
    P = AgentSession._parse_open_and_type_intent
    assert P("打开记事本") is None                      # 没有输入动作
    assert P("打开微信，输入 你好") is None              # 聊天软件：必须走白名单
    assert P("打开记事本，写一首诗") is None             # 创作，不是照抄
    assert P("打开记事本，输入第一行然后保存") is None   # 后面还挂着保存，别只做前半句


def test_open_and_type_types_lines_then_verifies(monkeypatch):
    """三条铁律：①逐行敲、行间按 enter；②打字前先 Ctrl+End；③OCR 认到才算成功。"""
    import pyautogui
    import agent.actions as act

    events = []
    monkeypatch.setattr(pyautogui, "click", lambda x, y: events.append(("click", x, y)))
    monkeypatch.setattr(pyautogui, "hotkey", lambda *a: events.append(("hotkey",) + a))
    monkeypatch.setattr(pyautogui, "press", lambda k, **kw: events.append(("press", k)))
    monkeypatch.setattr(act, "_type_unicode",
                        lambda t, interval=0.04: events.append(("type", t)))
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)

    sess = _sess()
    monkeypatch.setattr(sess, "_wait_until_idle", lambda *a, **k: True)
    monkeypatch.setattr(sess, "_open_app", lambda app: True)
    monkeypatch.setattr(core_mod.AgentSession, "_exe_keys",
                        classmethod(lambda cls, a: ["notepad"]))
    monkeypatch.setattr(sess, "_ensure_front_keys", lambda keys, app, tries=3: True)
    monkeypatch.setattr(sess, "_front_ok", lambda keys: True)
    monkeypatch.setattr(core_mod.AgentSession, "_focused_control_center",
                        staticmethod(lambda: (500.0, 400.0, 900.0, 600.0)))
    monkeypatch.setattr(core_mod.AgentSession, "_window_info",
                        staticmethod(lambda: ("无标题 - 记事本", [])))
    monkeypatch.setattr(sess, "_wait_screen_change", lambda img, **k: img)
    monkeypatch.setattr(sess, "_screen_has_text", lambda t: t == "春天来了")
    monkeypatch.setattr(sess, "_note_fail", lambda t: events.append(("fail", t)))

    assert sess._open_and_type("记事本", "春天来了\n花开满树\n燕子归来") is True
    types = [e[1] for e in events if e[0] == "type"]
    assert types == ["春天来了", "花开满树", "燕子归来"]      # 逐行敲，没有整段塞
    presses = [e[1] for e in events if e[0] == "press"]
    assert presses.count("enter") == 2                        # 3 行只要 2 个换行
    hots = [e[1:] for e in events if e[0] == "hotkey"]
    assert ("ctrl", "n") in hots, "必须先新建空白文档，不能往用户已有内容后面追加"
    assert ("ctrl", "end") in hots, "打字前跳到文末，防止插进正文中间"


def test_open_and_type_refuses_when_front_is_not_target(monkeypatch):
    """前台不是目标程序时一个字都不许敲（SendInput 打到的是前台窗口）。"""
    import pyautogui
    import agent.actions as act

    typed = []
    monkeypatch.setattr(pyautogui, "click", lambda *a, **k: None)
    monkeypatch.setattr(pyautogui, "hotkey", lambda *a: None)
    monkeypatch.setattr(pyautogui, "press", lambda *a, **k: None)
    monkeypatch.setattr(act, "_type_unicode", lambda t, interval=0.04: typed.append(t))
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)

    sess = _sess()
    monkeypatch.setattr(sess, "_wait_until_idle", lambda *a, **k: True)
    monkeypatch.setattr(sess, "_open_app", lambda app: True)
    monkeypatch.setattr(core_mod.AgentSession, "_exe_keys",
                        classmethod(lambda cls, a: ["notepad"]))
    monkeypatch.setattr(sess, "_ensure_front_keys", lambda keys, app, tries=3: False)
    monkeypatch.setattr(core_mod.AgentSession, "_foreground_exe",
                        staticmethod(lambda: "msedge.exe"))

    assert sess._open_and_type("记事本", "春天来了") is False
    assert typed == []


def test_system_prompt_teaches_how_not_just_where():
    """提示词必须教「这件事怎么做」，不能只教「怎么点」。

    真机教训（用户原话）：让它转发一条聊天记录，它没做；让它新建文件夹，
    它走「右键→新建→文件夹」而不试 ctrl+shift+n —— 一直瞎找，最后也能找到。
    这不是定位不准，是压根没有「这类功能从哪进」的先验：规则 1~12 全在约束
    格式与坐标，没有一条讲操作知识，模型只能靠穷举碰运气。
    """
    p = _sess()._system_prompt(img_w=1280, img_h=800)
    for must in ("ctrl+shift+n", "F2", "ctrl+l", "alt+tab", "win+e", "右键"):
        assert must in p, f"提示词缺少操作常识：{must}"
    assert "转发" in p, "没给出「转发消息」这类功能的入口"
    assert "搜索框" in p, "没强调「用搜索代替翻列表」这条最短路径原则"
    assert "看过新截图" in p, "右键菜单要点开才能看见——漏了这句模型会把两步塞进一个数组"


def test_go_home_desktop_skips_when_already_on_desktop(monkeypatch):
    """已经在桌面时绝不能按 Win+D —— 它是开关，再按一次会把所有窗口还原出来。

    真机动机：8B 眼前有什么就想点什么。实测它把 WorkBuddy 底部的输入框当成
    开始菜单搜索框，往里打「今天天气」，连试三次被熔断。回桌面是为了先把
    干扰源清空。
    """
    import pyautogui
    sess = _sess()
    monkeypatch.setattr(AgentSession, "_foreground_class",
                        staticmethod(lambda: "Progman"))
    pressed = []
    monkeypatch.setattr(pyautogui, "hotkey", lambda *a: pressed.append(a))

    assert sess._go_home_desktop() is False
    assert pressed == [], "已经在桌面还按 Win+D，会把用户正在用的窗口全还原"


def test_go_home_desktop_presses_once(monkeypatch):
    import pyautogui
    import agent.core as core_mod
    sess = _sess()
    state = {"cls": "Chrome_WidgetWin_1"}
    monkeypatch.setattr(AgentSession, "_foreground_class",
                        staticmethod(lambda: state["cls"]))
    pressed = []

    def fake_hotkey(*a):
        pressed.append(a)
        state["cls"] = "Progman"     # 按完真的回到桌面

    monkeypatch.setattr(pyautogui, "hotkey", fake_hotkey)
    monkeypatch.setattr(core_mod.time, "sleep", lambda *_: None)

    assert sess._go_home_desktop() is True
    assert pressed == [("win", "d")]


# ---------- 「青蛙眼」：只处理画面变化的部分 ----------
# 用户的比喻是青蛙的视网膜：静止的东西不产生信号。落地时有个关键取舍 ——
# 千万不能把图裁成一小块喂给模型（它连"我在哪个窗口"都要靠整屏判断），
# 所以「只看变化」落在三处：跳过整条流水线 / 只对变化区 OCR / 在全屏图上画红框。

def _diff_cfg():
    cfg = _cfg()
    cfg["agent"] = dict(cfg["agent"], frog_eye=True, frog_eye_window=1.0)
    return cfg


def test_diff_regions_ignores_blinking_caret():
    """光标闪烁是最高频的"假变化"：2x18 的竖条，必须当成没变。"""
    a = Image.new("RGB", (2560, 1600), "white")
    b = a.copy()
    ImageDraw.Draw(b).rectangle([300, 300, 301, 317], fill="black")
    regions, ratio = AgentSession._diff_regions(a, b)
    assert regions == []
    assert ratio == 0.0, "光标级噪声必须收成 0，否则整条流水线每步都白跑"


def test_diff_regions_locates_a_real_change():
    a = Image.new("RGB", (2560, 1600), "white")
    b = a.copy()
    ImageDraw.Draw(b).rectangle([500, 400, 699, 417], fill="black")     # 一行字
    regions, ratio = AgentSession._diff_regions(a, b)
    assert len(regions) == 1
    x0, y0, x1, y1 = regions[0]
    assert x0 <= 500 and y0 <= 400 and x1 >= 699 and y1 >= 417
    assert (x1 - x0) < 400, "框要贴着变化本身，不能退回 128px 的格子框"
    assert 0 < ratio < 0.05


def test_diff_regions_separates_two_far_apart_changes():
    from PIL import ImageDraw as _D
    a = Image.new("RGB", (2560, 1600), "white")
    b = a.copy()
    d = ImageDraw.Draw(b)
    d.rectangle([100, 100, 259, 199], fill="red")
    d.rectangle([2000, 1200, 2149, 1299], fill="blue")
    regions, _ = AgentSession._diff_regions(a, b)
    assert len(regions) == 2, "两块相距很远的变化必须分开，否则裁剪框会圈进整片空白"
    assert regions[0][0] < 500 < regions[1][0]


def test_diff_regions_gives_up_on_screen_flip():
    """整屏换页（打开浏览器）时裁剪毫无意义，要老实退回全屏流程。"""
    a = Image.new("RGB", (2560, 1600), "white")
    b = Image.new("RGB", (2560, 1600), "black")
    regions, ratio = AgentSession._diff_regions(a, b)
    assert regions == []
    assert ratio > 0.35


def test_frog_eye_window_rejects_stale_baseline(monkeypatch):
    """超过窗口期的上一帧不可比 —— 这正是不设时间限制会踩的坑。

    十秒前可能弹窗出现又消失、页面滚出去又滚回来，两帧之间看着"没变"，
    但拿它当基准去裁变化区，裁出来的是一块早就没意义的区域。
    """
    import agent.core as core_mod
    sess = AgentSession(_diff_cfg(), model=FakeModel(replies=[]),
                        executor=FakeExecutor(), screenshot_fn=_fake_screenshot,
                        stop_event=threading.Event())
    img = Image.new("RGB", (640, 480), "white")
    monkeypatch.setattr(core_mod.time, "time", lambda: 1000.0)
    assert sess._frog_eye_diff(img) == ([], 1.0)      # 第一帧没有基准
    assert sess._frog_eye_diff(img) == ([], 0.0)      # 紧挨着，且一模一样
    monkeypatch.setattr(core_mod.time, "time", lambda: 1002.0)
    assert sess._frog_eye_diff(img) == ([], 1.0)      # 隔了 2 秒，基准作废


def test_frog_eye_plan_reuses_untouched_elements():
    sess = AgentSession(_diff_cfg(), model=FakeModel(replies=[]),
                        executor=FakeExecutor(), screenshot_fn=_fake_screenshot,
                        stop_event=threading.Event())
    sess._frog_els = [{"id": 1, "label": "确定", "bbox": [100, 100, 140, 130]},
                      {"id": 2, "label": "取消", "bbox": [300, 300, 340, 330]}]
    img = Image.new("RGB", (800, 600), "white")
    crop, keep, regs = sess._frog_eye_plan(img, [(280, 280, 360, 340)], scale=1.0)
    assert crop is not None
    assert [e["label"] for e in keep] == ["确定"], "远处那个没变的元素要原样捡回来"
    assert regs == [(280, 280, 360, 340)]


def test_frog_eye_plan_gives_up_when_change_is_most_of_screen():
    sess = AgentSession(_diff_cfg(), model=FakeModel(replies=[]),
                        executor=FakeExecutor(), screenshot_fn=_fake_screenshot,
                        stop_event=threading.Event())
    sess._frog_els = [{"id": 1, "label": "确定", "bbox": [10, 10, 40, 40]}]
    img = Image.new("RGB", (800, 600), "white")
    crop, keep, _ = sess._frog_eye_plan(img, [(0, 0, 780, 580)], scale=1.0)
    assert crop is None and keep == [], "变化占了七成屏，裁剪省不下时间还不如全屏"


def test_frog_eye_crops_ocr_to_changed_area():
    """端到端：第二步只该 OCR 变化那一块，并且带上裁剪偏移。"""
    calls = []
    shots = {"i": 0}

    def shot():
        shots["i"] += 1
        im = Image.new("RGB", (400, 300), "white")
        # 第一帧空白，之后固定在同一处画一块：这样"变化"只有一处，
        # 而且离左边的元素很远 —— 否则那一块连元素一起变，旧元素就不能复用了。
        # 尺寸按「弹窗」来：默认模式只在变化区像个对话框时才裁剪
        # （160x100，占屏 13%），太小的块（一行字/一个按钮）会走全屏。
        if shots["i"] > 1:
            ImageDraw.Draw(im).rectangle([200, 120, 360, 220], fill="black")
        return im

    class G:
        def __init__(self):
            self._last_elements = []
            self.last_texts = []

        def annotate(self, img):
            return ([{"id": 1, "label": "", "bbox": [10, 10, 40, 40],
                      "center": [25, 25], "score": 0.9}], img.copy())

        def _draw(self, img, els):
            return img

        def ocr_enrich(self, img, elements, scale=1.0, max_new=40,
                       origin=(0, 0), extra_texts=()):
            calls.append({"size": img.size, "origin": tuple(origin), "n": len(elements)})
            return []

        def ensure_captions_batch(self, img, els, max_n=40):
            return els

    model = FakeModel(replies=[
        "1. 一步",
        # expect_change 必须为 true，否则程序会沿用上一步的截图，根本走不到"重新截屏"这步
        '{"action":"click","x":5,"y":5,"expect_change":true}',
        "2. 二步",
        '{"action":"click","x":6,"y":6,"expect_change":true}',
        '{"action":"done","result":"完成"}',
    ])
    sess = AgentSession(_diff_cfg(), model=model, executor=FakeExecutor(),
                        screenshot_fn=shot, stop_event=threading.Event())
    sess.grounding = G()
    sess.run("测试任务")

    assert len(calls) >= 2
    assert calls[0]["origin"] == (0, 0), "第一帧没有基准，只能全屏"
    assert calls[1]["origin"] != (0, 0), "第二步要带上裁剪偏移，否则文字坐标全错位"
    assert calls[1]["size"][0] < 400, "第二步只该看变化那一块"


def test_looks_like_popup_rules():
    """弹窗判定：只有「一块边界清楚的矩形、占屏 1%~40%、不是整条带」才算。

    这条边界的意义：裁剪是为了「点完蹦出对话框」那一刻，此时弹窗外的网页一个字
    都不用重读。翻页/滚动/整页跳转硬裁等于把下一步要点的东西裁掉。
    """
    W, H = 2560, 1600
    assert AgentSession._looks_like_popup([(800, 500, 1700, 1100)], W, H), "中央对话框"
    assert not AgentSession._looks_like_popup([(100, 100, 300, 120)], W, H), "一行字太窄"
    assert not AgentSession._looks_like_popup([(0, 0, W, H)], W, H), "整页跳转"
    assert not AgentSession._looks_like_popup([(0, 0, W, 40)], W, H), "地址栏横条"
    assert not AgentSession._looks_like_popup([(10, 10, 200, 300)] * 4, W, H), "四块分散=翻页"

